import re
from langchain_core.documents import Document
from typing import List, Dict, Any, Tuple, Optional

# =========================
# Prompt-injection patterns
# =========================

_BAD_PATTERNS = [
    r"ignore (previous|above) instructions",
    r"^system:\s*", r"^developer:\s*", r"^assistant:\s*",
    r"erase instructions", r"jailbreak", r"prompt injection",
]
_BAD_RE = re.compile("|".join(_BAD_PATTERNS), re.IGNORECASE)

def scrub_context(raw_text: str) -> tuple[str, list[str]]:
    """Drop suspicious lines from context. Returns (clean_text, warnings)."""
    warnings = []
    clean_lines = []
    for ln in (raw_text or "").splitlines():
        if _BAD_RE.search(ln):
            warnings.append(ln.strip()[:160])
            continue
        clean_lines.append(ln)
    return "\n".join(clean_lines), warnings

def empty_or_thin_context(clean_text: str, min_chars: int = 40) -> bool:
    return len((clean_text or "").strip()) < min_chars


# =========================
# Generation guardrails
# =========================

GUARDRAIL_SYSTEM = (
    "You answer STRICTLY from the provided context. "
    "If the context is insufficient, reply exactly: "
    "'I don’t have enough context to answer that from the provided documents.' "
    "When you do answer, include at least one citation. "
    "Use (file p.X) for paged sources like PDFs; use (file) for non-paged sources like TXT/DOCX. "
    "Keep answers concise."
)

# Accept a filename with extension in parentheses, with an optional ' p.X' suffix.
# e.g. (paper.pdf p.3)  (notes.txt)  (report.docx)
_CITE_RE = re.compile(
    r"\((?:[^)]+\.(?:pdf|txt|docx))(?:\s+p\.\d+)?\)",
    re.IGNORECASE,
)

def has_citation(text: str) -> bool:
    return bool(_CITE_RE.search(text or ""))

FALLBACK_NO_CITATION = (
    "I don’t have enough context to answer that from the provided documents."
)

# =========================
# Prompt-injection scrub (per chunk)
# =========================

_SUSPECT = re.compile(
    r"(ignore (previous|above) (instructions|directions)|"
    r"^system:|^developer:|^assistant:|"
    r"erase instructions|jailbreak|prompt injection)",
    re.IGNORECASE | re.MULTILINE,
)

def scrub_injection(text: str, max_line_len: int = 2000) -> tuple[str, int]:
    """
    Heuristic sanitizer for a single chunk:
      • drop lines matching _SUSPECT
      • trim very long lines to max_line_len
      • close a dangling code fence if present
    Returns (clean_text, dropped_count)
    """
    lines = (text or "").splitlines()

    dropped = 0
    kept = []
    for ln in lines:
        if _SUSPECT.search(ln):
            dropped += 1
            continue
        if len(ln) > max_line_len:
            ln = ln[:max_line_len] + " …"
        kept.append(ln)

    clean = "\n".join(kept)
    if clean.count("```") % 2 == 1:
        clean += "\n```"  # close stray fence

    return clean, dropped

def sanitize_chunks(chunks: List[Document]) -> tuple[List[Document], Dict[str, int]]:
    """
    Apply scrub_injection() to each retrieved Document.
    Returns (clean_chunks, telemetry) where telemetry is:
      {'chunks_with_drops': X, 'lines_dropped': Y}
    """
    clean = []
    chunks_with_drops = 0
    lines_dropped = 0

    for ch in chunks or []:
        new_text, dropped = scrub_injection(ch.page_content)
        if dropped:
            chunks_with_drops += 1
            lines_dropped += dropped
        clean.append(Document(page_content=new_text, metadata=ch.metadata))

    telemetry = {
        "chunks_with_drops": chunks_with_drops,
        "lines_dropped": lines_dropped,
    }
    return clean, telemetry


# =========================
# Export guard (provenance)
# =========================

def guard_export_settings(run_settings: dict) -> dict:
    """
    Extra safety check for exporting run_settings.
    Keeps only whitelisted keys and redacts sensitive ones.
    """
    if not run_settings:
        return {}

    allowed = {
        "model", "provider", "top_k", "retrieval_mode",
        "chunk_size", "chunk_overlap", "mmr_lambda",
        "use_history", "max_history_turns",
    }

    safe = {}
    for k, v in run_settings.items():
        kl = str(k).lower()
        if k in allowed:
            safe[k] = v
        elif any(term in kl for term in ["key", "token", "secret"]):
            safe[k] = "[REDACTED]"
        # ignore anything unexpected
    return safe


# =========================
# Guardrail status API
# =========================

# Codes (prefer WARN over BLOCK; only 'no_context' blocks)
# - ok
# - no_context (BLOCK)
# - prompt_injection_warning (WARN)
# - source_conflict (WARN)
# - no_citation (WARN; checked post-LLM)

def _status(code: str, message: str, severity: str = "warn", details: Optional[dict] = None) -> Dict[str, Any]:
    return {
        "code": code,
        "severity": severity,  # "warn" | "block" | "info"
        "message": message,
        "details": details or {},
    }

_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_CONFLICT_MARKERS = re.compile(r"\b(contradict|conflict|disagree|inconsistent)\b", re.IGNORECASE)
_NUM_RE = re.compile(r"\b\d+(?:\.\d+)?\b")

def _detect_possible_conflict(docs: List[Document]) -> bool:
    """Very light heuristic to avoid false positives."""
    if not docs or len(docs) < 2:
        return False

    sources = [ (d.metadata or {}).get("source", "src") for d in docs ]
    distinct_sources = len(set(sources)) >= 2

    # 1) explicit conflict words in any chunk
    if any(_CONFLICT_MARKERS.search((d.page_content or "")) for d in docs):
        return True if distinct_sources else False

    # 2) too many distinct years across top chunks → likely timeline disagreement
    years = set()
    for d in docs:
        years.update(_YEAR_RE.findall(d.page_content or ""))  # findall returns tuples due to group; normalize below
    # normalize tuple matches ('19','90') → just join digits in the text by re-finditer instead
    years = set(m.group(0) for d in docs for m in _YEAR_RE.finditer(d.page_content or ""))

    if distinct_sources and len(years) >= 4:
        return True

    # 3) Large numeric spread across sources (coarse)
    nums_by_src = {}
    for d in docs:
        src = (d.metadata or {}).get("source", "src")
        vals = [float(x) for x in _NUM_RE.findall(d.page_content or "") if len(x) <= 6]  # avoid giant IDs
        if vals:
            nums_by_src.setdefault(src, []).extend(vals)
    if len(nums_by_src) >= 2:
        # compare source medians; if the spread is huge, flag (coarse)
        meds = []
        for v in nums_by_src.values():
            v = sorted(v)
            m = v[len(v)//2]
            meds.append(m)
        if meds and (max(meds) - min(meds)) > 1e6:  # very high spread → likely not comparable; skip
            return False
        if meds and (max(meds) - min(meds)) >= 10 and distinct_sources:
            return True

    return False

def evaluate_guardrails(
    *,
    question: str,
    context_text: str,
    docs_used: List[Document],
    sanitize_telemetry: Optional[Dict[str, int]] = None,
    scrubbed_lines: Optional[List[str]] = None,
    answer_text: Optional[str] = None,
    min_chars: int = 40,
) -> List[Dict[str, Any]]:
    """
    Return a list of guardrail_status dicts (ordered; most important first).
    Caller can decide whether to block on 'no_context' or just warn.
    """
    statuses: List[Dict[str, Any]] = []

    # A) No/weak context → BLOCK
    if empty_or_thin_context(context_text, min_chars=min_chars):
        statuses.append(_status(
            "no_context",
            "Declined — not enough supporting context in your documents.",
            severity="block",
            details={"min_chars": min_chars}
        ))
        # still continue to gather WARNs (so UI can show stacked banners)

    # B) Prompt-injection scrub happened → WARN (only if meaningful drops)
    tel = sanitize_telemetry or {}
    lines_dropped = int(tel.get("lines_dropped", 0))
    chunks_with_drops = int(tel.get("chunks_with_drops", 0))
    if lines_dropped > 0:
        msg = f"Possible prompt injection removed ({lines_dropped} line(s) across {chunks_with_drops} chunk(s))."
        statuses.append(_status(
            "prompt_injection_warning",
            msg,
            severity="warn",
            details={"lines_dropped": lines_dropped, "chunks_with_drops": chunks_with_drops}
        ))

    # C) Scrubbed explicit bad lines during context cleaning → WARN
    if scrubbed_lines:
        statuses.append(_status(
            "prompt_injection_warning",
            "Suspicious lines were removed from the retrieved context.",
            severity="warn",
            details={"examples": scrubbed_lines[:3]}
        ))

    # D) Conflicting sources heuristic → WARN
    try:
        if _detect_possible_conflict(docs_used or []):
            statuses.append(_status(
                "source_conflict",
                "Potential conflict between sources — proceed with caution.",
                severity="warn",
            ))
    except Exception:
        # Never block on a failed heuristic
        pass

    # E) Post-LLM check: missing citation → WARN (only if answer is non-empty)
    if answer_text is not None:
        if (answer_text.strip() != "") and not has_citation(answer_text):
            statuses.append(_status(
                "no_citation",
                "Answer has no explicit citation. Consider citing at least one source.",
                severity="warn",
            ))

    # If nothing triggered, return a single OK
    if not statuses:
        statuses.append(_status("ok", "OK", severity="info"))

    # Re-order: block first, then others (already mostly in that order)
    blocks = [s for s in statuses if s.get("severity") == "block"]
    warns  = [s for s in statuses if s.get("severity") == "warn"]
    infos  = [s for s in statuses if s.get("severity") == "info"]
    return blocks + warns + infos

def pick_primary_status(statuses: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Select the single most important status to headline in the UI."""
    if not statuses:
        return _status("ok", "OK", severity="info")
    for s in statuses:
        if s.get("severity") == "block":
            return s
    for s in statuses:
        if s.get("severity") == "warn":
            return s
    return statuses[0]
