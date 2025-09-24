import re
from langchain_core.documents import Document

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

# --- Guardrail constants & citation check ---

# Short, explicit system rules for generation
GUARDRAIL_SYSTEM = (
    "You answer STRICTLY from the provided context. "
    "If the context is insufficient, reply exactly: "
    "'I don’t have enough context to answer that from the provided documents.' "
    "When you do answer, include at least one citation. "
    "Use (file p.X) for paged sources like PDFs; use (file) for non-paged sources like TXT/DOCX. "
    "Keep answers concise."
)

# Accept a filename with extension in parentheses, with an optional ' p.X' suffix.
# Examples that match:
#   (paper.pdf p.3)   (notes.txt)   (report.docx)   (my file.pdf p.12)
_CITE_RE = re.compile(
    r"\((?:[^)]+\.(?:pdf|txt|docx))(?:\s+p\.\d+)?\)",
    re.IGNORECASE,
)

def has_citation(text: str) -> bool:
    return bool(_CITE_RE.search(text or ""))

FALLBACK_NO_CITATION = (
    "I don’t have enough context to answer that from the provided documents."
)

# --- Prompt-injection scrub (retrieved text) ---

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

def sanitize_chunks(chunks):
    """
    Apply scrub_injection() to each retrieved Document.
    Returns (clean_chunks, telemetry) where telemetry is:
      {'chunks_with_drops': X, 'lines_dropped': Y}
    """
    clean = []
    chunks_with_drops = 0
    lines_dropped = 0

    for ch in chunks:
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