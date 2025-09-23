import re

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
