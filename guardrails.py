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
