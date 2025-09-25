from typing import List, Tuple
import os
from langchain_core.documents import Document

from rag_core import (
    _get_all_docs_from_chroma,   # full-corpus read (best-effort)
    read_pdf_pages, read_docx, read_txt,
    chunk_documents,
    write_manifest,
    # Index pointer helpers + constants
    read_active_pointer, save_active_pointer, find_latest_index_dir,
)
from utils.settings import PERSIST_DIR
# ---------- Public API ----------

def list_sources_in_vs(vs) -> list[str]:
    """Unique 'source' values present in the current collection."""
    docs = _get_all_docs_from_chroma(vs)
    return sorted({(d.metadata or {}).get("source", "unknown") for d in docs})

def delete_source(vs, source: str) -> bool:
    """
    Delete all chunks whose metadata.source == source.
    Returns True if, after the call, no chunks with that source remain.
    """
    try:
        vs.delete(where={"source": source})
    except Exception:
        # Some langchain-chroma versions expose the underlying collection
        try:
            vs._collection.delete(where={"source": source})
        except Exception:
            pass

    # Verify removal via quick scan
    remaining = [d for d in _get_all_docs_from_chroma(vs) if (d.metadata or {}).get("source") == source]
    return len(remaining) == 0

def recount_stats(vs) -> dict:
    """
    Recompute {'num_docs','num_chunks','sources','per_file'} by scanning the collection.
    'num_docs' means distinct files (sources).
    """
    docs = _get_all_docs_from_chroma(vs)
    per_file = {}
    page_set = {}
    chunk_count = {}
    for d in docs:
        md = d.metadata or {}
        src = md.get("source", "unknown")
        pg = md.get("page")
        chunk_count[src] = chunk_count.get(src, 0) + 1
        if pg:
            page_set.setdefault(src, set()).add(pg)

    char_count = {}
    for d in docs:
        src = (d.metadata or {}).get("source", "unknown")
        char_count[src] = char_count.get(src, 0) + len(d.page_content or "")

    for src, c in chunk_count.items():
        total_c = char_count.get(src, 0)
        per_file[src] = {
            "pages": len(page_set.get(src, {1})) if page_set.get(src) else 1,
            "chunks": c,
            "chars": total_c,
            "avg_chunk_len": round(total_c / c, 1) if c else 0,
        }


    total_chars = 0
    for d in docs:
        total_chars += len(d.page_content or "")

    avg_chunk_len = round(total_chars / sum(chunk_count.values()), 1) if chunk_count else 0

    return {
        "num_docs": len(per_file),
        "num_chunks": sum(chunk_count.values()) if chunk_count else 0,
        "sources": sorted(per_file.keys()),
        "per_file": per_file,
        "total_chars": total_chars,
        "avg_chunk_len": avg_chunk_len,
    }


def rebuild_manifest_from_vs(persist_dir: str, vs, params: dict | None = None) -> dict:
    """
    Rewrite manifest.json based on current collection contents.
    Returns the computed stats.
    """
    stats = recount_stats(vs)
    write_manifest(persist_dir, stats, params=params or {})
    return stats

def add_or_replace_file(
    vs,
    uploaded_file,          # Streamlit UploadedFile-like
    embed_model: str,       # kept for symmetry; not used directly here
    chunk_size: int,
    chunk_overlap: int,
) -> Tuple[bool, int]:
    """
    Delete any existing chunks for this file name, then add fresh chunks from the upload.
    Returns (deleted_ok, added_chunks_count).
    """
    name = uploaded_file.name
    # 1) delete old
    del_ok = delete_source(vs, name)

    # 2) read + chunk
    ext = os.path.splitext(name)[1].lower()
    if ext == ".pdf":
        docs = read_pdf_pages(uploaded_file, name)
    elif ext == ".docx":
        text = read_docx(uploaded_file)
        docs = [Document(page_content=text, metadata={"source": name, "ext": "docx"})] if text.strip() else []
    elif ext == ".txt":
        text = read_txt(uploaded_file)
        docs = [Document(page_content=text, metadata={"source": name, "page": 1, "ext": "txt"})] if text.strip() else []
    else:
        docs = []

    chunks = chunk_documents(docs, size=chunk_size, overlap=chunk_overlap) if docs else []
    if not chunks:
        return del_ok, 0

    # 3) add to existing vs
    vs.add_documents(chunks)
    return del_ok, len(chunks)

# ---------- Index pointer API (new) ----------

def normalize_base(root: str, name: str) -> str:
    """
    Join the vector root with a base name.
    name="" means the root itself (default base).
    """
    name = (name or "").strip()
    return os.path.join(root, name) if name else root


def list_indexes(root: str = PERSIST_DIR) -> list[str]:
    """
    Return immediate subfolder names under the vector root that look like bases.
    Hidden folders and files are ignored.
    """
    try:
        entries = os.listdir(root)
    except FileNotFoundError:
        return []

    out = []
    for e in entries:
        if e.startswith("."):
            continue
        full = os.path.join(root, e)
        if os.path.isdir(full):
            out.append(e)
    return sorted(out)


def index_exists(name: str, root: str = PERSIST_DIR) -> bool:
    """Whether a base folder exists under the root."""
    return os.path.isdir(normalize_base(root, name))


def get_active_index(base: str | None = None) -> str | None:
    """
    Given a base folder path (e.g., rag_store/user), return the active sub-index folder.
    Falls back to the latest timestamped index if no pointer exists.
    """
    base = base or PERSIST_DIR
    pointer = read_active_pointer(base)
    if pointer:
        return pointer
    return find_latest_index_dir(base)


def set_active_index(base: str, active_dir: str) -> None:
    """
    Persist the active sub-index pointer for a base.
    """
    save_active_pointer(base, active_dir)
