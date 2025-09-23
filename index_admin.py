"""
Lightweight, optional maintenance helpers for a built Chroma index:
- list_sources_in_vs(vs)
- delete_source(vs, source)
- recount_stats(vs)
- rebuild_manifest_from_vs(persist_dir, vs, params)
- add_or_replace_file(vs, uploaded_file, embed_model, chunk_size, chunk_overlap)
"""

from typing import List, Tuple
import os

from langchain_core.documents import Document

# Reuse your own helpers from rag_core (no duplication).
from rag_core import (
    _get_all_docs_from_chroma,   # full-corpus read (best-effort)
    read_pdf_pages, read_docx, read_txt,
    chunk_documents,
    write_manifest,
)

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

    for src, c in chunk_count.items():
        per_file[src] = {
            "pages": len(page_set.get(src, {1})) if page_set.get(src) else 1,
            "chunks": c,
        }

    return {
        "num_docs": len(per_file),
        "num_chunks": sum(chunk_count.values()) if chunk_count else 0,
        "sources": sorted(per_file.keys()),
        "per_file": per_file,
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
