from typing import List, Tuple
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

from pypdf import PdfReader
from collections import defaultdict  
import io, os, re, math, json, time

# cache BM25 per-vectorstore to avoid re-tokenizing
_BM25_CACHE = {}  # key: id(vs) -> SimpleBM25

_TOKENIZER = re.compile(r"\w+").findall

def _tokenize(text: str):
    return [t.lower() for t in _TOKENIZER(text or "")]

def _doc_key(doc: Document) -> tuple:
    md = doc.metadata or {}
    return (
        md.get("source", ""),
        md.get("page", None),
        (doc.page_content or "")[:64],  # short prefix to keep keys stable
    )

class SimpleBM25:
    def __init__(self, docs: List[Document], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = docs
        self.N = len(docs)
        self.tf = []              # list[dict[token] -> freq]
        self.df = defaultdict(int)
        self.doc_len = []
        for d in docs:
            toks = _tokenize(d.page_content)
            counts = defaultdict(int)
            for t in toks:
                counts[t] += 1
            self.tf.append(counts)
            self.doc_len.append(sum(counts.values()))
            for t in counts.keys():
                self.df[t] += 1
        self.avgdl = (sum(self.doc_len) / self.N) if self.N else 0.0

    def _idf(self, t: str) -> float:
        n = self.df.get(t, 0)
        # slight smoothing to avoid negatives on rare corpora
        return math.log((self.N - n + 0.5) / (n + 0.5) + 1.0)

    def score(self, query: str, top_m: int = 50) -> List[tuple]:
        if not self.docs:
            return []
        q_toks = _tokenize(query)
        scores = [0.0] * self.N
        for i, counts in enumerate(self.tf):
            dl = self.doc_len[i] or 1
            norm = 1 - self.b + self.b * (dl / (self.avgdl or 1))
            s = 0.0
            for t in q_toks:
                f = counts.get(t, 0)
                if not f:
                    continue
                idf = self._idf(t)
                s += idf * ((f * (self.k1 + 1)) / (f + self.k1 * norm))
            scores[i] = s
        idxs = sorted(range(self.N), key=lambda j: scores[j], reverse=True)[:top_m]
        return [(self.docs[j], scores[j]) for j in idxs if scores[j] > 0]

def _rrf_fuse(
    lists: List[List[tuple]],  # each: [(Document, score), ...] in rank order
    rrf_k: int = 60,
    top_k: int = 4,
) -> List[tuple]:
    acc = defaultdict(float)
    pick = {}  # key -> Document (first seen)
    for result_list in lists:
        for rank, (doc, _score) in enumerate(result_list, start=1):
            key = _doc_key(doc)
            acc[key] += 1.0 / (rrf_k + rank)
            if key not in pick:
                pick[key] = doc
    fused = sorted(acc.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    return [(pick[key], fused_score) for key, fused_score in fused]

def _get_all_docs_from_chroma(vs) -> List[Document]:
    """
    Try to read the full corpus from Chroma.
    Falls back to empty list if the backend doesn’t support a full dump.
    """
    try:
        raw = vs.get(include=["documents", "metadatas", "ids"])
    except Exception:
        try:
            raw = vs._collection.get(include=["documents", "metadatas", "ids"])
        except Exception:
            return []
    docs = []
    for text, meta in zip(raw.get("documents", []) or [], raw.get("metadatas", []) or []):
        docs.append(Document(page_content=text or "", metadata=meta or {}))
    return docs

def _ensure_bm25_for_vs(vs, candidate_docs: List[Document] | None = None) -> SimpleBM25:
    """
    Build or return cached BM25 for this vectorstore.
    If candidate_docs is provided, we build BM25 over that pool (fallback).
    """
    if candidate_docs is not None:
        return SimpleBM25(candidate_docs)
    key = id(vs)
    if key in _BM25_CACHE:
        return _BM25_CACHE[key]
    corpus = _get_all_docs_from_chroma(vs)
    bm25 = SimpleBM25(corpus)
    _BM25_CACHE[key] = bm25
    return bm25

# # ---------- HELPERS (stubs) ----------
def read_txt(file_obj: io.BytesIO) -> str:
    """Decode a .txt upload to plain text (UTF-8 fallback)."""
    try:
        file_obj.seek(0) 
        return file_obj.read().decode("utf-8", errors="ignore")
    except Exception:
        return ""

def read_pdf_pages(file_obj: io.BytesIO, name: str):
    """Extract text per page from a text-based PDF (no OCR)."""
    try:
        file_obj.seek(0)
        reader = PdfReader(file_obj)
        out = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                out.append(Document(
                    page_content=text,
                    metadata={"source": name, "page": i + 1}
                ))
        return out
    except Exception:
        return []

def files_to_documents(uploaded_files) -> Tuple[List[Document], List[str]]:
    """Turn Streamlit UploadedFile objects into LangChain Documents.
    Returns: (docs, skipped) where skipped is a list of 'filename — reason'.
    """
    docs: List[Document] = []
    skipped: List[str] = []

    for uf in uploaded_files:
        name = uf.name
        ext = os.path.splitext(name)[1].lower()

        if ext == ".txt":
            text = read_txt(uf)
        elif ext == ".pdf":
            pdf_docs = read_pdf_pages(uf, name)
            if pdf_docs:
                docs.extend(pdf_docs)
                continue
            else:
                skipped.append(f"{name} — no extractable text (empty or parse error)")
                continue
        else:
            skipped.append(f"{name} — unsupported file type ({ext})")
            continue  # ignore other types

        if not text or not text.strip():
            skipped.append(f"{name} — no extractable text (empty or parse error)")
            continue  # skip empty files

        docs.append(
            Document(
                page_content=text,
                metadata={"source": name, "page": 1}
            )
        )
    return docs, skipped

def chunk_documents(docs, size: int, overlap: int):
    """
    Split Documents into overlapping chunks.
    - size: target characters per chunk (e.g., 800)
    - overlap: shared chars between neighboring chunks (e.g., 120)
    Returns a new list[Document] (each chunk keeps metadata).
    """
    if not docs:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", " ", ""],  # coarse → fine
    )
    return splitter.split_documents(docs)

def get_embeddings(model_name: str):
    """Return an OllamaEmbeddings instance."""
    try:
        return OllamaEmbeddings(model=model_name)
    except Exception as e:
        raise RuntimeError(
            f"Could not initialize embeddings for '{model_name}'. "
            f"Make sure you've pulled it with `ollama pull {model_name}`."
        ) from e
    
def build_or_load_vectorstore(chunks, embedding, persist_dir: str):
    """
    Build a Chroma vector store when chunks are provided (embeds & persists).
    If chunks is empty, return a handle to an existing store (no embedding).
    Note: Newer langchain-chroma persists on create; no .persist() method exists.
    """
    if chunks and len(chunks) > 0:
        # Creates/updates a persisted collection immediately.
        return Chroma.from_documents(
            chunks, embedding=embedding, persist_directory=persist_dir
        )
    # Open existing collection (no embed pass)
    return Chroma(embedding_function=embedding, persist_directory=persist_dir)

# --- post-retrieval helpers (UI-agnostic) ---

def normalize_hits(hits):
    """Return list[(Document, score|None)]."""
    out = []
    for item in hits or []:
        if isinstance(item, tuple) and len(item) == 2:
            out.append(item)
        else:
            out.append((item, None))
    return out

def filter_by_score(pairs, threshold: float):
    return [(d, s) for (d, s) in pairs if (isinstance(s, (float, int)) and s >= threshold) or s is None]

def cap_per_source(pairs, cap: int):
    counts, kept = {}, []
    for d, s in pairs:
        src = (d.metadata or {}).get("source", "unknown")
        counts[src] = counts.get(src, 0) + 1
        if counts[src] <= cap:
            kept.append((d, s))
    return kept

def make_chunk_rows(pairs, snippet_len: int = 240):
    """Rows for the Chunk Inspector table (no Streamlit calls)."""
    rows = []
    for i, (doc, score) in enumerate(pairs, start=1):
        meta = doc.metadata or {}
        full = (doc.page_content or "").replace("\n", " ")
        snippet = full[:snippet_len] + ("…" if len(full) > snippet_len else "")
        rows.append({
            "Rank": i,
            "Score": f"{score:.4f}" if isinstance(score, (float, int)) else "—",
            "File": meta.get("source", "unknown"),
            "Page": meta.get("page", ""),
            "Snippet": snippet,
        })
    return rows

def build_citation_tags(docs):
    """Stable, de-duped `source p.X` tags for captions."""
    cited_pairs = []
    for d in docs:
        m = d.metadata or {}
        cited_pairs.append((m.get("source", "unknown"), m.get("page", None)))
    seen = {}
    for src, pg in cited_pairs:
        key = (src, pg)
        if key not in seen:
            seen[key] = None
    ordered = sorted(seen.keys(), key=lambda t: (t[0], t[1] or 0))
    return [f"{src} p.{pg}" if pg else src for src, pg in ordered]

# --- retrieval ---
def retrieve(
    vs,
    query: str,
    k: int,
    mmr_lambda: float = 0.7,
    mode: str = "dense",  # NEW: "dense" | "hybrid"
):
    """
    Return top-k relevant chunks with scores.
    Output: list of (Document, score).
    - dense: current dense-only behavior (similarity + MMR, status quo).
    - hybrid: BM25 + Dense fused via RRF (no extra deps).
    """
    if vs is None:
        return []

    # --- Dense path: keep your existing behavior intact ---
    dense_fetch = max(3 * k, 20)
    try:
        dense_results = vs.similarity_search_with_score(query, k=dense_fetch)
        # dense_results: List[(Document, float)]
    except Exception:
        dense_docs = vs.similarity_search(query, k=dense_fetch)
        dense_results = [(d, 0.0) for d in dense_docs]

    if mode == "dense":
        # Preserve your MMR diversification + score mapping
        retriever = vs.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": max(10, 3 * k), "lambda_mult": mmr_lambda},
        )
        mmr_docs = retriever.invoke(query)  # modern API
        score_map = {d.page_content: s for d, s in dense_results}
        return [(d, score_map.get(d.page_content)) for d in mmr_docs]

    # --- Hybrid path: BM25 + Dense via RRF ---
    # Try BM25 over the full corpus; if empty, build over dense candidates
    full_corpus = _get_all_docs_from_chroma(vs)
    bm25_index = _ensure_bm25_for_vs(
        vs,
        candidate_docs=[d for (d, _s) in dense_results] if not full_corpus else None,
    )
    bm25_top = bm25_index.score(query, top_m=dense_fetch)  # [(Document, score)]

    # RRF uses rank positions, so we can ignore raw score magnitudes
    dense_ranked = [(doc, score) for (doc, score) in dense_results]

    fused = _rrf_fuse([bm25_top, dense_ranked], rrf_k=60, top_k=k)
    return fused

# --- load existing store on app start ---
def load_vectorstore_if_exists(embed_model: str, persist_dir: str):
    """Try to open an existing Chroma store; return vs or None."""
    if not os.path.exists(persist_dir):
        return None
    embedding = OllamaEmbeddings(model=embed_model)
    try:
        return Chroma(embedding_function=embedding, persist_directory=persist_dir)
    except Exception:
        return None
    
# ---------- Functions ----------

def build_index_from_files(
    uploaded_files,
    embed_model: str,
    chunk_size: int,
    chunk_overlap: int,
    persist_dir: str,
    embedding_obj=None,  
):
    """
    Orchestrator for index building:
      1) Convert uploaded files -> Documents
      2) Split documents into chunks
      3) Get embeddings
      4) Build/load Chroma vector store and persist it

    Returns:
        vs: the Chroma vector store object
        stats: dict with summary info
            {'num_docs': int, 'num_chunks': int, 'sources': list[str]}
    """

    docs, skipped = files_to_documents(uploaded_files)
    chunks = chunk_documents(docs, size=chunk_size, overlap=chunk_overlap)
    
    embedding = embedding_obj or get_embeddings(embed_model)
    
    vs = build_or_load_vectorstore(
        chunks=chunks,
        embedding=embedding,
        persist_dir=persist_dir,
    )

    stats = {
    "num_docs": len(docs),
    "num_chunks": len(chunks),
    "sources": list({d.metadata.get("source", "unknown") for d in docs}),
    "skipped_files": skipped,
    "per_file": {}
    }

    # Count pages and chunks per file
    page_count = defaultdict(set)
    chunk_count = defaultdict(int)
    for d in chunks:
        src = d.metadata.get("source", "unknown")
        pg = d.metadata.get("page", None)
        if pg:
            page_count[src].add(pg)
        chunk_count[src] += 1

    for src in stats["sources"]:
        stats["per_file"][src] = {
            "pages": len(page_count[src]) if src in page_count else 1,
            "chunks": chunk_count[src],
        }

    # Attach persist_dir and write a manifest next to the vectors
    stats["persist_dir"] = persist_dir
    write_manifest(
        persist_dir,
        stats,
        params={
            "embed_model": embed_model,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
        },
    )
    return vs, stats


# --- Index management (fresh dir + manifest) ---
def make_fresh_index_dir(base_dir: str) -> str:
    """
    Create a new subfolder under base_dir for a clean index build.
    Example: rag_store/demo/idx_20250919_121530
    Also clears BM25 cache so hybrid retrieval can’t see stale corpora.
    """
    ts = time.strftime("%Y%m%d_%H%M%S")
    new_dir = os.path.join(base_dir, f"idx_{ts}")
    os.makedirs(new_dir, exist_ok=True)
    _BM25_CACHE.clear()
    return new_dir

def _manifest_path(persist_dir: str) -> str:
    return os.path.join(persist_dir, "manifest.json")

def write_manifest(persist_dir: str, stats: dict, params: dict | None = None):
    payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "persist_dir": persist_dir,
        "num_docs": stats.get("num_docs", 0),
        "num_chunks": stats.get("num_chunks", 0),
        "sources": stats.get("sources", []),
        "per_file": stats.get("per_file", {}),
        "params": params or {},
    }
    try:
        with open(_manifest_path(persist_dir), "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception:
        pass  # non-fatal

def read_manifest(persist_dir: str) -> dict | None:
    try:
        with open(_manifest_path(persist_dir), "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None
    
# --- Active pointer (per base) & index discovery ---

def _active_pointer_path(base_dir: str) -> str:
    return os.path.join(base_dir, "_ACTIVE.json")

def save_active_pointer(base_dir: str, index_dir: str) -> None:
    """Persist the active index for this base so the app can auto-load on startup."""
    payload = {"active_index_dir": index_dir}
    try:
        with open(_active_pointer_path(base_dir), "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception:
        pass  # non-fatal

def read_active_pointer(base_dir: str) -> str | None:
    """Return the saved active index dir if it exists and is a valid folder."""
    try:
        with open(_active_pointer_path(base_dir), "r", encoding="utf-8") as f:
            data = json.load(f)
        cand = data.get("active_index_dir")
        if cand and os.path.isdir(cand):
            return cand
    except Exception:
        return None
    return None

def list_index_dirs(base_dir: str) -> list[str]:
    """Return absolute paths to subfolders matching idx_YYYYMMDD_HHMMSS under base_dir."""
    if not os.path.isdir(base_dir):
        return []
    out = []
    for name in os.listdir(base_dir):
        p = os.path.join(base_dir, name)
        if os.path.isdir(p) and name.startswith("idx_"):
            out.append(p)
    return sorted(out)

def find_latest_index_dir(base_dir: str) -> str | None:
    """Pick the latest idx_* folder by lexicographic order (timestamped names)."""
    idxs = list_index_dirs(base_dir)
    return idxs[-1] if idxs else None

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

# --- Citation Enforcer (tiny safety net) ---

# Accept "(anything p.<digits>)" inside parentheses as a valid citation,
# e.g., (myfile.pdf p.3)
_CITATION_RE = re.compile(r"\([^)]+ p\.\d+\)")

def enforce_citation(answer: str, fallback: str) -> str:
    """
    Ensure the final answer contains at least one '(file p.X)'-style citation.
    If none found, return the provided fallback string (same as WU1).
    """
    txt = answer or ""
    if _CITATION_RE.search(txt):
        return txt
    return fallback

