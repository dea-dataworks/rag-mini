from typing import List, Tuple
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama

from pypdf import PdfReader
import io, os

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
                metadata={"source": name}
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
    """
    if chunks and len(chunks) > 0:
        vs = Chroma.from_documents(
            chunks, embedding=embedding, persist_directory=persist_dir
        )
    else:
        vs = Chroma(embedding_function=embedding, persist_directory=persist_dir)
    vs.persist()
    return vs

# --- retrieval ---
def retrieve(vs, query: str, k: int, mmr_lambda: float = 0.7):
    """
    Return top-k relevant chunks with scores.
    Output: list of (Document, score).
    """
    if vs is None:
        return []

    # First, get top-k via similarity with scores
    results = vs.similarity_search_with_score(query, k=k)

    # Optionally rerank/diversify via MMR (using lambda_mult)
    retriever = vs.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": max(10, 3*k), "lambda_mult": mmr_lambda},
    )
    mmr_docs = retriever.get_relevant_documents(query)

    # Merge scores: map by content
    score_map = {d.page_content: s for d, s in results}
    return [(d, score_map.get(d.page_content)) for d in mmr_docs]

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
    
# --- answering ---    
def build_prompt(context: str, question: str) -> str:
    return (
        "You are a helpful assistant. Answer ONLY from the provided context.\n"
        "Use ALL relevant context; if multiple documents apply, combine them. Answer concisely.\n"
        "If the answer isn't in the context, say you don't know.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    )

def call_llm(
    prompt: str,
    provider: str = "ollama",
    model_name: str = "mistral",
    openai_api_key: str | None = None,
    temperature: float = 0.2,
) -> str:
    """
    Call LLM by provider.
    provider: "ollama" | "openai"
    """
    if provider == "openai":
        if not openai_api_key:
            raise ValueError("OpenAI key not provided.")
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model=model_name, temperature=temperature, api_key=openai_api_key)
    else:
        llm = ChatOllama(model=model_name, temperature=temperature)

    resp = llm.invoke(prompt)
    return getattr(resp, "content", str(resp))

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
    from collections import defaultdict
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
