from typing import List
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pypdf import PdfReader
import io
import os

# # ---------- HELPERS (stubs) ----------
def read_txt(file_obj: io.BytesIO) -> str:
    """Decode a .txt upload to plain text (UTF-8 fallback)."""
    try:
        return file_obj.read().decode("utf-8", errors="ignore")
    except Exception:
        return ""

def read_pdf(file_obj: io.BytesIO) -> str:
    """Extract text from a text-based PDF (no OCR)."""
    try:
        reader = PdfReader(file_obj)
        pages = []
        for page in reader.pages:
            text = page.extract_text() or ""
            pages.append(text)
        return "\n\n".join(pages).strip()
    except Exception:
        return ""

def files_to_documents(uploaded_files) -> List[Document]:
    """Turn Streamlit UploadedFile objects into LangChain Documents."""
    docs: List[Document] = []
    for uf in uploaded_files:
        name = uf.name
        ext = os.path.splitext(name)[1].lower()

        if ext == ".txt":
            text = read_txt(uf)
        elif ext == ".pdf":
            text = read_pdf(uf)
        else:
            continue  # ignore other types

        if not text or not text.strip():
            continue  # skip empty files

        docs.append(
            Document(
                page_content=text,
                metadata={"source": name}
            )
        )
    return docs

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

# def _get_embeddings(model_name: str):
#     """Return an OllamaEmbeddings instance."""
#     # TODO M1: implement (lazy import langchain_ollama + langchain_community)
#     return None

# def _build_or_load_vectorstore(chunks: List[Document], embedding, persist_dir: str, overwrite: bool):
#     """Create or load a Chroma vector store and persist it."""
#     # TODO M1: implement; if overwrite=True, clear persist_dir first
#     return None

# ---------- Functions ----------

def build_index_from_files(
    uploaded_files,
    embed_model: str,
    chunk_size: int,
    chunk_overlap: int,
    persist_dir: str,
    overwrite: bool = False,
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

    docs = files_to_documents(uploaded_files)
    chunks = chunk_documents(docs, size=chunk_size, overlap=chunk_overlap)

    # TODO: get embeddings
    embedding = None

    # TODO: build/load vector store
    vs = None

    stats = {
        "num_docs": len(docs),
        "num_chunks": len(chunks),
        "sources": list({d.metadata.get("source", "unknown") for d in docs}),
    }

    return vs, stats
