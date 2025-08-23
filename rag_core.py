from typing import List
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama

from pypdf import PdfReader
import io, shutil, os

# # ---------- HELPERS (stubs) ----------
def read_txt(file_obj: io.BytesIO) -> str:
    """Decode a .txt upload to plain text (UTF-8 fallback)."""
    try:
        file_obj.seek(0) 
        return file_obj.read().decode("utf-8", errors="ignore")
    except Exception:
        return ""

def read_pdf(file_obj: io.BytesIO) -> str:
    """Extract text from a text-based PDF (no OCR)."""
    try:
        file_obj.seek(0) 
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

def get_embeddings(model_name: str):
    """Return an OllamaEmbeddings instance."""
    try:
        return OllamaEmbeddings(model=model_name)
    except Exception as e:
        raise RuntimeError(
            f"Could not initialize embeddings for '{model_name}'. "
            f"Make sure you've pulled it with `ollama pull {model_name}`."
        ) from e

def build_or_load_vectorstore(chunks, embedding, persist_dir: str, overwrite: bool):
    """Create or load a Chroma vector store and persist it."""
    if overwrite and os.path.exists(persist_dir):
        shutil.rmtree(persist_dir, ignore_errors=True)
    if chunks and len(chunks) > 0:
        vs = Chroma.from_documents(
            chunks, embedding=embedding, persist_directory=persist_dir
        )
    else:
        vs = Chroma(embedding_function=embedding, persist_directory=persist_dir)
    vs.persist()
    return vs

# --- retrieval ---
def retrieve(vs, query: str, k: int):
    """Return top-k relevant chunks (LangChain Documents)."""
    if vs is None:
        return []
    retriever = vs.as_retriever(
        search_type="mmr",
        search_kwargs={"k": int(k), "fetch_k": max(10, 3*int(k)), "lambda_mult": 0.7},
    )
    return retriever.get_relevant_documents(query)

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

def call_llm(prompt: str, model_name: str = "mistral", temperature: float = 0.2) -> str:
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
    overwrite: bool = False,
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

    docs = files_to_documents(uploaded_files)
    chunks = chunk_documents(docs, size=chunk_size, overlap=chunk_overlap)
    
    embedding = embedding_obj or get_embeddings(embed_model)
    
    vs = build_or_load_vectorstore(
        chunks=chunks,
        embedding=embedding,
        persist_dir=persist_dir,
        overwrite=overwrite,
    )

    stats = {
        "num_docs": len(docs),
        "num_chunks": len(chunks),
        "sources": list({d.metadata.get("source", "unknown") for d in docs}),
    }

    return vs, stats
