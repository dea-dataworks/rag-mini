import streamlit as st
from rag_core import build_index_from_files
from rag_core import load_vectorstore_if_exists
from rag_core import retrieve, build_prompt, call_llm

# ---------- CONFIG ----------
APP_TITLE = "🔎 RAG Mini v0.1"
PERSIST_DIR = "rag_store"  
EMBED_MODEL = "nomic-embed-text"

st.set_page_config(page_title="RAG Mini v0.1", layout="wide")
st.title(APP_TITLE)
st.caption("Local, simple Retrieval-Augmented Q&A (scope-first)")


# ---------- HELPERS ----------

@st.cache_resource
def get_cached_embeddings(embed_model: str):
    # calls your rag_core.get_embeddings under the hood
    from rag_core import get_embeddings
    return get_embeddings(embed_model)

if "vs" not in st.session_state:
    st.session_state["vs"] = None

# Try to load an existing store on startup so users can retrieve without re-uploading
if st.session_state["vs"] is None:
    st.session_state["vs"] = load_vectorstore_if_exists(
        embed_model= EMBED_MODEL,  
        persist_dir= PERSIST_DIR,
    )

# ---------- SIDEBAR SETTINGS ----------
with st.sidebar:
    st.header("⚙️ Settings")

    llm_model = st.selectbox("LLM (Ollama)", ["mistral"], index=0, disabled=True)
    embed_model = EMBED_MODEL
    chunk_size = st.number_input("Chunk size", 200, 4000, 800, 50, disabled=True)
    chunk_overlap = st.number_input("Chunk overlap", 0, 1000, 120, 10, disabled=True)
    top_k = st.slider("Top-k retrieval", 1, 10, 4)
    overwrite_store = st.checkbox("Overwrite vector store", value=False, disabled=True)

    st.markdown("---")
    st.markdown("**Implementation steps**")
    st.caption("1) Ingest → chunk → index\n2) Retrieve → answer → cite\n3) UI polish + docs")

# ---------- FILE UPLOAD ----------
st.subheader("1) Upload documents")
uploaded_files = st.file_uploader(
    "Upload .pdf or .txt files",
    type=["pdf", "txt"],
    accept_multiple_files=True,
)

# ---------- INDEX BUILD (M1) ----------
col_a, col_b = st.columns([2, 1])
with col_a:
    build_btn = st.button("🧱 Build / Load Index", type="primary", use_container_width=True)
with col_b:
    st.info("Step 1: Build the index before asking questions.", icon="ℹ️")

if build_btn:
    if not uploaded_files:
        st.warning("Please upload at least one .pdf or .txt file.")
    else:
        with st.spinner("Reading & chunking…"):
            try:
                embedding = get_cached_embeddings(embed_model=EMBED_MODEL)
                vs, stats = build_index_from_files(
                    uploaded_files=uploaded_files,
                    embed_model=embed_model,    
                    chunk_size=800,
                    chunk_overlap=120,
                    persist_dir=PERSIST_DIR,
                    overwrite=overwrite_store,
                    embedding_obj=embedding,
                )
                st.session_state["vs"] = vs
                st.success(f"Index built — docs: {stats['num_docs']}, chunks: {stats['num_chunks']}")
                st.caption(f"Sources: {', '.join(stats['sources']) or 'None'}")
            except Exception as e:
                st.error(f"Index build failed: {e}")
                st.info(f"Tip: run `ollama pull {EMBED_MODEL}` (or another embedding model) and try again.")


# ---------- Q&A (M2) ----------
st.subheader("2) Ask questions about your docs")
question = st.text_input("Your question", placeholder="e.g., What are the main conclusions?")

# Retrieve & Answer button
answer_btn = st.button("💬 Retrieve & Answer", use_container_width=True)
# Preview Top Sources button
preview_btn = st.button("Preview Top Sources", use_container_width=True)

vs = st.session_state.get("vs")

if preview_btn:
    if not question.strip():
        st.warning("Please type a question.")
    elif vs is None:
        st.error("No vector store found. Build the index first (Step 1).")
    else:
        with st.spinner("Retrieving…"):
            hits = retrieve(vs, question, k=top_k)

        if not hits:
            st.info("No results. Try a simpler question or rebuild the index.")
        else:
            st.markdown("### Sources (Top-k)")
            # list unique sources
            seen = set()
            for h in hits:
                src = h.metadata.get("source", "unknown")
                if src not in seen:
                    st.write(f"- {src}")
                    seen.add(src)

            # show retrieved chunks
            st.markdown("### Retrieved Chunks")
            for i, h in enumerate(hits, start=1):
                with st.expander(f"Chunk {i} — {h.metadata.get('source','unknown')}"):
                    st.write(h.page_content)

if answer_btn:
    if not question.strip():
        st.warning("Please type a question.")
    elif vs is None:
        st.error("No vector store found. Build the index first (Step 1).")
    else:
        with st.spinner("Retrieving…"):
            hits = retrieve(vs, question, k=top_k)  
        if not hits:
            st.info("No results. Try a simpler question or rebuild the index.")
        else:
            context_text = "\n\n---\n\n".join([h.page_content for h in hits])
            prompt = build_prompt(context_text, question)
            with st.spinner("Thinking…"):
                answer = call_llm(prompt, model_name=llm_model)  
            st.markdown("### Answer")
            st.write(answer)

            st.markdown("### Sources")
            seen = set()
            for h in hits:
                src = h.metadata.get("source", "unknown")
                if src not in seen:
                    st.write(f"- {src}")
                    seen.add(src)

# ---------- FOOTER ----------
st.markdown("---")
st.caption("Scope-first scaffold. Implement M1 → M2 in small commits. Keep it simple.")