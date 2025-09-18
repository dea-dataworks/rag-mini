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

# --- Provider & API key ---
with st.sidebar:
    st.subheader("Provider")
    # store in session to avoid clearing during reruns
    openai_key = st.text_input("OpenAI API Key (optional)", type="password")
    if "OPENAI_API_KEY" not in st.session_state or openai_key:
        st.session_state["OPENAI_API_KEY"] = openai_key.strip()

    provider = st.selectbox(
        "LLM Provider",
        options=["Ollama (local)", "OpenAI (API)"],
        index=0 if not st.session_state.get("OPENAI_API_KEY") else 1,
        help="Default is fully local. OpenAI requires an API key."
    )

    # pick model per provider (keep short curated lists)
    if provider.startswith("Ollama"):
        llm_model = st.selectbox("LLM Model", ["mistral"], index=0)
    else:
        disabled = not bool(st.session_state.get("OPENAI_API_KEY"))
        if disabled:
            st.info("Enter a valid OpenAI key to enable cloud models.", icon="🔐")
        llm_model = st.selectbox("LLM Model", ["gpt-4o-mini", "gpt-4o"], index=0, disabled=disabled)

    st.session_state["LLM_PROVIDER"] = "openai" if provider.startswith("OpenAI") else "ollama"
    st.session_state["LLM_MODEL"] = llm_model

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

    embed_model   = EMBED_MODEL  # keep embeddings local (Ollama) in v0.2
    chunk_size    = st.number_input("Chunk size", 200, 4000, 800, 50)
    chunk_overlap = st.number_input("Chunk overlap", 0, 1000, 120, 10)
    top_k         = st.slider("Top-k retrieval", 1, 10, 4)
    
    st.session_state.update({
        "CHUNK_SIZE": int(chunk_size),
        "CHUNK_OVERLAP": int(chunk_overlap),
        "TOP_K": int(top_k),
    })

    st.markdown("---")
    st.markdown("**Implementation steps**")
    st.caption("1) Ingest → chunk → index\n2) Retrieve → answer → cite\n3) UI polish + docs")

# ---------- FILE UPLOAD ----------
# st.subheader("1) Upload documents")
# uploaded_files = st.file_uploader(
#     "Upload .pdf or .txt files",
#     type=["pdf", "txt"],
#     accept_multiple_files=True,
# )

# ---------- FILE UPLOAD ----------
# create a resettable key so we can clear the uploader after a build
if "UPLOAD_KEY" not in st.session_state:
    st.session_state["UPLOAD_KEY"] = 0

uploaded_files = st.file_uploader(
    "Upload .pdf or .txt",
    type=["pdf", "txt"],
    accept_multiple_files=True,
    key=f"uploader_{st.session_state['UPLOAD_KEY']}",
)

# small control to clear uploads without restarting
clear_col, _ = st.columns([1, 3])
with clear_col:
    if st.button("Clear uploads"):
        st.session_state["UPLOAD_KEY"] += 1  # re-key the widget -> clears files
        st.rerun()


# ---------- INDEX BUILD (M1) ----------
# --- Build / Load controls ---
rebuild_from_uploads = st.checkbox(
    "Rebuild from current uploads",
    value=False,
    help="If OFF, this will just load an existing index (fast). Turn ON to re-index the files above."
)

col_a, col_b = st.columns([2, 1])
with col_a:
    build_btn = st.button("Build / Load Index", type="primary", use_container_width=True)
with col_b:
    st.info("Step 1: Build the index before asking questions.", icon="ℹ️")

if build_btn:
    try:
        if rebuild_from_uploads and uploaded_files:
            # ----- REBUILD PATH (heavy) -----
            with st.spinner("Reading, chunking, and indexing…"):
                embedding = get_cached_embeddings(embed_model=EMBED_MODEL)
                vs, stats = build_index_from_files(
                    uploaded_files=uploaded_files,
                    embed_model=embed_model,
                    chunk_size=st.session_state.get("CHUNK_SIZE", 800),
                    chunk_overlap=st.session_state.get("CHUNK_OVERLAP", 120),
                    persist_dir=PERSIST_DIR,
                    embedding_obj=embedding,
                )
            st.session_state["vs"] = vs
            st.success(f"Index built — docs: {stats['num_docs']}, chunks: {stats['num_chunks']}")
            st.caption(f"Sources: {', '.join(stats['sources']) or 'None'}")

            # Guard (a): warn if no valid chunks
            if stats["num_chunks"] == 0:
                st.warning("No valid text chunks were created. The index was not rebuilt.")
                st.stop()

            # Guard (b): show skipped files
            if stats.get("skipped_files"):
                st.info("Skipped files: " + ", ".join(stats["skipped_files"]))

            # Per-file table (pages & chunks)
            if stats.get("per_file"):
                st.markdown("##### Per-file stats")
                st.table([
                    {"File": fname, "Pages": meta.get("pages", ""), "Chunks": meta.get("chunks", "")}
                    for fname, meta in stats["per_file"].items()
                ])
        else:
            # ----- LOAD-ONLY PATH (fast) -----
            with st.spinner("Loading existing index…"):
                vs = load_vectorstore_if_exists(embed_model=EMBED_MODEL, persist_dir=PERSIST_DIR)
            if vs is None:
                st.warning("No existing index found. Enable 'Rebuild from current uploads' and provide files.")
            else:
                st.session_state["vs"] = vs
                st.success("Loaded existing index.")

    except Exception as e:
        st.error(f"Index operation failed: {e}")
        st.info(f"Tip: run `ollama pull {EMBED_MODEL}` (or another embedding model) and try again.")

# ---------- Q&A (M2) ----------
st.subheader("2) Ask questions about your docs")
question = st.text_input("Your question", placeholder="e.g., What are the main conclusions?")

# Retrieve & Answer button
answer_btn = st.button("Retrieve & Answer", use_container_width=True)
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
            hits_raw = retrieve(vs, question, k=top_k)

        if not hits_raw:
            st.info("No results. Try a simpler question or rebuild the index.")
        else:
            # normalize to [(doc, score)]
            norm = []
            for item in hits_raw:
                if isinstance(item, tuple) and len(item) == 2:
                    norm.append(item)
                else:
                    norm.append((item, None))

            st.markdown("### Chunk Inspector")
            rows = []
            for i, (doc, score) in enumerate(norm, start=1):
                meta = doc.metadata or {}
                snippet = (doc.page_content or "")[:240].replace("\n", " ")
                if len(doc.page_content or "") > 240:
                    snippet += "…"
                rows.append({
                    "Rank": i,
                    "Score": f"{score:.4f}" if isinstance(score, (float, int)) else "—",
                    "File": meta.get("source", "unknown"),
                    "Page": meta.get("page", ""),
                    "Snippet": snippet,
                })
            st.dataframe(rows, use_container_width=True)

            st.markdown("### Retrieved Chunks")
            for i, (doc, _) in enumerate(norm, start=1):
                with st.expander(f"Chunk {i} — {doc.metadata.get('source','unknown')} p.{doc.metadata.get('page','')}"):
                    st.write(doc.page_content)

if answer_btn:
    if not question.strip():
        st.warning("Please type a question.")
    elif vs is None:
        st.error("No vector store found. Build the index first (Step 1).")
    else:
        with st.spinner("Retrieving…"):
            hits_raw = retrieve(vs, question, k=top_k)
        if not hits_raw:
            st.info("No results. Try a simpler question or rebuild the index.")
        else:
            # normalize to docs list
            docs_only = [h[0] if isinstance(h, tuple) else h for h in hits_raw]
            context_text = "\n\n---\n\n".join([d.page_content for d in docs_only])
            prompt = build_prompt(context_text, question)

            with st.spinner("Thinking…"):
                answer = call_llm(
                    prompt,
                    provider=st.session_state.get("LLM_PROVIDER", "ollama"),
                    model_name=st.session_state.get("LLM_MODEL", "mistral"),
                    openai_api_key=st.session_state.get("OPENAI_API_KEY"),
                )

            st.markdown("### 🧠 Answer")
            st.write(answer)

            # compact citations
            cited = []
            for d in docs_only:
                m = d.metadata or {}
                tag = m.get("source", "unknown")
                if m.get("page"):
                    tag += f" p. {m['page']}"
                cited.append(tag)
            cited = sorted(set(cited))
            if cited:
                st.caption("Sources: " + "; ".join(cited))

# ---------- FOOTER ----------
st.markdown("---")
st.caption("Scope-first scaffold. Implement M1 → M2 in small commits. Keep it simple.")