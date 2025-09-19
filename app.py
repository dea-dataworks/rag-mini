import os
import pandas as pd
import streamlit as st

import guardrails
from llm_chain import build_prompt, call_llm
from rag_core import (load_vectorstore_if_exists, retrieve, normalize_hits, filter_by_score, cap_per_source, make_chunk_rows,
                      build_index_from_files, build_citation_tags, sanitize_chunks)

# ---------- CONFIG ----------
APP_TITLE = "🔎 RAG Mini v0.1"
PERSIST_DIR = "rag_store"  
EMBED_MODEL = "nomic-embed-text"

st.set_page_config(page_title="RAG Mini v0.1", layout="wide")
st.title(APP_TITLE)
st.caption("Local, simple Retrieval-Augmented Q&A (scope-first)")

# ---------- SESSION DEFAULTS ----------
st.session_state.setdefault("PERSIST_DIR", PERSIST_DIR)
st.session_state.setdefault("OPENAI_API_KEY", "")
st.session_state.setdefault("LLM_PROVIDER", "ollama")
st.session_state.setdefault("LLM_MODEL", "mistral")

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
        persist_dir= st.session_state["PERSIST_DIR"],
    )

# ---------- SIDEBAR SETTINGS ----------
with st.sidebar:
    st.header("⚙️ Settings")

    # --- Regular (simple) controls ---
    embed_model   = EMBED_MODEL  # keep embeddings local (Ollama) in v0.2
    chunk_size    = st.number_input("Chunk size", 200, 4000, 800, 50)
    chunk_overlap = st.number_input("Chunk overlap", 0, 1000, 120, 10)
    top_k         = st.slider("Top-k retrieval", 1, 10, 4)

    # Persist basic controls
    st.session_state.update({
        "CHUNK_SIZE": int(chunk_size),
        "CHUNK_OVERLAP": int(chunk_overlap),
        "TOP_K": int(top_k),
    })

    # --- Advanced (foldable) ---
    with st.expander("Advanced", expanded=False):
        st.markdown("**Provider (hidden by default)**")

        use_openai = st.checkbox("Use OpenAI (cloud)", value=False,
                                 help="Default stays local with Ollama/mistral.")
        if use_openai:
            # API key + model select only when enabled
            openai_key = st.text_input("OpenAI API Key", type="password")
            if openai_key:
                st.session_state["OPENAI_API_KEY"] = openai_key.strip()
            disabled = not bool(st.session_state.get("OPENAI_API_KEY"))
            if disabled:
                st.info("Enter a valid OpenAI key to enable models.", icon="🔐")
            llm_model = st.selectbox("OpenAI model", ["gpt-4o-mini", "gpt-4o"], index=0, disabled=disabled)
            st.session_state["LLM_PROVIDER"] = "openai"
            st.session_state["LLM_MODEL"]    = llm_model
        else:
            # Local default
            llm_model = st.selectbox("Ollama model", ["mistral"], index=0)
            st.session_state["LLM_PROVIDER"] = "ollama"
            st.session_state["LLM_MODEL"]    = llm_model

        st.markdown("---")
        st.markdown("**Retrieval tuning**")

        # Hybrid retrieval toggle (BM25 + Dense via RRF)
        use_hybrid = st.checkbox(
            "Hybrid retrieval (BM25 + Dense via RRF)",
            value=False,
            help="BM25 over chunk text + dense (Chroma). Results fused by Reciprocal Rank Fusion."
        )
        st.session_state["RETRIEVE_MODE"] = "hybrid" if use_hybrid else "dense"

        mmr_lambda = st.slider("MMR λ (0–1)", 0.0, 1.0, 0.7, 0.05,
                               help="Balance relevance (→1) vs diversity (→0).")

        enable_thresh = st.checkbox("Enable score threshold (hide low-similarity chunks)", value=False)
        score_thresh  = st.slider("Score threshold", 0.0, 1.0, 0.4, 0.05, disabled=not enable_thresh)
        enable_cap    = st.checkbox("Enable per-source cap", value=False,
                                    help="Limit how many chunks can come from the same file.")
        per_source_cap = st.number_input("Max chunks per source", 1, 10, 2, 1, disabled=not enable_cap)

        st.session_state.update({
            "MMR_LAMBDA": float(mmr_lambda),
            "USE_SCORE_THRESH": bool(enable_thresh),
            "SCORE_THRESH": float(score_thresh),
            "USE_SOURCE_CAP": bool(enable_cap),
            "PER_SOURCE_CAP": int(per_source_cap),
        })

        # Anti-injection sanitizer (default ON)
        st.session_state.setdefault("SANITIZE_RETRIEVED", True)
        st.checkbox("Sanitize retrieved text (anti-injection)", key="SANITIZE_RETRIEVED",
                    help="Drop lines like 'ignore previous instructions', trim extreme lines, and fix stray code fences.")

        st.markdown("---")
        st.markdown("**UX / Display**")


        snippet_len = st.slider("Snippet length (chars)", 160, 600, 240, 10,
                                help="UI only; does not change retrieval.")
        show_debug  = st.checkbox("Show timings & debug", value=False,
                                  help="Reveals load/split/embed/persist timing and skipped files detail.")
        st.session_state.update({
            "SNIPPET_LEN": int(snippet_len),
            "SHOW_DEBUG": bool(show_debug),
        })

        st.markdown("---")
        st.markdown("**Persistence (low-touch)**")

        suffix = st.text_input("Index name (suffix only)", value="",
                               placeholder="e.g., demo or client-A",
                               help="Writes to rag_store/<suffix>. Leave blank to use the default store.")
        active_persist_dir = os.path.join(PERSIST_DIR, suffix) if suffix.strip() else PERSIST_DIR
        st.session_state["PERSIST_DIR"] = active_persist_dir
        st.caption(f"Active index: `{active_persist_dir}`")

    st.markdown("---")
    st.markdown("**Implementation steps**")
    st.caption("1) Ingest → chunk → index\n2) Retrieve → answer → cite\n3) UI polish + docs")

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
                    persist_dir=st.session_state["PERSIST_DIR"],
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
                rows = [
                    {"File": fname, "Pages": meta.get("pages", 0), "Chunks": meta.get("chunks", 0)}
                    for fname, meta in stats["per_file"].items()
                ]
                df_pf = pd.DataFrame(rows).sort_values("File").reset_index(drop=True)
                try:
                    st.dataframe(df_pf, use_container_width=True, hide_index=True)
                except TypeError:
                    df_pf.index = [""] * len(df_pf)
                    st.dataframe(df_pf, use_container_width=True)

            # Optional timings/debug (always outside per_file block)
            if st.session_state.get("SHOW_DEBUG") and stats and isinstance(stats, dict):
                if "timings" in stats:
                    t = stats["timings"]
                    st.caption(
                        f"Timing — load: {t.get('load_docs','?')}s | "
                        f"split: {t.get('split','?')}s | "
                        f"embed: {t.get('embed','?')}s | "
                        f"persist: {t.get('persist','?')}s | "
                        f"total: {t.get('total','?')}s"
                    )

        else:
            # ----- LOAD-ONLY PATH (fast) -----
            with st.spinner("Loading existing index…"):
                vs = load_vectorstore_if_exists(embed_model=EMBED_MODEL, persist_dir=st.session_state["PERSIST_DIR"])
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
            # Try to pass mmr if supported; otherwise fall back silently
            mode = st.session_state.get("RETRIEVE_MODE", "dense")
            try:
                hits_raw = retrieve(
                    vs, question,
                    k=top_k,
                    mmr_lambda=st.session_state.get("MMR_LAMBDA", 0.7),
                    mode=mode,
                )
            except TypeError:
                # Fallback for older retrieve() signatures
                hits_raw = retrieve(vs, question, k=top_k)


        if not hits_raw:
            st.info("No results. Try a simpler question or rebuild the index.")
        else:
            # normalize to [(doc, score)]
            norm = normalize_hits(hits_raw)
            if st.session_state.get("USE_SCORE_THRESH"):
                norm = filter_by_score(norm, st.session_state.get("SCORE_THRESH", 0.4))
            if st.session_state.get("USE_SOURCE_CAP"):
                norm = cap_per_source(norm, st.session_state.get("PER_SOURCE_CAP", 2))
            st.markdown("### Chunk Inspector")
            rows = make_chunk_rows(norm, st.session_state.get("SNIPPET_LEN", 240))
            st.dataframe(rows, use_container_width=True)

            # --- Sanitize retrieved chunks (Preview) ---
            docs_only = [d for (d, _) in norm]
            sanitize_stats = {"chunks_with_drops": 0, "lines_dropped": 0}
            if st.session_state.get("SANITIZE_RETRIEVED", True):
                docs_only, sanitize_stats = sanitize_chunks(docs_only)

            # Optional badge about sanitization (Preview)
            if sanitize_stats.get("lines_dropped", 0) > 0:
                st.caption(
                    f"Sanitized (preview): {sanitize_stats['lines_dropped']} line(s) "
                    f"in {sanitize_stats['chunks_with_drops']} chunk(s)."
                )

            st.markdown("### Retrieved Chunks")
            for i, d in enumerate(docs_only, start=1):
                with st.expander(f"Chunk {i} — {d.metadata.get('source','unknown')} p.{d.metadata.get('page','')}"):
                    st.write(d.page_content)


if answer_btn:
    if not question.strip():
        st.warning("Please type a question.")
    elif vs is None:
        st.error("No vector store found. Build the index first (Step 1).")
    else:
        with st.spinner("Retrieving…"):
            mode = st.session_state.get("RETRIEVE_MODE", "dense")
            try:
                hits_raw = retrieve(
                    vs, question,
                    k=top_k,
                    mmr_lambda=st.session_state.get("MMR_LAMBDA", 0.7),
                    mode=mode,
                )
            except TypeError:
                hits_raw = retrieve(vs, question, k=top_k)


        if not hits_raw:
            st.info("No results. Try a simpler question or rebuild the index.")
        else:
            # normalize to [(doc, score)]
            norm = normalize_hits(hits_raw)
            if st.session_state.get("USE_SCORE_THRESH"):
                norm = filter_by_score(norm, st.session_state.get("SCORE_THRESH", 0.4))
            if st.session_state.get("USE_SOURCE_CAP"):
                norm = cap_per_source(norm, st.session_state.get("PER_SOURCE_CAP", 2))
            docs_only = [d for (d, _) in norm]

            # sanitize retrieved chunks (pre-prompt)
            sanitize_stats = {"chunks_with_drops": 0, "lines_dropped": 0}
            if st.session_state.get("SANITIZE_RETRIEVED", True):
                docs_only, sanitize_stats = sanitize_chunks(docs_only)

            raw_context = "\n\n---\n\n".join(d.page_content for d in docs_only)
            context_text, bad_lines = guardrails.scrub_context(raw_context)


            # Early exit if too thin
            if guardrails.empty_or_thin_context(context_text):
                st.info("I don’t have enough context to answer that from your documents.")
                st.stop()

            prompt = build_prompt(context_text, question)

            # If scrub removed a lot, warn (Phase 3 acceptance: be explicit).
            if bad_lines:
                st.warning("Some retrieved lines were removed for safety.")
                if st.session_state.get("SHOW_DEBUG"):
                    st.caption("Scrubbed lines:")
                    for ln in bad_lines[:5]:
                        st.code(ln)

            # If context is too thin, exit early with the required phrasing.
            if guardrails.empty_or_thin_context(context_text):
                st.info("I don’t have enough context to answer that from your documents.")
                st.stop()

            prompt = build_prompt(context_text, question)


            with st.spinner("Thinking…"):
                answer = call_llm(
                    prompt,
                    provider=st.session_state.get("LLM_PROVIDER", "ollama"),
                    model_name=st.session_state.get("LLM_MODEL", "mistral"),
                    openai_api_key=st.session_state.get("OPENAI_API_KEY"),
                )

            st.markdown("### Answer")
            st.write(answer)

            # Optional badge about sanitization
            if sanitize_stats.get("lines_dropped", 0) > 0:
                st.caption(f"Sanitized: {sanitize_stats['lines_dropped']} line(s) in "
                           f"{sanitize_stats['chunks_with_drops']} chunk(s).")

            # --- Citations (dedup + page-anchored) ---
            tags = build_citation_tags(docs_only)
            if tags:
                st.caption("Sources: " + "; ".join(tags))

            # --- Optional: Show cited chunks (subset used in the prompt) ---
            with st.expander("Show cited chunks", expanded=False):
                maxlen = st.session_state.get("SNIPPET_LEN", 240)
                for i, d in enumerate(docs_only, start=1):
                    m = d.metadata or {}
                    src = m.get("source", "unknown")
                    pg  = m.get("page", None)
                    header = f"{src} p.{pg}" if pg else src
                    st.markdown(f"**Chunk {i} — {header}**")
                    st.write(d.page_content)

# ---------- FOOTER ----------
st.markdown("---")
st.caption("Scope-first scaffold. Implement M1 → M2 in small commits. Keep it simple.")