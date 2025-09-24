import os
import time
import pandas as pd
import streamlit as st
import guardrails
from llm_chain import build_prompt, call_llm
from rag_core import (load_vectorstore_if_exists, retrieve, normalize_hits, filter_by_score, cap_per_source, make_chunk_rows,
                      build_index_from_files, build_citation_tags, build_qa_result,
                      make_fresh_index_dir, read_manifest)
from guardrails import sanitize_chunks
from index_admin import (
    list_sources_in_vs, delete_source, add_or_replace_file, rebuild_manifest_from_vs
)
from utils.settings import seed_session_from_settings, save_settings, apply_persisted_defaults
from utils.ui import (sidebar_pipeline_diagram, render_export_buttons, render_copy_row , render_cited_chunks_expander,
                    render_pdf_limit_note_for_uploads, render_pdf_limit_note_for_docs, render_why_this_answer,
                    render_dev_metrics, render_session_export, get_exportable_settings)
from eval.run_eval import run_eval_snapshot
from exports import chat_to_markdown
from utils.helpers import _attempt_with_timeout, RETRIEVAL_TIMEOUT_S, LLM_TIMEOUT_S


# ---------- CONFIG ----------
APP_TITLE = "RAG Mini"
PERSIST_DIR = "rag_store"  
EMBED_MODEL = "nomic-embed-text"

st.set_page_config(page_title="RAG Mini", layout="wide")
# Load last-used settings into session (persisted in settings.json)
seed_session_from_settings(st)
apply_persisted_defaults(st)
st.title(APP_TITLE)
st.caption("v0.2") 
st.caption("Local, simple Retrieval-Augmented Q&A (scope-first)")

# ---------- SESSION DEFAULTS ----------
st.session_state.setdefault("BASE_DIR", PERSIST_DIR)       # sidebar-selected base folder
st.session_state.setdefault("ACTIVE_INDEX_DIR", None)      # specific subfolder in use (idx_YYYYMMDD_HHMMSS)
st.session_state.setdefault("OPENAI_API_KEY", "")
st.session_state.setdefault("LLM_PROVIDER", "ollama")
st.session_state.setdefault("LLM_MODEL", "mistral")
st.session_state.setdefault("chat_history", [])   # list[dict]: prior turns
st.session_state.setdefault("use_history", False) # UI toggle: condition LLM on prior turns
st.session_state.setdefault("max_history_turns", 4)  # small cap to avoid token bloat
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
    from rag_core import read_active_pointer, find_latest_index_dir
    base_dir = st.session_state.get("BASE_DIR", PERSIST_DIR)

    # 1) Try saved pointer for this base
    active_dir = st.session_state.get("ACTIVE_INDEX_DIR") or read_active_pointer(base_dir)

    # 2) Fallback: newest idx_* under base
    if not active_dir:
        active_dir = find_latest_index_dir(base_dir)

    if active_dir:
        vs0 = load_vectorstore_if_exists(embed_model=EMBED_MODEL, persist_dir=active_dir)
        if vs0 is not None:
            # Switch the app to the loaded index and save pointer
            st.session_state["vs"] = vs0
            st.session_state["ACTIVE_INDEX_DIR"] = active_dir
            from rag_core import save_active_pointer
            save_active_pointer(base_dir, active_dir)

# ---------- SIDEBAR SETTINGS ----------
with st.sidebar:
    st.header("Settings")
    # Compact pipeline diagram (utils.ui)
    sidebar_pipeline_diagram()

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

    # ---- Instructions (collapsible, above Advanced) ----
    with st.expander("Instructions", expanded=False):
        st.markdown(
        """
1. **Upload docs** in the main area (`.pdf` / `.txt`).
- **Limitation:** PDF **images and tables aren’t parsed yet** — only the text layer is indexed.
2. **Build / Load Index**  
   - Turn **Rebuild from current uploads** ON → creates a fresh index.  
   - Leave it OFF → loads the last active index for the selected base folder.
3. **Ask & Cite**  
   - Type your question and press **Enter** (or click **Retrieve & Answer**).  
   - Use **Preview Top Sources** to inspect retrieved chunks.
 - **Advanced:**  
   - The app auto-loads your last active index next session.
   - Optional: Set **Index name (suffix)** to keep separate bases (e.g., `client-a`).  
           """
    )

    # --- Advanced (foldable) ---
    with st.expander("Advanced", expanded=False):
        st.markdown("**Provider**")

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
        st.caption("Conversation")

        # Disable history use when there are no prior turns; also clear a stale True.
        has_history = len(st.session_state.get("chat_history", [])) > 0
        if not has_history and st.session_state.get("use_history", False):
            st.session_state["use_history"] = False

        st.checkbox(
            "Use previous turns as context",
            key="use_history",
            disabled=not has_history,
            help="Keeps continuity between questions. Facts still come from retrieved sources."
        )

        st.number_input(
            "Turns to include",
            min_value=1, max_value=10, step=1,
            key="max_history_turns",
            disabled=(not st.session_state.get("use_history", False) or not has_history)
        )

        if not has_history:
            st.caption("Add at least one Q&A turn to enable history.")

        st.markdown("---")
        st.markdown("**Persistence (low-touch)**")

        suffix = st.text_input("Index name (suffix only)", value="",
                    placeholder="e.g., demo or client-A",
                    help="Base folder under rag_store/. Each rebuild creates a fresh subfolder.")
        base_dir = os.path.join(PERSIST_DIR, suffix) if suffix.strip() else PERSIST_DIR
        os.makedirs(base_dir, exist_ok=True)

        # --- Preferences persistence (save last-used settings to settings.json) ---
        st.markdown("---")
        st.markdown("###### Preferences")
        auto_save = st.checkbox(
            "Persist settings to disk",
            value=True,
            help="Save last-used settings to settings.json"
        )

        _current = {
            "chunk_size": st.session_state.get("CHUNK_SIZE", 800),
            "chunk_overlap": st.session_state.get("CHUNK_OVERLAP", 120),
            "k": st.session_state.get("TOP_K", 4),
            "provider": st.session_state.get("LLM_PROVIDER", "ollama"),
            "use_history": st.session_state.get("use_history", False),
            "max_history_turns": st.session_state.get("max_history_turns", 4),
            "mmr_lambda": st.session_state.get("MMR_LAMBDA", 0.7),
            "score_threshold": st.session_state.get("SCORE_THRESH", 0.4),
            "sanitize": st.session_state.get("SANITIZE_RETRIEVED", True),
            "debug": st.session_state.get("SHOW_DEBUG", False),
        }

        if auto_save:
            try:
                save_settings(_current)
            except Exception as e:
                st.caption(f"Could not save settings: {e}")

        # Important: sidebar only sets the BASE_DIR. The active index is set by the Build step.
        st.session_state["BASE_DIR"] = base_dir
        st.caption(f"Active base: `{base_dir}`")

# ---------- FILE UPLOAD ----------
# create a resettable key so we can clear the uploader after a build
if "UPLOAD_KEY" not in st.session_state:
    st.session_state["UPLOAD_KEY"] = 0

uploaded_files = st.file_uploader(
    "Upload .pdf, .txt, or .docx",
    type=["pdf", "txt", "docx"],
    accept_multiple_files=True,
    key=f"uploader_{st.session_state['UPLOAD_KEY']}",
)

# --- UX note for PDFs (images/tables not parsed yet) ---
render_pdf_limit_note_for_uploads(uploaded_files)

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
    help="ON = create a fresh index in a new folder and re-index the current uploads (previous indices are kept). OFF = load the last active index (fast)."
)

build_btn = st.button("Build / Load Index", type="primary", use_container_width=True)

if build_btn:
    try:
        from rag_core import (
            read_active_pointer, find_latest_index_dir, list_index_dirs, save_active_pointer
        )
        base_dir = st.session_state.get("BASE_DIR", PERSIST_DIR)

        # Decide behavior up-front for an empty base
        base_has_any = bool(list_index_dirs(base_dir))

        # Case A — Rebuild explicitly requested AND uploads present → rebuild
        # Case B — Base is empty AND uploads present → build even if toggle is OFF (helpful default)
        if (rebuild_from_uploads and uploaded_files) or (not base_has_any and uploaded_files):
            # ----- REBUILD PATH (heavy) -----
            with st.spinner("Reading, chunking, and indexing…"):
                fresh_dir = make_fresh_index_dir(base_dir)

                embedding = get_cached_embeddings(embed_model=EMBED_MODEL)
                vs, stats = build_index_from_files(
                    uploaded_files=uploaded_files,
                    embed_model=embed_model,
                    chunk_size=st.session_state.get("CHUNK_SIZE", 800),
                    chunk_overlap=st.session_state.get("CHUNK_OVERLAP", 120),
                    persist_dir=fresh_dir,
                    embedding_obj=embedding,
                )

                # Switch the app to the new index (persist across reruns) + save pointer
                st.session_state["vs"] = vs
                st.session_state["ACTIVE_INDEX_DIR"] = fresh_dir
                save_active_pointer(base_dir, fresh_dir)

                st.success(
                    f"Index built — docs: {stats['num_docs']}, chunks: {stats['num_chunks']}, "
                    f"avg chunk len: {stats.get('avg_chunk_len', 0)} chars"
                )
                st.caption(f"Active index: `{fresh_dir}`")
                st.caption(f"Sources: {', '.join(stats['sources']) or 'None'}")


            # Guard (a): warn if no valid chunks
            if stats["num_chunks"] == 0:
                st.warning("No valid text chunks were created. The index was not rebuilt.")
                st.stop()

            # Guard (b): show skipped files 
            if stats.get("skipped_files"):
                st.info("Skipped files: " + ", ".join(stats["skipped_files"]))


            # Optional timings/debug
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
            active_dir = st.session_state.get("ACTIVE_INDEX_DIR") or read_active_pointer(base_dir)
            if not active_dir:
                active_dir = find_latest_index_dir(base_dir)

            if not active_dir:
                if uploaded_files:
                    st.info("No index exists yet. Turn ON 'Rebuild from current uploads' to create one.")
                else:
                    st.info("No index exists yet. Upload files above, then click Build.")
            else:
                with st.spinner("Loading active index…"):
                    vs = load_vectorstore_if_exists(embed_model=EMBED_MODEL, persist_dir=active_dir)
                if vs is None:
                    st.warning("Couldn’t load that index. Try rebuilding from uploads.")
                else:
                    st.session_state["vs"] = vs
                    st.session_state["ACTIVE_INDEX_DIR"] = active_dir
                    save_active_pointer(base_dir, active_dir)
                    st.success(f"Loaded active index: `{active_dir}`")

    except Exception as e:
        st.error(f"Index operation failed: {e}")
        st.info(f"Tip: run `ollama pull {EMBED_MODEL}` (or another embedding model) and try again.")

# --- Index Inspector (pre-Q&A) ---
st.markdown("### Index Inspector")
active_dir = st.session_state.get("ACTIVE_INDEX_DIR")

if not active_dir:
    st.info("No active index selected yet. Rebuild from uploads to create one.", icon="ℹ️")
else:
    st.caption(f"Active index path: `{active_dir}`")
    mf = read_manifest(active_dir)
    if mf:
        # Manifest summary (what the index was built with)
        st.write(
            f"**Built:** {mf.get('timestamp','?')} — "
            f"**Docs:** {mf.get('num_docs',0)} — "
            f"**Chunks:** {mf.get('num_chunks',0)} — "
            f"**Avg chunk len:** {mf.get('avg_chunk_len',0)} chars"
        )

        # --- Settings mismatch warning (manifest params vs current UI) ---
        mf_params = (mf.get("params") or {})
        current_params = {
            "embed_model": EMBED_MODEL,
            "chunk_size": st.session_state.get("CHUNK_SIZE", 800),
            "chunk_overlap": st.session_state.get("CHUNK_OVERLAP", 120),
        }
        # Collect keys where the values differ (including missing in manifest)
        diff_keys = [k for k, v in current_params.items() if mf_params.get(k) != v]

        if diff_keys:
            # Build a human-friendly diff list
            lines = []
            for k in diff_keys:
                lines.append(
                    f"- **{k}**: index = `{mf_params.get(k, '∅')}` | UI = `{current_params[k]}`"
                )
            st.warning(
                "Settings differ from the active index. "
                "Rebuild with the current UI settings to apply changes, or continue using the index as-is.\n\n"
                + "\n".join(lines)
            )

        pf = mf.get("per_file", {}) or {}
        if pf:
            rows = [
                        {
                            "File": k,
                            "Pages": v.get("pages", 0),
                            "Chunks": v.get("chunks", 0),
                            "Chars": v.get("chars", 0),
                            "Avg chunk len": v.get("avg_chunk_len", 0),
                        }
                        for k, v in sorted(pf.items(), key=lambda x: x[0].lower())
                    ]
            st.dataframe(rows, use_container_width=True)

        # --- Manage files in this index ---
        st.markdown("#### Manage files in index")
        vs = st.session_state.get("vs")
        if vs is None:
            st.info("Load or build the index to manage files.", icon="ℹ️")
        else:
            sources_in_index = list_sources_in_vs(vs)
            if not sources_in_index:
                st.info("No files currently in this index.", icon="ℹ️")
            else:
                sel = st.selectbox("Select a file", sources_in_index)

                c1, c2 = st.columns(2)

                with c1:
                    if st.button("Delete from index", disabled=not bool(sel)):
                        ok = delete_source(vs, sel)
                        mf_params = (mf.get("params") or {}) if mf else {}
                        rebuild_manifest_from_vs(active_dir, vs, params=mf_params)
                        if ok:
                            st.success(f"Removed '{sel}' from the index.")
                        else:
                            st.warning(f"Couldn’t remove '{sel}'.")
                        st.rerun()

                with c2:
                    # Replacement requires an uploaded file with the same name
                    matching = None
                    for uf in (uploaded_files or []):
                        if uf.name == sel:
                            matching = uf
                            break
                    if st.button("Replace from uploads", disabled=not bool(sel)):
                        if not matching:
                            st.warning("No matching uploaded file found. Upload with the same name first.")
                        else:
                            del_ok, added = add_or_replace_file(
                                vs, matching,
                                embed_model=EMBED_MODEL,
                                chunk_size=st.session_state.get("CHUNK_SIZE", 800),
                                chunk_overlap=st.session_state.get("CHUNK_OVERLAP", 120),
                            )
                            mf_params = (mf.get("params") or {}) if mf else {}
                            rebuild_manifest_from_vs(active_dir, vs, params=mf_params)
                            st.success(f"Replaced '{sel}' — added {added} fresh chunk(s).")
                            st.rerun()

    else:
        st.info("No manifest found for the active index. Rebuild to generate one.", icon="ℹ️")

# ---------- Q&A (M2) ----------
st.subheader("2) Ask questions about your docs")
st.caption("Tip: press Enter to run, or click Retrieve & Answer.")

# fire answer on Enter
st.session_state.setdefault("TRIGGER_ANSWER", False)
def _on_enter_answer():
    st.session_state["TRIGGER_ANSWER"] = True
st.text_input(
    "Your question",
    key="QUESTION",
    placeholder="e.g., What are the main conclusions?",
    on_change=_on_enter_answer
)
question = st.session_state.get("QUESTION", "").strip()

# Retrieve & Answer button (still available)
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
            mode = st.session_state.get("RETRIEVE_MODE", "dense")

            def _do_retrieve():
                try:
                    return retrieve(
                        vs, question,
                        k=top_k,
                        mmr_lambda=st.session_state.get("MMR_LAMBDA", 0.7),
                        mode=mode,
                    )
                except TypeError:
                    # Fallback for older retrieve() signatures
                    return retrieve(vs, question, k=top_k)

            t0 = time.perf_counter()
            ok, hits_raw, err = _attempt_with_timeout(_do_retrieve, RETRIEVAL_TIMEOUT_S, retries=1)
            t_retrieve = (time.perf_counter() - t0) * 1000  # ms


        if not ok:
            st.info(f"Retrieval {err or 'failed'}. Try again, reduce Top-k, or rebuild the index.")
        elif not hits_raw:
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
            # One-time note if any retrieved chunk comes from a PDF
            render_pdf_limit_note_for_docs(docs_only)

            for i, d in enumerate(docs_only, start=1):
                m = d.metadata or {}
                src = m.get("source", "unknown")
                pg  = m.get("page", None)
                cid = m.get("chunk_id") or m.get("id")
                hdr = f"Chunk {i} — {src}" + (f" p.{pg}" if pg else "") + (f"  [{cid}]" if cid else "")
                with st.expander(hdr):
                    st.write(d.page_content)

if answer_btn or st.session_state.get("TRIGGER_ANSWER"):
    # reset the Enter-trigger for next time
    st.session_state["TRIGGER_ANSWER"] = False
    if not question:
        st.warning("Please type a question.")
    elif vs is None:
        st.error("No vector store found. Build the index first (Step 1).")
    else:
        with st.spinner("Retrieving…"):
            mode = st.session_state.get("RETRIEVE_MODE", "dense")

            def _do_retrieve():
                try:
                    return retrieve(
                        vs, question,
                        k=top_k,
                        mmr_lambda=st.session_state.get("MMR_LAMBDA", 0.7),
                        mode=mode,
                    )
                except TypeError:
                    return retrieve(vs, question, k=top_k)

            ok, hits_raw, err = _attempt_with_timeout(_do_retrieve, RETRIEVAL_TIMEOUT_S, retries=1)

        if not ok:
            st.info(f"Retrieval {err or 'failed'}. Try again, reduce Top-k, or rebuild the index.")
            st.stop()
        elif not hits_raw:
            st.info("No results. Try a simpler question or rebuild the index.")
            st.stop()
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

            # If scrub removed a lot, warn (Phase 3 acceptance: be explicit).
            if bad_lines:
                st.warning("Some retrieved lines were removed for safety.")
                if st.session_state.get("SHOW_DEBUG"):
                    st.caption("Scrubbed lines:")
                    for ln in bad_lines[:5]:
                        st.code(ln)

            prompt = build_prompt(
                context_text,
                question,
                chat_history=st.session_state.get("chat_history", []),
                use_history=st.session_state.get("use_history", False),
                max_history_turns=st.session_state.get("max_history_turns", 4),
            )

            with st.spinner("Thinking…"):
                def _do_llm():
                    return call_llm(
                        prompt,
                        provider=st.session_state.get("LLM_PROVIDER", "ollama"),
                        model_name=st.session_state.get("LLM_MODEL", "mistral"),
                        openai_api_key=st.session_state.get("OPENAI_API_KEY"),
                    )

                t1 = time.perf_counter()
                ok, answer, err = _attempt_with_timeout(_do_llm, LLM_TIMEOUT_S, retries=1)
                t_llm = (time.perf_counter() - t1) * 1000  # ms

            if not ok:
                st.info(f"Model {err or 'failed'}. It was cancelled to keep the app responsive. Try again.")
                st.stop()

            st.markdown("### Answer")
            st.write(answer)

            # Quick copy buttons (utils.ui)
            try:
                citations_str = "; ".join(build_citation_tags(docs_only)) if docs_only else ""
            except Exception:
                citations_str = ""

            render_copy_row(answer, citations_str)

            # Optional badge about sanitization
            if sanitize_stats.get("lines_dropped", 0) > 0:
                st.caption(f"Sanitized: {sanitize_stats['lines_dropped']} line(s) in "
                           f"{sanitize_stats['chunks_with_drops']} chunk(s).")

            # --- Citations (dedup + page-anchored) ---
            tags = build_citation_tags(docs_only)
            if tags:
                st.caption("Sources: " + "; ".join(tags))

            # === Build QA payload (for exports/history) ===
            try:
                qa = build_qa_result(
                    question=question,
                    answer=answer,
                    docs_used=docs_only,        # sanitized docs actually sent to the LLM
                    pairs=norm,                 # normalized [(Document, score)] used for ranking
                    meta={
                        "model": st.session_state.get("LLM_MODEL"),
                        "top_k": st.session_state.get("TOP_K", top_k),
                        "retrieval_mode": st.session_state.get("RETRIEVE_MODE"),
                     },
                    timings={
                        "retrieve_ms": round(t_retrieve, 1),
                        "llm_ms": round(t_llm, 1),
                    },
                )                
                st.session_state["last_qa"] = qa
            except Exception as e:
                st.info(f"Couldn’t package the Q&A for export: {e}")
                qa = None
            
             # --- Why-this-answer panel (compact; always visible) ---
            if qa:
                render_why_this_answer(qa)
                render_dev_metrics(qa)
            
            # # --- Dev / observability panel ---
            # with st.expander("Evaluation / dev metrics", expanded=False):
            #     st.caption("Latency breakdown and retrieval score stats (per question).")

            #     metrics = qa.get("metrics", {})
            #     times = metrics.get("timings", {})
            #     scores = metrics.get("scores", {})

            #     # Small timings table
            #     t_rows = [{"Step": k, "ms": v} for k, v in times.items()]
            #     if t_rows:
            #         st.markdown("**Timings (ms)**")
            #         st.dataframe(t_rows, use_container_width=True, hide_index=True)

            #     # Small score stats table
            #     s_rows = [{"Stat": k, "Value": v} for k, v in scores.items()]
            #     if s_rows:
            #         st.markdown("**Score stats**")
            #         st.dataframe(s_rows, use_container_width=True, hide_index=True)

            #     # Tiny bar of scores (if you want to visualize dispersion)
            #     vals = [r.get("score") for r in qa.get("retrieved_chunks", []) if r.get("score") is not None]
            #     if vals:
            #         st.bar_chart(vals, use_container_width=True, height=120)

            # Build one history turn and append
            try:
                answer_text = (qa.get("answer") or "").strip()
                gist = (answer_text[:200] + "…") if len(answer_text) > 200 else answer_text
                history_item = {
                    "question": qa.get("question", ""),
                    "answer": answer_text,
                    "answer_gist": gist,
                    "sources": qa.get("sources", []),
                    "created_at": (qa.get("meta", {}) or {}).get("timestamp", None),
                    "run_settings": get_exportable_settings(st.session_state),
                }
                st.session_state["chat_history"].append(history_item)

                # Cap to last N turns
                k = int(st.session_state.get("max_history_turns", 4)) or 4
                if len(st.session_state["chat_history"]) > k:
                    st.session_state["chat_history"] = st.session_state["chat_history"][-k:]
            except Exception:
                pass  # history is best-effort; don't block the UI

            # === Exports (latest turn + full session) ===
            if qa:
                st.markdown("#### Exports")
                render_export_buttons(qa)

                idx_label = st.session_state.get("ACTIVE_INDEX_DIR") or st.session_state.get("BASE_DIR", "rag_store")
                render_session_export(st.session_state.get("chat_history", []), idx_label)

            # --- Optional: Show cited chunks (subset used in the prompt) ---
            render_cited_chunks_expander(docs_only, snippet_len=240)

            st.markdown("### Chat (History)")

            hist = st.session_state.get("chat_history", [])
            if not hist:
                st.info("No history yet. Ask a question to start.", icon="💬")
            else:
                # Show newest first, up to the configured cap
                k = int(st.session_state.get("max_history_turns", 4)) or 4
                turns = list(reversed(hist[-k:]))
                for i, t in enumerate(turns, start=1):
                    st.markdown(f"**Q{i}:** {t.get('question','')}")
                    st.markdown(f"**A{i}:** {t.get('answer','')}")
                    # tiny source badge line (optional)
                    if t.get("sources"):
                        st.caption("Sources: " + "; ".join({s.get('tag', s.get('source','src')) for s in t["sources"] if isinstance(s, dict)}))

                # Tools row under Chat (History): Clear + Export
                c1, c2 = st.columns([1, 1])

                with c1:
                    if st.button("Clear chat history"):
                        st.session_state["chat_history"] = []
                        # Also ensure history usage is off once cleared
                        st.session_state["use_history"] = False
                        st.rerun()

                with c2:
                    has_history = len(st.session_state.get("chat_history", [])) > 0
                    # Nice title uses the active index dir (fallback to base dir if none)
                    idx_label = st.session_state.get("ACTIVE_INDEX_DIR") or st.session_state.get("BASE_DIR", "rag_store")
                    transcript_title = f"Chat Transcript — {os.path.basename(idx_label)}"
                    chat_md = chat_to_markdown(st.session_state.get("chat_history", []), title=transcript_title)

                    st.download_button(
                        "Export chat (.md)",
                        data=chat_md.encode("utf-8"),
                        file_name="chat_transcript.md",
                        mime="text/markdown",
                        disabled=not has_history,
                    )


            st.divider()
            st.caption(
                "Follow-up question (reuses the same retriever). "
                + ("Using last turns for continuity." if st.session_state.get("use_history") else "History not used for answering.")
            )

            # Enter-to-run for Chat input (mirrors main Q&A)
            st.session_state.setdefault("CHAT_TRIGGER_ANSWER", False)
            def _on_enter_chat():
                st.session_state["CHAT_TRIGGER_ANSWER"] = True

            st.text_input(
                "Your follow-up",
                key="CHAT_QUESTION",
                placeholder="Ask a follow-up…",
                on_change=_on_enter_chat
            )

            chat_go = st.button("Follow-up: Retrieve & Answer", use_container_width=True)
            if chat_go or st.session_state.get("CHAT_TRIGGER_ANSWER"):
                st.session_state["CHAT_TRIGGER_ANSWER"] = False
                q = (st.session_state.get("CHAT_QUESTION") or "").strip()
                if not q:
                    st.warning("Type a follow-up question first.")
                else:
                    # Reuse the same pipeline by setting QUESTION + trigger
                    st.session_state["QUESTION"] = q
                    st.session_state["TRIGGER_ANSWER"] = True
                    st.rerun()

# ---------- EVAL SNAPSHOT (Retrieval-only) ----------
with st.expander("Eval — retrieval snapshot (hit@k & MRR)", expanded=False):
    st.caption("Runs on eval/qa.jsonl (retrieval-only; no LLM). Matches filename + page (if given).")

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        k_eval = st.slider("k (top docs)", 1, 20, int(st.session_state.get("TOP_K", 4)), 1)
    with c2:
        modes = st.multiselect(
            "Retrievers",
            options=["BM25", "Hybrid", "Dense"],
            default=["BM25", "Hybrid", "Dense"],
            help="Compare multiple retrievers side-by-side",
        )
    with c3:
        qpath = st.text_input("Questions file", value="eval/qa.jsonl")

    run_eval = st.button("Run eval", type="primary", use_container_width=True)

    if run_eval:
        persist_dir = st.session_state.get("ACTIVE_INDEX_DIR") or st.session_state.get("BASE_DIR", "rag_store")
        mode_tokens = [m.lower() for m in modes]

        with st.spinner("Evaluating…"):
            summary_df, details_by_mode = run_eval_snapshot(
                qa_path=qpath,
                persist_dir=persist_dir,
                embed_model=EMBED_MODEL,
                modes=mode_tokens,
                k=int(k_eval),
            )

        if summary_df.empty:
            st.warning("No results. Check that an index is loaded and eval/qa.jsonl has data.")
        else:
            st.subheader("Metrics")
            st.write(summary_df.style.format({"hit@k": "{:.2f}", "mrr": "{:.3f}"}))

            st.subheader("MRR by retriever")
            st.bar_chart(summary_df["mrr"])

            with st.expander("Per-question details", expanded=False):
                for mode, df in details_by_mode.items():
                    st.markdown(f"**{mode.upper()}**")
                    st.dataframe(
                        df[["question", "gold_source", "gold_page", "rank", "hit", "mrr"]],
                        use_container_width=True,
                        hide_index=True,
                    )
                    st.divider()

# ---------- FOOTER ----------
st.markdown("---")
st.caption("Scope-first scaffold. Implement M1 → M2 in small commits. Keep it simple.")