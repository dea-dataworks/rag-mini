import os
import uuid
import streamlit.components.v1 as components
import streamlit as st
from exports import to_markdown, to_csv_bytes, to_excel_bytes, chat_to_markdown

def render_copy_button(label: str, text: str, key: str | None = None):
    _key = key or str(uuid.uuid4()).replace("-", "")
    escaped = (text or "").replace("\\", "\\\\").replace("'", "\\'").replace("\n", "\\n")
    html = f"""
    <button id="btn-{_key}" style="margin-right:8px; padding:4px 8px; font-size:12px; cursor:pointer;">
      {label}
    </button>
    <script>
      const btn = document.getElementById('btn-{_key}');
      if (btn) {{
        btn.onclick = async () => {{
          try {{
            await navigator.clipboard.writeText('{escaped}');
            btn.innerText = '{label} ✓';
            setTimeout(() => btn.innerText = '{label}', 1200);
          }} catch (e) {{
            btn.innerText = '{label} (copy failed)';
            setTimeout(() => btn.innerText = '{label}', 1500);
          }}
        }};
      }}
    </script>
    """
    components.html(html, height=38)

# def sidebar_pipeline_diagram():
#     with st.expander("**Instructions**", expanded=False):
#         st.markdown(
#             """
#             1. Upload one or more files.
#             2. Type your question in the box.
#             3. Read the answer and check cited sources below.

#             _Tip: Switch active indexes in the sidebar if you’ve built more than one._
#             """
#         )

def render_export_buttons(qa: dict):
    """Render MD/CSV/Excel download buttons for a structured QA dict."""
    md = to_markdown(qa)
    csv_b = to_csv_bytes(qa)
    try:
        xls_b = to_excel_bytes(qa)
        have_xls = True
    except Exception:
        xls_b = b""
        have_xls = False

    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button("Download Markdown", data=md, file_name="answer.md", mime="text/markdown")
    with c2:
        st.download_button("Download CSV", data=csv_b, file_name="answer.csv", mime="text/csv")
    with c3:
        st.download_button(
            "Download Excel",
            data=xls_b,
            file_name="answer.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            disabled=not have_xls,
        )

def render_copy_row(answer_text: str, citations_text: str):
    """Two copy-to-clipboard buttons in a row (answer + citations)."""
    c1, c2 = st.columns(2)
    with c1:
        render_copy_button("Copy answer", answer_text or "", key="copy_answer_btn")
    with c2:
        render_copy_button("Copy citations", citations_text or "", key="copy_cites_btn")

# --- Why-this-answer panel ---
def render_why_this_answer(qa: dict, min_items: int = 3, max_items: int = 5):
    """
    Compact, always-visible panel that lists top retrieved chunks under the answer.
    Shows file/page, score, role label, and a one-liner. Roles are annotations only.
    """
    items = (qa.get("retrieved_chunks") or qa.get("chunks") or []) or []
    if not items:
        return
    # enforce bounds from settings if available
    try:
        from utils.settings import WHY_PANEL_MIN, WHY_PANEL_MAX
        min_items = WHY_PANEL_MIN
        max_items = WHY_PANEL_MAX
    except Exception:
        pass

    items = items[:max(max_items, min_items)] if len(items) >= min_items else items

    st.markdown("#### Why this answer")
    st.caption("Top retrieved chunks that informed the response (roles are annotations, not citations).")

    for r in items[:max_items]:
        src  = r.get("source", "unknown")
        pg   = r.get("page")
        sc   = r.get("score")
        role = (r.get("role") or "context").capitalize()
        # UX copy kept in UI layer on purpose
        role_line = {
            "Definitional": "Defines the key term(s) the answer relies on.",
            "Fact source": "Provides numbers/dates quoted or referenced.",
            "Context": "Adds surrounding context that shaped the phrasing.",
        }.get(role, "Helps ground the answer.")
        left  = f"**{src}**" + (f" p.{pg}" if pg else "")
        right = f"score: {sc:.4f}" if isinstance(sc, (float, int)) else "score: —"
        st.markdown(f"- {left} — _{role}_ · {right}\n  ↳ {role_line}")

def render_cited_chunks_expander(
    docs_only,
    snippet_len: int = 240,
    title: str = "Sources used in the answer",
    expanded: bool = False,
):
    """
    Compact expander that lists the cited chunks (Documents) used for the answer.
    Shows file/page and a short snippet. Pure presentation; no side effects.
    """
    if not docs_only:
        return

    with st.expander(title, expanded=expanded):
        # One-time PDF note inside the expander
        render_pdf_limit_note_for_docs(docs_only)

        for d in docs_only:
            md = d.metadata or {}
            src = md.get("source", "unknown")
            pg = md.get("page")
            text = (d.page_content or "").replace("\n", " ")
            snippet = text[:snippet_len] + ("…" if len(text) > snippet_len else "")
            left = f"**{src}**" + (f" p.{pg}" if pg else "")
            st.markdown(f"- {left}\n\n  {snippet}")

def render_pdf_limit_note_for_uploads(uploaded_files):
    """One-time UX note when user has uploaded PDFs."""
    if uploaded_files and any(f.name.lower().endswith(".pdf") for f in uploaded_files):
        st.info(
            "Heads-up: PDF **images and tables aren’t parsed yet**. "
            "Only the text layer is indexed (scanned PDFs may yield little/no text)."
        )

def render_pdf_limit_note_for_docs(docs):
    """One-time UX note when a list of LangChain Documents includes any PDF source."""
    if any(((d.metadata or {}).get("source","").lower().endswith(".pdf")) for d in (docs or [])):
        st.info("Note: For PDFs, **images/tables aren’t parsed**—snippets and citations refer to text only.")

def render_dev_metrics(qa: dict):
    """
    Collapsible panel showing latency breakdown + retrieval score stats for one QA turn.
    Intended for Advanced/dev use only.
    """
    if not qa:
        return

    metrics = qa.get("metrics", {}) or {}
    times = metrics.get("timings", {}) or {}
    scores = metrics.get("scores", {}) or {}

    with st.expander("Evaluation / dev metrics", expanded=False):
        st.caption("Latency breakdown and retrieval score stats (per question).")

        if times:
            st.markdown("**Timings (ms)**")
            t_rows = [{"Step": k, "ms": v} for k, v in times.items()]
            st.dataframe(t_rows, width='stretch', hide_index=True)

        if scores:
            st.markdown("**Score stats**")
            s_rows = [{"Stat": k, "Value": v} for k, v in scores.items()]
            st.dataframe(s_rows, width='stretch', hide_index=True)

        vals = [
            r.get("score") for r in qa.get("retrieved_chunks", [])
            if isinstance(r.get("score"), (float, int))
        ]
        if vals:
            st.bar_chart(vals, width='stretch', height=120)

def render_session_export(chat_history, idx_label: str):
    """Render a download button for the full chat transcript with provenance."""
    if not chat_history:
        return
    transcript_title = f"Chat Transcript — {os.path.basename(idx_label)}"
    chat_md = chat_to_markdown(chat_history, title=transcript_title)
    st.download_button(
        "Export full session (.md)",
        data=chat_md.encode("utf-8"),
        file_name="chat_transcript.md",
        mime="text/markdown",
    )


# ------------------------------
# Guardrail banners + settings export
# ------------------------------

def render_guardrail_banner(status: dict | None):
    """
    Show a compact banner based on a guardrail status:
      - severity: "block" -> st.error, "warn" -> st.warning, "info" -> st.info
      - code: "no_context" | "prompt_injection_warning" | "source_conflict" | "no_citation" | "ok"
    """
    if not status:
        return
    code = (status.get("code") or "ok").lower()
    sev  = (status.get("severity") or "info").lower()
    msg  = status.get("message") or "OK"

    # Prefer warn over block; only block on explicit "block"
    if sev == "block":
        st.error(msg, icon="🚫")
    elif sev == "warn":
        # Make copy a touch clearer per code
        if code == "prompt_injection_warning":
            st.warning(msg, icon="🛡️")
        elif code == "source_conflict":
            st.warning(msg, icon="⚖️")
        elif code == "no_citation":
            st.warning(msg, icon="🔎")
        else:
            st.warning(msg, icon="⚠️")
    else:
        if code != "ok":
            st.info(msg, icon="ℹ️")
        # if ok, show nothing—keeps UI clean

# def get_exportable_settings(state) -> dict:
#     """
#     Snapshot of run settings safe for persistence/export. 
#     Redacts secrets and skips transient UI-only fields.
#     """
#     s = state or {}

#     out = {
#         "provider": s.get("LLM_PROVIDER", "ollama"),
#         "model": s.get("LLM_MODEL", "mistral"),
#         "top_k": int(s.get("TOP_K", 4) or 4),
#         "retrieval_mode": s.get("RETRIEVE_MODE", "dense"),
#         "chunk_size": int(s.get("CHUNK_SIZE", 800) or 800),
#         "chunk_overlap": int(s.get("CHUNK_OVERLAP", 120) or 120),
#         "mmr_lambda": float(s.get("MMR_LAMBDA", 0.7) or 0.7),
#         "use_history": bool(s.get("use_history", False)),
#         "max_history_turns": int(s.get("max_history_turns", 4) or 4),
#         "use_score_threshold": bool(s.get("USE_SCORE_THRESH", False)),
#         "score_threshold": float(s.get("SCORE_THRESH", 0.4) or 0.4),
#         "use_per_source_cap": bool(s.get("USE_SOURCE_CAP", False)),
#         "per_source_cap": int(s.get("PER_SOURCE_CAP", 2) or 2),
#         "sanitize_retrieved": bool(s.get("SANITIZE_RETRIEVED", True)),
#     }

#     # Never include secrets; if present, mark redacted.
#     if s.get("OPENAI_API_KEY"):
#         out["openai_key"] = "[REDACTED]"

#     return out

# --- Provider / fallback toast helper ---
def render_provider_fallback_toast(reason: str, from_provider: str, to_provider: str):
    """
    Show a small, consistent toast when we auto-fallback providers.
    Keep it low-key on purpose (info-level).
    """
    msg = f"Switched from **{from_provider}** → **{to_provider}** automatically."
    if reason:
        msg += f" ({reason})"
    st.info(msg)

# --- Index switcher helpers ---

def render_index_switcher_header():
    """
    Small, consistent header for the Index switcher block.
    """
    st.markdown("**Active index (base)**")
    st.caption("Switch among named bases under `rag_store/`. "
               "Each base can hold multiple timestamped sub-indexes.")

def render_index_switched(prev: str | None, new: str | None):
    """
    One-liner toast when the user switches the base/index.
    """
    prev_lbl = (prev or "default")
    new_lbl = (new or "default")
    st.info(f"Active index switched: `{prev_lbl}` → `{new_lbl}`")
