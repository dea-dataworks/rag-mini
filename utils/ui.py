import uuid
import streamlit.components.v1 as components
import streamlit as st
from exports import to_markdown, to_csv_bytes, to_excel_bytes

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

def sidebar_pipeline_diagram():
    with st.expander("📈 How it works (pipeline)"):
        st.markdown(
            """
            **Ingest → Chunk → Embed → Index → Retrieve → Answer → Cite**

            ```
            Upload files
                 │
                 ▼
            Split into chunks (size/overlap)
                 │
                 ▼
            Embed chunks → Vector store (persisted)
                 │
                 ▼
            Query → BM25 + Dense (RRF/MMR)
                 │
                 ▼
            Prompt LLM with top-k chunks
                 │
                 ▼
            Answer + sources (filename p.X)
            ```
            """
        )

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

def render_cited_chunks_expander(docs_only, snippet_len: int = 240):
    """
    Compact expander that lists the cited chunks (Documents) used for the answer.
    Shows file/page and a short snippet. Pure presentation; no side effects.
    """
    if not docs_only:
        return

    with st.expander("Show cited chunks", expanded=False):
        for d in docs_only:
            md = d.metadata or {}
            src = md.get("source", "unknown")
            pg = md.get("page")
            text = (d.page_content or "").replace("\n", " ")
            snippet = text[:snippet_len] + ("…" if len(text) > snippet_len else "")
            left = f"**{src}**" + (f" p.{pg}" if pg else "")
            st.markdown(f"- {left}\n\n  {snippet}")

