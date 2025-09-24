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
