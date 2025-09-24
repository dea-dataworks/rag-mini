import uuid
import streamlit as st
import streamlit.components.v1 as components

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
