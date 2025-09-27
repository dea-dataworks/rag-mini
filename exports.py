import io
from typing import List, Dict
from guardrails import guard_export_settings

try:
    import pandas as pd
except ImportError as e:
    pd = None  # CSV can still be produced without pandas; Excel will require it.

# ---------- core flatten ----------

def _flatten_qa_rows(qa: Dict) -> List[Dict]:
    """
    Flatten a QA dict to one row per retrieved chunk.
    Repeats QA-level columns so CSV/Excel are analysis-friendly.
    """
    base = {
        "timestamp": qa.get("meta", {}).get("timestamp"),
        "question": qa.get("question", ""),
        "answer": qa.get("answer", ""),
        "model": qa.get("meta", {}).get("model"),
        "retrieval_mode": qa.get("meta", {}).get("retrieval_mode"),
        "top_k": qa.get("meta", {}).get("top_k"),
        "index_name": qa.get("meta", {}).get("index_name"),
        # --- provider provenance ---
        "provider": qa.get("meta", {}).get("provider"),
        "provider_used": qa.get("meta", {}).get("provider_used"),
        "fallback": qa.get("meta", {}).get("fallback"),
        "fallback_reason": qa.get("meta", {}).get("fallback_reason"),
        # --------------------------------
        "num_chunks_used": len(qa.get("chunks", []) or []),
        # optional: a compact citations field
        "citations": ", ".join(
            [f"{c.get('source')}" + (f" p.{c.get('page')}" if c.get('page') else "")
            for c in qa.get("citations", [])]
        ),
    }

    rows = []
    chunks = qa.get("chunks", []) or []
    if not chunks:
        rows.append({**base, "rank": None, "score": None, "source": None, "page": None, "chunk_id": None, "snippet": ""})
        return rows
    for ch in chunks:
        rows.append({
            **base,
            "rank": ch.get("rank"),
            "score": ch.get("score"),
            "source": ch.get("source"),
            "page": ch.get("page"),
            "chunk_id": ch.get("chunk_id"),
            "snippet": ch.get("snippet", ""),
        })
    return rows

# ---------- markdown ----------
def to_markdown(qa: Dict, snippet_max: int = 500) -> str:
    """
    Render a readable Markdown block:
      - header meta
      - guardrail one-liner (if present)
      - answer block
      - table of retrieved chunks
    """
    meta = qa.get("meta", {}) or {}
    header = [
        "### Q&A Export",
        f"- **Timestamp:** {meta.get('timestamp')}",
        f"- **Model:** {meta.get('model') or ''}",
        f"- **Retrieval:** {meta.get('retrieval_mode') or ''} · top_k={meta.get('top_k')}",
        f"- **Index:** {meta.get('index_name') or 'default'}",
    ]

    # Provider provenance (match chat export style)
    prov_sel = meta.get("provider")
    prov_used = meta.get("provider_used") or prov_sel
    if prov_sel or prov_used:
        header.append(f"- **Provider selected:** {prov_sel or ''}")
        suffix = " (fallback)" if meta.get("fallback") else ""
        header.append(f"- **Provider used:** {prov_used or ''}{suffix}")
        if meta.get("fallback_reason"):
            header.append(f"- **Reason:** {meta.get('fallback_reason')}")

    # Add citations line if present
    citations = qa.get("citations", []) or []
    if citations:
        parts = []
        for c in citations:
            src = c.get("source") if isinstance(c, dict) else str(c)
            page = c.get("page") if isinstance(c, dict) else None
            if src:
                if page is not None:
                    parts.append(f"{src} p.{page}")
                else:
                    parts.append(src)
        if parts:
            header.append("- **Citations:** " + ", ".join(parts))

    # --- Guardrail one-liner (if present and not ok) ---
    gr = qa.get("guardrail_primary_status") or {}
    gr_code = (gr.get("code") or "ok").lower()
    gr_msg = gr.get("message") or ""
    guard_line = f"- **Guardrail:** {gr_msg}" if gr_code != "ok" and gr_msg else None

    question = qa.get("question", "")
    answer = qa.get("answer", "")

    # Chunk table
    rows = qa.get("chunks", []) or []
    table_lines = [
        "| # | Score | Source | Page | Snippet |",
        "|---:|-----:|--------|-----:|---------|",
    ]
    for ch in rows:
        snip = (ch.get("snippet") or "")[:snippet_max].replace("\n", " ")
        score_val = ch.get("score")
        score = "" if score_val is None else f"{score_val:.4f}"
        page = "" if ch.get("page") in (None, "") else str(ch["page"])
        table_lines.append(
            f"| {ch.get('rank','')} | {score} | {ch.get('source','')} | {page} | {snip} |"
        )

    # Build Markdown sections
    md = []
    md.append("\n".join(header))
    if guard_line:
        md.append(guard_line)
    md.append("\n**Question**\n\n> " + question)
    md.append("\n**Answer**\n\n" + answer)
    md.append("\n**Retrieved Chunks**\n")
    md.append("\n".join(table_lines))

    return "\n".join(md).strip()

# ---------- CSV / Excel ----------

def to_csv_bytes(qa: Dict) -> bytes:
    """
    Return a UTF-8 CSV (one row per chunk).
    Uses pandas if available; falls back to manual CSV otherwise.
    """
    rows = _flatten_qa_rows(qa)
    if pd is not None:
        df = pd.DataFrame(rows)
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        return buf.getvalue().encode("utf-8")
    # fallback without pandas
    import csv
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue().encode("utf-8")

def to_excel_bytes(qa: Dict) -> bytes:
    """
    Return an Excel file (XLSX) in-memory.
    Requires pandas + openpyxl.
    """
    if pd is None:
        raise RuntimeError("Excel export requires pandas (and openpyxl).")
    rows = _flatten_qa_rows(qa)
    df = pd.DataFrame(rows)
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as xw:
        df.to_excel(xw, index=False, sheet_name="QA")
    return out.getvalue()

# --- Chat export helper ---
def chat_to_markdown(chat_history, title="Chat Transcript"):
    """
    Convert st.session_state['chat_history'] into a Markdown string.
    Each turn includes Q, A, sources, run_settings, and (if present) a guardrail note.
    """
    lines = [f"# {title}", ""]
    for i, turn in enumerate(chat_history or [], 1):
        ts = turn.get("ts") or turn.get("timestamp") or turn.get("time") or turn.get("created_at") or ""
        q = (turn.get("question") or "").strip()
        a = (turn.get("answer") or "").strip()
        sources = turn.get("sources") or []
        run_settings = guard_export_settings(turn.get("run_settings") or {})

        # --- NEW: primary guardrail per turn (if present)
        gr = turn.get("guardrail_primary_status") or {}
        gr_code = (gr.get("code") or "ok").lower()
        gr_msg  = gr.get("message") or ""

        lines.append(f"## Turn {i}")
        if ts:
            lines.append(f"*Time:* {ts}")

        if run_settings:
            lines.append("")
            lines.append("**Run settings:**")
            for k, v in run_settings.items():
                lines.append(f"- {k}: {v}")

        # --- Provider provenance (if present) ---
        prov_sel = turn.get("provider_selected")
        prov_used = turn.get("provider_used") or prov_sel
        if prov_sel or prov_used:
            lines.append("")
            lines.append("**Provider:**")
            if prov_sel:
                lines.append(f"- selected: {prov_sel}")
            if prov_used:
                suffix = " (fallback)" if turn.get("fallback") else ""
                lines.append(f"- used: {prov_used}{suffix}")
            if turn.get("fallback_reason"):
                lines.append(f"- reason: {turn.get('fallback_reason')}")

        # add a compact guardrail note if not OK
        if gr_code != "ok" and gr_msg:
            lines.append("")
            lines.append(f"**Guardrail:** {gr_msg}")


        lines.append("")
        lines.append("**Q:** " + (q if q else "_(empty)_"))
        lines.append("")
        lines.append("**A:** " + (a if a else "_(empty)_"))

        if sources:
            lines.append("")
            lines.append("**Sources:**")
            for s in sources:
                if isinstance(s, dict):
                    ref = s.get("source") or s.get("path") or s.get("doc_id") or str(s)
                    page = s.get("page")
                    if page is not None:
                        lines.append(f"- {ref} (p.{page})")
                    else:
                        lines.append(f"- {ref}")
                else:
                    lines.append(f"- {s}")
        lines.append("")
    return "\n".join(lines)

