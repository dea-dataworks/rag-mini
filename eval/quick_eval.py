import os
import json
import pandas as pd
from langchain_core.documents import Document
from index_admin import get_active_index
from rag_core import retrieve, load_vectorstore_if_exists  # uses your core helpers


def _tag(d: Document) -> str:
    m = d.metadata or {}
    src = m.get("source", "unknown")
    pg = m.get("page", None)
    return f"{src} p.{pg}" if pg else src

def _read_jsonl(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def _baseline_knn(vs, query: str, k: int):
    try:
        pairs = vs.similarity_search_with_score(query, k=k)
        docs = [d for (d, _s) in pairs]
    except Exception:
        docs = vs.similarity_search(query, k=k)
    return docs

def _hit_mrr(gold_tags, ranked_docs, k: int):
    tags = [_tag(d) for d in ranked_docs]
    first_rank = None
    for i, t in enumerate(tags, start=1):
        if any(gt == t for gt in gold_tags):
            first_rank = i
            break
    hit_at_k = 1 if (first_rank is not None and first_rank <= k) else 0
    mrr = 1.0 / first_rank if first_rank is not None else 0.0
    return hit_at_k, mrr

def _resolve_active_dir(persist_dir: str) -> str | None:
    """
    If 'persist_dir' is a base (e.g., rag_store/user), return its active idx_* dir.
    If it's already an idx_* folder, return it as-is.
    """
    base = os.path.basename(persist_dir.rstrip(os.sep))
    if base.startswith("idx_"):
        return persist_dir
    return get_active_index(base=persist_dir)

def run_quick_eval(qpath: str, k_eval: int, mmr_lambda: float, embed_model: str, persist_dir: str = "rag_store"):
    """Compare baseline KNN vs hybrid retrieval on hit@k and MRR for a JSONL QA set."""
    qa_rows = _read_jsonl(qpath)
    if not qa_rows:
        return pd.DataFrame(), {"msg": "No questions", "k": k_eval}

    active_dir = _resolve_active_dir(persist_dir)
    if not active_dir:
        return pd.DataFrame(), {"msg": f"No active index under base '{persist_dir}'", "k": k_eval}

    vs = load_vectorstore_if_exists(embed_model=embed_model, persist_dir=active_dir)
    if vs is None:
        return pd.DataFrame(), {"msg": f"No index at '{active_dir}'", "k": k_eval}

    rows = []
    for i, ex in enumerate(qa_rows, start=1):
        q = (ex.get("question") or "").strip()
        gold = ex.get("must_include_sources", []) or []
        if not q:
            continue

        baseline_docs = _baseline_knn(vs, q, k=k_eval)
        b_hit, b_mrr = _hit_mrr(gold, baseline_docs, k_eval)

        try:
            fused_pairs = retrieve(vs, q, k=k_eval, mmr_lambda=mmr_lambda, mode="hybrid")
        except TypeError:
            fused_pairs = retrieve(vs, q, k=k_eval)
        hybrid_docs = [d for (d, _s) in fused_pairs]
        h_hit, h_mrr = _hit_mrr(gold, hybrid_docs, k_eval)

        rows.append({
            "qid": i,
            "question": q,
            "gold": "; ".join(gold),
            "k": k_eval,
            "baseline_hit@k": b_hit,
            "baseline_mrr": round(b_mrr, 4),
            "hybrid_hit@k": h_hit,
            "hybrid_mrr": round(h_mrr, 4),
            "improved_mrr": 1 if h_mrr > b_mrr else 0,
        })

    df = pd.DataFrame(rows)
    summary = {
        "k": k_eval,
        "mean_baseline_hit": float(df["baseline_hit@k"].mean()) if not df.empty else 0.0,
        "mean_baseline_mrr": float(df["baseline_mrr"].mean()) if not df.empty else 0.0,
        "mean_hybrid_hit": float(df["hybrid_hit@k"].mean()) if not df.empty else 0.0,
        "mean_hybrid_mrr": float(df["hybrid_mrr"].mean()) if not df.empty else 0.0,
        "improved_mrr_count": int(df["improved_mrr"].sum()) if not df.empty else 0,
        "total": int(len(df)),
    }
    return df, summary
