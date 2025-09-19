"""
Mini eval runner: Baseline k-NN vs Hybrid (BM25 + Dense via RRF)
Outputs: eval/results_YYYYMMDD.csv
Usage:
  python eval/run_eval.py --questions eval/qa.jsonl --k 4 --persist_dir rag_store --embed_model nomic-embed-text
"""

from __future__ import annotations
import argparse, json, os, sys
from datetime import datetime
from typing import List, Tuple

import pandas as pd

# Import your core pieces
# - retrieve (hybrid/dense), normalize_hits (utility)
# - load_vectorstore_if_exists (open existing index)
from rag_core import retrieve, normalize_hits, get_embeddings, build_or_load_vectorstore  # hybrid path uses this under the hood
from rag_core import load_vectorstore_if_exists  # to open an existing store
from langchain_chroma import Chroma  # baseline direct access
from langchain_core.documents import Document


def _tag(doc: Document) -> str:
    """Make a 'filename p.X' tag from metadata."""
    m = doc.metadata or {}
    src = m.get("source", "unknown")
    pg = m.get("page", None)
    return f"{src} p.{pg}" if pg else src


def _read_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _hit_mrr(gold_tags: List[str], ranked_docs: List[Document], k: int) -> Tuple[int, float]:
    """
    gold_tags: list of strings like 'udhr.pdf p.1'
    ranked_docs: rank-ordered list of Documents
    Returns (hit@k, MRR)
    """
    # map each rank to tag
    tags = [_tag(d) for d in ranked_docs]
    # First relevant rank (1-based); else None
    first_rank = None
    for i, t in enumerate(tags, start=1):
        if any(gt == t for gt in gold_tags):
            first_rank = i
            break
    hit_at_k = 1 if (first_rank is not None and first_rank <= k) else 0
    mrr = 1.0 / first_rank if first_rank is not None else 0.0
    return hit_at_k, mrr


def _baseline_knn(vs: Chroma, query: str, k: int) -> List[Document]:
    """
    Baseline: straight k-NN over dense vectors (no MMR, no BM25, no fusion).
    We fetch exactly k results using Chroma's k-NN.
    """
    try:
        pairs = vs.similarity_search_with_score(query, k=k)  # [(Document, score)]
        docs = [d for (d, _s) in pairs]
    except Exception:
        docs = vs.similarity_search(query, k=k)
    return docs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=str, default="eval/qa.jsonl", help="Path to eval/qa.jsonl")
    parser.add_argument("--k", type=int, default=4, help="top-k to evaluate")
    parser.add_argument("--persist_dir", type=str, default="rag_store", help="Directory of the persisted Chroma store")
    parser.add_argument("--embed_model", type=str, default="nomic-embed-text", help="Embedding model name for opening the store")
    parser.add_argument("--mmr_lambda", type=float, default=0.7, help="MMR lambda (hybrid path)")
    args = parser.parse_args()

    # 1) Load questions
    qa = _read_jsonl(args.questions)
    if not qa:
        print(f"[eval] No questions found at {args.questions}")
        sys.exit(1)

    # 2) Open existing vectorstore (no re-embedding)
    vs = load_vectorstore_if_exists(embed_model=args.embed_model, persist_dir=args.persist_dir)
    if vs is None:
        print(f"[eval] No existing index at '{args.persist_dir}'. Build it first in the app.")
        sys.exit(1)

    # 3) Evaluate each question
    rows = []
    improved_count = 0
    for i, ex in enumerate(qa, start=1):
        q = ex.get("question", "").strip()
        gold = ex.get("must_include_sources", []) or []
        if not q:
            continue

        # Baseline: k-NN (dense only, no MMR)
        baseline_docs = _baseline_knn(vs, q, k=args.k)
        b_hit, b_mrr = _hit_mrr(gold, baseline_docs, k=args.k)

        # Hybrid (BM25 + Dense via RRF) using your retrieve()
        try:
            fused_pairs = retrieve(
                vs, q, k=args.k, mmr_lambda=args.mmr_lambda, mode="hybrid"
            )  # -> list[(Document, score)]
        except TypeError:
            fused_pairs = retrieve(vs, q, k=args.k)  # fallback signature
        hybrid_docs = [d for (d, _s) in fused_pairs]
        h_hit, h_mrr = _hit_mrr(gold, hybrid_docs, k=args.k)

        improved = 1 if (h_mrr > b_mrr) else 0
        improved_count += improved

        rows.append({
            "qid": i,
            "question": q,
            "gold": "; ".join(gold),
            "k": args.k,
            "baseline_hit@k": b_hit,
            "baseline_mrr": round(b_mrr, 4),
            "hybrid_hit@k": h_hit,
            "hybrid_mrr": round(h_mrr, 4),
            "improved_mrr": improved,
        })

    # 4) Save CSV
    ts = datetime.now().strftime("%Y%m%d")
    out_path = os.path.join("eval", f"results_{ts}.csv")
    os.makedirs("eval", exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)

    # 5) Print short summary
    mean_b_hit = df["baseline_hit@k"].mean() if not df.empty else 0.0
    mean_b_mrr = df["baseline_mrr"].mean() if not df.empty else 0.0
    mean_h_hit = df["hybrid_hit@k"].mean() if not df.empty else 0.0
    mean_h_mrr = df["hybrid_mrr"].mean() if not df.empty else 0.0

    print(f"[eval] Saved: {out_path}")
    print(f"[eval] mean hit@{args.k} — baseline: {mean_b_hit:.3f} | hybrid: {mean_h_hit:.3f}")
    print(f"[eval] mean MRR      — baseline: {mean_b_mrr:.3f} | hybrid: {mean_h_mrr:.3f}")
    print(f"[eval] improved MRR on {int(df['improved_mrr'].sum())}/{len(df)} queries")


if __name__ == "__main__":
    main()
