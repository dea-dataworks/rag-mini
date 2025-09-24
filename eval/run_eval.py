import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import pandas as pd
from langchain_core.documents import Document

from rag_core import load_vectorstore_if_exists, get_embeddings, retrieve

@dataclass
class QAPair:
    question: str
    source: Optional[str] = None
    page: Optional[int] = None
    answer: Optional[str] = None  # unused for scoring

def _load_qa_jsonl(path: str | Path) -> List[QAPair]:
    items: List[QAPair] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            items.append(
                QAPair(
                    question=obj.get("question", ""),
                    answer=obj.get("answer"),
                    # tolerate "filename" or "source"
                    source=(obj.get("source") or obj.get("filename")),
                    page=obj.get("page"),
                )
            )
    return items

def _is_match(doc: Document, gold_source: Optional[str], gold_page: Optional[int]) -> bool:
    if not gold_source:
        return False
    md = doc.metadata or {}
    src = (md.get("source") or "").strip()
    if not src:
        return False
    if Path(src).name != Path(gold_source).name:
        return False
    if gold_page is not None:
        try:
            return int(md.get("page")) == int(gold_page)
        except Exception:
            return False
    return True

def _rank_of_first_match(docs: List[Document], gold_source: Optional[str], gold_page: Optional[int]) -> Optional[int]:
    for i, d in enumerate(docs, start=1):
        if _is_match(d, gold_source, gold_page):
            return i
    return None

def run_eval_snapshot(
    qa_path: str | Path,
    persist_dir: str,
    embed_model: str,
    modes: List[str],            # ["bm25","hybrid","dense"] any subset
    k: int = 5,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Returns:
      summary_df: index=mode, columns=["n","hit@k","mrr"]
      details_by_mode: dict[mode]-> DataFrame with per-question ranks/hits
    """
    qa = _load_qa_jsonl(qa_path)
    if not qa:
        return pd.DataFrame(), {}

    vs = load_vectorstore_if_exists(embed_model=embed_model, persist_dir=persist_dir)
    if vs is None:
        # graceful empty result if there is no active index
        return pd.DataFrame(), {}

    results_summary = []
    details_by_mode: Dict[str, pd.DataFrame] = {}

    for mode in modes:
        rows = []
        for item in qa:
            pairs = retrieve(vs, item.question, k=k, mode=mode)
            docs = [d for (d, _s) in pairs]
            rank = _rank_of_first_match(docs, item.source, item.page)
            hit = 1 if rank is not None else 0
            mrr = (1.0 / rank) if rank is not None else 0.0
            rows.append({
                "question": item.question,
                "gold_source": item.source,
                "gold_page": item.page,
                "rank": rank,
                "hit": hit,
                "mrr": mrr,
            })
        df = pd.DataFrame(rows)
        n = len(df)
        hit_at_k = float(df["hit"].mean()) if n else 0.0
        mrr = float(df["mrr"].mean()) if n else 0.0
        results_summary.append({"mode": mode, "n": n, "hit@k": hit_at_k, "mrr": mrr})
        details_by_mode[mode] = df

    summary_df = pd.DataFrame(results_summary).set_index("mode").sort_values("mrr", ascending=False)
    return summary_df, details_by_mode
