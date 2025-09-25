from statistics import mean
from rag_core import SimpleBM25, chunk_documents

def _hit_at_k(bm25, query, gold_sources, k):
    top = bm25.score(query, top_m=k)
    top_srcs = [ (d.metadata or {}).get("source") for d, _s in top ]
    return any(src in gold_sources for src in top_srcs)

def _mrr(bm25, query, gold_sources, k):
    top = bm25.score(query, top_m=k)
    for rank, (d, _s) in enumerate(top, start=1):
        src = (d.metadata or {}).get("source")
        if src in gold_sources:
            return 1.0 / rank
    return 0.0

def test_bm25_hit_and_mrr(mini_docs, qa_pairs):
    # Use chunks as retrieval units (closer to app behavior)
    chunks = chunk_documents(mini_docs, size=160, overlap=30)
    bm25 = SimpleBM25(chunks)

    ks = 5
    hits = []
    mrrs = []
    for item in qa_pairs:
        q = item["q"]
        gold = set(item["gold"])
        hits.append(1 if _hit_at_k(bm25, q, gold, ks) else 0)
        mrrs.append(_mrr(bm25, q, gold, ks))

    # sanity: at least 4/6 queries hit@5 on this curated corpus
    assert sum(hits) >= 4
    # mrr should be healthy (>0.5 on average for these easy queries)
    assert mean(mrrs) >= 0.5
