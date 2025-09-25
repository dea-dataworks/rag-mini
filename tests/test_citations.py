from langchain_core.documents import Document
from rag_core import build_qa_result
from exports import to_markdown

def test_build_qa_and_markdown_has_citations():
    # Minimal two "retrieved" chunks from different pages of same file
    d1 = Document(page_content="Python created by Guido van Rossum in 1991.", metadata={"source": "doc1.txt", "page": 1})
    d2 = Document(page_content="Python emphasizes readability.", metadata={"source": "doc1.txt", "page": 2})

    pairs = [(d1, 1.2), (d2, 0.8)]
    qa = build_qa_result(
        question="Who created Python?",
        answer="It was created by Guido van Rossum. (doc1.txt p.1)",
        docs_used=[d1, d2],
        pairs=pairs,
        meta={"model": "noop", "retrieval_mode": "bm25", "top_k": 2},
        context_text=d1.page_content + "\n" + d2.page_content,
    )

    # Citations exist and are de-duplicated with page info
    cits = qa.get("citations") or []
    assert len(cits) >= 1
    assert any(c.get("source") == "doc1.txt" for c in cits)

    md = to_markdown(qa)
    # Markdown header contains citations line
    assert "Citations:" in md
    assert "doc1.txt p.1" in md
