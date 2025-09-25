from langchain_core.documents import Document
from guardrails import evaluate_guardrails, pick_primary_status

def test_no_context_blocks():
    statuses = evaluate_guardrails(
        question="Tell me everything about the universe.",
        context_text="",  # too thin
        docs_used=[],
        min_chars=40,
    )
    primary = pick_primary_status(statuses)
    assert primary.get("code") == "no_context"
    assert primary.get("severity") == "block"

def test_missing_citation_warns():
    d = Document(page_content="Some factual text.", metadata={"source": "doc.txt", "page": 1})
    statuses = evaluate_guardrails(
        question="What is in the doc?",
        context_text=d.page_content,
        docs_used=[d],
        answer_text="Here is an answer without citation.",
        min_chars=10,
    )
    codes = [s["code"] for s in statuses]
    assert "no_citation" in codes
