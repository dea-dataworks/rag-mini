import json
from pathlib import Path
import random
from langchain_core.documents import Document
import pytest

# Keep tests deterministic (even if we don't use RNG directly)
@pytest.fixture(autouse=True)
def _seed_everything():
    random.seed(0)
    return None

@pytest.fixture(scope="session")
def mini_corpus_dir():
    here = Path(__file__).parent / "fixtures" / "mini_corpus"
    assert here.exists(), f"Missing fixtures at: {here}"
    return here

@pytest.fixture(scope="session")
def mini_docs(mini_corpus_dir):
    """Load .txt files into LangChain Documents with consistent metadata."""
    docs = []
    for p in sorted(mini_corpus_dir.glob("*.txt")):
        text = p.read_text(encoding="utf-8").strip()
        if not text:
            continue
        # mimic .txt ingestion in your app: page=1, ext=txt
        meta = {"source": p.name, "page": 1, "ext": "txt"}
        docs.append(Document(page_content=text, metadata=meta))
    assert len(docs) >= 8
    return docs

@pytest.fixture(scope="session")
def qa_pairs(mini_corpus_dir):
    """Load small question set with gold sources."""
    data = []
    with open(mini_corpus_dir / "qa_small.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    assert 6 <= len(data) <= 12
    return data
