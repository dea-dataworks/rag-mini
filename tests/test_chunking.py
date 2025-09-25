from rag_core import chunk_documents

def test_chunking_metadata_and_ids(mini_docs):
    chunks = chunk_documents(mini_docs, size=120, overlap=20)
    assert len(chunks) >= len(mini_docs)  # each file → ≥1 chunk

    for ch in chunks:
        md = ch.metadata or {}
        # Required metadata present
        assert "source" in md
        assert "chunk_id" in md or "id" in md
        # Stable chunk index
        assert isinstance(md.get("chunk_index"), int) and md["chunk_index"] >= 1
        # TXT path keeps page=1
        if md.get("ext") == "txt":
            assert md.get("page") == 1

def test_avg_chunk_len_reasonable(mini_docs):
    chunks = chunk_documents(mini_docs, size=120, overlap=20)
    total_chars = sum(len(c.page_content or "") for c in chunks)
    avg = total_chars / len(chunks)
    # sanity window for tiny texts
    assert 40 <= avg <= 240
