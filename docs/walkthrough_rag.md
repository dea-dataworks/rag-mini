# RAG Explorer — Retrieval Pipeline Walkthrough

## 🔹 Step 1: High-Level Map (Upload → Chunk → Embed → Store → Retrieve → Answer)

1. **User uploads documents**
   - Streamlit `file_uploader()` collects `.pdf`, `.txt`, or `.docx` files.
   - Files held temporarily in memory until indexing.

2. **Index management**
   - “Manage index” panel controls whether to **Rebuild** or **Load**.
   - `make_fresh_index_dir()` in `rag_core.py` creates new timestamped folder.
   - `load_vectorstore_if_exists()` loads previous FAISS index.

3. **File ingestion & preprocessing**
   - `build_index_from_files()` parses files (PyMuPDF, docx, txt).
   - Returns raw text content ready for chunking.

4. **Chunking**
   - Text split into overlapping segments via `RecursiveCharacterTextSplitter()`.
   - Parameters: `chunk_size`, `chunk_overlap` (from sidebar inputs).

5. **Embedding generation**
   - `get_embeddings()` wraps `OllamaEmbeddings("nomic-embed-text")`.
   - Produces vector embeddings for each chunk.

6. **Vector store creation**
   - `FAISS.from_texts(chunks, embedding)` builds the index.
   - Index saved under `rag_store/idx_YYYYMMDD_HHMMSS/`.
   - Manifest records chunk count, per-file stats.

7. **Active index pointer**
   - `save_active_pointer()` stores active folder path for reloading on startup.

8. **User query**
   - Input captured by `st.text_input(key="QUESTION")`.
   - Buttons “Preview Top Sources” or “Retrieve & Answer” trigger retrieval.

9. **Retrieval**
   - `_retrieve_hits()` calls `retrieve()` in `rag_core.py`.
   - Runs FAISS (dense) or hybrid search with `k=top_k`, MMR optional.
   - Returns top-k results with similarity scores.

10. **Filtering & sanitization**
    - `normalize_hits()`, `filter_by_score()`, `cap_per_source()` tidy results.
    - `sanitize_chunks()` removes unsafe or irrelevant text.

11. **Prompt building**
    - `build_prompt()` merges question, retrieved chunks, and optional chat history.

12. **LLM inference**
    - `call_llm()` runs the chosen provider (Ollama or OpenAI).
    - `_attempt_with_timeout()` ensures responsive execution.

13. **Answer packaging**
    - `build_qa_result()` creates Q&A payload (answer, docs, timings, metadata).

14. **Display & export**
    - `st.write(answer)` shows output.
    - `render_cited_chunks_expander()` reveals retrieved sources.
    - `render_export_buttons()` allows saving results (Markdown, CSV, Excel, JSON).

15. **Evaluation (optional)**
    - “Evaluation” expander runs `run_eval_snapshot()` to compute hit@k and MRR metrics.

---

## ⚙️ Step 2: Light Instrumentation

Add `logging.info()` at key steps to trace pipeline activity in console:

```python
import logging
logging.basicConfig(level=logging.INFO)

logging.info(f"📂 Uploaded {len(uploaded_files)} files")
logging.info(f"🧩 Created {len(chunks)} chunks")
logging.info("🔢 Generating embeddings...")
logging.info("💾 Index built and saved")
logging.info(f"❓ Query: {question}")
logging.info(f"📚 Retrieved {len(docs)} top results")
logging.info("🧠 Calling LLM...")
logging.info("✅ Answer generated successfully")
```

---

## 🧩 Step 3: Critical Path (Single Query Trace)

| Stage | File | Function | Description |
|-------|------|-----------|-------------|
| Upload | `app.py` | `file_uploader()` | User uploads document(s) |
| Build Index | `rag_core.py` | `build_index_from_files()` | Load, chunk, embed, persist |
| Retrieve | `app.py` → `rag_core.py` | `_retrieve_hits()` → `retrieve()` | Query FAISS index |
| Prompt | `llm_chain.py` | `build_prompt()` | Combine context + question |
| LLM | `llm_chain.py` | `call_llm()` | Run Mistral/OpenAI model |
| Output | `app.py` | `render_cited_chunks_expander()` | Display answer + sources |

---

**Notes:**
- This walkthrough corresponds to version v0.2 of *RAG Explorer*.
- Logs and structure can be reused across your other apps (Dashboard, Forecasting) with minor renaming.
- Keep this document updated after major refactors.
