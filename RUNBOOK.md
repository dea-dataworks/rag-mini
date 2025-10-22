# RAG Explorer — RUNBOOK

## Overview
This document captures environment details, model configuration, and versioning information for the **RAG Explorer** app.

_Last updated: 2025-10-22_

---

## 1. Environment Summary

| Component | Detail |
|------------|--------|
| **Python** | 3.12.x |
| **Platform** | Windows 10/11 |
| **Virtual Environment** | `.venv` or `.venv_labs` |
| **App entry** | `streamlit run app.py` |
| **Primary Purpose** | Retrieval-Augmented Generation (RAG) document QA |

---

## 2. Library Versions (as of current environment)

| Library | Version | Notes |
|----------|----------|-------|
| **streamlit** | 1.50.0 | UI framework (Snowflake Inc) |
| **langchain** | 0.3.26 | Core LLM chaining |
| **langchain-ollama** | 0.3.3 | Ollama integration for LangChain |
| **faiss-cpu** | 1.12.0 | Vector store backend |
| **numpy** | ≥1.26 | Array ops |
| **pandas** | ≥2.2 | Data handling |
| **protobuf / pyarrow / pydeck** | — | Streamlit dependencies |
| **openai** | (optional) | Cloud LLM fallback |

---

## 3. Model & Embeddings

| Item | Setting |
|------|----------|
| **Provider** | Ollama (local) |
| **Model** | `mistral:latest` — ID `6577803aa9a0` (4.4 GB, pulled ~3 weeks ago) |
| **Temperature** | 0.2 |
| **Embedding Model** | `nomic-embed-text:latest` — ID `0a109f422b47` (274 MB, pulled ~3 weeks ago) |
| **Chunk Size** | 800 |
| **Chunk Overlap** | 120 |
| **Retrieval Mode** | Dense (FAISS) — hybrid optional |
| **Top‑K Retrieval** | 4 |

---

## 4. Data & Indexing

- **Base Directory:** `rag_store/`
- Each rebuild creates subfolders `idx_YYYYMMDD_HHMMSS/`
- Active index pointer managed by `index_admin.py`
- Manifest summary (docs, chunks, timestamps) written to each index folder

---

## 5. Notes & Maintenance

- Re‑capture environment with:
  ```bash
  pip freeze > requirements.txt
  ollama list > model_versions.txt
  ```

- **Lock a specific model digest**
  ```bash
  ollama pull mistral
  ollama list  # copy ID hash
  echo "FROM mistral@6577803aa9a0" > Modelfile
  ollama create mistral-rag -f Modelfile
  ```

- **Re‑run retrieval eval** after index rebuilds:
  ```bash
  python eval/run_eval.py
  ```

---

## 6. Contacts / Ownership

| Role | Name | Notes |
|------|------|-------|
| Maintainer | Daniel | Core dev / maintainer |
| Created | Oct 2025 | DS/LLM/AI 3‑Month Program |
