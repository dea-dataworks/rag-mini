# RAG Explorer — RUNBOOK

## Overview
This document captures environment details, model configuration, and versioning information for the **RAG Explorer** app.

_Last updated: 2025-10-22_

---

## 0. Install & Run

### Installation
```bash
python -m venv .venv
.venv\Scripts\activate      # or source .venv/bin/activate (Mac/Linux)
pip install -r requirements.txt
```

### Launch
```bash
streamlit run app.py
```

### Verification
- Open browser → [http://localhost:8501](http://localhost:8501)  
- Upload one short `.pdf` or `.txt` file  
- Confirm retrieval and answer appear under **Response**  
- (Optional) Take screenshot of main UI → save as `docs/ui_main.png`

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
| **Model** | `mistral:latest` — ID `6577803aa9a0` (4.4 GB, pulled ~3 weeks ago) |
| **Temperature** | 0.2 |
| **Embedding Model** | `nomic-embed-text:latest` — ID `0a109f422b47` (274 MB, pulled ~3 weeks ago) |
| **Chunk Size** | 800 |
| **Chunk Overlap** | 120 |
| **Retrieval Mode** | Dense (FAISS) — hybrid optional |
| **Top-K Retrieval** | 4 |

---

## 4. Data & Indexing
- **Base Directory:** `rag_store/`  
- Each rebuild creates subfolders `idx_YYYYMMDD_HHMMSS/`  
- Active index pointer managed by `index_admin.py`  
- Manifest summary (docs, chunks, timestamps) written to each index folder  

---

## 4.1 Runtime Inputs
| Input Type | Variable / Control | Description | Default / Example |
|-------------|--------------------|--------------|-------------------|
| **Document Upload** | `uploaded_file` | PDF, TXT, DOCX input for retrieval | — |
| **Rebuild Flag** | `overwrite_index` | Forces FAISS/Chroma index rebuild | False |
| **Model Selector** | `MODEL_NAME` (env var) | LLM to use (e.g. mistral:latest) | mistral |
| **Embedding Selector** | `EMBED_MODEL` | Embedding backend | nomic-embed-text |
| **Top-K** | `TOP_K` | Number of retrieved chunks | 4 |
| **Temperature** | `TEMP` | LLM generation temperature | 0.2 |
| **Database Path** | `DB_PATH` | Vector index directory | `./rag_store/` |
| **Provider Toggle** | `USE_OPENAI` | Optional fallback to OpenAI API | False |

---

## 5. Notes & Maintenance
- Re-capture environment with:
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
- **Re-run retrieval eval** after index rebuilds:
  ```bash
  python eval/run_eval.py
  ```

---

## 6. Contacts / Ownership
| Role | Name | Notes |
|------|------|-------|
| **Maintainer** | Daniel | Core dev / maintainer |
| **Created** | Oct 2025 | DS/LLM/AI 3-Month Program |
