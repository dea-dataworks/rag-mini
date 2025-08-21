# RAG Mini v0.1 (Scope-First Skeleton)

A tiny, local Retrieval-Augmented Generation app (Streamlit + LangChain + Chroma + Ollama).  
This repository starts with a scope-approved scaffold and TODOs. Implement features incrementally.

## v0.1 Goals
- Upload `.pdf` / `.txt`
- Chunk → Embed (Ollama) → Index (Chroma)
- Retrieve top‑k and answer with sources
- Local‑only (no API keys)

## Prerequisites
- Python 3.11 or 3.12 recommended
- [Ollama](https://ollama.com) installed
- Pull models:
  
  ```bash
  ollama pull mistral
  ollama pull nomic-embed-text
  ```
## Quickstart

```
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

## Milestones
- M1: ingest + chunk + index
- M2: retrieve + answer + cite
- M3: polish UI + README examples

## Notes

- This is a minimal single‑file app for simplicity.

- OCR/scanned PDFs and multi‑turn chat are out of scope for v0.1.



## License

MIT — see LICENSE.