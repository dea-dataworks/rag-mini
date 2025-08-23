# RAG Mini v0.1 (Scope-First Skeleton)

A tiny, local Retrieval-Augmented Generation app (Streamlit + LangChain + Chroma + Ollama).  
This repository starts with a scope-approved scaffold and TODOs. Implement features incrementally.

## v0.1 Goals
- Upload `.pdf` / `.txt`
- Chunk → Embed (Ollama) → Index (Chroma)
- Retrieve top‑k and answer with sources
- Local‑only (no API keys)

## Prerequisites
- Python 3.11–3.13 (tested). If you hit install issues on 3.13, try 3.12.
- [Ollama](https://ollama.com) installed
- Pull models:
  
  ```bash
  ollama pull mistral
  ollama pull nomic-embed-text
  ```
## Quickstart

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

# pull models 
ollama pull mistral
ollama pull nomic-embed-text

streamlit run app.py
```

## Milestones
- M1: ingest + chunk + index
- M2: retrieve + answer + cite
- M3: polish UI + README examples

## Notes

- This is a minimal single‑file app for simplicity.
- PDF parsing is text-only via pypdf. Scanned PDFs (images) won’t work unless you OCR them first.
- OCR/scanned PDFs and multi‑turn chat are out of scope for v0.1.

## Troubleshooting

- **“Index build failed: … Could not initialize embeddings for 'nomic-embed-text'”**  
Ollama model isn’t available. Run:

```
ollama pull nomic-embed-text
ollama pull mistral
 ```

Make sure the Ollama service is running, then rebuild the index. (This error is raised from the embedding init in code.) 

- **“No vector store found. Build the index first (Step 1).”**  
Click **uild / Load Index** after uploading files. Then try your question again.

- **“No results. Try a simpler question or rebuild the index.”**  
Your query didn’t match the chunks. Try a simpler/shorter question or rebuild the index with more/larger documents.

- **PDF uploaded but answer is empty / chunks look blank**  
The PDF likely contains images (scanned) instead of text. This app uses text-only extraction via `pypdf`. Run OCR on the PDF, or use a text-based PDF. 

- **Chroma directory issues or stale index**  
Close the app, delete the local `rag_store/` directory, and rebuild the index. (It’s ignored by Git.) 

```
rm -rf rag_store # macOS/Linux
rmdir /S /Q rag_store # Windows
```


- **Port already in use**  
Run Streamlit on another port:

```
streamlit run app.py --server.port 8502
```


- **Version conflicts during pip install**  
If you hit resolver errors, try Python 3.12 and the pinned ranges in `requirements.txt`, then upgrade one-by-one if needed.



## License

MIT — see LICENSE.