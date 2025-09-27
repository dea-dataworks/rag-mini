# RAG Explorer

**A lightweight, local Retrieval-Augmented Generation app**  
Built with Streamlit · LangChain · FAISS · Ollama

![demo screenshot or gif here once you capture one]

---

## Features (v0.2)

- Upload `.pdf`, `.txt`, or `.docx` files.  
- Build and manage local indexes (FAISS): inspect stats, delete/replace files, rebuild manifest. 
- Answer questions with **cited sources** (Ollama mistral + nomic-embed-text).  
- **Conversation mode**: toggle chat history, export full transcript to Markdown.  
- Switch between **multiple indexes** (projects / datasets).  
- **Provider toggle**: default Ollama, optional OpenAI with graceful fallback.  
- **Guardrail banners**: inline warnings for no citations, thin context, prompt injection.  
- **Advanced retrieval tuning**: BM25, Dense, Hybrid (RRF), score thresholds, MMR λ, per-source caps.  
- **Exports**: per-turn and full session to Markdown, CSV, Excel, including provenance + guardrail notes.  
- **Evaluation (retrieval quality)**: run QA sets across BM25 / Dense / Hybrid and see hit@k & MRR.  
- **Local-only by default** — no API keys required.

_Not included (yet): OCR/scanned PDFs, full web search._

---

## Quickstart

### Prerequisites
- Python 3.12+ (tested on 3.12)  
- [Ollama](https://ollama.com) installed and running  
- Optional: [OpenAI](https://platform.openai.com) key for GPT models  

### Installation

```bash
# create virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -U pip
pip install -r requirements.txt

# pull models (one-time)
ollama pull mistral
ollama pull nomic-embed-text
```

### Optional: OpenAI provider support

By default the app runs fully local with Ollama. To enable **OpenAI** models:

```bash
pip install -r requirements-openai.txt
# set your key (mac/linux)
export OPENAI_API_KEY=sk-...
# or (Windows PowerShell)
setx OPENAI_API_KEY "sk-..."
```

Then select **OpenAI** in the provider dropdown. If the key is missing or invalid, 
the app will automatically fall back to Ollama and show a small toast.


### Run the app
```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Testing

Minimal test suite with `pytest` is included:

```bash
pytest -q
```

Runs in <2s against small fixture corpus (`tests/fixtures`).  
A fuller manual checklist is in [TESTING.md](./TESTING.md).

---

## Troubleshooting

- **Embedding init failed** → `ollama pull nomic-embed-text mistral` and ensure Ollama is running.  
- **No vector store found** → Upload files, then click **Build / Load Index**.  
- **Empty answers from PDFs** → Likely scanned images; this app only extracts text. Run OCR first.  
- **DOCX import errors** → Ensure `python-docx` is installed (already in `requirements.txt`).  
- **Stale index issues** → Delete `rag_store/` and rebuild.  
- **Port already in use** → Run `streamlit run app.py --server.port 8502`.  
- **Install conflicts** → Prefer Python 3.12, use pinned versions in `requirements.txt`.  

---

## Project Structure
```
app.py              # Streamlit entry point
rag_core.py         # core retrieval / prompt assembly
llm_chain.py        # provider toggle, LLM client factory
index_admin.py      # index building / switching / manifest
exports.py          # export QA + provenance + sessions
guardrails.py       # guardrail checks
utils/              # settings, ui, helpers
eval/               # eval snapshot scripts + qa.jsonl
tests/              # pytest suite
sample_data/        # demo files
```

---

## License
MIT — see [LICENSE](./LICENSE).
