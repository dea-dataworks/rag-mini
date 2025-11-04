# RAG Explorer — v0.3 TODOs

This file tracks pending improvements and refactors planned for the next version (v0.3: Docker + FastAPI).

---

## Core Refactors
- [ ] **Create RetrieverEngine class**  
  Consolidate `retrieve()`, `_ensure_bm25_for_vs()`, and `_rrf_fuse()` into a single class handling all retrieval modes (Dense / BM25 / Hybrid).

- [ ] **Add latency logging and metrics hooks**  
  Measure per-stage timings (`embedding`, `retrieval`, `LLM`) and expose them to the Dev Metrics panel.

- [ ] **Implement embedding cache**  
  Persist computed embeddings to disk (`.pkl` or SQLite) to speed up reloading of previously indexed files.

- [ ] **Add reranker option (BM25 or Cross-Encoder)**  
  Experiment with re-ranking retrieved chunks before fusion to improve answer precision.

---

## Infrastructure & Integration
- [ ] **Add Docker support**  
  Write a `Dockerfile` for clean, reproducible deployment. Validate container with both Ollama and OpenAI modes.

- [ ] **Add FastAPI wrapper**  
  Expose a minimal REST API (`api.py`) with endpoints for `upload`, `query`, and `feedback`.

- [ ] **Set up CI/CD pipeline**  
  GitHub Actions workflow to lint, test, and build the Docker image on each push.

---

## UI & Usability
- [ ] **Polish UI consistency**  
  Standardize fonts, spacing, and chunk-table styling. Add per-file chunk stats to sidebar.

- [ ] **Add user feedback controls**  
  Introduce thumbs-up/down buttons for answers and store results in `exports/feedback.jsonl`.

- [ ] **Improve guardrail banner visuals**  
  Refine icons/colors and group warnings for better UX readability.

---

## Notes
These tasks prepare the app for v0.3 “productionization” — containerized, API-ready, and capable of switching between local (Ollama) and remote (OpenAI) modes seamlessly.

Todo (v0.3 cycle): bump LangChain packages to 0.3.*, re-test RAG pipeline compatibility.