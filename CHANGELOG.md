# Changelog
All notable changes to this project will be documented here.

## [0.2.0] - 2025-09-25
### Added
- DOCX ingestion alongside PDF and TXT.
- Index switcher for multiple bases, with stats and file management (delete, replace, rebuild manifest).
- Conversation mode: optional chat history with transcript export.
- Provider toggle (Ollama default; optional OpenAI). Graceful fallback to Ollama when OpenAI is unavailable.
- Guardrail banners: inline warnings for prompt injection, thin context, and missing citations.
- Advanced retrieval tuning: BM25, Dense, Hybrid (RRF), score threshold, per-source caps, and MMR λ control.
- Evaluation snapshot: retrieval quality metrics (hit@k, MRR) with summary table and charts.
- Expanded exports: per-turn and full session to Markdown, CSV, Excel with provenance (model, provider, retrieval mode, top_k, index_name).
- Settings persistence (`settings.json`) to restore defaults across sessions.
- Manual regression checklist in `TESTING.md`.

### Changed
- Requirements split: core (`requirements.txt`) and optional OpenAI extras (`requirements-openai.txt`).
- README updated with polished presentation and optional OpenAI setup.
- Clearer UX for index build/load and cache invalidation when switching bases.

### Fixed
- Provenance metadata in exports now consistent (`index_name`, provider selected vs. used).
- Silenced metric warnings on undefined edge cases.

---

## [0.1.0] - 2025-08-22
### Added
- Initial prototype release:
  - Ingest `.txt`/`.pdf` files, chunking, embeddings, and Chroma persistence.
  - “Build / Load Index” UI with overwrite toggle.
  - Retrieve → Answer → Cite flow with ChatOllama.
  - Preview top sources and citations panel.
  - Retrieval tuned with MMR and basic caching.
- Basic project skeleton: `app.py`, `rag_core.py`, README, requirements, .gitignore, LICENSE.
- Early pytest suite and sample data.

### Fixed
- Streamlit widget key conflicts and early error handling improvements.
