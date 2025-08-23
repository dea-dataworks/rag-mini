
# Changelog
All notable changes to this project will be documented here.

## [0.1.3] - 2025-08-22
### Changed
- Retrieval switched to **MMR** for more diverse results across documents.
- **Cached embeddings** via `st.cache_resource` for snappier reruns.
### Fixed
- Minor Streamlit widget ID conflicts (unique keys) and friendlier error messages.

## [0.1.2] - 2025-08-22
### Added
- **M2:** Retrieve → Answer → Cite flow with ChatOllama.
- “Preview Top Sources” (inspect retrieved chunks before answering).
- Basic citations section listing unique source files.
### Changed
- Store vector store handle in `st.session_state["vs"]` for use after reruns.

## [0.1.1] - 2025-08-22
### Added
- **M1:** Ingest `.txt`/`.pdf`, chunking, embeddings, and Chroma persistence.
- “Build / Load Index” UI, with overwrite toggle.

## [0.1.0] - 2025-08-22
### Added
- Scope‑first repo skeleton: `app.py` scaffold, `rag_core.py` stubs, README, requirements, .gitignore, LICENSE.
