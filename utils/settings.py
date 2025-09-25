import json, os

SETTINGS_PATH = "settings.json"
DEFAULT_SETTINGS = {
    "chunk_size": 800,
    "chunk_overlap": 120,
    "k": 4,
    "provider": "ollama",          # default; OpenAI optional
    "use_history": False,
    "max_history_turns": 3,
    "mmr_lambda": 0.7,
    "score_threshold": 0.0,
    "sanitize": True,
    "debug": False,
    # --- guardrail controls ---
    "guardrails_enabled": True,     # feature flag (UI can honor this)
    "guardrails_strict": False,     # if True, decline on warnings; else warn-first
    "min_context_chars": 40,        # thin-context threshold for 'has context'
}

def load_settings():
    if os.path.exists(SETTINGS_PATH):
        try:
            with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                return {**DEFAULT_SETTINGS, **data}
        except Exception:
            pass
    return DEFAULT_SETTINGS.copy()

def save_settings(d: dict):
    """
    Persist only the known, safe settings (no secrets).
    Anything not in DEFAULT_SETTINGS is ignored on save.
    """
    safe = {k: d.get(k, DEFAULT_SETTINGS[k]) for k in DEFAULT_SETTINGS}
    with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(safe, f, indent=2, ensure_ascii=False)

def seed_session_from_settings(st):
    """Idempotently seed st.session_state with persisted settings (lower-case schema)."""
    settings = load_settings()
    for k, v in settings.items():
        if k not in st.session_state:
            st.session_state[k] = v

def apply_persisted_defaults(st):
    """Bridge lower-case persisted keys -> existing UPPERCASE app keys."""
    _lc = st.session_state
    st.session_state.setdefault("CHUNK_SIZE", int(_lc.get("chunk_size", 800)))
    st.session_state.setdefault("CHUNK_OVERLAP", int(_lc.get("chunk_overlap", 120)))
    st.session_state.setdefault("TOP_K", int(_lc.get("k", 4)))
    st.session_state.setdefault("LLM_PROVIDER", _lc.get("provider", "ollama"))
    st.session_state.setdefault("use_history", bool(_lc.get("use_history", False)))
    st.session_state.setdefault("max_history_turns", int(_lc.get("max_history_turns", 3)))
    st.session_state.setdefault("MMR_LAMBDA", float(_lc.get("mmr_lambda", 0.7)))
    st.session_state.setdefault("SCORE_THRESH", float(_lc.get("score_threshold", 0.0)))
    st.session_state.setdefault("SANITIZE_RETRIEVED", bool(_lc.get("sanitize", True)))
    st.session_state.setdefault("SHOW_DEBUG", bool(_lc.get("debug", False)))
    # --- NEW: guardrail keys surfaced to runtime ---
    st.session_state.setdefault("GUARDRAILS_ENABLED", bool(_lc.get("guardrails_enabled", True)))
    st.session_state.setdefault("GUARDRAILS_STRICT", bool(_lc.get("guardrails_strict", False)))
    st.session_state.setdefault("GUARDRAILS_MIN_CONTEXT_CHARS", int(_lc.get("min_context_chars", 40)))

# --- Provenance / export config ---

# Keys we explicitly consider part of run_settings provenance.
EXPORTABLE_SETTINGS = [
    "model",
    "provider",
    "top_k",
    "retrieval_mode",
    "chunk_size",
    "chunk_overlap",
    "mmr_lambda",
    "use_history",
    "max_history_turns",
    # --- NEW (optional in exports) ---
    "guardrails_enabled",
    "guardrails_strict",
    "min_context_chars",
]

def get_exportable_settings(session_state) -> dict:
    """
    Snapshot the current Streamlit session_state into a clean provenance dict.
    Uses the EXPORTABLE_SETTINGS allow-list and maps to human-friendly keys.
    """
    mapping = {
        "model": session_state.get("LLM_MODEL"),
        "provider": session_state.get("LLM_PROVIDER"),
        "top_k": session_state.get("TOP_K"),
        "retrieval_mode": session_state.get("RETRIEVE_MODE"),
        "chunk_size": session_state.get("CHUNK_SIZE"),
        "chunk_overlap": session_state.get("CHUNK_OVERLAP"),
        "mmr_lambda": session_state.get("MMR_LAMBDA"),
        "use_history": session_state.get("use_history"),
        "max_history_turns": session_state.get("max_history_turns"),
        # --- NEW guardrail knobs (optional provenance) ---
        "guardrails_enabled": session_state.get("GUARDRAILS_ENABLED"),
        "guardrails_strict": session_state.get("GUARDRAILS_STRICT"),
        "min_context_chars": session_state.get("GUARDRAILS_MIN_CONTEXT_CHARS"),
    }
    return mapping

