import json, os

SETTINGS_PATH = "settings.json"
DEFAULT_SETTINGS = {
    "chunk_size": 800,
    "chunk_overlap": 120,
    "k": 4,
    "provider": "ollama",
    "use_history": False,
    "max_history_turns": 3,
    "mmr_lambda": 0.7,
    "score_threshold": 0.0,
    "sanitize": True,
    "debug": False,
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
    safe = {k: d.get(k, DEFAULT_SETTINGS[k]) for k in DEFAULT_SETTINGS}
    with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(safe, f, indent=2, ensure_ascii=False)

def seed_session_from_settings(st):
    """Idempotently seed st.session_state with persisted settings."""
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

