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
