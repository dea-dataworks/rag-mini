from typing import Optional
from langchain_core.messages import SystemMessage, HumanMessage
from guardrails import GUARDRAIL_SYSTEM, has_citation, FALLBACK_NO_CITATION
from rag_core import load_vectorstore_if_exists, retrieve

# helper to compress prior turns for prompt conditioning
from typing import List, Dict

def _format_history_for_prompt(
    chat_history: List[Dict],
    max_turns: int = 4,
    max_chars: int = 800,
) -> str:
    """
    Turn recent Q/A into compact lines for continuity, not as evidence.
    Takes the most recent 'max_turns', newest last. Truncates to 'max_chars'.
    Expects items like {"question": str, "answer": str, "answer_gist": str}.
    """
    if not chat_history:
        return ""

    # keep only the last N, preserve chronological order
    turns = chat_history[-int(max_turns):]

    lines = []
    for t in turns:
        q = (t.get("question") or "").strip()
        # Prefer the gist if present; fallback to a short answer slice
        a = (t.get("answer_gist") or t.get("answer") or "").strip()
        if len(a) > 220:
            a = a[:220] + "…"
        if q:
            lines.append(f"Q: {q}")
        if a:
            lines.append(f"A (gist): {a}")
        lines.append("---")

    text = "\n".join(lines).strip()
    if len(text) > max_chars:
        text = text[:max_chars].rstrip() + "…"
    return text


def build_prompt(
    context: str,
    question: str,
    chat_history: Optional[list] = None,
    use_history: bool = False,
    max_history_turns: int = 4,
) -> str:
    """
    Build the final prompt. History is optional and strictly secondary to retrieved context.
    If 'use_history' is False or 'chat_history' is empty, behavior is identical to before.
    """
    history_block = ""
    if use_history and chat_history:
        hist_txt = _format_history_for_prompt(
            chat_history=chat_history,
            max_turns=max_history_turns,
            max_chars=800,
        )
        if hist_txt:
            # History is for continuity ONLY. Documents remain the ground truth.
            history_block = (
                "\n\nConversation (recent, for continuity only — do not cite):\n" + hist_txt
            )

    return (
        f"Context (authoritative; cite from this):\n{context}"
        f"{history_block}\n\n"
        f"Question: {question}\n\n"
        "Instructions:\n"
        "- Answer concisely.\n"
        "- Ground all factual claims in the Context. Do NOT use Conversation as evidence.\n"
        "- If you answer, include at least one citation.\n"
        "- Use (file p.X) for paged PDFs; use (file) for non-paged sources like TXT/DOCX.\n\n"
        "Answer:"
    )

# ---- helpers for model construction and response handling ----
def _extract_text(resp) -> str:
    """Get text content from a LangChain chat return."""
    try:
        txt = getattr(resp, "content", None)
        if txt is None:
            txt = str(resp)
        return (txt or "").strip()
    except Exception:
        return ""

def _make_llm(provider: str, model_name: str, temperature: float, openai_api_key: Optional[str]):
    """
    Build an LLM client for the requested provider.
    Raises an Exception if it can't be constructed.
    """
    p = (provider or "ollama").lower()
    if p == "openai":
        if not openai_api_key:
            raise ValueError("OpenAI key missing")
        try:
            from langchain_openai import ChatOpenAI
        except Exception as e:
            raise RuntimeError(f"OpenAI client unavailable: {e}")
        return ChatOpenAI(model=model_name, temperature=temperature, api_key=openai_api_key), "openai"

    # default to Ollama
    try:
        from langchain_ollama import ChatOllama
    except Exception as e:
        raise RuntimeError(f"Ollama client unavailable: {e}")
    return ChatOllama(model=model_name, temperature=temperature), "ollama"

def call_llm(
    prompt: str,
    provider: str = "ollama",
    model_name: str = "mistral",
    openai_api_key: Optional[str] = None,
    temperature: float = 0.2,
):
    """
    Build the requested provider; on init or invoke error, gracefully fall back to Ollama once.
    Returns either:
      - dict: {"text": str, "meta": {"provider_used": str, "fallback": bool, "reason": Optional[str]}}
      - or a plain string (e.g., FALLBACK_NO_CITATION) for guardrail short-circuits.
    """
    selected = (provider or "ollama").lower()
    llm = None
    provider_used = selected
    fallback = False
    reason = None

    # --- Try primary provider ---
    try:
        llm, provider_used = _make_llm(selected, model_name, temperature, openai_api_key)
        messages = [SystemMessage(content=GUARDRAIL_SYSTEM), HumanMessage(content=prompt)]
        resp = llm.invoke(messages)
        text = _extract_text(resp)
    except Exception as e_primary:
        reason = f"{selected} error: {e_primary}"
        # If already ollama, don't loop; surface the error up.
        if selected == "ollama":
            raise
        # --- Single-hop fallback to Ollama ---
        try:
            llm, provider_used = _make_llm("ollama", model_name, temperature, openai_api_key=None)
            messages = [SystemMessage(content=GUARDRAIL_SYSTEM), HumanMessage(content=prompt)]
            resp = llm.invoke(messages)
            text = _extract_text(resp)
            fallback = True
        except Exception as e_fallback:
            # Both providers failed; surface the original + fallback error context.
            raise RuntimeError(f"{reason}; fallback(ollama) error: {e_fallback}") from e_fallback

    # Post-answer guardrail: no citation pattern → return special fallback payload
    if not has_citation(text):
        # allow non-cited answers through, but tag them
        return {
            "text": text or FALLBACK_NO_CITATION,
            "meta": {
                "provider_used": provider_used,
                "fallback": fallback,
                "reason": reason if fallback else "no_citation_warn",
            },
        }

    return {
        "text": text,
        "meta": {
            "provider_used": provider_used,
            "fallback": fallback,
            "reason": reason if fallback else None,
        },
    }


# ========== Index-aware retriever factory (cacheable) ==========

# cache by persist_dir so switching base/index rebinds cleanly
_CHAIN_CACHE = {}  # key: persist_dir -> {"cfg": {...}, "run": callable}

def invalidate_chain_cache(persist_dir: Optional[str] = None) -> None:
    """
    Clear the cached chain(s).
    - If persist_dir is given, drop only that entry.
    - If None, drop all entries.
    Call this when the active index changes.
    """
    if persist_dir:
        _CHAIN_CACHE.pop(persist_dir, None)
    else:
        _CHAIN_CACHE.clear()


def _make_context(docs, max_chars: int = 3200) -> str:
    """
    Build a plain text context from retrieved docs. Keep it simple & compact.
    """
    parts = []
    total = 0
    for d in docs:
        txt = (d.page_content or "").strip().replace("\n", " ")
        if not txt:
            continue
        # annotate minimal provenance inline to help citation behavior
        m = d.metadata or {}
        src = m.get("source", "file")
        pg  = m.get("page", None)
        head = f"[{src}{f' p.{pg}' if pg else ''}] "
        piece = (head + txt)
        if total + len(piece) > max_chars:
            remain = max_chars - total
            if remain <= 0:
                break
            piece = piece[:remain] + "…"
        parts.append(piece)
        total += len(piece)
        if total >= max_chars:
            break
    return "\n\n".join(parts)


def get_or_build_chain(
    *,
    persist_dir: str,
    embed_model: str,
    provider: str = "ollama",
    model_name: str = "mistral",
    temperature: float = 0.2,
    openai_api_key: Optional[str] = None,
    retrieval_mode: str = "dense",   # "dense" | "hybrid" | "bm25"
    k: int = 4,
    mmr_lambda: float = 0.7,
):
    """
    Returns a callable: run(question, chat_history=None, use_history=False) -> dict
    The callable performs: retrieve -> build prompt -> call LLM.
    Cached per 'persist_dir' so switching indices rebinds the chain.
    """
    cfg = {
        "provider": (provider or "ollama").lower(),
        "model_name": model_name,
        "temperature": float(temperature),
        "retrieval_mode": retrieval_mode,
        "k": int(k),
        "mmr_lambda": float(mmr_lambda),
        "embed_model": embed_model,
    }

    # Cache hit with identical config
    cached = _CHAIN_CACHE.get(persist_dir)
    if cached and cached.get("cfg") == cfg:
        return cached["run"]

    # Build (or open) vectorstore bound to this persist_dir
    vs = load_vectorstore_if_exists(embed_model=embed_model, persist_dir=persist_dir)
    if vs is None:
        # Create a "cold" runner that returns a guardrail message if index is missing
        def run_missing(question: str, chat_history=None, use_history: bool = False):
            return {
                "text": "No active index is loaded for this selection. Build or load an index first.",
                "pairs": [],
                "docs": [],
                "meta": {"provider_used": None, "fallback": False, "reason": "no_index"},
            }
        _CHAIN_CACHE[persist_dir] = {"cfg": cfg, "run": run_missing}
        return run_missing

    def _run(question: str, chat_history=None, use_history: bool = False):
        # 1) retrieve
        pairs = retrieve(
            vs=vs,
            query=question,
            k=cfg["k"],
            mmr_lambda=cfg["mmr_lambda"],
            mode=cfg["retrieval_mode"],
        )
        docs = [d for (d, _s) in pairs]

        # 2) build prompt (history optional, never treated as evidence)
        context = _make_context(docs)
        prompt = build_prompt(
            context=context,
            question=question,
            chat_history=chat_history or [],
            use_history=bool(use_history),
            max_history_turns=4,
        )

        # 3) call LLM (provider with graceful fallback already handled)
        out = call_llm(
            prompt=prompt,
            provider=cfg["provider"],
            model_name=cfg["model_name"],
            openai_api_key=openai_api_key,
            temperature=cfg["temperature"],
        )
        # Normalize to dict with text/meta
        if isinstance(out, str):
            payload = {"text": out, "meta": {"provider_used": cfg["provider"], "fallback": False, "reason": None}}
        else:
            payload = out

        # include retrieval artifacts so the caller can build exports/stats
        payload.update({"pairs": pairs, "docs": docs})
        return payload

    _CHAIN_CACHE[persist_dir] = {"cfg": cfg, "run": _run}
    return _run
