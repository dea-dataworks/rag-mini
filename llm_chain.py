from typing import Optional
from langchain_core.messages import SystemMessage, HumanMessage
from guardrails import GUARDRAIL_SYSTEM, has_citation, FALLBACK_NO_CITATION

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
            history_block = f"\n\nConversation (recent, for continuity only — do not cite):\n{hist_txt}"

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

def call_llm(
    prompt: str,
    provider: str = "ollama",
    model_name: str = "mistral",
    openai_api_key: Optional[str] = None,
    temperature: float = 0.2,
) -> str:
    if provider == "openai":
        if not openai_api_key:
            raise ValueError("OpenAI key not provided.")
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model=model_name, temperature=temperature, api_key=openai_api_key)
    else:
        from langchain_ollama import ChatOllama
        llm = ChatOllama(model=model_name, temperature=temperature)

    messages = [
        SystemMessage(content=GUARDRAIL_SYSTEM),
        HumanMessage(content=prompt),
    ]
    resp = llm.invoke(messages)
    answer = getattr(resp, "content", str(resp))

    # Post-answer guardrail: no citation pattern → fallback
    if not has_citation(answer):
        return FALLBACK_NO_CITATION

    return answer

