from typing import Optional
from langchain_core.messages import SystemMessage, HumanMessage
from guardrails import GUARDRAIL_SYSTEM, has_citation, FALLBACK_NO_CITATION


def build_prompt(context: str, question: str) -> str:
    return (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer concisely. If you answer, include at least one citation like (file p.X).\n\n"
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

