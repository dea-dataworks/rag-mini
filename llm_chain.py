from typing import Optional

def build_prompt(context: str, question: str) -> str:
    return (
        "You are a helpful assistant. Answer ONLY from the provided context.\n"
        "Use ALL relevant context; if multiple documents apply, combine them. Answer concisely.\n"
        "If the answer isn't in the context, say you don't know.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
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

    resp = llm.invoke(prompt)
    return getattr(resp, "content", str(resp))
