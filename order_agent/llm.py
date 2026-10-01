from langchain_openai import ChatOpenAI

from order_agent import config


def get_llm() -> ChatOpenAI:
    if not config.OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set (see .env.example)")
    return ChatOpenAI(
        model=config.MODEL,
        base_url=config.LLM_BASE_URL,
        api_key=config.OPENROUTER_API_KEY,
        temperature=0,
        timeout=config.LLM_TIMEOUT_S,
        max_retries=config.LLM_TRANSPORT_RETRIES,
        extra_body={"reasoning": {"effort": config.REASONING_EFFORT}} if config.REASONING_EFFORT else None,
    )
