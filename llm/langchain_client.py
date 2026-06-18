"""
LangChain Universal LLM Client
================================
Wraps every major LLM provider under a single BaseChatModel interface.
No "free-first" bias — all providers are equal. Priority is set by LLM_PROVIDER
env var, with a full cascade fallback if set to "auto".

References
----------
Chase, H. (2022). "LangChain." github.com/langchain-ai/langchain.
    BaseChatModel abstraction; provider-specific wrappers (ChatOpenAI,
    ChatAnthropic, ChatGoogleGenerativeAI, ChatGroq, etc.).

Brown, T., et al. (2020). "Language Models are Few-Shot Learners."
    NeurIPS 2020. arXiv:2005.14165.
    Foundation for GPT-series provider integrations.

Ouyang, L., et al. (2022). "Training Language Models to Follow Instructions
    with Human Feedback." NeurIPS 2022. arXiv:2203.02155.
    InstructGPT — basis for instruction-tuned models used across providers.

Touvron, H., et al. (2023). "Llama 2: Open Foundation and Fine-Tuned Chat
    Models." arXiv:2307.09288.
    Llama-series models via Groq, Together, Ollama providers.

Architecture
------------
Provider cascade order (auto mode): Groq (free-tier speed) → OpenAI →
    Anthropic → Google/Gemini → Together → OpenRouter → Mistral → Cohere →
    Bedrock → Azure → LM Studio → Ollama → HuggingFace.
Supported: 14 providers, 2 model tiers per provider (primary + fast),
    configurable via env vars.

Supported providers (configure via .env):
  openai      OPENAI_API_KEY
  anthropic   ANTHROPIC_API_KEY
  google      GOOGLE_API_KEY
  groq        GROQ_API_KEY
  together    TOGETHER_API_KEY
  openrouter  OPENROUTER_API_KEY
  mistral     MISTRAL_API_KEY
  cohere      COHERE_API_KEY
  ollama      OLLAMA_BASE_URL (default http://localhost:11434)
  lmstudio    LMSTUDIO_BASE_URL (default http://localhost:1234/v1)
  bedrock     AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY + AWS_REGION_NAME
  azure       AZURE_OPENAI_API_KEY + AZURE_OPENAI_ENDPOINT
"""

from __future__ import annotations

import os
from functools import lru_cache
from loguru import logger

# ── Cascade order when LLM_PROVIDER=auto ──────────────────────────────
# Adjust this list to change the default try-order.
_DEFAULT_CASCADE = [
    "openai",
    "anthropic",
    "google",
    "groq",
    "together",
    "openrouter",
    "mistral",
    "cohere",
    "bedrock",
    "azure",
    "ollama",
    "lmstudio",
]


def get_langchain_llm(fast: bool = False, provider: str = "auto"):
    """
    Return a LangChain BaseChatModel for the resolved provider.

    Args:
        fast:     Use the cheaper/faster variant (e.g. gpt-4o-mini vs gpt-4o).
        provider: Override provider name, or "auto" for cascade.

    The result is NOT cached at module level because API keys can change
    between runs (Colab secrets, CI injection).  Callers that want a cached
    handle should cache it themselves.
    """
    explicit = os.getenv("LLM_PROVIDER", provider).lower().strip()
    cascade = _build_cascade(explicit)

    for provider_name, loader in cascade:
        try:
            model = loader(fast=fast)
            logger.info(f"[LangChainClient] {provider_name} ({'fast' if fast else 'primary'})")
            return model
        except Exception as exc:
            logger.debug(f"[LangChainClient] {provider_name} unavailable: {exc}")

    raise RuntimeError(
        "No LangChain LLM provider is available.  "
        "Set at least one API key in .env or run Ollama / LM Studio locally."
    )


def _build_cascade(explicit: str) -> list[tuple[str, callable]]:
    all_loaders = {
        "openai":     _load_openai,
        "anthropic":  _load_anthropic,
        "google":     _load_google,
        "groq":       _load_groq,
        "together":   _load_together,
        "openrouter": _load_openrouter,
        "mistral":    _load_mistral,
        "cohere":     _load_cohere,
        "bedrock":    _load_bedrock,
        "azure":      _load_azure,
        "ollama":     _load_ollama,
        "lmstudio":   _load_lmstudio,
    }

    if explicit == "auto":
        return [(p, all_loaders[p]) for p in _DEFAULT_CASCADE if p in all_loaders]

    # Explicit provider first, rest as fallback
    ordered = [explicit] + [p for p in _DEFAULT_CASCADE if p != explicit]
    return [(p, all_loaders[p]) for p in ordered if p in all_loaders]


# ── Provider loaders ────────────────────────────────────────────────────

def _load_openai(fast: bool = False):
    from langchain_openai import ChatOpenAI
    key = os.getenv("OPENAI_API_KEY", "")
    if not key:
        raise ValueError("OPENAI_API_KEY not set")
    model = os.getenv("OPENAI_FAST_MODEL" if fast else "OPENAI_MODEL",
                      "gpt-4o-mini" if fast else "gpt-4o")
    return ChatOpenAI(model=model, api_key=key, temperature=0.1, max_retries=2)


def _load_anthropic(fast: bool = False):
    from langchain_anthropic import ChatAnthropic
    key = os.getenv("ANTHROPIC_API_KEY", "")
    if not key:
        raise ValueError("ANTHROPIC_API_KEY not set")
    model = os.getenv("ANTHROPIC_FAST_MODEL" if fast else "ANTHROPIC_MODEL",
                      "claude-haiku-4-5-20251001" if fast else "claude-sonnet-4-6")
    return ChatAnthropic(model=model, api_key=key, temperature=0.1, max_retries=2)


def _load_google(fast: bool = False):
    from langchain_google_genai import ChatGoogleGenerativeAI
    key = os.getenv("GOOGLE_API_KEY", "")
    if not key:
        raise ValueError("GOOGLE_API_KEY not set")
    model = os.getenv("GEMINI_FAST_MODEL" if fast else "GEMINI_MODEL",
                      "gemini-2.0-flash" if fast else "gemini-1.5-pro-latest")
    return ChatGoogleGenerativeAI(model=model, google_api_key=key, temperature=0.1)


def _load_groq(fast: bool = False):
    from langchain_groq import ChatGroq
    key = os.getenv("GROQ_API_KEY", "")
    if not key:
        raise ValueError("GROQ_API_KEY not set")
    model = os.getenv("GROQ_FAST_MODEL" if fast else "GROQ_MODEL",
                      "llama-3.1-8b-instant" if fast else "llama-3.3-70b-versatile")
    return ChatGroq(model=model, api_key=key, temperature=0.1, max_retries=2)


def _load_together(fast: bool = False):
    from langchain_together import ChatTogether
    key = os.getenv("TOGETHER_API_KEY", "")
    if not key:
        raise ValueError("TOGETHER_API_KEY not set")
    model = os.getenv(
        "TOGETHER_FAST_MODEL" if fast else "TOGETHER_MODEL",
        "meta-llama/Llama-3.1-8B-Instruct-Turbo" if fast
        else "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    )
    return ChatTogether(model=model, api_key=key, temperature=0.1)


def _load_openrouter(fast: bool = False):
    from langchain_openai import ChatOpenAI
    key = os.getenv("OPENROUTER_API_KEY", "")
    if not key:
        raise ValueError("OPENROUTER_API_KEY not set")
    model = os.getenv(
        "OPENROUTER_FAST_MODEL" if fast else "OPENROUTER_MODEL",
        "google/gemini-flash-1.5" if fast else "anthropic/claude-3.5-sonnet",
    )
    return ChatOpenAI(
        model=model, api_key=key, temperature=0.1,
        base_url="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": "https://github.com/anubhav0499-bit/forensic-ai",
            "X-Title": "Forensic AI",
        },
    )


def _load_mistral(fast: bool = False):
    from langchain_mistralai import ChatMistralAI
    key = os.getenv("MISTRAL_API_KEY", "")
    if not key:
        raise ValueError("MISTRAL_API_KEY not set")
    model = os.getenv("MISTRAL_MODEL",
                      "mistral-small-latest" if fast else "mistral-large-latest")
    return ChatMistralAI(model=model, api_key=key, temperature=0.1)


def _load_cohere(fast: bool = False):
    from langchain_cohere import ChatCohere
    key = os.getenv("COHERE_API_KEY", "")
    if not key:
        raise ValueError("COHERE_API_KEY not set")
    model = "command-r" if fast else "command-r-plus"
    return ChatCohere(model=model, cohere_api_key=key, temperature=0.1)


def _load_bedrock(fast: bool = False):
    from langchain_aws import ChatBedrock
    import boto3
    region = os.getenv("AWS_REGION_NAME", os.getenv("AWS_DEFAULT_REGION", "us-east-1"))
    access_key = os.getenv("AWS_ACCESS_KEY_ID", "")
    if not access_key:
        raise ValueError("AWS_ACCESS_KEY_ID not set")
    model = os.getenv("BEDROCK_MODEL",
                      "anthropic.claude-3-haiku-20240307-v1:0" if fast
                      else "anthropic.claude-3-5-sonnet-20241022-v2:0")
    client = boto3.client(
        "bedrock-runtime",
        region_name=region,
        aws_access_key_id=access_key,
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", ""),
    )
    return ChatBedrock(model_id=model, client=client)


def _load_azure(fast: bool = False):
    from langchain_openai import AzureChatOpenAI
    key = os.getenv("AZURE_OPENAI_API_KEY", "")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    if not key or not endpoint:
        raise ValueError("AZURE_OPENAI_API_KEY or AZURE_OPENAI_ENDPOINT not set")
    deployment = os.getenv("AZURE_OPENAI_FAST_DEPLOYMENT" if fast else "AZURE_OPENAI_DEPLOYMENT",
                           "gpt-4o-mini" if fast else "gpt-4o")
    return AzureChatOpenAI(
        azure_deployment=deployment,
        azure_endpoint=endpoint,
        api_key=key,
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"),
        temperature=0.1,
    )


def _load_ollama(fast: bool = False):
    from langchain_ollama import ChatOllama
    import httpx
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    httpx.get(f"{base_url}/api/tags", timeout=3).raise_for_status()
    model = os.getenv("OLLAMA_FAST_MODEL" if fast else "OLLAMA_MODEL",
                      "phi3.5:3.8b" if fast else "qwen2.5:7b")
    return ChatOllama(model=model, base_url=base_url, temperature=0.1)


def _load_lmstudio(fast: bool = False):
    from langchain_openai import ChatOpenAI
    import httpx
    base_url = os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
    httpx.get(base_url.replace("/v1", ""), timeout=3).raise_for_status()
    model = os.getenv("LMSTUDIO_MODEL", "local-model")
    return ChatOpenAI(model=model, base_url=base_url, api_key="lmstudio", temperature=0.1)


# ── Convenience: invoke text-in / text-out ─────────────────────────────

def lc_invoke(prompt: str, system: str = "", fast: bool = False) -> str:
    """
    One-shot text generation via LangChain.
    Returns the response content string.
    """
    from langchain_core.messages import SystemMessage, HumanMessage
    llm = get_langchain_llm(fast=fast)
    messages = []
    if system:
        messages.append(SystemMessage(content=system))
    messages.append(HumanMessage(content=prompt))
    response = llm.invoke(messages)
    return response.content if hasattr(response, "content") else str(response)
