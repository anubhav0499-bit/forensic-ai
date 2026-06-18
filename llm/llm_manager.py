"""
LLM Manager — Universal multi-provider interface
=================================================
Legacy native client. For new Agentic RAG code, prefer llm/langchain_client.py.

References
----------
LangChain (Chase 2022): New Agentic RAG code prefers llm/langchain_client.py;
    this legacy module is kept for backward compatibility with agents that use
    self.llm.generate().

Ollama (2023). "Ollama: Get up and running with local LLMs." ollama.ai.
    Local model support for air-gapped deployments.

LM Studio (2023). "LM Studio: Discover, download, and run local LLMs."
    lmstudio.ai. Alternative local inference via OpenAI-compatible API.

Anthropic (2024). Claude API — claude-sonnet-4-6,
    claude-haiku-4-5-20251001 supported via native SDK.

Architecture
------------
Purpose: Pre-LangChain client; provides self.llm.generate(prompt, system,
    max_tokens) interface consumed by BaseForensicAgent._analyze_with_llm()
    and _analyze_and_extract(). For new code, use langchain_client.lc_invoke()
    or get_langchain_llm() instead.

Provider cascade in auto mode (first available key wins — all providers equal):
  1. OpenAI        — GPT-4o / GPT-4o-mini
  2. Anthropic     — Claude Sonnet / Haiku
  3. Google Gemini — Gemini 1.5 Pro / 2.0 Flash
  4. Groq          — ultra-fast inference
  5. Together AI   — hosted open-source models
  6. OpenRouter    — meta-router over 100+ models
  7. LM Studio     — local OpenAI-compatible server (port 1234)
  8. Ollama        — local model server (port 11434)
  9. HuggingFace   — Transformers on GPU (Colab / local)
 10. Template      — deterministic fallback (no LLM)

Set LLM_PROVIDER env var to force a specific provider.
Set *_API_KEY env vars to enable cloud providers.
"""

from __future__ import annotations
import json
import os
import re
import time
from typing import Optional
from loguru import logger

from config import (
    LLM_CONFIG, LLM_PROVIDER, PROVIDER_MODELS,
    OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY,
    GROQ_API_KEY, TOGETHER_API_KEY, OPENROUTER_API_KEY,
)


# Providers that use the OpenAI REST schema (same client, different base_url + key)
_OPENAI_COMPAT: dict[str, dict] = {
    "openai":     {"base_url": "https://api.openai.com/v1",          "api_key": OPENAI_API_KEY},
    "groq":       {"base_url": "https://api.groq.com/openai/v1",     "api_key": GROQ_API_KEY},
    "together":   {"base_url": "https://api.together.xyz/v1",        "api_key": TOGETHER_API_KEY},
    "openrouter": {"base_url": "https://openrouter.ai/api/v1",       "api_key": OPENROUTER_API_KEY},
    "lmstudio":   {"base_url": LLM_CONFIG.lmstudio_base_url + "/v1", "api_key": "lm-studio"},
}

# Detection priority in auto mode (all providers equal — first available key wins)
_AUTO_ORDER = ["openai", "anthropic", "gemini", "groq", "together", "openrouter", "lmstudio", "ollama", "hf"]


class LLMManager:
    """
    Universal LLM interface.  Works identically on:
    - Google Colab (Groq free API or Gemini free API)
    - VSCode / local (Ollama or LM Studio)
    - Cloud / CI (OpenAI, Anthropic, Gemini paid)
    """

    def __init__(self):
        self.backend: str = "template"
        self._provider_models: dict = {}
        self._openai_client = None   # used for all openai-compat providers
        self._anthropic_client = None
        self._gemini_model = None
        self._hf_model = None
        self._hf_tokenizer = None
        self._has_gpu = False

        self._detect_backend()

    # ─── Detection ────────────────────────────────────────────

    def _detect_backend(self) -> None:
        provider = LLM_PROVIDER.lower().strip()

        if provider != "auto":
            if self._try_provider(provider):
                return
            logger.warning(f"Forced provider '{provider}' unavailable. Falling back to auto.")

        for p in _AUTO_ORDER:
            if self._try_provider(p):
                return

        logger.warning("No LLM provider available. Running in template mode.")

    def _try_provider(self, name: str) -> bool:
        try:
            if name in _OPENAI_COMPAT:
                return self._check_openai_compat(name)
            if name == "anthropic":
                return self._check_anthropic()
            if name == "gemini":
                return self._check_gemini()
            if name == "ollama":
                return self._check_ollama()
            if name == "hf":
                return self._check_hf()
        except Exception as e:
            logger.debug(f"Provider '{name}' check failed: {e}")
        return False

    def _check_openai_compat(self, provider: str) -> bool:
        cfg = _OPENAI_COMPAT[provider]
        api_key = cfg["api_key"]

        # LM Studio doesn't need a real key
        if provider != "lmstudio" and not api_key:
            return False

        # For local providers (lmstudio) verify server is running
        if provider == "lmstudio":
            try:
                import httpx
                r = httpx.get(cfg["base_url"].replace("/v1", "/v1/models"), timeout=3.0)
                if r.status_code not in (200, 401):
                    return False
            except Exception:
                return False

        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key or "dummy", base_url=cfg["base_url"])
            self._openai_client = client
            self._openai_provider = provider
            self.backend = provider
            self._provider_models = PROVIDER_MODELS[provider]
            logger.info(f"LLM backend: {provider} | primary={self._provider_models['primary']}")
            return True
        except ImportError:
            logger.debug("openai package not installed. pip install openai")
            return False

    def _check_anthropic(self) -> bool:
        if not ANTHROPIC_API_KEY:
            return False
        try:
            import anthropic
            self._anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            self.backend = "anthropic"
            self._provider_models = PROVIDER_MODELS["anthropic"]
            logger.info(f"LLM backend: Anthropic | primary={self._provider_models['primary']}")
            return True
        except ImportError:
            logger.debug("anthropic package not installed. pip install anthropic")
            return False

    def _check_gemini(self) -> bool:
        if not GOOGLE_API_KEY:
            return False
        try:
            import google.generativeai as genai
            genai.configure(api_key=GOOGLE_API_KEY)
            model_name = PROVIDER_MODELS["gemini"]["primary"]
            self._gemini_model = genai.GenerativeModel(model_name)
            self.backend = "gemini"
            self._provider_models = PROVIDER_MODELS["gemini"]
            logger.info(f"LLM backend: Google Gemini | primary={model_name}")
            return True
        except ImportError:
            logger.debug("google-generativeai not installed. pip install google-generativeai")
            return False

    def _check_ollama(self) -> bool:
        try:
            import httpx
            r = httpx.get(f"{LLM_CONFIG.ollama_base_url}/api/tags", timeout=4.0)
            if r.status_code != 200:
                return False
            models = r.json().get("models", [])
            names = [m.get("name", "") for m in models]
            preferred = [PROVIDER_MODELS["ollama"]["primary"], "qwen2.5:7b", "qwen2.5:3b",
                         "phi3.5:3.8b", "llama3.2:3b", "mistral:7b", "gemma2:2b"]
            chosen = next((m for p in preferred for m in names if p in m), None)
            if not chosen and names:
                chosen = names[0] if isinstance(names[0], str) else names[0].get("name")
            if not chosen:
                return False
            self._ollama_model = chosen
            self.backend = "ollama"
            self._provider_models = {**PROVIDER_MODELS["ollama"], "primary": chosen}
            logger.info(f"LLM backend: Ollama | model={chosen}")
            return True
        except Exception:
            return False

    def _check_hf(self) -> bool:
        try:
            import torch
            self._has_gpu = torch.cuda.is_available()
            self.backend = "hf"
            self._provider_models = PROVIDER_MODELS["hf"]
            logger.info(f"LLM backend: HuggingFace Transformers | GPU={self._has_gpu}")
            return True
        except ImportError:
            return False

    # ─── Main Interface ───────────────────────────────────────

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        max_tokens: int = None,
        temperature: float = None,
        use_fast_model: bool = False,
    ) -> str:
        max_tokens  = max_tokens  or LLM_CONFIG.max_tokens
        temperature = temperature if temperature is not None else LLM_CONFIG.temperature

        for attempt in range(LLM_CONFIG.max_retries):
            try:
                return self._dispatch(prompt, system_prompt, max_tokens, temperature, use_fast_model)
            except Exception as e:
                err = str(e)
                logger.warning(f"LLM attempt {attempt+1}/{LLM_CONFIG.max_retries} failed: {err}")
                if attempt < LLM_CONFIG.max_retries - 1:
                    wait = self._parse_retry_wait(err, attempt)
                    logger.info(f"Waiting {wait:.1f}s before retry...")
                    time.sleep(wait)
        logger.error("All LLM retries exhausted. Returning template response.")
        return self._template_response(prompt)

    @staticmethod
    def _parse_retry_wait(error_msg: str, attempt: int) -> float:
        """Extract wait time from rate-limit error messages, with exponential fallback."""
        # Groq/OpenAI: "try again in 3.26s" or "try again in 1m30s"
        # Gemini:      "Please retry in 32.57s"
        m = re.search(r'(?:try again|retry) in\s+((?:\d+m)?\d+(?:\.\d+)?s)', error_msg, re.IGNORECASE)
        if m:
            raw = m.group(1)
            minutes = float(re.search(r'(\d+)m', raw).group(1)) if 'm' in raw else 0.0
            seconds = float(re.search(r'([\d.]+)s', raw).group(1)) if 's' in raw else 0.0
            suggested = minutes * 60 + seconds
            return suggested + 1.0  # 1s buffer
        # Fallback: exponential backoff (5s, 15s, 45s)
        return LLM_CONFIG.retry_delay * (3 ** attempt)

    def _dispatch(self, prompt, system_prompt, max_tokens, temperature, fast: bool) -> str:
        if self.backend in _OPENAI_COMPAT:
            return self._generate_openai_compat(prompt, system_prompt, max_tokens, temperature, fast)
        if self.backend == "anthropic":
            return self._generate_anthropic(prompt, system_prompt, max_tokens, temperature, fast)
        if self.backend == "gemini":
            return self._generate_gemini(prompt, system_prompt, max_tokens, temperature, fast)
        if self.backend == "ollama":
            return self._generate_ollama(prompt, system_prompt, max_tokens, temperature, fast)
        if self.backend == "hf":
            return self._generate_hf(prompt, system_prompt, max_tokens, temperature)
        return self._template_response(prompt)

    # ─── Provider Implementations ────────────────────────────

    def _generate_openai_compat(
        self, prompt, system_prompt, max_tokens, temperature, fast: bool
    ) -> str:
        model = self._provider_models["fast" if fast else "primary"]
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        extra = {}
        # OpenRouter needs extra headers
        if self.backend == "openrouter":
            extra["extra_headers"] = {
                "HTTP-Referer": "https://forensic-ai.local",
                "X-Title": "Forensic AI Platform",
            }

        resp = self._openai_client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=LLM_CONFIG.timeout,
            **extra,
        )
        return resp.choices[0].message.content

    def _generate_anthropic(self, prompt, system_prompt, max_tokens, temperature, fast: bool) -> str:
        model = self._provider_models["fast" if fast else "primary"]
        kwargs = dict(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        if system_prompt:
            kwargs["system"] = system_prompt
        resp = self._anthropic_client.messages.create(**kwargs)
        return resp.content[0].text

    def _generate_gemini(self, prompt, system_prompt, max_tokens, temperature, fast: bool) -> str:
        import google.generativeai as genai
        model_name = self._provider_models["fast" if fast else "primary"]
        model = genai.GenerativeModel(
            model_name,
            system_instruction=system_prompt or None,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=temperature,
            ),
        )
        resp = model.generate_content(prompt)
        try:
            return resp.text
        except (AttributeError, ValueError) as e:
            logger.warning(f"Gemini response has no text (safety filter or empty): {e}")
            return self._template_response(prompt)

    def _generate_ollama(self, prompt, system_prompt, max_tokens, temperature, fast: bool) -> str:
        import httpx
        model = self._provider_models.get("fast" if fast else "primary",
                                          getattr(self, "_ollama_model", "qwen2.5:7b"))
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": model, "messages": messages, "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens,
                        "num_ctx": LLM_CONFIG.context_window},
        }
        resp = httpx.post(
            f"{LLM_CONFIG.ollama_base_url}/api/chat",
            json=payload, timeout=LLM_CONFIG.timeout,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]

    def _generate_hf(self, prompt, system_prompt, max_tokens, temperature) -> str:
        model_name = self._provider_models["primary"]
        if self._hf_model is None:
            self._load_hf_model(model_name)
        if self._hf_model is None:
            return self._template_response(prompt)

        import torch
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        inputs = self._hf_tokenizer(
            full_prompt, return_tensors="pt", truncation=True, max_length=2048
        )
        if self._has_gpu:
            inputs = {k: v.cuda() for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self._hf_model.generate(
                **inputs, max_new_tokens=max_tokens,
                temperature=temperature, do_sample=temperature > 0,
                pad_token_id=self._hf_tokenizer.eos_token_id,
            )
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        return self._hf_tokenizer.decode(generated, skip_special_tokens=True)

    def _load_hf_model(self, model_name: str) -> None:
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
            logger.info(f"Loading HF model: {model_name}…")
            self._hf_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            load_kw = {"trust_remote_code": True}
            if self._has_gpu:
                load_kw.update({"torch_dtype": torch.float16, "device_map": "auto"})
            else:
                load_kw["torch_dtype"] = torch.float32
            self._hf_model = AutoModelForCausalLM.from_pretrained(model_name, **load_kw)
            logger.info(f"HF model loaded: {model_name}")
        except Exception as e:
            logger.error(f"HF model load failed: {e}")
            self._hf_model = None
            self.backend = "template"

    def _template_response(self, prompt: str) -> str:
        return (
            "[NO LLM AVAILABLE — TEMPLATE MODE]\n\n"
            "To enable analysis, set any one of these environment variables:\n"
            "  OPENAI_API_KEY=...     platform.openai.com\n"
            "  ANTHROPIC_API_KEY=...  console.anthropic.com\n"
            "  GOOGLE_API_KEY=...     aistudio.google.com (free tier available)\n"
            "  GROQ_API_KEY=...       console.groq.com (free tier available)\n"
            "Or run Ollama locally: https://ollama.com → ollama pull qwen2.5:7b\n\n"
            f"Prompt received (first 200 chars): {prompt[:200]}…"
        )

    # ─── Convenience Methods ──────────────────────────────────

    def fast_generate(self, prompt: str, system_prompt: str = "", max_tokens: int = 1024) -> str:
        return self.generate(prompt, system_prompt, max_tokens=max_tokens, use_fast_model=True)

    def extract_json(self, prompt: str, system_prompt: str = "", schema: dict = None) -> dict:
        if schema:
            prompt += f"\n\nRespond ONLY with valid JSON matching: {json.dumps(schema)}"
        prompt += "\n\nReturn ONLY valid JSON. No markdown fences."
        response = self.generate(prompt, system_prompt, temperature=0.0)
        import re
        # Strip markdown fences if present
        clean = re.sub(r"```(?:json)?|```", "", response).strip()
        for candidate in [clean, response]:
            m = re.search(r"\{.*\}", candidate, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group())
                except json.JSONDecodeError:
                    pass
        return {"raw_response": response, "parse_error": True}

    def analyze_sentiment(self, text: str) -> dict:
        prompt = (
            "Analyze the sentiment and tone of this financial text.\n"
            "Return JSON: {sentiment: positive/negative/neutral, confidence: 0-1, "
            "evasion_score: 0-1, optimism_score: 0-1, uncertainty_score: 0-1, "
            "key_phrases: [list of 5 notable phrases]}\n\n"
            f"Text: {text[:3000]}"
        )
        return self.extract_json(prompt)

    def is_available(self) -> bool:
        if self.backend == "hf":
            return self._hf_model is not None
        return self.backend != "template"

    def get_backend_info(self) -> dict:
        return {
            "backend": self.backend,
            "primary_model": self._provider_models.get("primary", "N/A"),
            "fast_model": self._provider_models.get("fast", "N/A"),
            "gpu_available": self._has_gpu,
        }
