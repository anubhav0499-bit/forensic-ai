"""
Streaming — Server-Sent Events (SSE) for real-time token streaming.

FastAPI integration:
    from fastapi import FastAPI
    from fastapi.responses import StreamingResponse
    from api.streaming import StreamingManager

    app = FastAPI()
    streamer = StreamingManager(llm_manager)

    @app.post("/stream/analyze")
    async def stream_analyze(body: AnalysisRequest):
        return StreamingResponse(
            streamer.stream_sse(body.prompt, body.system_role),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

Streamlit integration (synchronous):
    for chunk in streamer.stream_sync(prompt, system_role):
        st.write(chunk)
"""

from __future__ import annotations
import asyncio
import json
from typing import AsyncGenerator, Generator

from loguru import logger


class StreamingManager:
    """
    Token-streaming wrapper supporting:
      - async SSE generator (FastAPI / ASGI)
      - sync chunk generator (Streamlit / scripts)

    Provider-awareness:
      Tries self.llm.stream() first (Anthropic, OpenAI, Groq support it).
      Falls back to full-response chunked delivery (20-char chunks) when
      the provider does not expose a streaming interface.
    """

    _SYNC_CHUNK = 25      # chars per chunk in fallback sync mode
    _ASYNC_CHUNK = 20     # chars per chunk in fallback async mode
    _ASYNC_SLEEP = 0.01   # seconds between async chunks (simulate cadence)

    def __init__(self, llm_manager=None):
        self.llm = llm_manager

    # ── Async SSE ─────────────────────────────────────────────────────

    async def stream_sse(
        self,
        prompt: str,
        system_role: str = "forensic_accountant",
        max_tokens: int = 2048,
    ) -> AsyncGenerator[str, None]:
        """
        Async generator for FastAPI StreamingResponse (media_type="text/event-stream").
        Each yield is a complete SSE frame:
          data: {"token": "...", "done": false}\\n\\n
          data: {"token": "", "done": true}\\n\\n
          data: {"error": "...", "done": true}\\n\\n   (on failure)
        """
        try:
            async for token in self._async_tokens(prompt, system_role, max_tokens):
                frame = json.dumps({"token": token, "done": False})
                yield f"data: {frame}\n\n"
            yield f"data: {json.dumps({'token': '', 'done': True})}\n\n"
        except Exception as exc:
            logger.error(f"SSE stream error: {exc}")
            yield f"data: {json.dumps({'error': str(exc), 'done': True})}\n\n"

    async def _async_tokens(
        self,
        prompt: str,
        system_role: str,
        max_tokens: int,
    ) -> AsyncGenerator[str, None]:
        system_prompt = self._system_prompt(system_role)

        # Try native streaming
        if self.llm:
            try:
                gen = self.llm.stream(prompt, system_prompt, max_tokens=max_tokens)
                if gen is not None:
                    async for token in self._wrap_sync(gen):
                        yield token
                    return
            except (AttributeError, NotImplementedError):
                pass

        # Fallback: chunked full response
        full = self._generate_full(prompt, system_prompt, max_tokens)
        for i in range(0, len(full), self._ASYNC_CHUNK):
            yield full[i: i + self._ASYNC_CHUNK]
            await asyncio.sleep(self._ASYNC_SLEEP)

    @staticmethod
    async def _wrap_sync(gen: Generator) -> AsyncGenerator[str, None]:
        """Wrap a synchronous token generator in an async generator."""
        for token in gen:
            yield token
            await asyncio.sleep(0)

    # ── Sync streaming (Streamlit / scripts) ─────────────────────────

    def stream_sync(
        self,
        prompt: str,
        system_role: str = "forensic_accountant",
        max_tokens: int = 2048,
    ) -> Generator[str, None, None]:
        """
        Synchronous chunk generator for Streamlit st.write_stream() or scripts.
        Yields str chunks of progressive response text.
        """
        system_prompt = self._system_prompt(system_role)

        if self.llm:
            try:
                gen = self.llm.stream(prompt, system_prompt, max_tokens=max_tokens)
                if gen is not None:
                    yield from gen
                    return
            except (AttributeError, NotImplementedError):
                pass

        full = self._generate_full(prompt, system_prompt, max_tokens)
        for i in range(0, len(full), self._SYNC_CHUNK):
            yield full[i: i + self._SYNC_CHUNK]

    # ── Helpers ───────────────────────────────────────────────────────

    def _system_prompt(self, role: str) -> str:
        try:
            from llm.prompts import SYSTEM_PROMPTS
            return SYSTEM_PROMPTS.get(role, SYSTEM_PROMPTS["forensic_accountant"])
        except ImportError:
            return "You are a senior forensic accountant."

    def _generate_full(self, prompt: str, system_prompt: str, max_tokens: int) -> str:
        if self.llm:
            try:
                return self.llm.generate(prompt, system_prompt, max_tokens=max_tokens) or ""
            except Exception as exc:
                logger.warning(f"LLM generate failed in streaming fallback: {exc}")
        return "LLM not available."
