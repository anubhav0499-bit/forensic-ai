"""
Context Compressor — LLM-based extractive compression and redundancy removal.
Reduces retrieved context to only the snippets most relevant to the query,
improving answer quality while making room for more source diversity.
"""

from __future__ import annotations
import re
from loguru import logger


class ContextCompressor:
    """
    Two-stage compression pipeline:
      Stage 1 — Extractive: keep only sentences relevant to the query
      Stage 2 — Dedup: remove near-identical sentences across chunks

    Uses LLM for semantic relevance when available; falls back to
    keyword-overlap scoring when the LLM is not configured.

    Inputs / outputs use the same string format as ContextBuilder.build(),
    so the compressor can be dropped in as a post-processing step.
    """

    _SOURCE_HEADER = re.compile(r"\[Source \d+:[^\]]+\]")
    _SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

    def __init__(self, llm_manager=None, compression_ratio: float = 0.5):
        self.llm = llm_manager
        self.compression_ratio = compression_ratio  # target: keep ~50% of input

    # ── Public API ────────────────────────────────────────────────────

    def compress(
        self,
        context: str,
        query: str,
        max_tokens: int = 1500,
    ) -> str:
        """
        Compress context to roughly max_tokens while maximising query relevance.
        Returns a formatted string in the same [Source N: ...] style as input.
        Falls back to truncated original if compression produces < 100 chars.
        """
        if not context or context.startswith("No relevant"):
            return context

        chunks = self._parse_chunks(context)
        if not chunks:
            return context

        if self.llm:
            result = self._llm_compress(chunks, query, max_tokens)
        else:
            result = self._keyword_compress(chunks, query, max_tokens)

        return result if len(result) >= 100 else context[: max_tokens * 4]

    # ── Parsing ───────────────────────────────────────────────────────

    def _parse_chunks(self, context: str) -> list[dict]:
        """Split the formatted context string into {header, body} pairs."""
        parts = self._SOURCE_HEADER.split(context)
        headers = self._SOURCE_HEADER.findall(context)
        chunks = []
        for header, body in zip(headers, parts[1:]):
            body = body.strip()
            if body:
                chunks.append({"header": header, "body": body})
        return chunks

    def _join(self, parts: list[str]) -> str:
        sep = "\n\n" + "─" * 40 + "\n\n"
        return sep.join(parts)

    # ── LLM-based compression ─────────────────────────────────────────

    def _llm_compress(
        self,
        chunks: list[dict],
        query: str,
        max_tokens: int,
    ) -> str:
        budget = max_tokens * 4  # chars
        used = 0
        out: list[str] = []

        for chunk in chunks:
            if used >= budget:
                break
            body = chunk["body"]
            if len(body) < 150:
                # Too short to compress — include as-is
                out.append(f"{chunk['header']}\n{body}")
                used += len(body)
                continue
            try:
                prompt = (
                    f"Extract ONLY the sentences directly relevant to: \"{query[:200]}\"\n\n"
                    f"Text:\n{body[:1200]}\n\n"
                    f"Output only the extracted sentences, nothing else. "
                    f"If nothing is relevant, output exactly: SKIP"
                )
                extracted = self.llm.generate(prompt, max_tokens=250)
                if not extracted or extracted.strip().upper() == "SKIP":
                    continue
                extracted = extracted.strip()
                out.append(f"{chunk['header']}\n{extracted}")
                used += len(extracted)
            except Exception as exc:
                logger.debug(f"LLM compress failed for chunk: {exc}")
                truncated = body[: int(len(body) * self.compression_ratio)]
                out.append(f"{chunk['header']}\n{truncated}")
                used += len(truncated)

        return self._join(out)

    # ── Keyword-overlap compression ───────────────────────────────────

    def _keyword_compress(
        self,
        chunks: list[dict],
        query: str,
        max_tokens: int,
    ) -> str:
        q_terms = set(re.findall(r"\b[a-zA-Z0-9]{3,}\b", query.lower()))
        budget = max_tokens * 4
        used = 0
        out: list[str] = []

        for chunk in chunks:
            if used >= budget:
                break
            sentences = self._SENT_SPLIT.split(chunk["body"])
            scored: list[tuple[float, str]] = []
            for sent in sentences:
                sent_terms = set(re.findall(r"\b[a-zA-Z0-9]{3,}\b", sent.lower()))
                if not sent_terms:
                    continue
                overlap = len(q_terms & sent_terms) / (len(q_terms) + 1)
                scored.append((overlap, sent))

            scored.sort(reverse=True)
            keep_n = max(1, int(len(scored) * self.compression_ratio))
            kept = " ".join(s for _, s in scored[:keep_n])
            if kept:
                out.append(f"{chunk['header']}\n{kept}")
                used += len(kept)

        return self._join(out)
