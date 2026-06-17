"""
Context Builder — Multi-query RAG retrieval with deduplication, relevance
scoring, token-budget-aware assembly, and optional LLM context compression.

Pipeline:
  retrieve → deduplicate → score-sort → assemble → [compress]
"""

from __future__ import annotations
import hashlib
import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class _ContextChunk:
    text: str
    source: str
    fiscal_year: str
    section: str
    score: float
    _hash: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._hash = hashlib.md5(self.text[:200].encode("utf-8", errors="replace")).hexdigest()


class ContextBuilder:
    """
    Multi-query context assembly for forensic agents.

    Pipeline:
      1. Run all queries against the hybrid retriever.
      2. Deduplicate near-identical chunks (Jaccard >= 0.65).
      3. Sort by relevance score (descending).
      4. Assemble within token budget.
      5. Optionally compress via ContextCompressor (LLM-based or keyword-based).
      6. Cache per (company_name, sorted query set).

    Usage:
        builder = ContextBuilder(retriever)
        context = builder.build(
            company_name="Infosys",
            queries=["revenue recognition policy", "deferred revenue contract assets"],
            budget_tokens=3000,
        )

        # With compression:
        from rag.context_compressor import ContextCompressor
        builder = ContextBuilder(retriever, compressor=ContextCompressor(llm))
    """

    _DEDUP_THRESHOLD = 0.65
    _CHARS_PER_TOKEN = 4
    _MIN_CHUNK_CHARS = 40

    def __init__(self, retriever, compressor=None) -> None:
        self.retriever = retriever
        self.compressor = compressor   # optional ContextCompressor
        self._cache: dict[str, str] = {}

    # ── Public API ─────────────────────────────────────────────────

    def build(
        self,
        company_name: str,
        queries: list[str] | str,
        budget_tokens: int = 3000,
        n_per_query: int = 6,
        compress: bool = False,
        primary_query: str = "",
    ) -> str:
        """
        Build context from one or more queries, deduplicated and budget-bounded.
        Returns formatted text ready for LLM prompt injection.

        compress=True: apply ContextCompressor after assembly (requires compressor set).
        primary_query: used as the compression target; defaults to queries[0].
        """
        if isinstance(queries, str):
            queries = [queries]
        queries = [q.strip() for q in queries if q.strip()]
        if not queries:
            return ""

        cache_key = f"{company_name}||{'|'.join(sorted(queries))}||{compress}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        chunks = self._retrieve_all(company_name, queries, n_per_query)
        chunks = self._deduplicate(chunks)
        chunks.sort(key=lambda c: c.score, reverse=True)
        context = self._assemble(chunks, budget_tokens)

        if compress and self.compressor and context:
            q_for_compression = primary_query or queries[0]
            context = self.compressor.compress(
                context, q_for_compression, max_tokens=budget_tokens
            )

        self._cache[cache_key] = context
        return context

    def invalidate(self, company_name: str | None = None) -> None:
        """Clear cache for one company (or all companies if None)."""
        if company_name is None:
            self._cache.clear()
        else:
            for k in [k for k in self._cache if k.startswith(company_name)]:
                del self._cache[k]

    # ── Retrieval ──────────────────────────────────────────────────

    def _retrieve_all(
        self, company_name: str, queries: list[str], n_per_query: int
    ) -> list[_ContextChunk]:
        chunks: list[_ContextChunk] = []
        for query in queries:
            try:
                results = self.retriever.search(
                    company_name=company_name, query=query, n_results=n_per_query
                )
                for r in results:
                    text = r.get("content", r.get("text", "")).strip()
                    if len(text) < self._MIN_CHUNK_CHARS:
                        continue
                    meta = r.get("metadata", {})
                    chunks.append(_ContextChunk(
                        text=text,
                        source=meta.get("source_document", r.get("source_document", "Unknown")),
                        fiscal_year=meta.get("fiscal_year", r.get("fiscal_year", "")),
                        section=meta.get("section", r.get("section", "")),
                        score=float(r.get("hybrid_score", r.get("score", 0.5))),
                    ))
            except Exception:
                continue
        return chunks

    # ── Deduplication ──────────────────────────────────────────────

    def _deduplicate(self, chunks: list[_ContextChunk]) -> list[_ContextChunk]:
        seen_hashes: set[str] = set()
        unique: list[_ContextChunk] = []
        for chunk in chunks:
            if chunk._hash in seen_hashes:
                continue
            dup_idx = self._find_near_duplicate(chunk, unique)
            if dup_idx is None:
                seen_hashes.add(chunk._hash)
                unique.append(chunk)
            elif chunk.score > unique[dup_idx].score:
                # Replace with higher-scoring version
                seen_hashes.discard(unique[dup_idx]._hash)
                unique[dup_idx] = chunk
                seen_hashes.add(chunk._hash)
        return unique

    def _find_near_duplicate(
        self, candidate: _ContextChunk, pool: list[_ContextChunk]
    ) -> int | None:
        cand_tokens = set(self._tokenize(candidate.text))
        if not cand_tokens:
            return None
        for i, existing in enumerate(pool):
            ex_tokens = set(self._tokenize(existing.text))
            if not ex_tokens:
                continue
            union = cand_tokens | ex_tokens
            intersection = cand_tokens & ex_tokens
            if intersection and len(intersection) / len(union) >= self._DEDUP_THRESHOLD:
                return i
        return None

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"\b[a-zA-Z0-9]{3,}\b", text.lower())

    # ── Assembly ───────────────────────────────────────────────────

    def _assemble(self, chunks: list[_ContextChunk], budget_tokens: int) -> str:
        if not chunks:
            return "No relevant document context found."
        parts: list[str] = []
        tokens_used = 0
        for i, chunk in enumerate(chunks, 1):
            chunk_tokens = len(chunk.text) // self._CHARS_PER_TOKEN
            if tokens_used + chunk_tokens > budget_tokens:
                remaining_chars = (budget_tokens - tokens_used) * self._CHARS_PER_TOKEN
                if remaining_chars > 200:
                    truncated = chunk.text[:remaining_chars].rsplit(" ", 1)[0] + " …"
                    parts.append(f"{self._header(i, chunk)}\n{truncated}")
                break
            parts.append(f"{self._header(i, chunk)}\n{chunk.text}")
            tokens_used += chunk_tokens

        sep = "\n\n" + "─" * 40 + "\n\n"
        return sep.join(parts)

    @staticmethod
    def _header(idx: int, chunk: _ContextChunk) -> str:
        parts = [f"Source {idx}: {chunk.source}"]
        if chunk.fiscal_year:
            parts.append(f"FY{chunk.fiscal_year}")
        if chunk.section:
            parts.append(f"§ {chunk.section}")
        return "[" + " | ".join(parts) + "]"
