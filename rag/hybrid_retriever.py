"""
Hybrid Retriever - Combines BM25 + Dense Retrieval with Reciprocal Rank Fusion
"""

from __future__ import annotations
from loguru import logger
from .bm25_retriever import BM25Retriever
from .vector_store import VectorStore
from processing.chunker import DocumentChunk


class HybridRetriever:
    """
    Hybrid retrieval: BM25 (keyword) + Dense (semantic) + RRF fusion.
    Weights: 0.4 BM25 + 0.6 Dense (optimized for financial documents).
    """

    def __init__(self, vector_store: VectorStore, bm25: BM25Retriever):
        self.vector_store = vector_store
        self.bm25 = bm25
        self.bm25_weight = 0.4
        self.dense_weight = 0.6
        self.rrf_k = 60  # Reciprocal Rank Fusion constant
        self._reranker = None  # lazy-loaded CrossEncoder

    def index(self, company_name: str, chunks: list[DocumentChunk]) -> None:
        """Index documents in both BM25 and vector store."""
        logger.info(f"Indexing {len(chunks)} chunks for {company_name}")
        self.bm25.index_chunks(company_name, chunks)
        self.vector_store.add_chunks(company_name, chunks)

    def search(
        self,
        company_name: str,
        query: str,
        n_results: int = 10,
        chunk_type: str | None = None,
    ) -> list[dict]:
        """Hybrid search with RRF fusion.

        chunk_type filtering is applied PRE-RRF to each individual retriever so
        that the fusion pool contains only relevant chunk types and the final
        n_results count is not silently reduced by post-hoc filtering.
        """
        dense_filter = {"chunk_type": chunk_type} if chunk_type else None
        dense_results = self.vector_store.search(
            company_name, query,
            n_results=n_results * 2,
            filter_metadata=dense_filter,
        )
        bm25_results = self.bm25.search(
            company_name, query,
            n_results=n_results * 2,
            chunk_type=chunk_type,
        )

        rrf_scores: dict[str, float] = {}
        result_map: dict[str, dict] = {}

        for rank, result in enumerate(dense_results):
            key = result["content"][:100]
            rrf_scores[key] = rrf_scores.get(key, 0) + self.dense_weight / (self.rrf_k + rank + 1)
            result_map[key] = result

        for rank, result in enumerate(bm25_results):
            key = result["content"][:100]
            rrf_scores[key] = rrf_scores.get(key, 0) + self.bm25_weight / (self.rrf_k + rank + 1)
            if key not in result_map:
                result_map[key] = result

        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for key, score in ranked[:n_results]:
            if key in result_map:
                r = result_map[key].copy()
                r["hybrid_score"] = score
                results.append(r)

        return results

    def _get_reranker(self):
        """Lazily load and cache the cross-encoder reranker (one instance per HybridRetriever)."""
        if self._reranker is None:
            from sentence_transformers import CrossEncoder
            self._reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        return self._reranker

    def search_with_reranking(
        self,
        company_name: str,
        query: str,
        n_results: int = 5,
        chunk_type: str | None = None,
    ) -> list[dict]:
        """Hybrid search + cross-encoder reranking for highest precision.

        The cross-encoder is loaded once and cached on the instance so
        subsequent calls pay no model-load overhead.
        """
        candidates = self.search(company_name, query, n_results=n_results * 3, chunk_type=chunk_type)

        try:
            reranker = self._get_reranker()
            pairs = [(query, c["content"]) for c in candidates]
            scores = reranker.predict(pairs)
            ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
            return [r for _, r in ranked[:n_results]]
        except Exception:
            return candidates[:n_results]

    def search_multi(
        self,
        company_name: str,
        queries: list[str],
        n_per_query: int = 6,
    ) -> list[dict]:
        """
        Run multiple queries and merge results via RRF, preserving diversity.
        Deduplication is handled downstream by ContextBuilder; here we just
        merge and re-rank across all queries.
        """
        merged_scores: dict[str, float] = {}
        result_map: dict[str, dict] = {}

        for q_idx, query in enumerate(queries):
            results = self.search(company_name, query, n_results=n_per_query)
            for rank, result in enumerate(results):
                key = result["content"][:100]
                # Boost earlier queries slightly (primary query is most relevant)
                query_weight = 1.0 / (1 + q_idx * 0.15)
                rrf_contribution = query_weight / (self.rrf_k + rank + 1)
                merged_scores[key] = merged_scores.get(key, 0) + rrf_contribution
                if key not in result_map:
                    result_map[key] = result

        ranked = sorted(merged_scores.items(), key=lambda x: x[1], reverse=True)
        results = []
        for key, score in ranked:
            if key in result_map:
                r = result_map[key].copy()
                r["hybrid_score"] = score
                results.append(r)
        return results

    def get_context_for_agent(
        self,
        company_name: str,
        agent_query: str,
        max_tokens: int = 3000,
    ) -> str:
        """
        Retrieve and format context for an agent prompt.
        Returns formatted text ready for LLM injection.
        """
        results = self.search(company_name, agent_query, n_results=8)

        if not results:
            return "No relevant document context found."

        context_parts = []
        total_words = 0

        for i, result in enumerate(results, 1):
            content = result["content"]
            words = content.split()

            if total_words + len(words) > max_tokens:
                remaining = max_tokens - total_words
                content = " ".join(words[:remaining])

            meta = result.get("metadata", {})
            context_parts.append(
                f"[Source {i}: {meta.get('source_document', 'Unknown')}, "
                f"FY{meta.get('fiscal_year', '?')}, "
                f"Section: {meta.get('section', 'Unknown')}]\n{content}"
            )
            total_words += len(content.split())

            if total_words >= max_tokens:
                break

        return "\n\n" + "─" * 40 + "\n\n".join(context_parts)
