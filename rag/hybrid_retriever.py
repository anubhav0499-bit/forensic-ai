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
        chunk_type: str = None,
    ) -> list[dict]:
        """Hybrid search with RRF fusion."""
        # Get results from both retrievers
        dense_results = self.vector_store.search(company_name, query, n_results=n_results * 2)
        bm25_results = self.bm25.search(company_name, query, n_results=n_results * 2)

        # Apply RRF
        rrf_scores: dict[str, float] = {}
        result_map: dict[str, dict] = {}

        # Dense scores
        for rank, result in enumerate(dense_results):
            key = result["content"][:100]
            rrf_scores[key] = rrf_scores.get(key, 0) + self.dense_weight / (self.rrf_k + rank + 1)
            result_map[key] = result

        # BM25 scores
        for rank, result in enumerate(bm25_results):
            key = result["content"][:100]
            rrf_scores[key] = rrf_scores.get(key, 0) + self.bm25_weight / (self.rrf_k + rank + 1)
            if key not in result_map:
                result_map[key] = result

        # Sort by RRF score
        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for key, score in ranked[:n_results]:
            if key in result_map:
                r = result_map[key].copy()
                r["hybrid_score"] = score
                results.append(r)

        # Filter by chunk type if specified
        if chunk_type:
            results = [r for r in results if r.get("metadata", {}).get("chunk_type") == chunk_type]

        return results

    def search_with_reranking(
        self,
        company_name: str,
        query: str,
        n_results: int = 5,
    ) -> list[dict]:
        """Hybrid search + cross-encoder reranking for highest precision."""
        # Get more candidates for reranking
        candidates = self.search(company_name, query, n_results=n_results * 3)

        # Try cross-encoder reranking
        try:
            from sentence_transformers import CrossEncoder
            reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            pairs = [(query, c["content"]) for c in candidates]
            scores = reranker.predict(pairs)
            ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
            return [r for _, r in ranked[:n_results]]
        except Exception:
            return candidates[:n_results]

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
