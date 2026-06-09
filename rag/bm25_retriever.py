"""
BM25 Retriever - Keyword-based retrieval for exact financial term matching
Complements dense retrieval for precise financial metric lookup.
"""

from __future__ import annotations
import re
from rank_bm25 import BM25Okapi
from processing.chunker import DocumentChunk


class BM25Retriever:
    """BM25 retriever for precise financial keyword matching."""

    def __init__(self):
        self._indexes: dict[str, tuple[BM25Okapi, list[DocumentChunk]]] = {}

    def _tokenize(self, text: str) -> list[str]:
        """Financial-aware tokenization."""
        # Normalize
        text = text.lower()
        text = re.sub(r"[^\w\s.]", " ", text)
        text = re.sub(r"\s+", " ", text)
        tokens = text.split()
        # Remove very short tokens but keep financial acronyms
        return [t for t in tokens if len(t) > 1 or t in ["m", "b", "k"]]

    def index_chunks(self, company_name: str, chunks: list[DocumentChunk]) -> None:
        """Build BM25 index for a company's documents."""
        tokenized = [self._tokenize(chunk.content) for chunk in chunks]
        bm25 = BM25Okapi(tokenized)
        self._indexes[company_name] = (bm25, chunks)

    def search(self, company_name: str, query: str, n_results: int = 10) -> list[dict]:
        """BM25 search."""
        if company_name not in self._indexes:
            return []

        bm25, chunks = self._indexes[company_name]
        query_tokens = self._tokenize(query)
        scores = bm25.get_scores(query_tokens)

        # Get top N
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n_results]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append({
                    "content": chunks[idx].content,
                    "metadata": {
                        "source_document": chunks[idx].source_document,
                        "section": chunks[idx].section,
                        "fiscal_year": chunks[idx].fiscal_year,
                        "chunk_type": chunks[idx].chunk_type,
                    },
                    "bm25_score": float(scores[idx]),
                })
        return results

    def has_index(self, company_name: str) -> bool:
        return company_name in self._indexes
