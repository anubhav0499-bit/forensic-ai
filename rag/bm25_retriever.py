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
        # Convert currency symbols to searchable codes before lowercasing
        for sym, code in [("₹", " rs "), ("$", " usd "), ("\xa3", " gbp "), ("€", " eur ")]:
            text = text.replace(sym, code)

        # Split CamelCase BEFORE lowercasing so uppercase boundaries are visible:
        # BeneishMScore → Beneish M Score
        text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
        text = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", text)

        text = text.lower()

        # Replace slash/backslash with space (Net Debt/EBITDA → net debt ebitda)
        text = re.sub(r"[/\\]", " ", text)

        # Replace non-word chars except hyphens, dots, underscores
        text = re.sub(r"[^\w\s.\-]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        tokens: list[str] = []
        for t in text.split():
            if "-" in t and len(t) > 3:
                # Hyphenated financial compound: keep joined form AND each part
                # e.g. m-score → ["m-score", "m", "score"]
                tokens.append(t)
                tokens.extend(p for p in t.split("-") if len(p) > 1)
            else:
                tokens.append(t)

        return [t for t in tokens if len(t) > 1 or t in ["m", "b", "k", "p", "q"]]

    def index_chunks(self, company_name: str, chunks: list[DocumentChunk]) -> None:
        """Build BM25 index for a company's documents."""
        tokenized = [self._tokenize(chunk.content) for chunk in chunks]
        bm25 = BM25Okapi(tokenized)
        self._indexes[company_name] = (bm25, chunks)

    def search(
        self,
        company_name: str,
        query: str,
        n_results: int = 10,
        chunk_type: str | None = None,
    ) -> list[dict]:
        """BM25 search with optional pre-filter on chunk_type."""
        if company_name not in self._indexes:
            return []

        bm25, chunks = self._indexes[company_name]
        query_tokens = self._tokenize(query)
        scores = bm25.get_scores(query_tokens)

        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        results = []
        for idx in top_indices:
            if len(results) >= n_results:
                break
            if scores[idx] <= 0:
                continue
            if chunk_type and chunks[idx].chunk_type != chunk_type:
                continue
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
