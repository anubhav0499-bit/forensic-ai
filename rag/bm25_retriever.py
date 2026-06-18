"""
BM25 Retriever — Keyword-based retrieval for exact financial term matching
==========================================================================
Complements dense retrieval for precise financial metric lookup.

References
----------
Robertson, S., & Zaragoza, H. (2009). "The Probabilistic Relevance
    Framework: BM25 and Beyond." Foundations and Trends in Information
    Retrieval, 3(4), 333-389.
    BM25Okapi: term-frequency saturation (k1=1.5) + document-length
    normalization (b=0.75).

Jones, K. S., Walker, S., & Robertson, S. E. (2000). "A Probabilistic
    Model of Information Retrieval." Information Processing & Management,
    36(6), 779-808.
    Original BM derivation underlying BM25.

Loughran, T., & McDonald, B. (2011). "When Is a Liability Not a Liability?
    Textual Analysis, Dictionaries, and 10-Ks." Journal of Finance, 66(1),
    35-65.
    Finance domain: BM25 excels on specialized vocabulary (EBITDA, NPA,
    PCAOB, SEBI) absent from general-domain IDF tables.

Architecture
------------
Role in hybrid retrieval: BM25 weight = 0.4 in the RRF fusion (vs. dense
    FAISS weight 0.6). BM25 provides exact-match recall for financial terms;
    dense retrieval provides semantic recall for paraphrased queries.
    RRF k=60 (Cormack et al. 2009).
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
