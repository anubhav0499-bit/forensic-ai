"""
Multi-Vector Retriever — Dual embeddings per chunk (raw content + LLM summary).
Improves recall for queries that match document gist rather than exact phrasing.
Both the raw chunk and a 1-sentence summary are embedded; results are merged via RRF.
"""

from __future__ import annotations
import re
from loguru import logger

from processing.chunker import DocumentChunk


class MultiVectorRetriever:
    """
    For every indexed chunk, stores TWO embeddings:
      raw     — full chunk text (precise term matching)
      summary — LLM-generated 1-sentence gist (semantic breadth)

    At query time, retrieves from both collections and merges via RRF(k=60).
    Raw results weighted 0.6, summary results weighted 0.4.

    Requires:
      raw_store    — any store with .add_chunks() / .search() (VectorStore or FAISSStore)
      summary_store — a second store instance for the summary collection
      embedder     — EmbeddingModel (shared, not duplicated)
      llm          — optional LLMManager; if absent, falls back to first-sentence extraction
    """

    RRF_K = 60
    RAW_W = 0.60
    SUM_W = 0.40
    _MAX_CONTENT_FOR_SUMMARY = 800

    def __init__(self, raw_store, summary_store, embedder=None, llm=None):
        self.raw_store = raw_store
        self.summary_store = summary_store
        self.embedder = embedder
        self.llm = llm

    # ── Indexing ──────────────────────────────────────────────────────

    def index(self, company_name: str, chunks: list[DocumentChunk]) -> None:
        """Index chunks into raw store and generate + index their summaries."""
        self.raw_store.add_chunks(company_name, chunks)

        summary_chunks = self._build_summary_chunks(chunks)
        if summary_chunks:
            self.summary_store.add_chunks(company_name, summary_chunks)

        logger.info(
            f"MultiVector [{company_name}]: {len(chunks)} raw + "
            f"{len(summary_chunks)} summary chunks indexed"
        )

    # ── Search ────────────────────────────────────────────────────────

    def search(
        self,
        company_name: str,
        query: str,
        n_results: int = 10,
    ) -> list[dict]:
        """Search raw + summary stores; merge via RRF."""
        raw_hits = self.raw_store.search(company_name, query, n_results=n_results * 2)
        sum_hits = self.summary_store.search(company_name, query, n_results=n_results * 2)
        return self._rrf_merge(raw_hits, sum_hits, n_results)

    def _rrf_merge(
        self,
        raw: list[dict],
        summaries: list[dict],
        n: int,
    ) -> list[dict]:
        scores: dict[str, float] = {}
        result_map: dict[str, dict] = {}

        for rank, r in enumerate(raw):
            key = r["content"][:100]
            scores[key] = scores.get(key, 0.0) + self.RAW_W / (self.RRF_K + rank + 1)
            result_map[key] = r

        for rank, r in enumerate(summaries):
            # Resolve back to the raw content stored in metadata
            raw_content = r.get("metadata", {}).get("raw_content") or r["content"]
            key = raw_content[:100]
            scores[key] = scores.get(key, 0.0) + self.SUM_W / (self.RRF_K + rank + 1)
            if key not in result_map:
                result_map[key] = {
                    "content":         raw_content,
                    "metadata":        r.get("metadata", {}),
                    "score":           r.get("score", 0.0),
                    "source_document": r.get("source_document", ""),
                    "section":         r.get("section", ""),
                    "fiscal_year":     r.get("fiscal_year", ""),
                }

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        out = []
        for key, score in ranked[:n]:
            if key in result_map:
                result = result_map[key].copy()
                result["hybrid_score"] = score
                out.append(result)
        return out

    # ── Summary generation ────────────────────────────────────────────

    def _build_summary_chunks(self, chunks: list[DocumentChunk]) -> list[DocumentChunk]:
        summary_chunks: list[DocumentChunk] = []
        for chunk in chunks:
            summary = self._summarise(chunk.content)
            if not summary:
                continue
            sc = DocumentChunk(
                chunk_id=f"{chunk.chunk_id}_summary",
                content=summary,
                chunk_type=chunk.chunk_type,
                source_document=chunk.source_document,
                page_num=chunk.page_num,
                fiscal_year=chunk.fiscal_year,
                section=chunk.section,
                word_count=len(summary.split()),
                metadata={
                    **chunk.metadata,
                    "is_summary":   True,
                    "raw_chunk_id": chunk.chunk_id,
                    "raw_content":  chunk.content,
                },
            )
            summary_chunks.append(sc)
        return summary_chunks

    def _summarise(self, content: str) -> str:
        """1-sentence summary via LLM; falls back to first-sentence extraction."""
        if self.llm:
            try:
                prompt = (
                    "Summarise the following financial document excerpt in exactly one "
                    "concise sentence, capturing the key financial metric or finding:\n\n"
                    + content[: self._MAX_CONTENT_FOR_SUMMARY]
                )
                result = self.llm.generate(prompt, max_tokens=80)
                if result and len(result.strip()) > 15:
                    return result.strip()
            except Exception:
                pass

        # Fallback: first sentence
        m = re.search(r"[^.!?]*[.!?]", content)
        return m.group(0).strip() if m else content[:200]
