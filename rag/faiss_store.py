"""
FAISS Vector Store — High-performance local similarity search with persistence
==============================================================================

References
----------
Johnson, J., Douze, M., & Jégou, H. (2019). "Billion-Scale Similarity
    Search with GPUs." IEEE Transactions on Big Data, 7(3), 535–547.
    FAISS (Facebook AI Similarity Search) — IndexFlatIP used here
    (inner product = cosine similarity on L2-normalised vectors).

Douze, M., et al. (2024). "The FAISS Library." arXiv:2401.08281.
    Comprehensive overview of FAISS index types, quantisation methods,
    and GPU acceleration strategies.

Index Type
----------
IndexFlatIP: exact inner-product search (no quantisation, full recall).
Input vectors are L2-normalised so inner-product == cosine similarity.
Metadata is co-stored in a companion .pkl file for filtering support.

Persistence Layout
------------------
{DATA_DIR}/faiss_db/{company}.faiss     — FAISS binary index
{DATA_DIR}/faiss_db/{company}_meta.pkl  — list[dict] metadata records

Drop-in alternative to ChromaDB for faster in-memory retrieval.
"""

from __future__ import annotations
import pickle
from pathlib import Path
from loguru import logger

from processing.chunker import DocumentChunk
from config import DATA_DIR


class FAISSStore:
    """
    FAISS IndexFlatIP (cosine on normalized vectors) with full disk persistence.
    Same .add_chunks() / .search() interface as VectorStore for easy swapping.

    Lazy-imports faiss so the module loads without error even if faiss-cpu is
    not installed — calls will raise ImportError only at runtime.
    """

    PERSIST_DIR = DATA_DIR / "faiss_db"

    def __init__(self, embedding_model=None):
        from rag.embeddings import EmbeddingModel
        self.embedder = embedding_model or EmbeddingModel()
        self.PERSIST_DIR.mkdir(parents=True, exist_ok=True)
        self._indexes: dict[str, object] = {}        # company_name → faiss.Index
        self._metadata: dict[str, list[dict]] = {}   # parallel to FAISS rows

    # ── Internal helpers ──────────────────────────────────────────────

    def _slug(self, company_name: str) -> str:
        return company_name.lower().replace(" ", "_").replace("/", "_")[:40]

    def _index_path(self, slug: str) -> Path:
        return self.PERSIST_DIR / f"{slug}.faiss"

    def _meta_path(self, slug: str) -> Path:
        return self.PERSIST_DIR / f"{slug}_meta.pkl"

    def _require_faiss(self):
        try:
            import faiss
            return faiss
        except ImportError:
            raise ImportError(
                "faiss-cpu not installed. Run: pip install faiss-cpu"
            )

    def _load_or_create(self, company_name: str) -> None:
        """Load index from disk if it exists; otherwise create an empty one."""
        if company_name in self._indexes:
            return

        faiss = self._require_faiss()
        slug = self._slug(company_name)
        idx_path = self._index_path(slug)
        meta_path = self._meta_path(slug)

        if idx_path.exists() and meta_path.exists():
            logger.info(f"FAISS: loading index from {idx_path}")
            self._indexes[company_name] = faiss.read_index(str(idx_path))
            with open(meta_path, "rb") as f:
                self._metadata[company_name] = pickle.load(f)
        else:
            dim = self.embedder.get_dimension()
            # Inner-product index → cosine on unit-normalised vectors
            self._indexes[company_name] = faiss.IndexFlatIP(dim)
            self._metadata[company_name] = []

    def _save(self, company_name: str) -> None:
        faiss = self._require_faiss()
        slug = self._slug(company_name)
        faiss.write_index(self._indexes[company_name], str(self._index_path(slug)))
        with open(self._meta_path(slug), "wb") as f:
            pickle.dump(self._metadata[company_name], f)

    # ── Public API ────────────────────────────────────────────────────

    def add_chunks(self, company_name: str, chunks: list[DocumentChunk]) -> None:
        """Embed and add chunks. Skips chunks already indexed (by chunk_id)."""
        import numpy as np
        self._load_or_create(company_name)

        existing_ids = {m["chunk_id"] for m in self._metadata[company_name]}
        new_chunks = [c for c in chunks if c.chunk_id not in existing_ids]
        if not new_chunks:
            return

        texts = [c.content for c in new_chunks]
        embeddings = self.embedder.encode(texts).astype("float32")

        self._indexes[company_name].add(embeddings)
        for chunk in new_chunks:
            self._metadata[company_name].append({
                "chunk_id":        chunk.chunk_id,
                "content":         chunk.content,
                "source_document": chunk.source_document,
                "section":         chunk.section,
                "fiscal_year":     chunk.fiscal_year,
                "chunk_type":      chunk.chunk_type,
                "word_count":      chunk.word_count,
            })

        self._save(company_name)
        logger.info(f"FAISS: indexed {len(new_chunks)} new chunks for {company_name}")

    def search(
        self,
        company_name: str,
        query: str,
        n_results: int = 10,
        filter_metadata: dict | None = None,
    ) -> list[dict]:
        """Cosine-similarity search; returns top-n matches with score."""
        import numpy as np
        self._load_or_create(company_name)
        index = self._indexes[company_name]
        meta = self._metadata[company_name]

        if index.ntotal == 0:
            return []

        q_emb = self.embedder.encode_query(query).astype("float32")
        q_emb = q_emb.reshape(1, -1)

        k = min(n_results * 3, index.ntotal)
        scores, indices = index.search(q_emb, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            m = meta[idx]
            if filter_metadata and not all(m.get(k) == v for k, v in filter_metadata.items()):
                continue
            results.append({
                "content":         m["content"],
                "metadata":        m,
                "score":           float(score),
                "source_document": m.get("source_document", ""),
                "section":         m.get("section", ""),
                "fiscal_year":     m.get("fiscal_year", ""),
            })
            if len(results) >= n_results:
                break

        return results

    def collection_exists(self, company_name: str) -> bool:
        slug = self._slug(company_name)
        return self._index_path(slug).exists()

    def drop_collection(self, company_name: str) -> None:
        slug = self._slug(company_name)
        for path in [self._index_path(slug), self._meta_path(slug)]:
            if path.exists():
                path.unlink()
        self._indexes.pop(company_name, None)
        self._metadata.pop(company_name, None)

    def stats(self, company_name: str) -> dict:
        self._load_or_create(company_name)
        index = self._indexes.get(company_name)
        return {
            "company":     company_name,
            "total_chunks": index.ntotal if index else 0,
            "dimension":   self.embedder.get_dimension(),
            "persist_dir": str(self.PERSIST_DIR),
        }
