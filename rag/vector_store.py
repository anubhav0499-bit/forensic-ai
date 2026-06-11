"""
ChromaDB Vector Store - Persistent embedding storage for forensic knowledge base
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional
from loguru import logger

from config import VECTOR_DB_CONFIG
from .embeddings import EmbeddingModel
from processing.chunker import DocumentChunk


class VectorStore:
    """
    ChromaDB-backed vector store for forensic document retrieval.
    Supports company-specific collections.
    """

    def __init__(self, embedding_model: EmbeddingModel, persist_dir: Optional[Path] = None):
        self.embedding_model = embedding_model
        self.persist_dir = str(persist_dir or VECTOR_DB_CONFIG.persist_dir)
        self._client = None
        self._collections: dict = {}
        self._initialize()

    def _initialize(self) -> None:
        try:
            import chromadb
            from chromadb.config import Settings
            Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=self.persist_dir,
                settings=Settings(anonymized_telemetry=False),
            )
            logger.info(f"ChromaDB initialized at: {self.persist_dir}")
        except Exception as e:
            logger.error(f"ChromaDB initialization failed: {e}")
            self._client = None

    def get_or_create_collection(self, company_name: str) -> Optional[object]:
        """Get or create a ChromaDB collection for a company."""
        if self._client is None:
            return None

        collection_name = f"{VECTOR_DB_CONFIG.collection_prefix}_{company_name[:30].replace(' ', '_').lower()}"

        if collection_name not in self._collections:
            try:
                import chromadb
                self._collections[collection_name] = self._client.get_or_create_collection(
                    name=collection_name,
                    metadata={"hnsw:space": VECTOR_DB_CONFIG.distance_metric},
                )
            except Exception as e:
                logger.error(f"Collection creation failed: {e}")
                return None

        return self._collections[collection_name]

    def add_chunks(self, company_name: str, chunks: list[DocumentChunk]) -> int:
        """Add document chunks to the vector store."""
        collection = self.get_or_create_collection(company_name)
        if collection is None:
            logger.warning("Vector store not available - skipping embedding")
            return 0

        if not chunks:
            return 0

        added = 0
        batch_size = 100

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            contents = [c.content for c in batch]
            ids = [c.chunk_id for c in batch]
            metadatas = [
                {
                    "chunk_type": c.chunk_type,
                    "source_document": c.source_document,
                    "section": c.section,
                    "fiscal_year": c.fiscal_year,
                    "word_count": c.word_count,
                }
                for c in batch
            ]

            try:
                embeddings = self.embedding_model.encode(contents)
                collection.upsert(
                    ids=ids,
                    embeddings=embeddings.tolist(),
                    documents=contents,
                    metadatas=metadatas,
                )
                added += len(batch)
            except Exception as e:
                logger.error(f"Failed to add batch to vector store: {e}")

        logger.info(f"Added {added} chunks to vector store for {company_name}")
        return added

    def search(
        self,
        company_name: str,
        query: str,
        n_results: int = None,
        filter_metadata: dict = None,
    ) -> list[dict]:
        """Semantic search over company documents."""
        collection = self.get_or_create_collection(company_name)
        if collection is None:
            return []

        n_results = n_results or VECTOR_DB_CONFIG.n_results
        query_embedding = self.embedding_model.encode_query(query)

        try:
            where = filter_metadata if filter_metadata else None
            results = collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=n_results,
                where=where,
                include=["documents", "metadatas", "distances"],
            )

            hits = []
            for i, doc in enumerate(results["documents"][0]):
                hits.append({
                    "content": doc,
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i],
                    "similarity": 1 / (1 + results["distances"][0][i]) if results["distances"][0][i] >= 0 else 1 - results["distances"][0][i],
                })
            return hits
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

    def search_by_type(
        self,
        company_name: str,
        query: str,
        chunk_type: str,
        n_results: int = 5,
    ) -> list[dict]:
        """Search filtered by chunk type (e.g., 'financial_statement', 'mda')."""
        return self.search(
            company_name,
            query,
            n_results=n_results,
            filter_metadata={"chunk_type": chunk_type},
        )

    def get_collection_count(self, company_name: str) -> int:
        collection = self.get_or_create_collection(company_name)
        if collection is None:
            return 0
        try:
            return collection.count()
        except Exception:
            return 0

    def delete_company_collection(self, company_name: str) -> None:
        """Remove all documents for a company (for re-indexing)."""
        if self._client is None:
            return
        collection_name = f"{VECTOR_DB_CONFIG.collection_prefix}_{company_name[:30].replace(' ', '_').lower()}"
        try:
            self._client.delete_collection(collection_name)
            self._collections.pop(collection_name, None)
            logger.info(f"Deleted collection: {collection_name}")
        except Exception as e:
            logger.debug(f"Collection deletion: {e}")
