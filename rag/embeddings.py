"""
Embedding Model - BAAI/bge-large-en-v1.5 for financial document embeddings
Falls back to all-MiniLM-L6-v2 if the larger model is unavailable.
"""

from __future__ import annotations
import numpy as np
from typing import Union
from loguru import logger
from config import EMBEDDING_CONFIG


class EmbeddingModel:
    """
    Sentence embedding model for financial document similarity search.
    Primary: BAAI/bge-large-en-v1.5 (best for English financial text)
    Fallback: all-MiniLM-L6-v2 (fast, smaller)
    """

    def __init__(self):
        self.model = None
        self.model_name = None
        self._load_model()

    def _load_model(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model: {EMBEDDING_CONFIG.model_name}")
            self.model = SentenceTransformer(
                EMBEDDING_CONFIG.model_name,
                device=self._get_device(),
            )
            self.model_name = EMBEDDING_CONFIG.model_name
            logger.info(f"Embedding model loaded: {self.model_name} (dim={self.get_dimension()})")
        except Exception as e:
            logger.warning(f"Primary embedding model failed ({e}), trying fallback")
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(
                    EMBEDDING_CONFIG.fallback_model,
                    device=self._get_device(),
                )
                self.model_name = EMBEDDING_CONFIG.fallback_model
                logger.info(f"Using fallback model: {self.model_name}")
            except Exception as e2:
                logger.error(f"All embedding models failed: {e2}")
                self.model = None

    def _get_device(self) -> str:
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def encode(self, texts: Union[str, list[str]], batch_size: int = None) -> np.ndarray:
        """Encode text(s) into embedding vectors."""
        if self.model is None:
            return np.zeros((1 if isinstance(texts, str) else len(texts), 384))

        if isinstance(texts, str):
            texts = [texts]

        batch_size = batch_size or EMBEDDING_CONFIG.batch_size

        # BGE models need a query prefix for retrieval
        if "bge" in self.model_name.lower():
            prefixed = [f"Represent this financial text: {t}" for t in texts]
        else:
            prefixed = texts

        embeddings = self.model.encode(
            prefixed,
            batch_size=batch_size,
            normalize_embeddings=EMBEDDING_CONFIG.normalize_embeddings,
            show_progress_bar=len(texts) > 10,
        )
        return embeddings

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a search query (with query-specific prefix for BGE)."""
        if self.model is None:
            return np.zeros(384)

        if "bge" in (self.model_name or "").lower():
            query = f"Represent this financial question for retrieval: {query}"

        return self.model.encode(
            [query],
            normalize_embeddings=EMBEDDING_CONFIG.normalize_embeddings,
        )[0]

    def similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Cosine similarity between two embeddings."""
        if emb1.ndim == 1:
            emb1 = emb1.reshape(1, -1)
        if emb2.ndim == 1:
            emb2 = emb2.reshape(1, -1)
        # Normalized vectors: cosine sim = dot product
        return float(np.dot(emb1, emb2.T)[0, 0])

    def get_dimension(self) -> int:
        if self.model is None:
            return 384
        return self.model.get_sentence_embedding_dimension()

    def is_available(self) -> bool:
        return self.model is not None
