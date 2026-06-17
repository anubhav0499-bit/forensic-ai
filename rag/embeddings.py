"""
Embedding Model - BAAI/bge-large-en-v1.5 for financial document embeddings.
Falls back to bge-small-en-v1.5 (fast, ~130 MB) then all-MiniLM-L6-v2.

Speed/quality trade-off:
  EmbeddingModel()               → bge-large-en-v1.5 (1024-dim, best quality)
  EmbeddingModel(fast=True)      → bge-small-en-v1.5  (384-dim, ~4× faster, ~130 MB)
  EMBEDDING_CONFIG.model_name=…  → override via .env
"""

from __future__ import annotations
import numpy as np
from typing import Union
from loguru import logger
from config import EMBEDDING_CONFIG

_BGE_SMALL = "BAAI/bge-small-en-v1.5"


class EmbeddingModel:
    """
    Sentence embedding model for financial document similarity search.

    Quality tier  (default):  BAAI/bge-large-en-v1.5  — 1024-dim, best for financial text
    Speed tier    (fast=True): BAAI/bge-small-en-v1.5  — 384-dim, ~4× faster, ~130 MB
    Final fallback:            all-MiniLM-L6-v2         — 384-dim, universal fallback
    """

    def __init__(self, fast: bool = False):
        self.model = None
        self.model_name = None
        self._fast = fast
        self._load_model()

    def _load_model(self) -> None:
        from sentence_transformers import SentenceTransformer
        device = self._get_device()

        # Ordered candidate list
        candidates = (
            [_BGE_SMALL, EMBEDDING_CONFIG.fallback_model]
            if self._fast
            else [EMBEDDING_CONFIG.model_name, _BGE_SMALL, EMBEDDING_CONFIG.fallback_model]
        )

        for model_name in candidates:
            try:
                logger.info(f"Loading embedding model: {model_name}")
                self.model = SentenceTransformer(model_name, device=device)
                self.model_name = model_name
                logger.info(
                    f"Embedding model ready: {self.model_name} "
                    f"(dim={self.get_dimension()}, device={device})"
                )
                return
            except Exception as exc:
                logger.warning(f"  {model_name} failed: {exc} — trying next")

        logger.error("All embedding models failed; using zero-vector fallback")
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
