"""Embedding service — wraps sentence-transformers for fast semantic encoding.

Uses all-MiniLM-L6-v2: 384 dims, 14.2K sentences/sec, 22M params, CPU-only.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


class EmbeddingService:
    """Generates dense vector embeddings from text using sentence-transformers.

    Loads the model lazily on first use and caches it. The model is ~80MB,
    runs on CPU, and encodes ~14K sentences/sec.
    """

    def __init__(self, model_name: str = MODEL_NAME):
        self._model_name = model_name
        self._model = None
        self._dim = EMBEDDING_DIM

    def initialize(self) -> None:
        """Load the model. Call from Orchestrator.initialize()."""
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name)
            # Verify dimension
            test = self._model.encode("test", convert_to_numpy=True)
            self._dim = len(test)
            logger.info(f"Embedding model loaded: {self._model_name} (dim={self._dim})")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            self._model = None

    @property
    def is_available(self) -> bool:
        return self._model is not None

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_text(self, text: str) -> Optional[np.ndarray]:
        """Encode a single text string into a normalized float32 vector."""
        if self._model is None:
            return None
        try:
            embedding = self._model.encode(
                text, convert_to_numpy=True, normalize_embeddings=True,
            )
            return embedding.astype(np.float32)
        except Exception as e:
            logger.error(f"Embedding error: {e}")
            return None

    def embed_batch(self, texts: list[str]) -> Optional[np.ndarray]:
        """Encode multiple texts at once. Returns (N, dim) float32 array."""
        if self._model is None or not texts:
            return None
        try:
            embeddings = self._model.encode(
                texts, convert_to_numpy=True,
                normalize_embeddings=True, batch_size=64,
                show_progress_bar=len(texts) > 100,
            )
            return embeddings.astype(np.float32)
        except Exception as e:
            logger.error(f"Batch embedding error: {e}")
            return None
