"""Vector index — hnswlib wrapper for fast approximate nearest-neighbor search.

Uses inner product space (equivalent to cosine similarity for normalized vectors).
Persists to disk and supports incremental adds.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_EF_CONSTRUCTION = 200
DEFAULT_M = 16
DEFAULT_EF_SEARCH = 50


class VectorIndex:
    """HNSW-based vector index for fast similarity search.

    Stores vectors with integer IDs that map to database row IDs.
    Supports incremental add, persist to disk, and reload from disk.
    """

    def __init__(
        self,
        dimension: int = 384,
        index_path: Optional[str] = None,
        max_elements: int = 200_000,
        ef_construction: int = DEFAULT_EF_CONSTRUCTION,
        M: int = DEFAULT_M,
    ):
        self._dim = dimension
        self._index_path = index_path or "data/vector_index.bin"
        self._max_elements = max_elements
        self._ef_construction = ef_construction
        self._M = M
        self._index = None
        self._count = 0

    def initialize(self) -> None:
        """Load from disk if exists, or create a fresh index."""
        try:
            import hnswlib
        except ImportError:
            logger.error("hnswlib not installed. Run: pip install hnswlib")
            return

        self._index = hnswlib.Index(space="ip", dim=self._dim)

        if Path(self._index_path).exists():
            try:
                self._index.load_index(self._index_path, max_elements=self._max_elements)
                self._count = self._index.get_current_count()
                self._index.set_ef(DEFAULT_EF_SEARCH)
                logger.info(f"Vector index loaded: {self._index_path} ({self._count} vectors)")
                return
            except Exception as e:
                logger.warning(f"Failed to load vector index: {e}. Creating fresh.")

        self._index.init_index(
            max_elements=self._max_elements,
            ef_construction=self._ef_construction,
            M=self._M,
        )
        self._index.set_ef(DEFAULT_EF_SEARCH)
        logger.info(f"Fresh vector index created (dim={self._dim}, max={self._max_elements})")

    def add(self, vector: np.ndarray, item_id: int) -> None:
        """Add a single vector with its database row ID."""
        if self._index is None:
            return
        if self._count >= self._max_elements:
            self._max_elements *= 2
            self._index.resize_index(self._max_elements)
            logger.info(f"Vector index resized to {self._max_elements}")
        self._index.add_items(vector.reshape(1, -1), np.array([item_id]))
        self._count += 1

    def add_batch(self, vectors: np.ndarray, item_ids: np.ndarray) -> None:
        """Add multiple vectors at once."""
        if self._index is None or len(vectors) == 0:
            return
        needed = self._count + len(vectors)
        if needed > self._max_elements:
            self._max_elements = max(needed * 2, self._max_elements * 2)
            self._index.resize_index(self._max_elements)
        self._index.add_items(vectors, item_ids)
        self._count += len(vectors)

    def search(self, query_vector: np.ndarray, top_k: int = 10) -> list[tuple[int, float]]:
        """Search for nearest neighbors. Returns list of (item_id, score) tuples.

        Score is cosine similarity (higher = more similar) for normalized vectors.
        """
        if self._index is None or self._count == 0:
            return []
        k = min(top_k, self._count)
        labels, distances = self._index.knn_query(query_vector.reshape(1, -1), k=k)
        results = []
        for label, dist in zip(labels[0], distances[0]):
            # For inner product space with normalized vectors:
            # hnswlib returns 1 - inner_product as distance
            score = 1.0 - dist
            results.append((int(label), float(score)))
        return results

    def remove(self, item_id: int) -> None:
        """Mark an item as deleted."""
        if self._index is None:
            return
        try:
            self._index.mark_deleted(item_id)
            self._count = max(0, self._count - 1)
        except Exception:
            pass

    def save(self) -> None:
        """Persist index to disk."""
        if self._index is None:
            return
        Path(self._index_path).parent.mkdir(parents=True, exist_ok=True)
        self._index.save_index(self._index_path)
        logger.info(f"Vector index saved ({self._count} vectors)")

    @property
    def count(self) -> int:
        return self._count

    @property
    def is_available(self) -> bool:
        return self._index is not None
