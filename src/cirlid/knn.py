"""Exact kNN search with explicit FAISS squared-L2 handling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .types import KNNResult


class KNNError(RuntimeError):
    pass


def as_float32_matrix(array: np.ndarray, *, name: str) -> np.ndarray:
    value = np.asarray(array)
    if value.ndim != 2:
        raise ValueError(f"{name} must be rank 2, got shape {value.shape}")
    if value.shape[0] == 0 or value.shape[1] == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} contains non-finite values")
    return np.ascontiguousarray(value, dtype=np.float32)


@dataclass
class ExactL2Index:
    """A checked wrapper around `faiss.IndexFlatL2`."""

    dimension: int
    _index: object | None = None

    @classmethod
    def build(cls, gallery: np.ndarray) -> "ExactL2Index":
        vectors = as_float32_matrix(gallery, name="gallery")
        try:
            import faiss
        except ImportError as exc:  # pragma: no cover
            raise KNNError("faiss-cpu is required for the frozen exact search") from exc
        index = faiss.IndexFlatL2(vectors.shape[1])
        index.add(vectors)
        return cls(dimension=vectors.shape[1], _index=index)

    @property
    def ntotal(self) -> int:
        return int(getattr(self._index, "ntotal", 0))

    def search(self, queries: np.ndarray, k: int, *, batch_size: int | None = None) -> KNNResult:
        query = as_float32_matrix(queries, name="queries")
        if query.shape[1] != self.dimension:
            raise ValueError(f"Query dimension {query.shape[1]} != gallery dimension {self.dimension}")
        if not 1 <= k <= self.ntotal:
            raise ValueError(f"k must lie in [1, {self.ntotal}], got {k}")
        if self._index is None:
            raise KNNError("Index is not initialized")
        batch = query.shape[0] if batch_size is None else int(batch_size)
        if batch <= 0:
            raise ValueError("batch_size must be positive")
        distances: list[np.ndarray] = []
        indices: list[np.ndarray] = []
        for start in range(0, query.shape[0], batch):
            squared_l2, neighbors = self._index.search(query[start : start + batch], k)
            distances.append(np.asarray(squared_l2, dtype=np.float32))
            indices.append(np.asarray(neighbors, dtype=np.int64))
        d2 = np.concatenate(distances, axis=0)
        nn = np.concatenate(indices, axis=0)
        if np.any(d2 < -1e-6):
            raise KNNError("FAISS returned materially negative squared distances")
        d2 = np.maximum(d2, 0.0)
        return KNNResult(np.sqrt(d2, dtype=np.float32), nn, squared_distances=False)

    def write(self, path: str | Path) -> None:
        if self._index is None:
            raise KNNError("Index is not initialized")
        try:
            import faiss
        except ImportError as exc:  # pragma: no cover
            raise KNNError("faiss-cpu is required to save an index") from exc
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(destination))

    @classmethod
    def read(cls, path: str | Path) -> "ExactL2Index":
        try:
            import faiss
        except ImportError as exc:  # pragma: no cover
            raise KNNError("faiss-cpu is required to load an index") from exc
        index = faiss.read_index(str(Path(path)))
        if not isinstance(index, faiss.IndexFlatL2):
            raise KNNError(f"Frozen protocol requires IndexFlatL2, got {type(index).__name__}")
        return cls(int(index.d), index)


def numpy_exact_l2(gallery: np.ndarray, queries: np.ndarray, k: int, *, batch_size: int = 256) -> KNNResult:
    """Reference exact search used by the synthetic experiment and audits."""

    base = as_float32_matrix(gallery, name="gallery")
    query = as_float32_matrix(queries, name="queries")
    if base.shape[1] != query.shape[1]:
        raise ValueError("Gallery and query dimensions differ")
    if not 1 <= k <= base.shape[0]:
        raise ValueError("k is outside the gallery size")
    all_distances: list[np.ndarray] = []
    all_indices: list[np.ndarray] = []
    base_norm = np.sum(base * base, axis=1)
    for start in range(0, query.shape[0], batch_size):
        block = query[start : start + batch_size]
        d2 = np.sum(block * block, axis=1, keepdims=True) + base_norm[None, :] - 2.0 * block @ base.T
        d2 = np.maximum(d2, 0.0)
        candidates = np.argpartition(d2, kth=k - 1, axis=1)[:, :k]
        candidate_d2 = np.take_along_axis(d2, candidates, axis=1)
        order = np.argsort(candidate_d2, axis=1, kind="stable")
        nn = np.take_along_axis(candidates, order, axis=1)
        sorted_d2 = np.take_along_axis(candidate_d2, order, axis=1)
        all_distances.append(np.sqrt(sorted_d2, dtype=np.float32))
        all_indices.append(nn.astype(np.int64, copy=False))
    return KNNResult(np.concatenate(all_distances), np.concatenate(all_indices), squared_distances=False)


def self_knn(features: np.ndarray, k: int) -> KNNResult:
    """Exact within-frame neighbors with robust own-index exclusion."""

    vectors = as_float32_matrix(features, name="features")
    if not 1 <= k < vectors.shape[0]:
        raise ValueError("self kNN requires 1 <= k < number of tokens")
    raw = numpy_exact_l2(vectors, vectors, min(vectors.shape[0], k + 8))
    distances = np.empty((vectors.shape[0], k), dtype=np.float32)
    indices = np.empty((vectors.shape[0], k), dtype=np.int64)
    for row in range(vectors.shape[0]):
        keep = raw.indices[row] != row
        kept_indices = raw.indices[row][keep]
        kept_distances = raw.distances[row][keep]
        if kept_indices.size < k:
            full_order = np.argsort(np.linalg.norm(vectors - vectors[row], axis=1), kind="stable")
            full_order = full_order[full_order != row][:k]
            kept_indices = full_order
            kept_distances = np.linalg.norm(vectors[full_order] - vectors[row], axis=1)
        indices[row] = kept_indices[:k]
        distances[row] = kept_distances[:k]
    return KNNResult(distances, indices, squared_distances=False)
