"""CIR-LID, fair baselines, SP-LID, and local-neighborhood scores."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .knn import self_knn


@dataclass(frozen=True)
class ScoreDiagnostics:
    distance_floor_fraction: float
    cir_denominator_floor_fraction: float
    non_finite_count: int
    minimum: float
    median: float
    maximum: float


def validate_distances(distances: np.ndarray, *, k: int | None = None) -> np.ndarray:
    value = np.asarray(distances, dtype=np.float32)
    if value.ndim != 2:
        raise ValueError("distances must be a rank-2 matrix")
    if k is not None and value.shape[1] != k:
        raise ValueError(f"Expected {k} neighbors, got {value.shape[1]}")
    if value.shape[1] < 2:
        raise ValueError("At least two neighbors are required")
    if not np.isfinite(value).all():
        raise ValueError("Distances contain non-finite values")
    if np.any(value < 0):
        raise ValueError("Distances must be non-negative")
    tolerance = 1e-6 * np.maximum(1.0, np.max(value, axis=1, keepdims=True))
    if np.any(value[:, 1:] + tolerance < value[:, :-1]):
        raise ValueError("Neighbor distances are not sorted in non-decreasing order")
    return value


def cir_lid(distances: np.ndarray, *, distance_floor: float = 1e-12, denominator_floor: float = 1e-6) -> np.ndarray:
    r = np.maximum(validate_distances(distances), distance_floor)
    rk = r[:, -1:]
    denominator = np.mean(np.log(rk / r[:, :-1]), axis=1)
    return (1.0 / np.maximum(denominator, denominator_floor)).astype(np.float32)


def normalized_geometric_profile(distances: np.ndarray, *, distance_floor: float = 1e-12) -> np.ndarray:
    r = np.maximum(validate_distances(distances), distance_floor)
    return np.exp(np.mean(np.log(r[:, :-1] / r[:, -1:]), axis=1)).astype(np.float32)


def arithmetic_multi_neighbor(distances: np.ndarray, *, distance_floor: float = 1e-12) -> np.ndarray:
    r = np.maximum(validate_distances(distances), distance_floor)
    return np.mean(r[:, :-1] / r[:, -1:], axis=1).astype(np.float32)


def first_neighbor(distances: np.ndarray) -> np.ndarray:
    return validate_distances(distances)[:, 0].astype(np.float32, copy=True)


def mean_knn(distances: np.ndarray) -> np.ndarray:
    return np.mean(validate_distances(distances), axis=1).astype(np.float32)


def r1_over_rk(distances: np.ndarray, *, distance_floor: float = 1e-12) -> np.ndarray:
    r = np.maximum(validate_distances(distances), distance_floor)
    return (r[:, 0] / r[:, -1]).astype(np.float32)


def score_from_knn(
    distances: np.ndarray,
    *,
    distance_floor: float = 1e-12,
    cir_denominator_floor: float = 1e-6,
) -> dict[str, np.ndarray]:
    r = validate_distances(distances)
    return {
        "CIR_LID": cir_lid(r, distance_floor=distance_floor, denominator_floor=cir_denominator_floor),
        "arithmetic_multi_neighbor": arithmetic_multi_neighbor(r, distance_floor=distance_floor),
        "r1": first_neighbor(r),
        "mean_knn": mean_knn(r),
        "r1_over_rk": r1_over_rk(r, distance_floor=distance_floor),
    }


def sp_lid(features: np.ndarray, k: int = 20, *, distance_floor: float = 1e-12, denominator_floor: float = 1e-6) -> np.ndarray:
    return cir_lid(self_knn(features, k).distances, distance_floor=distance_floor, denominator_floor=denominator_floor)


def local_pca_participation_ratio(gallery: np.ndarray, neighbor_indices: np.ndarray) -> np.ndarray:
    base = np.asarray(gallery, dtype=np.float32)
    neighbors = np.asarray(neighbor_indices, dtype=np.int64)
    if base.ndim != 2 or neighbors.ndim != 2:
        raise ValueError("gallery and neighbor_indices must be matrices")
    if np.any((neighbors < 0) | (neighbors >= base.shape[0])):
        raise IndexError("neighbor index outside gallery")
    scores = np.empty(neighbors.shape[0], dtype=np.float32)
    for row, ids in enumerate(neighbors):
        local = base[ids].astype(np.float64)
        centered = local - local.mean(axis=0, keepdims=True)
        singular_values = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
        eigenvalues = np.maximum((singular_values * singular_values) / max(local.shape[0] - 1, 1), 0.0)
        scores[row] = (eigenvalues.sum() ** 2) / (np.square(eigenvalues).sum() + 1e-12)
    return scores


def lof_scores(gallery: np.ndarray, queries: np.ndarray, *, n_neighbors: int = 20) -> np.ndarray:
    try:
        from sklearn.neighbors import LocalOutlierFactor
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("scikit-learn is required for the LOF baseline") from exc
    estimator = LocalOutlierFactor(n_neighbors=n_neighbors, metric="euclidean", algorithm="brute", novelty=True)
    estimator.fit(np.asarray(gallery, dtype=np.float32))
    return (-estimator.score_samples(np.asarray(queries, dtype=np.float32))).astype(np.float32)


def diagnostics(
    distances: np.ndarray,
    scores: np.ndarray,
    *,
    distance_floor: float = 1e-12,
    cir_denominator_floor: float = 1e-6,
) -> ScoreDiagnostics:
    r = validate_distances(distances)
    clipped_distance = r <= distance_floor
    safe = np.maximum(r, distance_floor)
    denominator = np.mean(np.log(safe[:, -1:] / safe[:, :-1]), axis=1)
    values = np.asarray(scores, dtype=np.float64)
    finite = np.isfinite(values)
    finite_values = values[finite]
    if finite_values.size == 0:
        extrema = (float("nan"),) * 3
    else:
        extrema = (float(finite_values.min()), float(np.median(finite_values)), float(finite_values.max()))
    return ScoreDiagnostics(
        distance_floor_fraction=float(clipped_distance.mean()),
        cir_denominator_floor_fraction=float(np.mean(denominator <= cir_denominator_floor)),
        non_finite_count=int((~finite).sum()),
        minimum=extrema[0],
        median=extrema[1],
        maximum=extrema[2],
    )
