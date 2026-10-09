"""Deterministic synthetic manifold × nuisance experiment for Gate B.

The synthetic experiment is deliberately independent of FF++ and DINOv3, but it
still follows the frozen protocol: one shared gallery permutation per seed,
canonical query-dependent local scales, paired clean/anomaly queries, and
machine-readable nuisance-sensitivity summaries.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from .knn import ExactL2Index
from .scoring import score_from_knn


@dataclass(frozen=True)
class SyntheticConfig:
    ambient_dimension: int = 64
    latent_dimension: int = 4
    clean_queries: int = 2000
    gallery_sizes: tuple[int, ...] = (2000, 5000, 10000)
    densities: tuple[str, ...] = ("sparse", "uniform", "concentrated")
    radial_scales: tuple[float, ...] = (0.5, 1.0, 2.0)
    deltas: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0)
    seeds: tuple[int, ...] = tuple(range(20))
    k: int = 20


METHOD_C2 = "arithmetic_multi_neighbor"


def _beta_density(u: np.ndarray, name: str) -> np.ndarray:
    """Apply the preregistered Beta-PPF density transforms."""

    if name == "uniform":
        return u
    try:
        from scipy.special import betaincinv
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("scipy is required for the synthetic Beta-PPF densities") from exc
    if name == "sparse":
        return betaincinv(0.5, 0.5, u)
    if name == "concentrated":
        return betaincinv(2.0, 2.0, u)
    raise ValueError(f"Unknown density: {name}")


def _psi(z: np.ndarray) -> np.ndarray:
    """The frozen 18-dimensional explicit manifold chart."""

    if z.ndim != 2 or z.shape[1] != 4:
        raise ValueError("psi expects latent coordinates with shape [N, 4]")
    z1, z2, z3, z4 = z.T
    return np.column_stack(
        [
            z1,
            z2,
            z3,
            z4,
            0.5 * np.sin(np.pi * z1 / 2.0),
            0.5 * np.cos(np.pi * z1 / 2.0),
            0.5 * np.sin(np.pi * z2 / 2.0),
            0.5 * np.cos(np.pi * z2 / 2.0),
            0.5 * np.sin(np.pi * z3 / 2.0),
            0.5 * np.cos(np.pi * z3 / 2.0),
            0.5 * np.sin(np.pi * z4 / 2.0),
            0.5 * np.cos(np.pi * z4 / 2.0),
            0.25 * z1 * z2,
            0.25 * z1 * z3,
            0.25 * z1 * z4,
            0.25 * z2 * z3,
            0.25 * z2 * z4,
            0.25 * z3 * z4,
        ]
    )


def _psi_jacobian(point: np.ndarray) -> np.ndarray:
    """Return the 18×4 Jacobian of :func:`_psi` at one latent point."""

    z = np.asarray(point, dtype=np.float64)
    if z.shape != (4,):
        raise ValueError("point must have shape [4]")
    z1, z2, z3, z4 = z
    jacobian = np.zeros((18, 4), dtype=np.float64)
    jacobian[:4] = np.eye(4)
    frequencies = (z1, z2, z3, z4)
    for offset, value in enumerate(frequencies):
        row = 4 + 2 * offset
        jacobian[row, offset] = 0.5 * (np.pi / 2.0) * np.cos(np.pi * value / 2.0)
        jacobian[row + 1, offset] = -0.5 * (np.pi / 2.0) * np.sin(np.pi * value / 2.0)
    jacobian[12, :] = (0.25 * z2, 0.25 * z1, 0.0, 0.0)
    jacobian[13, :] = (0.25 * z3, 0.0, 0.25 * z1, 0.0)
    jacobian[14, :] = (0.25 * z4, 0.0, 0.0, 0.25 * z1)
    jacobian[15, :] = (0.0, 0.25 * z3, 0.25 * z2, 0.0)
    jacobian[16, :] = (0.0, 0.25 * z4, 0.0, 0.25 * z2)
    jacobian[17, :] = (0.0, 0.0, 0.25 * z4, 0.25 * z3)
    return jacobian


def _orthogonal_matrix(seed: int, ambient_dimension: int) -> np.ndarray:
    if ambient_dimension < 18:
        raise ValueError("ambient_dimension must be at least 18")
    rng = np.random.default_rng(seed + 1_000_003)
    matrix = rng.normal(size=(ambient_dimension, ambient_dimension))
    q, _ = np.linalg.qr(matrix)
    return q.astype(np.float64, copy=False)


def _chart(z: np.ndarray, ambient_dimension: int, rotation: np.ndarray) -> np.ndarray:
    """Evaluate phi(z)=R[psi(z), 0, ...] row-wise."""

    coordinates = np.asarray(z, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 4:
        raise ValueError("latent coordinates must have shape [N, 4]")
    if rotation.shape != (ambient_dimension, ambient_dimension):
        raise ValueError("rotation has the wrong shape")
    padded = np.zeros((coordinates.shape[0], ambient_dimension), dtype=np.float64)
    padded[:, :18] = _psi(coordinates)
    return padded @ rotation.T


def _jacobian(point: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    padded = np.zeros((rotation.shape[0], 4), dtype=np.float64)
    padded[:18] = _psi_jacobian(point)
    return rotation @ padded


def _normal_direction(point: np.ndarray, rotation: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    tangent, _ = np.linalg.qr(_jacobian(point, rotation), mode="complete")
    normal = tangent[:, 4:] @ rng.normal(size=rotation.shape[0] - 4)
    norm = np.linalg.norm(normal)
    return normal / max(norm, 1e-12)


def _gallery_latents(base_u: np.ndarray, density: str) -> np.ndarray:
    return 4.0 * _beta_density(base_u, density) - 2.0


def _search(gallery: np.ndarray, queries: np.ndarray, k: int) -> np.ndarray:
    """Run the frozen exact FAISS search and return rooted L2 distances."""

    if gallery.shape[0] < k:
        raise ValueError(f"Gallery has {gallery.shape[0]} points but k={k}")
    index = ExactL2Index.build(gallery)
    return index.search(queries, k, batch_size=4096).distances


def _binary_auroc(clean_scores: np.ndarray, anomaly_scores: np.ndarray) -> float:
    """Dependency-free AUROC with average ranks for ties."""

    clean = np.asarray(clean_scores, dtype=np.float64).reshape(-1)
    anomaly = np.asarray(anomaly_scores, dtype=np.float64).reshape(-1)
    if clean.size == 0 or anomaly.size == 0:
        raise ValueError("AUROC requires non-empty clean and anomaly scores")
    values = np.concatenate([clean, anomaly])
    if not np.isfinite(values).all():
        raise ValueError("AUROC scores must be finite")
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    positive_ranks = ranks[clean.size :]
    n_positive = float(anomaly.size)
    n_negative = float(clean.size)
    return float((positive_ranks.sum() - n_positive * (n_positive + 1.0) / 2.0) / (n_positive * n_negative))


def _method_scores(distances: np.ndarray) -> dict[str, np.ndarray]:
    """Return scores with the frozen convention: larger means more suspicious."""

    return score_from_knn(distances)


def run_synthetic(config: SyntheticConfig = SyntheticConfig()) -> list[dict[str, float | int | str | bool]]:
    """Run all synthetic conditions and return one row per seed/condition/method."""

    if config.latent_dimension != 4:
        raise ValueError("The frozen synthetic chart has latent_dimension=4")
    if config.ambient_dimension < 18:
        raise ValueError("ambient_dimension must be at least 18")
    if not config.gallery_sizes or min(config.gallery_sizes) < config.k:
        raise ValueError("gallery_sizes must all be at least k")
    if 5000 not in config.gallery_sizes or 1.0 not in config.radial_scales or "uniform" not in config.densities:
        raise ValueError("synthetic config must include canonical gallery=5000, uniform density, scale=1.0")
    if not config.seeds:
        raise ValueError("seeds must not be empty")

    rows: list[dict[str, float | int | str | bool]] = []
    maximum_gallery_size = max(config.gallery_sizes)
    canonical_size = 5000
    canonical_density = "uniform"
    canonical_scale = 1.0

    for seed in config.seeds:
        rng = np.random.default_rng(seed)
        rotation = _orthogonal_matrix(seed, config.ambient_dimension)
        query_u = rng.uniform(-0.5, 0.5, size=(config.clean_queries, 4))
        clean = _chart(query_u, config.ambient_dimension, rotation)
        directions = np.stack([_normal_direction(point, rotation, rng) for point in query_u])
        base_gallery_u = rng.uniform(1e-8, 1.0 - 1e-8, size=(maximum_gallery_size, 4))

        canonical_gallery = _chart(
            _gallery_latents(base_gallery_u[:canonical_size], canonical_density),
            config.ambient_dimension,
            rotation,
        )
        canonical_clean_distances = _search(canonical_gallery, clean, config.k)
        local_scale = np.median(canonical_clean_distances, axis=1)

        for density in config.densities:
            full_gallery = _chart(_gallery_latents(base_gallery_u, density), config.ambient_dimension, rotation)
            for radial_scale in config.radial_scales:
                for gallery_size in config.gallery_sizes:
                    gallery = radial_scale * full_gallery[:gallery_size]
                    index = ExactL2Index.build(gallery)
                    for delta in config.deltas:
                        clean_query = radial_scale * clean
                        anomaly_query = radial_scale * (clean + float(delta) * local_scale[:, None] * directions)
                        distances = index.search(
                            np.concatenate([clean_query, anomaly_query], axis=0),
                            config.k,
                            batch_size=4096,
                        ).distances
                        clean_distances = distances[: config.clean_queries]
                        anomaly_distances = distances[config.clean_queries :]
                        clean_scores = _method_scores(clean_distances)
                        anomaly_scores = _method_scores(anomaly_distances)
                        is_canonical = (
                            density == canonical_density
                            and gallery_size == canonical_size
                            and radial_scale == canonical_scale
                        )
                        for method in clean_scores:
                            clean_values = clean_scores[method]
                            anomaly_values = anomaly_scores[method]
                            rows.append(
                                {
                                    "seed": int(seed),
                                    "density": density,
                                    "radial_scale": float(radial_scale),
                                    "gallery_size": int(gallery_size),
                                    "delta": float(delta),
                                    "method": method,
                                    "canonical": bool(is_canonical),
                                    "clean_median": float(np.median(clean_values)),
                                    "anomaly_median": float(np.median(anomaly_values)),
                                    "median_score": float(np.median(anomaly_values)),
                                    "mean_score": float(np.mean(anomaly_values)),
                                    "anomaly_auroc": _binary_auroc(clean_values, anomaly_values),
                                    "canonical_local_scale_median": float(np.median(local_scale)),
                                }
                            )
    return rows


def summarize_synthetic(
    rows: Iterable[dict[str, Any]],
    *,
    reduction_min: float | None = None,
) -> dict[str, Any]:
    """Summarize Q/AUROC; an optional threshold is supplied by the command layer."""

    materialized = list(rows)
    if not materialized:
        raise ValueError("Synthetic rows must not be empty")
    grouped: dict[tuple[str, int, float], list[dict[str, Any]]] = defaultdict(list)
    for row in materialized:
        grouped[(str(row["method"]), int(row["seed"]), float(row["delta"]))].append(row)

    q_by_method_delta: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    auroc_by_method_delta: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    separation_by_method_delta: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for (method, seed, delta), condition_rows in grouped.items():
        anomaly_values = np.asarray([float(row["anomaly_median"]) for row in condition_rows])
        canonical_rows = [row for row in condition_rows if bool(row["canonical"])]
        if len(canonical_rows) != 1:
            raise ValueError(f"Expected exactly one canonical row for {method}, seed={seed}, delta={delta}")
        canonical = canonical_rows[0]
        separation = abs(float(canonical["anomaly_median"]) - float(canonical["clean_median"]))
        q_value = float(np.std(anomaly_values, ddof=0) / (separation + 1e-12))
        q_by_method_delta[method].setdefault(str(delta), []).append(q_value)
        auroc_by_method_delta[method].setdefault(str(delta), []).append(
            float(np.median([float(row["anomaly_auroc"]) for row in condition_rows]))
        )
        separation_by_method_delta[method].setdefault(str(delta), []).append(separation)

    q_summary = {
        method: {delta: float(np.median(values)) for delta, values in delta_values.items()}
        for method, delta_values in q_by_method_delta.items()
    }
    auroc_summary = {
        method: {delta: float(np.median(values)) for delta, values in delta_values.items()}
        for method, delta_values in auroc_by_method_delta.items()
    }
    separation_summary = {
        method: {delta: float(np.median(values)) for delta, values in delta_values.items()}
        for method, delta_values in separation_by_method_delta.items()
    }

    deltas = [delta for delta in q_summary.get(METHOD_C2, {}) if float(delta) > 0.0]
    if not deltas:
        raise ValueError("Synthetic results need at least one positive delta")
    cir_q = float(np.median([q_summary["CIR_LID"][delta] for delta in deltas]))
    c2_q = float(np.median([q_summary[METHOD_C2][delta] for delta in deltas]))
    relative_reduction = float((c2_q - cir_q) / max(abs(c2_q), 1e-12))
    summary = {
        "q_method": q_summary,
        "canonical_separation": separation_summary,
        "anomaly_auroc": auroc_summary,
        "positive_deltas": [float(delta) for delta in deltas],
        "q_cir": cir_q,
        "q_c2": c2_q,
        "relative_q_reduction_cir_vs_c2": relative_reduction,
    }
    if reduction_min is not None:
        if not np.isfinite(float(reduction_min)):
            raise ValueError("reduction_min must be finite")
        summary["synthetic_gate_b_passed"] = bool(relative_reduction >= float(reduction_min))
        summary["synthetic_gate_b_reduction_min"] = float(reduction_min)
    return summary
