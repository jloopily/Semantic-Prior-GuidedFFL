"""Manipulation-stratified paired video-cluster bootstrap."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class BootstrapSummary:
    point: float
    lower: float
    upper: float
    replicates: np.ndarray
    seed: int


def percentile_interval(values: np.ndarray, confidence_level: float = 0.95) -> tuple[float, float]:
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie in (0, 1)")
    alpha = (1.0 - confidence_level) / 2.0
    return tuple(float(v) for v in np.quantile(values, [alpha, 1.0 - alpha], method="linear"))


def stratified_cluster_bootstrap(
    values: Mapping[str, Mapping[str, float]],
    *,
    replicates: int = 2000,
    seed: int = 20260919,
    confidence_level: float = 0.95,
) -> BootstrapSummary:
    """Bootstrap macro means from `{manipulation: {video_id: value}}`."""

    if replicates <= 0:
        raise ValueError("replicates must be positive")
    groups = {str(group): dict(items) for group, items in values.items()}
    if not groups or any(not items for items in groups.values()):
        raise ValueError("Each manipulation must contain at least one video")
    point = float(np.mean([np.mean(list(items.values())) for items in groups.values()]))
    rng = np.random.default_rng(seed)
    samples = np.empty(replicates, dtype=np.float64)
    arrays = {group: np.asarray(list(items.values()), dtype=np.float64) for group, items in groups.items()}
    for replicate in range(replicates):
        manipulation_means = []
        for array in arrays.values():
            draw = rng.integers(0, array.size, size=array.size)
            manipulation_means.append(float(np.mean(array[draw])))
        samples[replicate] = np.mean(manipulation_means)
    lower, upper = percentile_interval(samples, confidence_level)
    return BootstrapSummary(point, lower, upper, samples, seed)


def paired_stratified_bootstrap(
    first: Mapping[str, Mapping[str, float]],
    second: Mapping[str, Mapping[str, float]],
    *,
    replicates: int = 2000,
    seed: int = 20260919,
    confidence_level: float = 0.95,
) -> BootstrapSummary:
    """Paired bootstrap of `first - second` using identical video draws."""

    if set(first) != set(second):
        raise ValueError("Methods have different manipulation groups")
    differences: dict[str, dict[str, float]] = {}
    for group in first:
        if set(first[group]) != set(second[group]):
            raise ValueError(f"Methods have different video IDs in manipulation {group}")
        differences[group] = {video: float(first[group][video]) - float(second[group][video]) for video in first[group]}
    return stratified_cluster_bootstrap(
        differences,
        replicates=replicates,
        seed=seed,
        confidence_level=confidence_level,
    )


def grouped_metric(rows: Sequence[Mapping[str, object]], metric: str) -> dict[str, dict[str, float]]:
    grouped: dict[str, dict[str, float]] = defaultdict(dict)
    for row in rows:
        group = str(row["manipulation"])
        video_id = str(row["video_id"])
        if video_id in grouped[group]:
            raise ValueError(f"Duplicate video {video_id!r} in manipulation {group!r}")
        grouped[group][video_id] = float(row[metric])
    return dict(grouped)
