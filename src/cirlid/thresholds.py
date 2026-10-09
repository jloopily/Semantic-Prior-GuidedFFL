"""Method-specific real-only threshold artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np

from .io import write_json
from .metrics import calibrate_threshold, false_positive_rate


def calibrate_method_thresholds(
    score_maps: Mapping[str, list[np.ndarray]],
    *,
    quantile: float = 0.99,
    numpy_method: str = "linear",
) -> dict[str, float]:
    if not score_maps:
        raise ValueError("No methods were supplied")
    return {
        method: calibrate_threshold(values, quantile=quantile, method=numpy_method)
        for method, values in score_maps.items()
    }


def audit_calibration(score_maps: Mapping[str, list[np.ndarray]], thresholds: Mapping[str, float]) -> dict[str, float]:
    missing = set(score_maps) - set(thresholds)
    if missing:
        raise ValueError(f"Missing thresholds for methods: {sorted(missing)}")
    return {
        method: false_positive_rate(np.concatenate([np.asarray(values).reshape(-1) for values in maps]), thresholds[method])
        for method, maps in score_maps.items()
    }


def save_thresholds(path: str | Path, thresholds: Mapping[str, float], *, metadata: Mapping[str, object] | None = None) -> None:
    payload = {"thresholds": {str(k): float(v) for k, v in thresholds.items()}, "metadata": dict(metadata or {})}
    write_json(path, payload)


def load_thresholds(path: str | Path) -> dict[str, float]:
    import json

    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    thresholds = payload.get("thresholds", payload)
    if not isinstance(thresholds, dict):
        raise ValueError("Threshold artifact must contain a mapping")
    return {str(key): float(value) for key, value in thresholds.items()}
