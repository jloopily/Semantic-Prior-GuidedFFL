"""Frozen threshold calibration and pixel/video evaluation metrics."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

from .types import VideoPrediction


@dataclass(frozen=True)
class PixelMetrics:
    pixel_ap: float
    pixel_auroc: float
    iou: float
    f1: float
    positive_pixels: int
    total_pixels: int


def calibrate_threshold(scores: Iterable[np.ndarray], quantile: float = 0.99, *, method: str = "linear") -> float:
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must lie strictly between 0 and 1")
    arrays = [np.asarray(score, dtype=np.float64).reshape(-1) for score in scores]
    if not arrays:
        raise ValueError("No scores were supplied for threshold calibration")
    pooled = np.concatenate(arrays)
    if pooled.size == 0 or not np.isfinite(pooled).all():
        raise ValueError("Calibration scores are empty or non-finite")
    return float(np.quantile(pooled, quantile, method=method))


def false_positive_rate(scores: np.ndarray, threshold: float) -> float:
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("scores must be non-empty and finite")
    if not np.isfinite(float(threshold)):
        raise ValueError("threshold must be finite")
    return float(np.mean(values > threshold))


def _binary_overlap(mask: np.ndarray, prediction: np.ndarray) -> tuple[float, float]:
    truth = np.asarray(mask, dtype=bool)
    pred = np.asarray(prediction, dtype=bool)
    tp = int(np.sum(truth & pred))
    fp = int(np.sum(~truth & pred))
    fn = int(np.sum(truth & ~pred))
    union = tp + fp + fn
    iou = tp / union if union else 1.0
    denominator = 2 * tp + fp + fn
    f1 = 2 * tp / denominator if denominator else 1.0
    return float(iou), float(f1)


def pixel_metrics(scores: np.ndarray, mask: np.ndarray, *, threshold: float) -> PixelMetrics:
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(mask, dtype=np.uint8).reshape(-1)
    if values.shape != labels.shape:
        raise ValueError("scores and mask must have equal flattened shapes")
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("scores are empty or non-finite")
    unique = np.unique(labels)
    if not np.all(np.isin(unique, [0, 1])):
        raise ValueError("mask must be binary")
    if unique.size < 2:
        raise ValueError("Pixel AP/AUROC require both positive and negative pixels")
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("scikit-learn is required for pixel metrics") from exc
    prediction = values > threshold
    iou, f1 = _binary_overlap(labels.astype(bool), prediction)
    return PixelMetrics(
        pixel_ap=float(average_precision_score(labels, values)),
        pixel_auroc=float(roc_auc_score(labels, values)),
        iou=iou,
        f1=f1,
        positive_pixels=int(labels.sum()),
        total_pixels=int(labels.size),
    )


def evaluate_videos(predictions: Sequence[VideoPrediction], *, threshold: float) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for item in predictions:
        if item.is_real:
            rows.append(
                {
                    "video_id": item.video_id,
                    "manipulation": item.manipulation,
                    "is_real": True,
                    "fpr": false_positive_rate(item.scores, threshold),
                    "total_pixels": int(item.scores.size),
                }
            )
        else:
            metric = pixel_metrics(item.scores, item.mask, threshold=threshold)
            rows.append(
                {
                    "video_id": item.video_id,
                    "manipulation": item.manipulation,
                    "is_real": False,
                    **asdict(metric),
                }
            )
    return rows


def macro_by_manipulation(rows: Sequence[Mapping[str, object]], metric: str = "pixel_ap") -> dict[str, object]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if bool(row.get("is_real", False)):
            continue
        value = row.get(metric)
        if value is None:
            continue
        grouped[str(row["manipulation"])].append(float(value))
    if not grouped:
        raise ValueError(f"No fake-video rows contain metric {metric!r}")
    per_manipulation = {name: float(np.mean(values)) for name, values in sorted(grouped.items())}
    return {
        "metric": metric,
        "per_manipulation": per_manipulation,
        "macro": float(np.mean(list(per_manipulation.values()))),
    }


def jaccard_top_fraction(first: np.ndarray, second: np.ndarray, fraction: float = 0.05) -> float:
    a = np.asarray(first).reshape(-1)
    b = np.asarray(second).reshape(-1)
    if a.shape != b.shape or a.size == 0:
        raise ValueError("score vectors must have equal non-empty shapes")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("score vectors must be finite")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must lie in (0, 1]")
    count = max(1, int(np.ceil(fraction * a.size)))
    top_a = set(np.argpartition(a, -count)[-count:].tolist())
    top_b = set(np.argpartition(b, -count)[-count:].tolist())
    return len(top_a & top_b) / len(top_a | top_b)


def decision_flip_rate(first: np.ndarray, second: np.ndarray, threshold: float) -> float:
    a = np.asarray(first).reshape(-1)
    b = np.asarray(second).reshape(-1)
    if a.shape != b.shape or a.size == 0:
        raise ValueError("score vectors must have equal shapes")
    if not np.isfinite(a).all() or not np.isfinite(b).all() or not np.isfinite(float(threshold)):
        raise ValueError("scores and threshold must be finite")
    return float(np.mean((a > threshold) != (b > threshold)))
