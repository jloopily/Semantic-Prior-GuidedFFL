"""Dependency-light pixel metrics used by the R0 report."""
from __future__ import annotations

import numpy as np


def average_precision(scores: np.ndarray, mask: np.ndarray) -> float | None:
    """Average precision with tied scores handled as one threshold group.

    Sorting individual equal-score pixels and accumulating them one at a time
    makes AP depend on arbitrary pixel order.  This implementation evaluates
    precision only after each unique score threshold, which is invariant to
    permutations inside a tie group.
    """
    y = np.asarray(mask, dtype=np.uint8).reshape(-1)
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    if y.size != s.size:
        raise ValueError("scores and mask sizes differ")
    if not np.isfinite(s).all():
        raise ValueError("scores contain non-finite values")
    if np.any((y != 0) & (y != 1)):
        raise ValueError("mask must be binary")
    positives = int(y.sum())
    if positives == 0 or positives == y.size:
        return None

    order = np.argsort(-s, kind="mergesort")
    sorted_scores = s[order]
    sorted_labels = y[order].astype(np.int64)
    boundaries = np.r_[np.flatnonzero(np.diff(sorted_scores)) + 1, sorted_scores.size]
    cumulative_tp = np.cumsum(sorted_labels)
    previous_tp = 0
    ap = 0.0
    for end in boundaries:
        true_positives = int(cumulative_tp[end - 1])
        positives_in_group = true_positives - previous_tp
        if positives_in_group:
            ap += (true_positives / float(end)) * positives_in_group
        previous_tp = true_positives
    return float(ap / positives)


def prevalence(mask: np.ndarray) -> float:
    value = np.asarray(mask, dtype=bool)
    return float(value.mean()) if value.size else 0.0


def flatten_map(score: np.ndarray, size: int = 448) -> np.ndarray:
    value = np.asarray(score, dtype=np.float32)
    if value.shape == (size, size):
        return value
    if value.size != 28 * 28:
        raise ValueError("expected 28x28 token map")
    try:
        import torch
        return torch.nn.functional.interpolate(
            torch.from_numpy(value.reshape(1, 1, 28, 28)),
            size=(size, size), mode="bilinear", align_corners=False,
        )[0, 0].numpy()
    except ImportError:
        return np.repeat(np.repeat(value.reshape(28, 28), 16, axis=0), 16, axis=1)
