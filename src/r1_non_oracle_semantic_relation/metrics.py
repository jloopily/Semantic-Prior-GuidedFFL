"""Deterministic pixel projection, pooled metrics, and independent-crop mapping."""
from __future__ import annotations

import numpy as np

from src.r0_semantic_relation.metrics import average_precision, prevalence


def pixel_map(tokens: np.ndarray, size: int = 448) -> np.ndarray:
    """Bilinear projection, align_corners=False; no environment-dependent fallback."""
    grid = np.asarray(tokens, dtype=np.float32)
    if grid.size != 784 or not np.isfinite(grid).all():
        raise ValueError('expected finite 28x28 token scores')
    grid = grid.reshape(28, 28)
    position = np.maximum((np.arange(size, dtype=np.float32) + 0.5) * (28.0 / size) - 0.5, 0)
    low = np.floor(position).astype(np.int64)
    high = np.minimum(low + 1, 27)
    weight = position - low
    horizontal = grid[:, low] * (1 - weight)[None, :] + grid[:, high] * weight[None, :]
    return (horizontal[low, :] * (1 - weight)[:, None]
            + horizontal[high, :] * weight[:, None]).astype(np.float32)


def pixel_metrics(scores: np.ndarray, truth: np.ndarray, threshold: float | None) -> dict:
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    truth = np.asarray(truth, dtype=np.uint8).reshape(-1)
    ap = average_precision(scores, truth)
    prev = prevalence(truth)
    output = {'ap': ap, 'prevalence': prev, 'ap_minus_prevalence': ap - prev if ap is not None else None,
              'pixels': int(truth.size), 'positive_pixels': int(truth.sum()),
              'f1': None, 'fpr': None, 'tp': None, 'fp': None, 'fn': None, 'tn': None}
    if threshold is not None:
        prediction = scores > threshold
        positive = truth.astype(bool)
        tp = int(np.count_nonzero(prediction & positive))
        fp = int(np.count_nonzero(prediction & ~positive))
        fn = int(np.count_nonzero(~prediction & positive))
        tn = int(np.count_nonzero(~prediction & ~positive))
        output.update(tp=tp, fp=fp, fn=fn, tn=tn,
                      f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
                      fpr=fp / (fp + tn) if fp + tn else None)
    return output


def warp_real_to_fake(values: np.ndarray, real_valid: np.ndarray, fake_row: dict,
                      real_row: dict) -> tuple[np.ndarray, np.ndarray]:
    """Map real's independent crop through source coordinates to fake's crop.

    Only a secondary evaluation operation AFTER both maps were committed.
    Never used in reference scoring or to change fake geometry.
    """
    import cv2
    fl, ft, fr, fb = fake_row['crop_bbox']
    rl, rt, rr, rb = real_row['crop_bbox']
    size = 448
    x = fl + (np.arange(size, dtype=np.float32) + 0.5) * (fr - fl) / size
    y = ft + (np.arange(size, dtype=np.float32) + 0.5) * (fb - ft) / size
    rx = x * real_row['width'] / fake_row['width']
    ry = y * real_row['height'] / fake_row['height']
    gx, gy = np.meshgrid((rx - rl) * size / (rr - rl) - 0.5,
                         (ry - rt) * size / (rb - rt) - 0.5)
    gx, gy = gx.astype(np.float32), gy.astype(np.float32)
    sampled = cv2.remap(np.asarray(values, np.float32), gx, gy, cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    valid = cv2.remap(np.asarray(real_valid, np.uint8), gx, gy, cv2.INTER_NEAREST,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    in_crop = (gx >= 0) & (gx <= size - 1) & (gy >= 0) & (gy <= size - 1)
    in_source = ((x >= 0) & (x < fake_row['width']))[None, :] & ((y >= 0) & (y < fake_row['height']))[:, None]
    return sampled, valid & in_crop & in_source
