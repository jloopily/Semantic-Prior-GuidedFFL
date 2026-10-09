"""Deterministic 448-pixel semantic regions and 28x28 token masks.

The geometry is defined without looking at manipulation masks.  Landmarks are
expected in crop coordinates (normalised [0, 1] MediaPipe coordinates).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

# MediaPipe Face Mesh landmark topology.  These are the ordered outer-lip
# points; the path does not jump between inner and outer lips.
OUTER_LIPS = (61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185)
FACE_OVAL = (10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109)
LEFT_EYE = (33, 7, 163, 144, 145, 153, 154, 155, 133)
RIGHT_EYE = (263, 249, 390, 373, 374, 380, 381, 382, 362)

@dataclass(frozen=True)
class SemanticRegions:
    mouth_core: np.ndarray
    perioral_ring: np.ndarray
    cheek_left: np.ndarray
    cheek_right: np.ndarray
    face: np.ndarray
    valid: np.ndarray
    token_mouth_core: np.ndarray
    token_perioral_ring: np.ndarray
    token_cheek_left: np.ndarray
    token_cheek_right: np.ndarray
    token_face: np.ndarray
    token_valid: np.ndarray
    metadata: dict


def _points(landmarks: Sequence[object], indices: Iterable[int], size: int) -> np.ndarray:
    result = []
    for index in indices:
        point = landmarks[int(index)]
        if hasattr(point, "x") and hasattr(point, "y"):
            x, y = float(point.x), float(point.y)
        elif isinstance(point, (tuple, list, np.ndarray)) and len(point) >= 2:
            x, y = float(point[0]), float(point[1])
        else:
            raise ValueError(f"landmark {index} has no x/y coordinates")
        result.append((min(max(x * size, 0.0), size - 1.0), min(max(y * size, 0.0), size - 1.0)))
    return np.asarray(result, dtype=np.float32)


def polygon_is_simple(points: np.ndarray) -> bool:
    """Return True for a positive-area, non-self-intersecting polygon."""
    value = np.asarray(points, dtype=np.float64)
    if value.ndim != 2 or value.shape[0] < 3 or not np.isfinite(value).all():
        return False
    area = 0.5 * float(np.sum(value[:, 0] * np.roll(value[:, 1], -1) - value[:, 1] * np.roll(value[:, 0], -1)))
    if abs(area) <= 1e-6:
        return False
    def orient(a, b, c):
        return float((b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0]))
    def proper(a, b, c, d):
        o1, o2, o3, o4 = orient(a,b,c), orient(a,b,d), orient(c,d,a), orient(c,d,b)
        return ((o1 > 0) != (o2 > 0)) and ((o3 > 0) != (o4 > 0))
    n = value.shape[0]
    for i in range(n):
        a, b = value[i], value[(i+1) % n]
        for j in range(i + 1, n):
            if j in (i, (i+1) % n, (i-1) % n):
                continue
            c, d = value[j], value[(j+1) % n]
            if proper(a,b,c,d):
                return False
    return True


def polygon_mask(points: np.ndarray, size: int) -> np.ndarray:
    value = np.asarray(points, dtype=np.float32)
    if not polygon_is_simple(value):
        raise ValueError("semantic polygon is degenerate or self-intersecting")
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    inside = np.zeros((size, size), dtype=bool)
    j = value.shape[0] - 1
    for i in range(value.shape[0]):
        xi, yi = value[i]
        xj, yj = value[j]
        crossing = ((yi > yy) != (yj > yy)) & (xx < (xj-xi) * (yy-yi) / (yj-yi + 1e-12) + xi)
        inside ^= crossing
        j = i
    return inside


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.copy()
    try:
        import cv2
        kernel = np.ones((2*radius+1, 2*radius+1), np.uint8)
        return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    except ImportError:
        padded = np.pad(mask, radius, mode="constant")
        out = np.zeros_like(mask, dtype=bool)
        for dy in range(2*radius+1):
            for dx in range(2*radius+1):
                out |= padded[dy:dy+mask.shape[0], dx:dx+mask.shape[1]]
        return out


def _to_tokens(mask: np.ndarray, grid: int) -> np.ndarray:
    h, w = mask.shape
    if h % grid or w % grid:
        raise ValueError("image size must be divisible by token grid")
    ph, pw = h // grid, w // grid
    return mask.reshape(grid, ph, grid, pw).mean(axis=(1, 3)).reshape(-1).astype(np.float32)


def build_semantic_regions(landmarks: Sequence[object], *, size: int = 448, token_grid: int = 28, ring_radius: int = 12) -> SemanticRegions:
    """Build the frozen mouth/ring/cheek regions from target-real landmarks."""
    if len(landmarks) < 468:
        raise ValueError(f"expected 468 MediaPipe landmarks, got {len(landmarks)}")
    mouth_points = _points(landmarks, OUTER_LIPS, size)
    face_points = _points(landmarks, FACE_OVAL, size)
    left_eye = polygon_mask(_points(landmarks, LEFT_EYE, size), size)
    right_eye = polygon_mask(_points(landmarks, RIGHT_EYE, size), size)
    face = polygon_mask(face_points, size)
    mouth = polygon_mask(mouth_points, size)
    ring = _dilate(mouth, int(ring_radius)) & ~mouth & face
    x = mouth_points[:, 0]
    y = mouth_points[:, 1]
    cx, cy = float(np.mean(x)), float(np.mean(y))
    # Fixed deterministic cheek quadrilaterals, clipped to the face and with
    # eye/central facial regions excluded.  No mask information is used.
    minx, maxx = float(np.min(face_points[:,0])), float(np.max(face_points[:,0]))
    miny, maxy = float(np.min(face_points[:,1])), float(np.max(face_points[:,1]))
    nose_top = cy - 0.22 * (maxy-miny)
    nose_bottom = cy + 0.20 * (maxy-miny)
    left_poly = np.asarray([(minx+0.05*(maxx-minx), miny+0.36*(maxy-miny)), (cx-0.10*(maxx-minx), miny+0.36*(maxy-miny)), (cx-0.10*(maxx-minx), maxy-0.20*(maxy-miny)), (minx+0.05*(maxx-minx), maxy-0.20*(maxy-miny))], np.float32)
    right_poly = np.asarray([(cx+0.10*(maxx-minx), miny+0.36*(maxy-miny)), (maxx-0.05*(maxx-minx), miny+0.36*(maxy-miny)), (maxx-0.05*(maxx-minx), maxy-0.20*(maxy-miny)), (cx+0.10*(maxx-minx), maxy-0.20*(maxy-miny))], np.float32)
    cheeks = [polygon_mask(left_poly, size) & face, polygon_mask(right_poly, size) & face]
    forbidden = mouth | ring | left_eye | right_eye
    cheeks = [item & ~forbidden for item in cheeks]
    if any(np.count_nonzero(item) == 0 for item in cheeks):
        raise ValueError("one or both deterministic cheek regions are empty")
    valid = face
    return SemanticRegions(
        mouth_core=mouth, perioral_ring=ring, cheek_left=cheeks[0], cheek_right=cheeks[1], face=face, valid=valid,
        token_mouth_core=_to_tokens(mouth, token_grid), token_perioral_ring=_to_tokens(ring, token_grid),
        token_cheek_left=_to_tokens(cheeks[0], token_grid), token_cheek_right=_to_tokens(cheeks[1], token_grid),
        token_face=_to_tokens(face, token_grid), token_valid=_to_tokens(valid, token_grid),
        metadata={"size": size, "token_grid": token_grid, "ring_radius": ring_radius, "outer_lips": list(OUTER_LIPS), "face_oval": list(FACE_OVAL), "cheek_definition": "face-clipped fixed side quadrilaterals; eyes and mouth/ring removed"},
    )
