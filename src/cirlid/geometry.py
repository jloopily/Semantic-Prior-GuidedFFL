"""Frame sampling, crop geometry, resizing, and patch-mask helpers."""

from __future__ import annotations

import math

import numpy as np

from .types import BBox


def fractional_frame_index(frame_count: int, fraction: float) -> int:
    """Frozen half-up frame rule: floor(p * (N - 1) + 0.5)."""

    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must lie in [0, 1]")
    return int(math.floor(fraction * (frame_count - 1) + 0.5))


def middle_frame_index(frame_count: int) -> int:
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    return (frame_count - 1) // 2


def remap_frame_index(index: int, source_count: int, target_count: int) -> int:
    """Map a zero-based temporal index by normalized video position."""

    if source_count <= 0 or target_count <= 0:
        raise ValueError("frame counts must be positive")
    if not 0 <= index < source_count:
        raise IndexError(f"frame index {index} outside [0, {source_count})")
    if source_count == 1 or target_count == 1:
        return 0
    return int(math.floor(index * (target_count - 1) / (source_count - 1) + 0.5))


def square_bbox(box: BBox | tuple[int, int, int, int], scale: float = 1.3) -> BBox:
    """Expand a detector box into the frozen square face crop."""

    bbox = box if isinstance(box, BBox) else BBox(*map(int, box))
    if bbox.width <= 0 or bbox.height <= 0:
        raise ValueError(f"Invalid detector box: {bbox}")
    if scale <= 0:
        raise ValueError("scale must be positive")
    center_x = (bbox.left + bbox.right) / 2.0
    center_y = (bbox.top + bbox.bottom) / 2.0
    side = scale * max(bbox.width, bbox.height)
    return BBox(
        left=math.floor(center_x - side / 2.0),
        top=math.floor(center_y - side / 2.0),
        right=math.ceil(center_x + side / 2.0),
        bottom=math.ceil(center_y + side / 2.0),
    )


def map_bbox(box: BBox, source_size: tuple[int, int], target_size: tuple[int, int]) -> BBox:
    """Map a bbox between frames with different `(width, height)` dimensions."""

    source_width, source_height = source_size
    target_width, target_height = target_size
    if min(source_width, source_height, target_width, target_height) <= 0:
        raise ValueError("source and target dimensions must be positive")
    scale_x = target_width / source_width
    scale_y = target_height / source_height
    return BBox(
        left=math.floor(box.left * scale_x),
        top=math.floor(box.top * scale_y),
        right=math.ceil(box.right * scale_x),
        bottom=math.ceil(box.bottom * scale_y),
    )


def _reflection_indices(coordinates: np.ndarray, length: int) -> np.ndarray:
    if length <= 0:
        raise ValueError("Cannot reflect indices for an empty axis")
    if length == 1:
        return np.zeros_like(coordinates, dtype=np.int64)
    period = 2 * length - 2
    folded = np.mod(coordinates, period)
    return np.where(folded < length, folded, period - folded).astype(np.int64)


def crop_with_reflection(array: np.ndarray, box: BBox) -> np.ndarray:
    """Crop HxW or HxWxC arrays using reflection outside image bounds."""

    image = np.asarray(array)
    if image.ndim not in (2, 3):
        raise ValueError("array must be HxW or HxWxC")
    if box.width <= 0 or box.height <= 0:
        raise ValueError("crop box must have positive area")
    ys = _reflection_indices(np.arange(box.top, box.bottom), image.shape[0])
    xs = _reflection_indices(np.arange(box.left, box.right), image.shape[1])
    return image[ys[:, None], xs[None, :]].copy()


def binarize_mask(mask: np.ndarray, *, policy: str = "nonzero", threshold: float = 0.5) -> np.ndarray:
    """Convert a resized mask to {0,1} without losing grayscale FF++ regions.

    ``nonzero`` is the policy recommended by the server audit for the local
    FF++ release: Face2Face, FaceSwap, and likely NeuralTextures masks contain
    non-black grayscale pixels inside the manipulated region. ``threshold``
    preserves the original v5.5 rule and interprets integer masks on a 0..255
    scale and floating masks on a 0..1 scale.
    """

    values = np.asarray(mask)
    if values.ndim != 2:
        raise ValueError("mask must be a two-dimensional array")
    if not np.isfinite(values).all():
        raise ValueError("mask contains non-finite values")
    if policy == "nonzero":
        return (values != 0).astype(np.uint8)
    if policy == "threshold":
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must lie in [0, 1]")
        scale = 255.0 if np.issubdtype(values.dtype, np.integer) else 1.0
        return (values >= threshold * scale).astype(np.uint8)
    raise ValueError("mask policy must be 'nonzero' or 'threshold'")


def resize_image(
    array: np.ndarray,
    size: int,
    *,
    is_mask: bool = False,
    mask_policy: str = "nonzero",
    mask_threshold: float = 0.5,
) -> np.ndarray:
    """Resize an image/mask with explicit interpolation and mask semantics."""

    if size <= 0:
        raise ValueError("size must be positive")
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("opencv-python-headless is required for crop resizing") from exc
    values = np.asarray(array)
    if is_mask:
        interpolation = cv2.INTER_NEAREST
    else:
        # Match the frozen bilinear+antialias intent: use area filtering only
        # when reducing a crop, and bilinear interpolation when upsampling.
        interpolation = cv2.INTER_AREA if values.shape[0] > size or values.shape[1] > size else cv2.INTER_LINEAR
    resized = cv2.resize(values, (size, size), interpolation=interpolation)
    if is_mask:
        return binarize_mask(resized, policy=mask_policy, threshold=mask_threshold)
    return resized


def crop_and_resize(
    array: np.ndarray,
    box: BBox,
    size: int,
    *,
    is_mask: bool = False,
    mask_policy: str = "nonzero",
    mask_threshold: float = 0.5,
) -> np.ndarray:
    return resize_image(
        crop_with_reflection(array, box),
        size,
        is_mask=is_mask,
        mask_policy=mask_policy,
        mask_threshold=mask_threshold,
    )


def patch_coverage(mask: np.ndarray, patch_size: int = 16) -> np.ndarray:
    """Average-pool a binary square mask into non-overlapping patch coverage."""

    binary = np.asarray(mask, dtype=np.float32)
    if binary.ndim != 2:
        raise ValueError("mask must be a two-dimensional array")
    height, width = binary.shape
    if height % patch_size or width % patch_size:
        raise ValueError("mask dimensions must be divisible by patch_size")
    return binary.reshape(height // patch_size, patch_size, width // patch_size, patch_size).mean(axis=(1, 3))


def token_labels(mask: np.ndarray, coverage_threshold: float = 0.25, patch_size: int = 16) -> np.ndarray:
    if not 0.0 <= coverage_threshold <= 1.0:
        raise ValueError("coverage_threshold must lie in [0, 1]")
    return (patch_coverage(mask, patch_size) >= coverage_threshold).reshape(-1)
