"""Shared typed records used by the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np


@dataclass(frozen=True)
class BBox:
    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top

    def as_tuple(self) -> tuple[int, int, int, int]:
        return self.left, self.top, self.right, self.bottom


@dataclass(frozen=True)
class ManifestRecord:
    sample_id: str
    split: str
    compression: str
    manipulation: str
    is_real: bool
    target_id: str
    source_id: str | None
    video_name: str
    video_path: str
    mask_path: str | None
    frame_count: int
    frame_fraction: float
    frame_index: int
    mask_frame_index: int | None
    crop_source_video: str
    crop_source_frame: int
    crop_left: int | None = None
    crop_top: int | None = None
    crop_right: int | None = None
    crop_bottom: int | None = None
    crop_scale: float | None = None
    exclusion_reason: str | None = None
    mask_frame_remapped: bool = False
    mask_frame_count: int | None = None
    seed: int = 20260919
    # Filled by the crop-cache stage.  These are optional because the first
    # manifest is deliberately written before any image data is decoded.
    crop_path: str | None = None
    mask_cache_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class KNNResult:
    distances: np.ndarray
    indices: np.ndarray
    squared_distances: bool = False

    def __post_init__(self) -> None:
        if self.distances.ndim != 2 or self.indices.ndim != 2:
            raise ValueError("KNN distances and indices must be rank-2 arrays")
        if self.distances.shape != self.indices.shape:
            raise ValueError("KNN distances and indices must have identical shapes")


@dataclass(frozen=True)
class VideoPrediction:
    video_id: str
    manipulation: str
    scores: np.ndarray
    mask: np.ndarray
    is_real: bool = False

    def __post_init__(self) -> None:
        if self.scores.ndim != 1 or self.mask.ndim != 1:
            raise ValueError("Video scores and masks must be flattened vectors")
        if self.scores.shape != self.mask.shape:
            raise ValueError("Video scores and masks must have equal length")
