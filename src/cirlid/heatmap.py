"""Score-map upsampling with the frozen bilinear definition."""

from __future__ import annotations

import numpy as np


def upsample_score_map(scores: np.ndarray, *, grid: tuple[int, int] = (28, 28), output_size: tuple[int, int] = (448, 448)) -> np.ndarray:
    vector = np.asarray(scores, dtype=np.float32)
    if vector.shape == grid:
        matrix = vector
    elif vector.size == grid[0] * grid[1]:
        matrix = vector.reshape(grid)
    else:
        raise ValueError(f"Scores cannot be reshaped to grid {grid}: {vector.shape}")
    try:
        import torch
        import torch.nn.functional as functional
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyTorch is required for align_corners=False bilinear upsampling") from exc
    tensor = torch.from_numpy(matrix).reshape(1, 1, *grid)
    upsampled = functional.interpolate(tensor, size=output_size, mode="bilinear", align_corners=False)
    return upsampled[0, 0].cpu().numpy().astype(np.float32, copy=False)
