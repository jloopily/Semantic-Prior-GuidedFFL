"""Integration with the frozen official DINOv3 repository."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .config import Config


class DINOv3IntegrationUnavailable(RuntimeError):
    """Raised when the authorized official DINOv3 integration cannot run."""


@dataclass(frozen=True)
class DINOv3Spec:
    architecture: str = "dinov3_vitb16"
    block_one_based: int = 6
    api_n: tuple[int, ...] = (5,)
    norm: bool = True
    reshape: bool = False
    input_size: int = 448
    expected_tokens: int = 784
    expected_dimension: int = 768


def spec_from_config(config: Config) -> DINOv3Spec:
    return DINOv3Spec(
        architecture=str(config.require("backbone.architecture")),
        block_one_based=int(config.require("backbone.block_one_based")),
        api_n=tuple(int(v) for v in config.require("backbone.api_n")),
        norm=bool(config.require("backbone.norm")),
        reshape=bool(config.require("backbone.reshape")),
        input_size=int(config.require("input.size")),
        expected_tokens=int(config.require("features.expected_tokens_per_frame")),
        expected_dimension=int(config.require("features.expected_dimension")),
    )


def _torch():
    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on the runtime
        raise DINOv3IntegrationUnavailable("PyTorch is required for DINOv3") from exc
    return torch


def load_official_model(repo: str | Path, checkpoint: str | Path, spec: DINOv3Spec, device: str) -> Any:
    """Load the official DINOv3 ViT through the repository's hubconf API.

    The official repository defines ``dinov3_vitb16`` in ``hubconf.py`` and
    accepts a local checkpoint through the ``weights`` argument.  ``api_n`` is
    zero-based in ``get_intermediate_layers``; therefore config block 6 maps to
    ``api_n=[5]``.
    """

    torch = _torch()
    repo_path = Path(repo).expanduser().resolve()
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    if not repo_path.is_dir():
        raise FileNotFoundError(f"DINOv3 repository does not exist: {repo_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"DINOv3 checkpoint does not exist: {checkpoint_path}")
    if spec.architecture != "dinov3_vitb16":
        raise DINOv3IntegrationUnavailable(
            f"Only the frozen dinov3_vitb16 integration is supported, got {spec.architecture!r}"
        )
    expected_api_n = (spec.block_one_based - 1,)
    if spec.api_n != expected_api_n:
        raise DINOv3IntegrationUnavailable(
            f"Frozen block/API mismatch: block {spec.block_one_based} requires api_n={list(expected_api_n)}, "
            f"got {list(spec.api_n)}"
        )
    try:
        model = torch.hub.load(
            str(repo_path),
            spec.architecture,
            source="local",
            weights=str(checkpoint_path),
        )
    except Exception as exc:  # pragma: no cover - depends on repository/runtime
        raise DINOv3IntegrationUnavailable(f"Official DINOv3 model load failed: {exc}") from exc
    model = model.to(torch.device(device))
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def preprocess_rgb(images: Sequence[np.ndarray], config: Config) -> Any:
    """Convert RGB HWC images to an ImageNet-normalized 448x448 tensor batch."""

    torch = _torch()
    try:
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms import functional as TF
    except Exception as exc:  # pragma: no cover - depends on the runtime
        raise DINOv3IntegrationUnavailable("torchvision is required for DINOv3 preprocessing") from exc

    size = int(config.require("input.size"))
    if config.require("input.color") != "RGB":
        raise DINOv3IntegrationUnavailable("DINOv3 preprocessing only supports the frozen RGB protocol")
    mean = tuple(float(v) for v in config.require("input.imagenet_mean"))
    std = tuple(float(v) for v in config.require("input.imagenet_std"))
    if len(mean) != 3 or len(std) != 3:
        raise ValueError("ImageNet mean/std must contain three channels")

    tensors = []
    for image in images:
        array = np.asarray(image)
        if array.ndim != 3 or array.shape[2] != 3:
            raise ValueError(f"Expected RGB HWC image, got {array.shape}")
        if np.issubdtype(array.dtype, np.integer):
            array = array.astype(np.float32) / 255.0
        else:
            array = array.astype(np.float32, copy=False)
            if array.size and (float(array.min()) < 0.0 or float(array.max()) > 1.0):
                array = array / 255.0
        tensor = torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1)
        tensor = TF.resize(
            tensor,
            [size, size],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        tensor = TF.normalize(tensor, mean=mean, std=std)
        tensors.append(tensor)
    if not tensors:
        return torch.empty((0, 3, size, size), dtype=torch.float32)
    return torch.stack(tensors, dim=0).contiguous()


def extract_patch_tokens(model: Any, batch: Any, spec: DINOv3Spec) -> np.ndarray:
    """Return ``[B, 784, 768]`` float32 patch tokens from official block 6."""

    torch = _torch()
    if spec.reshape:
        raise DINOv3IntegrationUnavailable("Frozen protocol requires reshape=False for patch tokens")
    if spec.api_n != (spec.block_one_based - 1,):
        raise DINOv3IntegrationUnavailable("Frozen block/API mismatch")
    if not hasattr(model, "get_intermediate_layers"):
        raise DINOv3IntegrationUnavailable("Official model has no get_intermediate_layers API")

    if not isinstance(batch, torch.Tensor):
        raise TypeError(f"Expected a torch.Tensor batch, got {type(batch).__name__}")
    try:
        model_device = next(model.parameters()).device
    except StopIteration as exc:
        raise DINOv3IntegrationUnavailable(
            "DINOv3 model has no parameters; cannot determine its device"
        ) from exc
    if batch.device != model_device:
        batch = batch.to(device=model_device, non_blocking=True)

    with torch.inference_mode():
        outputs = model.get_intermediate_layers(
            batch,
            n=spec.api_n,
            reshape=False,
            return_class_token=False,
            return_extra_tokens=False,
            norm=spec.norm,
        )
    if not isinstance(outputs, (tuple, list)) or len(outputs) != 1:
        raise ValueError(f"Expected one intermediate-layer output, got {type(outputs).__name__} length {len(outputs)}")
    output = outputs[0]
    if not isinstance(output, torch.Tensor):
        raise ValueError(f"Expected a torch.Tensor intermediate output, got {type(output).__name__}")
    return validate_patch_tokens(output.detach().cpu().numpy(), spec)


def validate_patch_tokens(tokens: np.ndarray, spec: DINOv3Spec) -> np.ndarray:
    array = np.asarray(tokens)
    expected_tail = (spec.expected_tokens, spec.expected_dimension)
    if array.ndim != 3 or array.shape[1:] != expected_tail:
        raise ValueError(f"Expected patch tokens [B,{expected_tail[0]},{expected_tail[1]}], got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("Patch token tensor contains non-finite values")
    return np.ascontiguousarray(array, dtype=np.float32)
