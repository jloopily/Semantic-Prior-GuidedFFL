"""Feature tensor persistence with shape and provenance metadata.

The feature stage writes one small, independently verifiable ``.npy`` artifact
per manifest row. Keeping samples separate makes interrupted GPU jobs
restartable without re-running completed forward passes.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .config import Config, config_digest
from .io import sha256_file, write_json


def validate_patch_features(
    features: np.ndarray,
    *,
    tokens: int = 784,
    dimension: int = 768,
) -> np.ndarray:
    """Validate and canonicalize a batch of patch features."""

    value = np.asarray(features)
    if value.ndim != 3 or value.shape[1:] != (tokens, dimension):
        raise ValueError(f"Expected feature tensor [B,{tokens},{dimension}], got {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError("Feature tensor contains non-finite values")
    return np.ascontiguousarray(value, dtype=np.float32)


def feature_artifact_key(sample_id: str) -> str:
    """Return a stable collision-resistant filename stem for a sample ID."""

    return hashlib.sha256(str(sample_id).encode("utf-8")).hexdigest()


def feature_artifact_path(root: str | Path, sample_id: str) -> Path:
    """Return the deterministic feature path for ``sample_id``."""

    return Path(root) / f"{feature_artifact_key(sample_id)}.npy"


def _atomic_save_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)
    temporary.replace(path)


def save_features(
    path: str | Path,
    features: np.ndarray,
    *,
    metadata: Mapping[str, Any] | None = None,
    config: Config | None = None,
) -> dict[str, Any]:
    """Atomically save validated features and a JSON sidecar."""

    tokens = int(config.get("features.expected_tokens_per_frame", 784)) if config is not None else 784
    dimension = int(config.get("features.expected_dimension", 768)) if config is not None else 768
    value = validate_patch_features(features, tokens=tokens, dimension=dimension)
    destination = Path(path)
    _atomic_save_npy(destination, value)
    meta: dict[str, Any] = {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "path": str(destination),
        "array_sha256": sha256_file(destination),
    }
    if metadata:
        meta.update(dict(metadata))
    if config is not None:
        meta["config_digest"] = config_digest(config)
    write_json(destination.with_suffix(destination.suffix + ".metadata.json"), meta)
    return meta


def load_features(path: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Load and validate one feature artifact and its optional sidecar."""

    source = Path(path)
    value = validate_patch_features(np.load(source, allow_pickle=False))
    metadata_path = source.with_suffix(source.suffix + ".metadata.json")
    import json

    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    if metadata.get("shape") not in (None, list(value.shape)):
        raise ValueError("Feature metadata shape disagrees with array")
    if metadata.get("dtype") not in (None, str(value.dtype)):
        raise ValueError("Feature metadata dtype disagrees with array")
    recorded_hash = metadata.get("array_sha256")
    if recorded_hash is not None and recorded_hash != sha256_file(source):
        raise ValueError("Feature metadata hash disagrees with array")
    return value, metadata


def reusable_feature(
    path: str | Path,
    *,
    sample_id: str,
    config_digest_value: str,
    tokens: int = 784,
    dimension: int = 768,
    crop_path: str | Path | None = None,
    required_metadata: Mapping[str, Any] | None = None,
) -> tuple[bool, dict[str, Any] | None]:
    """Check whether an existing feature artifact is safe to reuse."""

    source = Path(path)
    metadata_path = source.with_suffix(source.suffix + ".metadata.json")
    if not source.is_file() or not metadata_path.is_file():
        return False, None
    try:
        value, metadata = load_features(source)
        expected_shape = [1, int(tokens), int(dimension)]
        if list(value.shape) != expected_shape:
            return False, metadata
        if metadata.get("sample_id") != str(sample_id):
            return False, metadata
        if metadata.get("config_digest") != str(config_digest_value):
            return False, metadata
        if crop_path is not None:
            crop = str(Path(crop_path))
            if metadata.get("crop_path") != crop:
                return False, metadata
            current_crop_hash = sha256_file(crop_path)
            if metadata.get("crop_sha256") != current_crop_hash:
                return False, metadata
        if required_metadata:
            for key, expected in required_metadata.items():
                if metadata.get(str(key)) != expected:
                    return False, metadata
    except (OSError, ValueError, TypeError, KeyError):
        return False, None
    return True, metadata
