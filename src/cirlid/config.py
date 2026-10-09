"""Configuration loading and immutable-protocol helpers."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, MutableMapping

try:
    import yaml
except ImportError:  # pragma: no cover - dependency is documented, error is clearer at use
    yaml = None


class ConfigError(ValueError):
    """Raised when the experiment configuration is missing or malformed."""


@dataclass(frozen=True)
class Config:
    """Thin immutable wrapper around the YAML mapping."""

    data: Mapping[str, Any]
    source: Path | None = None

    def get(self, path: str, default: Any = None) -> Any:
        value: Any = self.data
        for part in path.split("."):
            if not isinstance(value, Mapping) or part not in value:
                return default
            value = value[part]
        return value

    def require(self, path: str) -> Any:
        value = self.get(path, None)
        if value is None:
            raise ConfigError(f"Missing required configuration key: {path}")
        return value

    @property
    def root(self) -> Path:
        return self.source.parent if self.source is not None else Path.cwd()


def _plain_mapping(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _plain_mapping(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_mapping(v) for v in value]
    return value


def load_config(path: str | Path) -> Config:
    """Load a YAML config and fail early on non-mapping roots."""

    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file does not exist: {config_path}")
    if yaml is None:
        raise ConfigError("PyYAML is required to load experiment configuration")
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, Mapping):
        raise ConfigError(f"Configuration root must be a mapping: {config_path}")
    return Config(_plain_mapping(payload), config_path)


def config_digest(config: Config | Mapping[str, Any]) -> str:
    """Return a stable SHA-256 digest of the semantic configuration."""

    payload = config.data if isinstance(config, Config) else config
    encoded = json.dumps(_plain_mapping(payload), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def resolve_config_path(config: Config, value: str | Path) -> Path:
    """Resolve a path relative to the config file, unless it is absolute."""

    path = Path(value).expanduser()
    return path if path.is_absolute() else (config.root / path).resolve()


def require_protocol(config: Config) -> None:
    """Validate the high-risk frozen settings before any expensive computation."""

    checks = {
        "input.size": 448,
        "input.color": "RGB",
        "input.test_time_augmentation": False,
        "face_crop.detector": "dlib_hog",
        "face_crop.detector_source": "original_target_c23_only",
        "face_crop.upsample_primary": 1,
        "face_crop.upsample_retry": 2,
        "face_crop.square_scale": 1.3,
        "face_crop.padding": "reflection",
        "face_crop.image_resize": "bilinear_antialias",
        "face_crop.mask_resize": "nearest",
        "face_crop.mask_binarization": "nonzero",
        "backbone.implementation": "official_facebookresearch_dinov3",
        "backbone.architecture": "dinov3_vitb16",
        "backbone.block_one_based": 6,
        "backbone.api_n": [5],
        "backbone.norm": True,
        "backbone.reshape": False,
        "backbone.extra_l2_normalization": False,
        "backbone.feature_dtype_on_disk": "float32",
        "features.expected_patch_grid": [28, 28],
        "features.expected_tokens_per_frame": 784,
        "features.expected_dimension": 768,
        "gallery.split": "train",
        "gallery.source": "original_sequences/youtube/c23",
        "gallery.videos": 64,
        "gallery.frame_per_video": "middle",
        "gallery.tokens_per_frame": 784,
        "gallery.expected_tokens": 50176,
        "knn.engine": "faiss.IndexFlatL2",
        "knn.exact_search": True,
        "knn.k": 20,
        "knn.faiss_output_is_squared_l2": True,
        "knn.take_square_root_before_scoring": True,
        "knn.distance_dtype": "float32",
        "threshold.source": "validation_real_c23_pooled_pixels",
        "threshold.quantile": 0.99,
        "threshold.numpy_method": "linear",
        "threshold.method_specific": True,
        "statistics.unit": "video",
        "statistics.bootstrap": "manipulation_stratified_video_cluster",
        "statistics.bootstrap_replicates": 2000,
    }
    mismatches: list[str] = []
    for key, expected in checks.items():
        actual = config.get(key)
        if actual != expected:
            mismatches.append(f"{key}={actual!r} (expected {expected!r})")
    if mismatches:
        raise ConfigError("Frozen protocol mismatch:\n" + "\n".join(mismatches))
