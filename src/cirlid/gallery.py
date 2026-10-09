"""Feature/gallery artifacts and exact FAISS index construction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .config import Config, config_digest
from .io import write_json
from .knn import ExactL2Index, as_float32_matrix


@dataclass(frozen=True)
class GalleryArtifact:
    vectors: np.ndarray
    index: ExactL2Index
    metadata: dict[str, object]


def flatten_patch_features(features: np.ndarray, *, tokens_per_frame: int = 784, dimension: int = 768) -> np.ndarray:
    value = np.asarray(features)
    if value.ndim == 2:
        if value.shape[1] != dimension:
            raise ValueError(f"Expected feature dimension {dimension}, got {value.shape[1]}")
        return as_float32_matrix(value, name="features")
    if value.ndim != 3 or value.shape[1:] != (tokens_per_frame, dimension):
        raise ValueError(f"Expected [frames,{tokens_per_frame},{dimension}], got {value.shape}")
    return as_float32_matrix(value.reshape(-1, dimension), name="features")


def build_gallery(
    features: np.ndarray,
    output_dir: str | Path,
    *,
    config: Config | None = None,
    source_ids: Sequence[str] | None = None,
) -> GalleryArtifact:
    tokens = int(config.get("features.expected_tokens_per_frame", 784)) if config is not None else 784
    dimension = int(config.get("features.expected_dimension", 768)) if config is not None else 768
    vectors = flatten_patch_features(features, tokens_per_frame=tokens, dimension=dimension)
    if config is not None:
        expected_tokens = int(config.get("gallery.expected_tokens", vectors.shape[0]))
        if vectors.shape[0] != expected_tokens:
            raise ValueError(f"Expected {expected_tokens} gallery tokens, got {vectors.shape[0]}")
    index = ExactL2Index.build(vectors)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    np.save(destination / "gallery.npy", vectors)
    index.write(destination / "gallery.index")
    metadata: dict[str, object] = {
        "shape": list(vectors.shape),
        "dtype": str(vectors.dtype),
        "ntotal": int(vectors.shape[0]),
        "dimension": int(vectors.shape[1]),
        "index": "faiss.IndexFlatL2",
        "source_ids": list(source_ids) if source_ids is not None else None,
    }
    if config is not None:
        metadata["config_digest"] = config_digest(config)
    write_json(destination / "gallery.metadata.json", metadata)
    return GalleryArtifact(vectors=vectors, index=index, metadata=metadata)


def load_gallery(directory: str | Path) -> GalleryArtifact:
    source = Path(directory)
    vectors = as_float32_matrix(np.load(source / "gallery.npy", allow_pickle=False), name="gallery")
    index = ExactL2Index.read(source / "gallery.index")
    if index.ntotal != vectors.shape[0] or index.dimension != vectors.shape[1]:
        raise ValueError("Gallery array and FAISS index metadata disagree")
    metadata_path = source / "gallery.metadata.json"
    import json

    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    return GalleryArtifact(vectors=vectors, index=index, metadata=metadata)
