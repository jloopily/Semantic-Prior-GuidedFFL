"""Command implementations used by the documented pipeline entry points.

The expensive external integrations are deliberately isolated behind the
DINOv3 and FaceForensics++ modules. Everything else in the numerical core can
be used with cached arrays/artifacts once those integrations are connected.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from .config import Config, config_digest, resolve_config_path
from .dinov3 import (
    DINOv3IntegrationUnavailable,
    extract_patch_tokens,
    load_official_model,
    preprocess_rgb,
    spec_from_config,
)
from .features import feature_artifact_path, load_features, reusable_feature, save_features
from .ffpp import (
    FFPPIntegrationUnavailable,
    FFPPPaths,
    audit_ffpp,
    directed_fake_names,
    enumerate_gallery_records,
    enumerate_records,
    split_original_ids,
)
from .io import read_json, sha256_file, write_json, write_jsonl
from .manifest import read_manifest, select_gallery_ids, split_validation_directed_names, write_manifest


def _mask_remapping_summary(rows: list[Any]) -> dict[str, Any]:
    """Return deterministic manifest-level statistics for mask-frame remapping.

    The summary is intentionally based on the records that are actually written
    to disk.  It therefore documents the temporal relationship between each
    sampled manipulated-video frame and its mask frame without probing masks a
    second time or depending on a particular FF++ split size.
    """

    from collections import defaultdict

    def field(row: Any, name: str) -> Any:
        if isinstance(row, dict):
            return row.get(name)
        return getattr(row, name)

    total_fake = 0
    remapped_fake = 0
    by_pair: dict[str, int] = defaultdict(int)
    by_group: dict[tuple[str, str, str], dict[str, int]] = defaultdict(
        lambda: {"fake_rows": 0, "remapped_rows": 0}
    )
    for row in rows:
        if bool(field(row, "is_real")):
            continue
        total_fake += 1
        video_count = int(field(row, "frame_count"))
        mask_count = field(row, "mask_frame_count")
        if mask_count is not None:
            pair = f"{video_count}->{int(mask_count)}"
            by_pair[pair] += 1
        remapped = bool(field(row, "mask_frame_remapped"))
        if remapped:
            remapped_fake += 1
        group = (
            str(field(row, "split")),
            str(field(row, "compression")),
            str(field(row, "manipulation")),
        )
        by_group[group]["fake_rows"] += 1
        by_group[group]["remapped_rows"] += int(remapped)

    def fraction(numerator: int, denominator: int) -> float:
        return float(numerator / denominator) if denominator else 0.0

    groups = []
    for (split, compression, manipulation), counts in sorted(by_group.items()):
        groups.append(
            {
                "split": split,
                "compression": compression,
                "manipulation": manipulation,
                "fake_rows": counts["fake_rows"],
                "remapped_rows": counts["remapped_rows"],
                "remapped_fraction": fraction(counts["remapped_rows"], counts["fake_rows"]),
            }
        )
    return {
        "total_fake_rows": total_fake,
        "remapped_fake_rows": remapped_fake,
        "remapped_fraction": fraction(remapped_fake, total_fake),
        "frame_count_pairs": dict(sorted(by_pair.items())),
        "by_split_compression_manipulation": groups,
    }


def audit_environment(config: Config) -> dict[str, Any]:
    import importlib.util
    import platform
    import subprocess
    import sys

    repo = resolve_config_path(config, config.require("paths.dinov3_repo"))
    checkpoint = resolve_config_path(config, config.require("paths.dinov3_weights"))
    modules = {
        name: importlib.util.find_spec(name) is not None
        for name in ("numpy", "torch", "torchvision", "cv2", "faiss", "dlib", "yaml")
    }

    def _git_revision(path: Path) -> str | None:
        if not path.is_dir():
            return None
        try:
            completed = subprocess.run(
                ["git", "-C", str(path), "rev-parse", "HEAD"],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        revision = completed.stdout.strip()
        return revision if completed.returncode == 0 and revision else None

    cuda_available = False
    cuda_device_count = 0
    cuda_devices: list[str] = []
    if modules["torch"]:
        try:
            import torch

            cuda_available = bool(torch.cuda.is_available())
            cuda_device_count = int(torch.cuda.device_count()) if cuda_available else 0
            cuda_devices = [str(torch.cuda.get_device_name(i)) for i in range(cuda_device_count)]
        except Exception as exc:  # pragma: no cover - depends on the server runtime
            cuda_error = f"{type(exc).__name__}: {exc}"
        else:
            cuda_error = None
    else:
        cuda_error = "torch is not installed"

    feature_smoke_ready = all(modules[name] for name in ("numpy", "torch", "torchvision"))
    crop_ready = all(modules[name] for name in ("numpy", "cv2", "dlib"))
    feature_extraction_ready = feature_smoke_ready and cuda_available
    blockers: list[str] = []
    if not repo.is_dir():
        blockers.append(f"missing DINOv3 repository: {repo}")
    if not checkpoint.is_file():
        blockers.append(f"missing DINOv3 checkpoint: {checkpoint}")
    if not crop_ready:
        blockers.append("crop stage dependencies are incomplete (numpy/cv2/dlib)")
    if not feature_smoke_ready:
        blockers.append("DINOv3 smoke-test dependencies are incomplete (numpy/torch/torchvision)")
    if feature_smoke_ready and not cuda_available:
        blockers.append("CUDA GPU is unavailable; feature extraction must wait for a GPU instance")

    if feature_extraction_ready and repo.is_dir() and checkpoint.is_file():
        status = "ready_for_gpu_pipeline"
    elif feature_smoke_ready and repo.is_dir() and checkpoint.is_file():
        status = "cpu_only_feature_extraction_blocked"
    elif repo.is_dir() and checkpoint.is_file():
        status = "dependencies_missing"
    else:
        status = "external_files_missing"

    report = {
        "python": sys.version,
        "platform": platform.platform(),
        "config": str(config.source) if config.source else None,
        "config_digest": config_digest(config),
        "status": status,
        "ffpp": "filesystem_and_crop_integration",
        "dinov3": {
            "repo_exists": repo.is_dir(),
            "repo_revision": _git_revision(repo),
            "checkpoint_exists": checkpoint.is_file(),
            "checkpoint_size_bytes": checkpoint.stat().st_size if checkpoint.is_file() else None,
            "checkpoint_sha256": sha256_file(checkpoint) if checkpoint.is_file() else None,
            "loader": "official_torch_hub_local_checkpoint",
        },
        "optional_modules": modules,
        "cuda": {
            "available": cuda_available,
            "device_count": cuda_device_count,
            "devices": cuda_devices,
            "error": cuda_error,
        },
        "readiness": {
            "crop": crop_ready,
            "feature_smoke": feature_smoke_ready and repo.is_dir() and checkpoint.is_file(),
            "feature_extraction": feature_extraction_ready and repo.is_dir() and checkpoint.is_file(),
        },
        "blockers": blockers,
    }
    output_dir = resolve_config_path(config, config.require("paths.environment"))
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "environment.audit.json", report)
    return report


def _ffpp_paths(config: Config) -> FFPPPaths:
    split_files = config.require("ffpp.split_files")
    if not isinstance(split_files, dict):
        raise FFPPIntegrationUnavailable("ffpp.split_files must be a mapping")
    resolved_splits = {
        str(name): resolve_config_path(config, path)
        for name, path in split_files.items()
    }
    return FFPPPaths.from_root(resolve_config_path(config, config.require("paths.ffpp_root")), resolved_splits)


def audit_dataset(config: Config) -> dict[str, Any]:
    paths = _ffpp_paths(config)
    report = audit_ffpp(
        paths,
        compressions=tuple(config.require("ffpp.compressions")),
        manipulations=tuple(config.require("ffpp.manipulations")),
    )
    report["config_digest"] = config_digest(config)
    report["config"] = str(config.source) if config.source else None
    output_dir = resolve_config_path(config, config.require("paths.manifests"))
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "dataset.audit.json", report)
    return report


def build_manifest(config: Config) -> dict[str, Any]:
    """Build the immutable path/frame manifest without decoding image data."""

    paths = _ffpp_paths(config)
    compressions = tuple(config.require("ffpp.compressions"))
    manipulations = tuple(config.require("ffpp.manipulations"))
    seed = int(config.require("experiment.seed"))
    validation_real_positions = tuple(
        float(value) for value in config.require("sampling.validation_real_positions")
    )
    validation_fake_positions = tuple(
        float(value) for value in config.require("sampling.validation_fake_positions")
    )
    test_positions = tuple(float(value) for value in config.require("sampling.test_positions"))
    all_rows = []
    per_split_counts: dict[str, int] = {}
    for split in ("train", "validation", "test"):
        if split == "validation":
            real_positions = validation_real_positions
            fake_positions = validation_fake_positions
        else:
            # The current frozen config uses the same three positions for
            # train/test bookkeeping.  Keeping the choice explicit avoids a
            # future config change being silently ignored.
            real_positions = test_positions
            fake_positions = test_positions
        rows = enumerate_records(
            paths,
            split=split,
            compressions=compressions,
            manipulations=manipulations,
            real_frame_fractions=real_positions,
            fake_frame_fractions=fake_positions,
            seed=seed,
        )
        all_rows.extend(rows)
        per_split_counts[split] = len(rows)

    output_dir = resolve_config_path(config, config.require("paths.manifests"))
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.jsonl"
    write_manifest(all_rows, manifest_path)
    write_manifest(all_rows, output_dir / "manifest.csv")

    train_ids = split_original_ids(paths.split_train)
    gallery_ids = select_gallery_ids(train_ids, count=int(config.require("gallery.videos")), seed=seed)
    validation_names = [name for name, _, _ in directed_fake_names(paths.split_validation)]
    pilot_names, reference_names = split_validation_directed_names(
        validation_names,
        count=int(config.require("sampling.gate_b_directed_names")),
        seed=seed,
    )
    gallery_manifest_path = output_dir / "gallery_manifest.jsonl"
    gallery_rows = enumerate_gallery_records(paths, gallery_ids=gallery_ids, seed=seed)
    write_manifest(gallery_rows, gallery_manifest_path)
    write_manifest(gallery_rows, output_dir / "gallery_manifest.csv")
    split_hashes = {
        split: sha256_file(paths.split_path(split))
        for split in ("train", "validation", "test")
    }
    metadata = {
        "status": "completed",
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "config_digest": config_digest(config),
        "split_hashes": split_hashes,
        "row_count": len(all_rows),
        "rows_by_split": per_split_counts,
        "frame_fractions": {
            "validation_real": list(validation_real_positions),
            "validation_fake": list(validation_fake_positions),
            "test": list(test_positions),
        },
        "gallery_ids": gallery_ids,
        "gallery_manifest": str(gallery_manifest_path),
        "gallery_manifest_sha256": sha256_file(gallery_manifest_path),
        "gallery_row_count": len(gallery_rows),
        "validation_gate_b_pilot_names": pilot_names,
        "validation_gate_beta_reference_names": reference_names,
        "mask_binarization": config.require("face_crop.mask_binarization"),
        "mask_frame_remapping": _mask_remapping_summary(all_rows),
        "notes": [
            "This stage only enumerates and probes videos; it does not create crops or features.",
            "Official test rows are recorded for reproducibility but must not be scored before Gates A/B pass.",
        ],
    }
    write_json(output_dir / "manifest.metadata.json", metadata)
    write_json(output_dir / "gallery_ids.json", {"seed": seed, "ids": gallery_ids})
    write_json(
        output_dir / "validation_subsets.json",
        {"seed": seed, "gate_b_pilot": pilot_names, "gate_beta_reference": reference_names},
    )
    write_json(output_dir / "split_hashes.json", split_hashes)
    return metadata


def cache_crops(config: Config) -> dict[str, Any]:
    """Materialize crops/masks and update the manifest with frozen bboxes."""

    from .crops import cache_manifest_crops
    manifests = resolve_config_path(config, config.require("paths.manifests"))
    manifest_path = manifests / "manifest.jsonl"
    if not manifest_path.is_file():
        raise FFPPIntegrationUnavailable(
            f"Manifest not found: {manifest_path}; run build_manifest before cache_crops"
        )
    summary = cache_manifest_crops(
        manifest_path,
        crops_dir=resolve_config_path(config, config.require("paths.crops")),
        masks_dir=resolve_config_path(config, config.require("paths.masks")),
        input_size=int(config.require("input.size")),
        detector_upsample_primary=int(config.require("face_crop.upsample_primary")),
        detector_upsample_retry=int(config.require("face_crop.upsample_retry")),
        square_scale=float(config.require("face_crop.square_scale")),
        padding=str(config.require("face_crop.padding")),
        mask_policy=str(config.require("face_crop.mask_binarization")),
        metadata_path=manifests / "manifest.metadata.json",
    )
    gallery_manifest_path = manifests / "gallery_manifest.jsonl"
    gallery_summary = None
    if gallery_manifest_path.is_file():
        # Gallery uses the exact middle frame, which is kept in a separate
        # manifest.  Cache it with the same detector/crop rules, but keep its
        # summary separate from the ordinary validation/test manifest.
        gallery_summary = cache_manifest_crops(
            gallery_manifest_path,
            crops_dir=resolve_config_path(config, config.require("paths.crops")),
            masks_dir=resolve_config_path(config, config.require("paths.masks")),
            input_size=int(config.require("input.size")),
            detector_upsample_primary=int(config.require("face_crop.upsample_primary")),
            detector_upsample_retry=int(config.require("face_crop.upsample_retry")),
            square_scale=float(config.require("face_crop.square_scale")),
            padding=str(config.require("face_crop.padding")),
            mask_policy=str(config.require("face_crop.mask_binarization")),
        )
        metadata_path = manifests / "manifest.metadata.json"
        payload = read_json(metadata_path) if metadata_path.is_file() else {}
        payload["gallery_crop_cache"] = gallery_summary.__dict__
        payload["gallery_manifest_sha256"] = sha256_file(gallery_manifest_path)
        write_json(metadata_path, payload)
    # Keep the CSV mirror synchronized with the JSONL manifest after bboxes
    # and exclusion reasons have been filled in.
    from .manifest import read_manifest, write_manifest
    write_manifest(read_manifest(manifest_path), manifests / "manifest.csv")
    if gallery_manifest_path.is_file():
        write_manifest(read_manifest(gallery_manifest_path), manifests / "gallery_manifest.csv")
    result = summary.__dict__.copy()
    if gallery_summary is not None:
        result["gallery"] = gallery_summary.__dict__
    return result


def smoke_test(config: Config) -> dict[str, Any]:
    """Load the frozen official DINOv3 model and verify one synthetic batch.

    This smoke test deliberately uses a deterministic synthetic RGB image.  It
    never reads FF++ test data and therefore cannot leak official test results
    into Gate A/B decisions.
    """

    import platform
    import torch

    repo = resolve_config_path(config, config.require("paths.dinov3_repo"))
    checkpoint = resolve_config_path(config, config.require("paths.dinov3_weights"))
    spec = spec_from_config(config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_size = int(config.require("input.size"))
    yy, xx = np.mgrid[:image_size, :image_size]
    image = np.stack(
        [
            (xx % 256).astype(np.uint8),
            (yy % 256).astype(np.uint8),
            ((xx + yy) % 256).astype(np.uint8),
        ],
        axis=-1,
    )
    model = load_official_model(repo, checkpoint, spec, device)
    batch = preprocess_rgb([image], config).to(device)
    tokens = extract_patch_tokens(model, batch, spec)
    result = {
        "status": "passed",
        "repo": str(repo),
        "checkpoint": str(checkpoint),
        "architecture": spec.architecture,
        "block_one_based": spec.block_one_based,
        "api_n": list(spec.api_n),
        "device": device,
        "device_name": torch.cuda.get_device_name(0) if device == "cuda" else platform.processor(),
        "input_shape": list(batch.shape),
        "token_shape": list(tokens.shape),
        "token_dtype": str(tokens.dtype),
        "tokens_finite": bool(np.isfinite(tokens).all()),
        "token_mean": float(tokens.mean()),
        "token_std": float(tokens.std()),
    }
    output_dir = resolve_config_path(config, config.require("paths.environment"))
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "dinov3.smoke.json", result)
    return result


def _manifest_for_feature_subset(config: Config, subset: str) -> tuple[Path, list[dict[str, Any]]]:
    if subset not in {"gallery", "validation", "test"}:
        raise ValueError(f"Unknown feature subset: {subset!r}")
    manifests = resolve_config_path(config, config.require("paths.manifests"))
    if subset == "gallery":
        path = manifests / "gallery_manifest.jsonl"
        if not path.is_file():
            raise FFPPIntegrationUnavailable(f"Gallery manifest not found: {path}; run build_manifest first")
        return path, read_manifest(path)
    path = manifests / "manifest.jsonl"
    if not path.is_file():
        raise FFPPIntegrationUnavailable(f"Manifest not found: {path}; run build_manifest first")
    return path, [row for row in read_manifest(path) if row["split"] == subset]


def _find_gate_decision_value(payload: Any, name: str) -> bool | None:
    """Find a named GateDecision in either the current or legacy JSON shape."""

    if isinstance(payload, dict):
        candidate_name = payload.get("name")
        if candidate_name in {name, name.replace("-", "_")} and "passed" in payload:
            return bool(payload["passed"])
        for key, value in payload.items():
            normalized = str(key).replace("_", "-")
            if normalized == name and isinstance(value, dict) and "passed" in value:
                return bool(value["passed"])
            found = _find_gate_decision_value(value, name)
            if found is not None:
                return found
    elif isinstance(payload, list):
        for value in payload:
            found = _find_gate_decision_value(value, name)
            if found is not None:
                return found
    return None


def _assert_test_gate_open(config: Config) -> Path:
    """Refuse official-test feature extraction until the frozen gates pass."""

    gates_dir = resolve_config_path(config, config.require("paths.results")) / "gates"
    candidates = (
        gates_dir / "gate_decision.json",
        gates_dir / "decide_gates.json",
        gates_dir / "gates.json",
    )
    decision_path = next((path for path in candidates if path.is_file()), None)
    if decision_path is None:
        raise RuntimeError(
            "Test feature extraction is locked: no Gate A/B decision artifact exists in "
            f"{gates_dir}"
        )
    payload = read_json(decision_path)
    expected_digest = config_digest(config)
    recorded_digest = payload.get("config_digest") if isinstance(payload, dict) else None
    if recorded_digest is None and isinstance(payload, dict):
        metadata = payload.get("metadata")
        if isinstance(metadata, dict):
            recorded_digest = metadata.get("config_digest")
    if recorded_digest != expected_digest:
        raise RuntimeError(
            f"Gate decision config digest mismatch: recorded={recorded_digest!r}, expected={expected_digest!r}"
        )
    through = payload.get("through") if isinstance(payload, dict) else None
    if str(through).upper() not in {"B", "D"}:
        raise RuntimeError(f"Gate decision only reaches {through!r}; test extraction requires through=B")
    overall = payload.get("overall_passed") if isinstance(payload, dict) else None
    if overall is None and isinstance(payload, dict):
        overall = payload.get("passed")
    if overall is not True:
        raise RuntimeError("Gate decision does not report overall_passed=true; test extraction remains locked")
    required = ("A-alpha", "A-beta", "B-synthetic", "B-real")
    missing_or_failed = [name for name in required if _find_gate_decision_value(payload, name) is not True]
    if missing_or_failed:
        raise RuntimeError(f"Required gates are not all passed: {missing_or_failed}")
    return decision_path


def _git_revision(path: Path) -> str | None:
    import subprocess

    if not path.is_dir():
        return None
    try:
        completed = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    value = completed.stdout.strip()
    return value if completed.returncode == 0 and value else None


def _load_crop_for_feature(row: dict[str, Any], *, input_size: int) -> tuple[Path, np.ndarray]:
    raw_path = row.get("crop_path")
    if not raw_path:
        raise RuntimeError(f"Active manifest row has no crop_path: {row['sample_id']}")
    path = Path(str(raw_path))
    if not path.is_file():
        raise FileNotFoundError(f"Crop artifact does not exist for {row['sample_id']}: {path}")
    try:
        crop = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Could not load crop for {row['sample_id']}: {path}") from exc
    expected = (int(input_size), int(input_size), 3)
    if crop.shape != expected or crop.dtype != np.dtype(np.uint8):
        raise ValueError(f"Crop {path} must be uint8 {expected}, got {crop.shape} {crop.dtype}")
    if not np.isfinite(crop).all():
        raise ValueError(f"Crop {path} contains non-finite values")
    return path, np.asarray(crop)


def extract_features(config: Config, *, subset: str | None = None) -> dict[str, Any]:
    """Extract frozen DINOv3 block-6 patch tokens on CUDA with resume support."""

    if subset is None:
        raise ValueError("extract_features requires subset=gallery|validation|test")
    if subset == "test":
        gate_path = _assert_test_gate_open(config)
    else:
        gate_path = None

    import torch

    if not torch.cuda.is_available():
        raise DINOv3IntegrationUnavailable("Feature extraction requires CUDA; CPU fallback is disabled")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    manifest_path, rows = _manifest_for_feature_subset(config, subset)
    active_rows = [row for row in rows if not row.get("exclusion_reason")]
    excluded_rows = [row for row in rows if row.get("exclusion_reason")]
    if subset == "gallery" and len(active_rows) != int(config.require("gallery.videos")):
        raise ValueError(
            f"Gallery manifest must contain exactly {config.require('gallery.videos')} active rows, "
            f"got {len(active_rows)} (excluded={len(excluded_rows)})"
        )
    if not active_rows:
        raise ValueError(f"No active rows available for feature subset {subset}")

    spec = spec_from_config(config)
    repo = resolve_config_path(config, config.require("paths.dinov3_repo"))
    checkpoint = resolve_config_path(config, config.require("paths.dinov3_weights"))
    manifest_sha = sha256_file(manifest_path)
    checkpoint_sha = sha256_file(checkpoint)
    repo_revision = _git_revision(repo)
    config_hash = config_digest(config)
    feature_root = resolve_config_path(config, config.require("paths.features")) / subset
    feature_root.mkdir(parents=True, exist_ok=True)
    batch_size = int(os.environ.get("CIRLID_FEATURE_BATCH_SIZE", "2"))
    if batch_size <= 0:
        raise ValueError("CIRLID_FEATURE_BATCH_SIZE must be positive")

    artifact_rows: list[dict[str, Any]] = []
    # Keep only paths in the pending queue. Validation/test can contain many
    # thousands of 448x448 RGB crops; retaining every decoded crop here would
    # unnecessarily consume several GB of host RAM before the first forward
    # pass. Crops are loaded and validated one batch at a time below.
    pending: list[tuple[dict[str, Any], Path, Path]] = []
    reused = 0
    for row in rows:
        if row.get("exclusion_reason"):
            artifact_rows.append(
                {"sample_id": row["sample_id"], "status": "excluded", "exclusion_reason": row["exclusion_reason"]}
            )
            continue
        raw_crop_path = row.get("crop_path")
        if not raw_crop_path:
            raise RuntimeError(f"Active manifest row has no crop_path: {row['sample_id']}")
        crop_path = Path(str(raw_crop_path))
        if not crop_path.is_file():
            raise FileNotFoundError(f"Crop artifact does not exist for {row['sample_id']}: {crop_path}")
        artifact_path = feature_artifact_path(feature_root, row["sample_id"])
        reusable, metadata = reusable_feature(
            artifact_path,
            sample_id=row["sample_id"],
            config_digest_value=config_hash,
            tokens=spec.expected_tokens,
            dimension=spec.expected_dimension,
            crop_path=crop_path,
            required_metadata={
                "subset": subset,
                "manifest_path": str(manifest_path),
                "manifest_sha256": manifest_sha,
                "checkpoint_path": str(checkpoint),
                "checkpoint_sha256": checkpoint_sha,
                "dinov3_repo": str(repo),
                "dinov3_commit": repo_revision,
                "architecture": spec.architecture,
                "block_one_based": spec.block_one_based,
                "api_n": list(spec.api_n),
                "norm": spec.norm,
                "reshape": spec.reshape,
                "device": "cuda",
            },
        )
        if reusable:
            reused += 1
            artifact_rows.append(
                {
                    "sample_id": row["sample_id"],
                    "status": "reused",
                    "feature_path": str(artifact_path),
                    "shape": metadata.get("shape") if metadata else None,
                    "dtype": metadata.get("dtype") if metadata else None,
                    "array_sha256": metadata.get("array_sha256") if metadata else None,
                }
            )
        else:
            pending.append((row, crop_path, artifact_path))

    model = load_official_model(repo, checkpoint, spec, "cuda") if pending else None
    extracted = 0
    device = torch.device("cuda")
    device_name = str(torch.cuda.get_device_name(0))
    for start in range(0, len(pending), batch_size):
        batch_rows = pending[start : start + batch_size]
        crops = []
        for row, crop_path, _ in batch_rows:
            loaded_path, crop = _load_crop_for_feature(
                row,
                input_size=int(config.require("input.size")),
            )
            if loaded_path != crop_path:
                raise RuntimeError(
                    f"Manifest crop path changed while extracting {row['sample_id']}: "
                    f"{loaded_path} != {crop_path}"
                )
            crops.append(crop)
        batch = preprocess_rgb(crops, config).to(device, non_blocking=True)
        tokens = extract_patch_tokens(model, batch, spec)
        expected_shape = (len(batch_rows), spec.expected_tokens, spec.expected_dimension)
        if tokens.shape != expected_shape:
            raise ValueError(f"DINOv3 returned unexpected batch shape: {tokens.shape}")
        if tokens.dtype != np.dtype(np.float32) or not np.isfinite(tokens).all():
            raise ValueError("DINOv3 feature batch must be finite float32")
        for offset, (row, crop_path, artifact_path) in enumerate(batch_rows):
            metadata = save_features(
                artifact_path,
                tokens[offset : offset + 1],
                config=config,
                metadata={
                    "sample_id": row["sample_id"],
                    "subset": subset,
                    "crop_path": str(crop_path),
                    "crop_sha256": sha256_file(crop_path),
                    "manifest_path": str(manifest_path),
                    "manifest_sha256": manifest_sha,
                    "checkpoint_path": str(checkpoint),
                    "checkpoint_sha256": checkpoint_sha,
                    "dinov3_repo": str(repo),
                    "dinov3_commit": repo_revision,
                    "architecture": spec.architecture,
                    "block_one_based": spec.block_one_based,
                    "api_n": list(spec.api_n),
                    "norm": spec.norm,
                    "reshape": spec.reshape,
                    "device": "cuda",
                    "device_name": device_name,
                },
            )
            artifact_rows.append(
                {
                    "sample_id": row["sample_id"],
                    "status": "extracted",
                    "feature_path": str(artifact_path),
                    "shape": metadata["shape"],
                    "dtype": metadata["dtype"],
                    "array_sha256": metadata["array_sha256"],
                }
            )
            extracted += 1
        print(
            f"[extract_features] subset={subset} batch={min(start + batch_size, len(pending))}/{len(pending)} "
            f"reused={reused}",
            flush=True,
        )

    artifact_rows.sort(key=lambda item: str(item["sample_id"]))
    index_path = feature_root / "index.jsonl"
    write_jsonl(index_path, artifact_rows)
    summary = {
        "status": "completed",
        "subset": subset,
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "config_digest": config_hash,
        "rows": len(rows),
        "active_rows": len(active_rows),
        "excluded_rows": len(excluded_rows),
        "reused": reused,
        "extracted": extracted,
        "batch_size": batch_size,
        "device": "cuda",
        "device_name": device_name,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "dinov3_repo": str(repo),
        "dinov3_commit": repo_revision,
        "architecture": spec.architecture,
        "block_one_based": spec.block_one_based,
        "api_n": list(spec.api_n),
        "norm": spec.norm,
        "reshape": spec.reshape,
        "expected_shape": [1, spec.expected_tokens, spec.expected_dimension],
        "index": str(index_path),
        "gate_decision": str(gate_path) if gate_path is not None else None,
    }
    write_json(feature_root / "summary.json", summary)
    return summary


def build_gallery_command(config: Config) -> dict[str, Any]:
    """Build the exact 64-video train-real/c23 gallery and FAISS index."""

    from .gallery import build_gallery

    manifest_path, rows = _manifest_for_feature_subset(config, "gallery")
    active = [row for row in rows if not row.get("exclusion_reason")]
    expected_videos = int(config.require("gallery.videos"))
    if len(active) != expected_videos:
        raise ValueError(f"Gallery requires {expected_videos} active rows, got {len(active)}")
    for row in active:
        if not (
            row["split"] == "train"
            and row["compression"] == "c23"
            and row["manipulation"] == "original"
            and row["is_real"]
        ):
            raise ValueError(f"Invalid gallery row: {row['sample_id']}")
        if not row.get("crop_path"):
            raise ValueError(f"Gallery row has no cached crop: {row['sample_id']}")

    active.sort(key=lambda row: (str(row["video_name"]), str(row["sample_id"])))
    feature_root = resolve_config_path(config, config.require("paths.features")) / "gallery"
    config_hash = config_digest(config)
    manifest_sha = sha256_file(manifest_path)
    checkpoint = resolve_config_path(config, config.require("paths.dinov3_weights"))
    checkpoint_sha = sha256_file(checkpoint)
    expected_shape = (
        1,
        int(config.require("features.expected_tokens_per_frame")),
        int(config.require("features.expected_dimension")),
    )
    feature_arrays: list[np.ndarray] = []
    feature_paths: list[str] = []
    for row in active:
        path = feature_artifact_path(feature_root, row["sample_id"])
        if not path.is_file():
            raise FileNotFoundError(f"Gallery feature artifact is missing: {path}")
        value, metadata = load_features(path)
        if value.shape != expected_shape or value.dtype != np.dtype(np.float32):
            raise ValueError(f"Invalid gallery feature {path}: {value.shape} {value.dtype}")
        if metadata.get("sample_id") != row["sample_id"]:
            raise ValueError(f"Gallery feature metadata sample mismatch: {path}")
        if metadata.get("config_digest") != config_hash:
            raise ValueError(f"Gallery feature config digest mismatch: {path}")
        expected_provenance = {
            "subset": "gallery",
            "manifest_path": str(manifest_path),
            "manifest_sha256": manifest_sha,
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha,
            "device": "cuda",
        }
        for key, expected in expected_provenance.items():
            if metadata.get(key) != expected:
                raise ValueError(f"Gallery feature provenance mismatch for {key}: {path}")
        crop_path = Path(str(row["crop_path"]))
        if metadata.get("crop_sha256") != sha256_file(crop_path):
            raise ValueError(f"Gallery crop hash mismatch: {crop_path}")
        feature_arrays.append(value)
        feature_paths.append(str(path))

    features = np.concatenate(feature_arrays, axis=0)
    indices_dir = resolve_config_path(config, config.require("paths.indices"))
    artifact = build_gallery(
        features,
        indices_dir,
        config=config,
        source_ids=[row["sample_id"] for row in active],
    )
    expected_tokens = int(config.require("gallery.expected_tokens"))
    expected_dimension = int(config.require("features.expected_dimension"))
    if artifact.vectors.shape != (expected_tokens, expected_dimension):
        raise ValueError(f"Gallery shape mismatch: {artifact.vectors.shape}")
    if artifact.index.ntotal != expected_tokens or artifact.index.dimension != expected_dimension:
        raise ValueError("Gallery FAISS index dimensions do not satisfy the frozen protocol")

    metadata_path = indices_dir / "gallery.metadata.json"
    metadata = read_json(metadata_path)
    metadata.update(
        {
            "manifest": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "config_digest": config_hash,
            "gallery_array_sha256": sha256_file(indices_dir / "gallery.npy"),
            "gallery_sample_ids": [row["sample_id"] for row in active],
            "gallery_video_ids": [row["video_name"] for row in active],
            "feature_paths": feature_paths,
        }
    )
    write_json(metadata_path, metadata)
    summary = {
        "status": "completed",
        "manifest": str(manifest_path),
        "manifest_sha256": metadata["manifest_sha256"],
        "config_digest": metadata["config_digest"],
        "videos": len(active),
        "tokens_per_frame": int(config.require("gallery.tokens_per_frame")),
        "shape": list(artifact.vectors.shape),
        "dtype": str(artifact.vectors.dtype),
        "ntotal": artifact.index.ntotal,
        "dimension": artifact.index.dimension,
        "gallery_array_sha256": metadata["gallery_array_sha256"],
        "sample_ids": metadata["gallery_sample_ids"],
        "feature_paths": feature_paths,
    }
    write_json(indices_dir / "gallery.summary.json", summary)
    return summary


def _upsample_score_map(values: np.ndarray, config: Config) -> np.ndarray:
    import torch

    grid = tuple(int(value) for value in config.require("features.expected_patch_grid"))
    image_size = int(config.require("input.size"))
    score_map = np.asarray(values, dtype=np.float32).reshape(grid)
    tensor = torch.from_numpy(score_map).view(1, 1, grid[0], grid[1])
    resized = torch.nn.functional.interpolate(
        tensor,
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    )
    return resized.numpy().reshape(-1).astype(np.float32, copy=False)


def _validation_score_maps(
    frame_features: np.ndarray,
    gallery: Any,
    lof: Any,
    methods: tuple[str, ...],
    *,
    k: int,
    query_batch_size: int,
) -> dict[str, np.ndarray]:
    from .scoring import local_pca_participation_ratio, score_from_knn, sp_lid

    features = np.asarray(frame_features, dtype=np.float32)
    if features.ndim != 2:
        raise ValueError(f"Expected frame features [784,768], got {features.shape}")
    knn = gallery.index.search(features, k, batch_size=query_batch_size)
    basic = score_from_knn(knn.distances)
    scores: dict[str, np.ndarray] = {}
    for method in methods:
        if method in basic:
            scores[method] = basic[method]
        elif method == "SP_LID":
            scores[method] = sp_lid(features, k=k)
        elif method == "LOF":
            scores[method] = (-lof.score_samples(features)).astype(np.float32)
        elif method == "local_PCA_PR":
            scores[method] = local_pca_participation_ratio(gallery.vectors, knn.indices)
        else:
            raise ValueError(f"Unsupported score method in frozen config: {method}")
        if not np.isfinite(scores[method]).all():
            raise ValueError(f"Non-finite scores for method {method}")
    return scores


def calibrate_thresholds_command(config: Config) -> dict[str, Any]:
    """Calibrate method-specific thresholds on validation-real c23 pixels only."""

    from .gallery import load_gallery
    from .thresholds import audit_calibration, calibrate_method_thresholds, save_thresholds

    manifest_path, all_rows = _manifest_for_feature_subset(config, "validation")
    rows = [
        row
        for row in all_rows
        if row["compression"] == "c23" and row["is_real"] and not row.get("exclusion_reason")
    ]
    if not rows:
        raise ValueError("No active validation-real c23 rows are available for threshold calibration")

    gallery_dir = resolve_config_path(config, config.require("paths.indices"))
    gallery = load_gallery(gallery_dir)
    expected_tokens = int(config.require("gallery.expected_tokens"))
    if gallery.index.ntotal != expected_tokens:
        raise ValueError("Gallery index does not match frozen expected token count")

    try:
        from sklearn.neighbors import LocalOutlierFactor
    except ImportError as exc:  # pragma: no cover - server dependency
        raise RuntimeError("scikit-learn is required for the LOF threshold baseline") from exc
    k = int(config.require("knn.k"))
    query_batch_size = int(config.require("knn.query_batch_size"))
    lof = LocalOutlierFactor(n_neighbors=k, metric="euclidean", algorithm="brute", novelty=True)
    lof.fit(gallery.vectors)
    methods = tuple(
        str(value)
        for value in tuple(config.require("scores.primary")) + tuple(config.require("scores.secondary"))
    )
    score_maps: dict[str, list[np.ndarray]] = {method: [] for method in methods}
    feature_root = resolve_config_path(config, config.require("paths.features")) / "validation"
    config_hash = config_digest(config)
    manifest_sha = sha256_file(manifest_path)
    checkpoint = resolve_config_path(config, config.require("paths.dinov3_weights"))
    checkpoint_sha = sha256_file(checkpoint)
    expected_shape = (
        1,
        int(config.require("features.expected_tokens_per_frame")),
        int(config.require("features.expected_dimension")),
    )
    index_rows: list[dict[str, Any]] = []
    for number, row in enumerate(rows, start=1):
        path = feature_artifact_path(feature_root, row["sample_id"])
        if not path.is_file():
            raise FileNotFoundError(f"Validation feature artifact is missing: {path}")
        value, metadata = load_features(path)
        if value.shape != expected_shape or value.dtype != np.dtype(np.float32):
            raise ValueError(f"Invalid validation feature artifact: {path}")
        if metadata.get("sample_id") != row["sample_id"] or metadata.get("config_digest") != config_hash:
            raise ValueError(f"Validation feature provenance mismatch: {path}")
        expected_provenance = {
            "subset": "validation",
            "manifest_path": str(manifest_path),
            "manifest_sha256": manifest_sha,
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha,
            "device": "cuda",
        }
        for key, expected in expected_provenance.items():
            if metadata.get(key) != expected:
                raise ValueError(f"Validation feature provenance mismatch for {key}: {path}")
        scores = _validation_score_maps(value[0], gallery, lof, methods, k=k, query_batch_size=query_batch_size)
        index_row: dict[str, Any] = {
            "sample_id": row["sample_id"],
            "feature_path": str(path),
            "frame_index": row["frame_index"],
        }
        for method in methods:
            pixels = _upsample_score_map(scores[method], config)
            score_maps[method].append(pixels)
            index_row[method] = {
                "token_min": float(np.min(scores[method])),
                "token_median": float(np.median(scores[method])),
                "token_max": float(np.max(scores[method])),
                "pixel_count": int(pixels.size),
            }
        index_rows.append(index_row)
        if number == len(rows) or number % 25 == 0:
            print(f"[calibrate_thresholds] processed={number}/{len(rows)}", flush=True)

    quantile = float(config.require("threshold.quantile"))
    numpy_method = str(config.require("threshold.numpy_method"))
    thresholds = calibrate_method_thresholds(score_maps, quantile=quantile, numpy_method=numpy_method)
    actual_fpr = audit_calibration(score_maps, thresholds)
    results_dir = resolve_config_path(config, config.require("paths.results")) / "thresholds"
    results_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "status": "completed",
        "source": "validation_real_c23_pooled_pixels",
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "gallery_dir": str(gallery_dir),
        "config_digest": config_hash,
        "quantile": quantile,
        "numpy_method": numpy_method,
        "methods": list(methods),
        "rows": len(rows),
        "pixels_per_row": int(config.require("input.size")) ** 2,
        "target_clean_pixel_fpr": float(config.require("threshold.target_clean_pixel_fpr")),
        "actual_clean_pixel_fpr": actual_fpr,
    }
    save_thresholds(results_dir / "thresholds.json", thresholds, metadata=metadata)
    write_jsonl(results_dir / "calibration_index.jsonl", index_rows)
    summary = {**metadata, "thresholds": thresholds}
    write_json(results_dir / "calibration_summary.json", summary)
    return summary


def run_synthetic(config: Config) -> dict[str, Any]:
    """Run the DINO/FF++-independent synthetic stage and save JSONL output."""

    from .gates import gate_synthetic
    from .synthetic import SyntheticConfig, run_synthetic as run, summarize_synthetic

    section = config.data.get("synthetic", {})
    seed_spec = section.get("seeds", 20)
    if isinstance(seed_spec, int):
        seeds = tuple(range(seed_spec))
    else:
        seeds = tuple(int(value) for value in seed_spec)
    synthetic_config = SyntheticConfig(
        ambient_dimension=int(section.get("ambient_dimension", 64)),
        latent_dimension=int(section.get("latent_dimension", 4)),
        clean_queries=int(section.get("clean_queries", 2000)),
        gallery_sizes=tuple(int(v) for v in section.get("gallery_sizes", [2000, 5000, 10000])),
        densities=tuple(str(v) for v in section.get("densities", ["sparse", "uniform", "concentrated"])),
        radial_scales=tuple(float(v) for v in section.get("radial_scales", [0.5, 1.0, 2.0])),
        deltas=tuple(float(v) for v in section.get("deltas", [0.0, 0.25, 0.5, 1.0])),
        seeds=seeds,
        k=int(config.require("knn.k")),
    )
    rows = run(synthetic_config)
    output_dir = resolve_config_path(config, config.require("paths.results")) / "synthetic"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "synthetic_scores.jsonl"
    write_jsonl(output_path, rows)
    reduction_min = float(config.require("gates.synthetic.relative_q_reduction_min"))
    synthetic_summary = summarize_synthetic(rows, reduction_min=reduction_min)
    decision = gate_synthetic(
        synthetic_summary["relative_q_reduction_cir_vs_c2"],
        reduction_min=reduction_min,
    )
    summary = {
        "status": "completed",
        "rows": len(rows),
        "output": str(output_path),
        "config": str(config.source) if config.source else None,
        "config_digest": config_digest(config),
        "synthetic": synthetic_summary,
        "gate": decision.to_dict(),
    }
    write_json(output_dir / "summary.json", summary)
    gate_dir = resolve_config_path(config, config.require("paths.results")) / "gates"
    gate_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        gate_dir / "synthetic_gate_b.json",
        {
            "decision": decision.to_dict(),
            "name": decision.name,
            "passed": decision.passed,
            "checks": dict(decision.checks),
            "observed": dict(decision.observed),
            "thresholds": dict(decision.thresholds),
            "config_digest": config_digest(config),
            "synthetic_config": {
                "ambient_dimension": synthetic_config.ambient_dimension,
                "latent_dimension": synthetic_config.latent_dimension,
                "clean_queries": synthetic_config.clean_queries,
                "gallery_sizes": list(synthetic_config.gallery_sizes),
                "densities": list(synthetic_config.densities),
                "radial_scales": list(synthetic_config.radial_scales),
                "deltas": list(synthetic_config.deltas),
                "seeds": list(synthetic_config.seeds),
                "k": synthetic_config.k,
            },
            "seed_spec": seed_spec,
            "row_count": len(rows),
            "methods": sorted({str(row["method"]) for row in rows}),
            "summary": synthetic_summary,
        },
    )
    return summary


def pending_experiment(name: str):
    def action(config: Config, **kwargs: Any) -> dict[str, Any]:
        return {
            "status": "pending",
            "command": name,
            "reason": "FF++/DINOv3 integration boundary is not connected",
            "options": kwargs,
        }

    return action


# Public aliases used by the documented `python -m src.<command>` entry points.
run_gate_alpha = pending_experiment("run_gate_alpha")
run_gate_beta = pending_experiment("run_gate_beta")
run_gate_b_pilot = pending_experiment("run_gate_b_pilot")
decide_gates = pending_experiment("decide_gates")
run_gate_d = pending_experiment("run_gate_d")
bootstrap_results = pending_experiment("bootstrap_results")
make_figures = pending_experiment("make_figures")
