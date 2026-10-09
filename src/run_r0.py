"""Run the isolated R0 semantic-relation oracle diagnostic.

Stages: prepare, crops, features, evaluate, finalize. This runner is
train-only, never opens validation/test data, and never unlocks FF++ official
test. It is an oracle paired target-real analysis, not a deployable detector.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from src.cirlid.config import config_digest, load_config, resolve_config_path
from src.cirlid.crops import cache_manifest_crops
from src.cirlid.dinov3 import extract_patch_tokens, load_official_model, preprocess_rgb, spec_from_config
from src.cirlid.features import feature_artifact_path, load_features, reusable_feature, save_features
from src.cirlid.ffpp import FFPPPaths, enumerate_records, read_split_pairs
from src.cirlid.io import read_json, read_jsonl, sha256_file, write_json, write_jsonl
from src.cirlid.manifest import normalize_record, read_manifest, write_manifest
from src.r0_semantic_relation.bootstrap import paired_group_bootstrap
from src.r0_semantic_relation.geometry import build_semantic_regions
from src.r0_semantic_relation.mediapipe_adapter import create_landmarker, detect_landmarks
from src.r0_semantic_relation.metrics import average_precision, flatten_map, prevalence
from src.r0_semantic_relation.protocol import require_r0_protocol, r0_source_hashes
from src.r0_semantic_relation.scoring import (
    broken_context_relation_score,
    local_difference_score,
    position_template_score,
    relation_score,
)


SCORES = ("relation", "local_difference", "position_template", "broken_context")


class R0ProvenanceError(ValueError):
    """Fatal mismatch in immutable model or feature provenance."""


def ctx(path):
    c = load_config(path)
    require_r0_protocol(c)
    root = resolve_config_path(c, c.require("paths.output_dir"))
    root.mkdir(parents=True, exist_ok=True)
    return c, root


def _assert_frozen(config, root: Path) -> None:
    receipt_path = root / "freeze_receipt.json"
    if not receipt_path.is_file():
        raise ValueError("R0 freeze_receipt.json is missing; run prepare first")
    receipt = read_json(receipt_path)
    if receipt.get("status") != "frozen" or receipt.get("config_digest") != config_digest(config):
        raise ValueError("R0 frozen config digest mismatch; do not evaluate a modified protocol")
    if receipt.get("source_sha256") != r0_source_hashes():
        raise ValueError("R0 source files changed since prepare; use a new output directory for a new run")
    checkpoint = resolve_config_path(config, config.require("paths.dinov3_weights"))
    if receipt.get("dinov3_checkpoint_sha256") != sha256_file(checkpoint):
        raise ValueError("R0 DINOv3 checkpoint changed since prepare")
    asset, digest = _verify_mediapipe_asset(config)
    if receipt.get("mediapipe_asset_sha256") != digest:
        raise ValueError(f"R0 MediaPipe asset changed since prepare: {asset}")


def _canonical_group(target_id: str, source_id: str) -> str:
    left, right = sorted((str(target_id), str(source_id)))
    return f"pair__{left}__{right}"


def _component_map(split_pairs: list[tuple[str, str]]) -> dict[str, str]:
    parent: dict[str, str] = {}

    def find(value: str) -> str:
        parent.setdefault(value, value)
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for a, b in split_pairs:
        union(str(a), str(b))
    roots = sorted({find(value) for value in parent})
    labels = {root: f"component-{i:04d}" for i, root in enumerate(roots, 1)}
    return {value: labels[find(value)] for value in parent}


def _partition_components(component_ids: list[str], config) -> dict[str, str]:
    enabled = bool(config.get("holdout.enabled", True))
    if not enabled:
        return {value: "all_train" for value in component_ids}
    fraction = float(config.get("holdout.fraction", 0.30))
    if not 0.0 < fraction < 1.0:
        raise ValueError("holdout.fraction must be strictly between zero and one")
    values = sorted(set(component_ids))
    if len(values) < 2:
        raise ValueError("holdout requires at least two independent source components")
    rng = np.random.default_rng(int(config.get("holdout.seed", 20260930)))
    shuffled = list(rng.permutation(values))
    count = max(1, min(len(values) - 1, int(round(len(values) * fraction))))
    holdout = set(shuffled[:count])
    return {value: ("holdout" if value in holdout else "exploration") for value in values}


def _pair_metadata(fake: dict, real: dict | None, source_group: str, component_id: str, partition: str) -> dict:
    fake_frame = fake.get("frame_index")
    real_frame = real.get("frame_index") if real else None
    fake_crop = fake.get("crop_source_frame")
    real_crop = real.get("crop_source_frame") if real else None
    crop_video_match = real is not None and fake.get("crop_source_video") == real.get("crop_source_video")
    fake_mask = fake.get("mask_frame_index")
    frame_match = real is not None and fake_frame == real_frame
    crop_match = real is not None and fake_crop == real_crop
    mask_match = fake_mask is not None and fake_mask == fake_frame and not bool(fake.get("mask_frame_remapped"))
    strict = bool(frame_match and crop_match and crop_video_match and mask_match)
    status = "strict_frame_match" if strict else ("missing_real_pair" if real is None else "index_mismatch")
    return {
        "fake_sample_id": fake["sample_id"],
        "real_sample_id": real["sample_id"] if real else None,
        "source_group": source_group,
        "component_id": component_id,
        "partition": partition,
        "target_id": fake["target_id"],
        "source_id": fake["source_id"],
        "compression": fake["compression"],
        "manipulation": fake["manipulation"],
        "frame_fraction": fake["frame_fraction"],
        "fake_frame_index": fake_frame,
        "real_frame_index": real_frame,
        "fake_crop_source_frame": fake_crop,
        "real_crop_source_frame": real_crop,
        "mask_frame_index": fake_mask,
        "fake_frame_count": fake.get("frame_count"),
        "real_frame_count": real.get("frame_count") if real else None,
        "mask_frame_count": fake.get("mask_frame_count"),
        "frame_index_match": bool(frame_match),
        "crop_source_frame_match": bool(crop_match),
        "crop_source_video_match": bool(crop_video_match),
        "mask_frame_index_match": bool(mask_match),
        "strict_frame_pair": strict,
        "pairing_status": status,
        "pairing_valid": real is not None,
    }


def prepare(config_path):
    c, root = ctx(config_path)
    if (root / "freeze_receipt.json").exists():
        _assert_frozen(c, root)
        if (root / "manifest.jsonl").exists():
            raise ValueError("R0 is already frozen; prepare cannot overwrite the existing manifest")
    asset_path, asset_digest = _verify_mediapipe_asset(c)
    checkpoint_path = resolve_config_path(c, c.require("paths.dinov3_weights"))
    checkpoint_digest = sha256_file(checkpoint_path)
    paths = FFPPPaths.from_root(
        resolve_config_path(c, c.require("paths.ffpp_root")),
        {k: resolve_config_path(c, v) for k, v in c.require("paths.split_files").items()},
    )
    split_pairs = read_split_pairs(paths.split_train)
    component_by_video = _component_map(split_pairs)
    partition_by_component = _partition_components(list(component_by_video.values()), c)
    rows = enumerate_records(
        paths,
        split="train",
        compressions=("c23",),
        manipulations=("Face2Face",),
        fake_frame_fractions=tuple(c.require("sampling.frame_fractions")),
        real_frame_fractions=tuple(c.require("sampling.frame_fractions")),
        seed=int(c.require("protocol.seed")),
    )
    normalized = [normalize_record(x) for x in rows]
    write_manifest(normalized, root / "manifest.jsonl")
    real_by = {(r["compression"], r["target_id"], float(r["frame_fraction"])): r for r in normalized if r["is_real"]}
    pairs: list[dict] = []
    groups: dict[str, dict] = {}
    for fake in normalized:
        if fake["is_real"]:
            continue
        source_group = _canonical_group(fake["target_id"], fake["source_id"])
        target_component = component_by_video.get(str(fake["target_id"]))
        source_component = component_by_video.get(str(fake["source_id"]))
        if target_component is None or target_component != source_component:
            raise ValueError(f"Missing or inconsistent source component for {fake['sample_id']}")
        component_id = target_component
        partition = partition_by_component[component_id]
        key = (fake["compression"], fake["target_id"], float(fake["frame_fraction"]))
        pair = _pair_metadata(fake, real_by.get(key), source_group, component_id, partition)
        pairs.append(pair)
        entry = groups.setdefault(source_group, {
            "source_group": source_group,
            "component_id": component_id,
            "partition": partition,
            "undirected_video_ids": sorted([str(fake["target_id"]), str(fake["source_id"])]),
            "directed_pairs": [],
            "fake_sample_count": 0,
        })
        directed = f"{fake['target_id']}__{fake['source_id']}"
        if directed not in entry["directed_pairs"]:
            entry["directed_pairs"].append(directed)
        entry["fake_sample_count"] += 1
    mismatch_counts = Counter(pair["pairing_status"] for pair in pairs)
    reverse_groups = 0
    multi_direction_groups = 0
    for entry in groups.values():
        directed = set(entry["directed_pairs"])
        if len(directed) > 1:
            multi_direction_groups += 1
        reverse_present = any(
            f"{right}__{left}" in directed
            for item in directed
            for left, right in [item.split("__", 1)]
        )
        if reverse_present:
            reverse_groups += 1
    source_overlap = {
        "canonical_source_group_shares_reverse_directions": True,
        "source_groups": len(groups),
        "groups_with_multiple_directions": multi_direction_groups,
        "groups_with_observed_reverse_directions": reverse_groups,
        "components": len(set(item["component_id"] for item in groups.values())),
        "component_sizes": dict(Counter(item["component_id"] for item in groups.values())),
        "pairing_status_counts": dict(mismatch_counts),
    }
    write_jsonl(root / "pairing_manifest.jsonl", pairs)
    write_json(root / "source_groups.json", groups)
    write_json(root / "source_group_audit.json", source_overlap)
    write_json(root / "mask_provenance.json", {
        "mask_used_only_for_validity_and_evaluation": True,
        "mask_used_for_region_design": False,
        "mask_used_for_tuning": False,
        "scope": "Face2Face/c23/train",
        "official_test_locked": True,
    })
    write_json(root / "freeze_receipt.json", {
        "status": "frozen",
        "protocol": "R0-semantic-relation-v1",
        "config_digest": config_digest(c),
        "config": str(config_path),
        "mediapipe_asset": str(asset_path),
        "mediapipe_asset_sha256": asset_digest,
        "dinov3_checkpoint_sha256": checkpoint_digest,
        "source_sha256": r0_source_hashes(),
        "rows_generated": len(normalized),
        "fake_pairs": len(pairs),
        "source_groups": len(groups),
        "components": len(set(component_by_video.values())),
        "partition_counts": dict(Counter(pair["partition"] for pair in pairs)),
        "pairing_status_counts": dict(mismatch_counts),
        "official_test_locked": True,
    })
    return {
        "status": "completed",
        "rows": len(normalized),
        "fake_pairs": len(pairs),
        "source_groups": len(groups),
        "partition_counts": dict(Counter(pair["partition"] for pair in pairs)),
        "pairing_status_counts": dict(mismatch_counts),
        "official_test_locked": True,
    }


def crops(config_path):
    c, root = ctx(config_path)
    _assert_frozen(c, root)
    summary = cache_manifest_crops(
        root / "manifest.jsonl",
        crops_dir=resolve_config_path(c, c.require("paths.crops")),
        masks_dir=resolve_config_path(c, c.require("paths.masks")),
        input_size=int(c.require("input.size")),
        detector_upsample_primary=int(c.require("face_crop.upsample_primary")),
        detector_upsample_retry=int(c.require("face_crop.upsample_retry")),
        square_scale=float(c.require("face_crop.square_scale")),
        padding=str(c.require("face_crop.padding")),
        mask_policy=str(c.require("face_crop.mask_binarization")),
        exclude_empty_fake_masks=False,
        metadata_path=root / "provenance.json",
        progress_every=int(c.get("runtime.progress_every", 250)),
    )
    write_json(root / "crop_summary.json", summary.__dict__)
    return summary.__dict__


def features(config_path):
    c, root = ctx(config_path)
    _assert_frozen(c, root)
    rows = read_manifest(root / "manifest.jsonl")
    repo = resolve_config_path(c, c.require("paths.dinov3_repo"))
    checkpoint = resolve_config_path(c, c.require("paths.dinov3_weights"))
    spec = spec_from_config(c)
    checkpoint_sha256 = read_json(root / "freeze_receipt.json")["dinov3_checkpoint_sha256"]
    model = load_official_model(repo, checkpoint, spec, str(c.get("runtime.device", "cuda")))
    batch_size = int(c.get("runtime.batch_size", 8))
    if batch_size < 1:
        raise ValueError("runtime.batch_size must be positive")
    feature_root = resolve_config_path(c, c.require("paths.features"))
    feature_root.mkdir(parents=True, exist_ok=True)
    done = 0
    failed = 0
    for start in range(0, len(rows), batch_size):
        chunk = [normalize_record(x) for x in rows[start : start + batch_size] if not normalize_record(x).get("exclusion_reason")]
        images, valid = [], []
        for row in chunk:
            if not row.get("crop_path"):
                failed += 1
                continue
            artifact = feature_artifact_path(feature_root, row["sample_id"])
            ok, _ = reusable_feature(
                artifact,
                sample_id=row["sample_id"],
                config_digest_value=config_digest(c),
                tokens=spec.expected_tokens,
                dimension=spec.expected_dimension,
                crop_path=row["crop_path"],
                required_metadata={"r0_feature_role": "paired_frame", "checkpoint_sha256": checkpoint_sha256},
            )
            if ok:
                done += 1
                continue
            images.append(np.load(row["crop_path"], allow_pickle=False))
            valid.append((row, artifact))
        if valid:
            tokens = extract_patch_tokens(model, preprocess_rgb(images, c), spec)
            for (row, artifact), value in zip(valid, tokens):
                save_features(
                    artifact,
                    value[None, ...],
                    metadata={
                        "sample_id": row["sample_id"],
                        "crop_path": row["crop_path"],
                        "crop_sha256": sha256_file(row["crop_path"]),
                        "r0_feature_role": "paired_frame",
                        "checkpoint_sha256": checkpoint_sha256,
                    },
                    config=c,
                )
                done += 1
        if int(c.get("runtime.progress_every", 250)) and (start + batch_size >= len(rows) or start % max(1, int(c.get("runtime.progress_every", 250))) == 0):
            print(f"[r0 features] {min(start + batch_size, len(rows))}/{len(rows)}", flush=True)
    write_json(root / "feature_summary.json", {"status": "completed", "features": done, "failed": failed, "config_digest": config_digest(c)})
    return {"status": "completed", "features": done, "failed": failed}


def _landmarker(asset, config):
    return create_landmarker(
        asset,
        num_faces=int(config.require("mediapipe.num_faces")),
        min_face_detection_confidence=float(config.require("mediapipe.min_face_detection_confidence")),
        min_face_presence_confidence=float(config.require("mediapipe.min_face_presence_confidence")),
        min_tracking_confidence=float(config.require("mediapipe.min_tracking_confidence")),
    )


def _context_derangement(seed: int, tokens: int = 784) -> np.ndarray:
    """Fixed, well-mixed permutation without fixed token indices."""
    rng = np.random.default_rng(seed)
    while True:
        perm = rng.permutation(tokens)
        if not np.any(perm == np.arange(tokens)):
            return perm


def _load_verified_feature(row, feature_root: Path, config, checkpoint_sha256: str):
    artifact = feature_artifact_path(feature_root, row["sample_id"])
    ok, _ = reusable_feature(
        artifact, sample_id=row["sample_id"], config_digest_value=config_digest(config),
        tokens=int(config.require("features.expected_tokens_per_frame")),
        dimension=int(config.require("features.expected_dimension")),
        crop_path=row["crop_path"],
        required_metadata={"r0_feature_role": "paired_frame", "checkpoint_sha256": checkpoint_sha256},
    )
    if not ok:
        raise R0ProvenanceError(f"R0 feature provenance mismatch: {row['sample_id']}")
    value, _ = load_features(artifact)
    return value[0]


def _verify_mediapipe_asset(config):
    asset = resolve_config_path(config, config.require("mediapipe.asset_path"))
    digest = sha256_file(asset)
    expected = str(config.require("mediapipe.asset_sha256"))
    if expected in {"", "REQUIRED_ON_SERVER"}:
        raise ValueError(f"Set the official MediaPipe asset SHA-256 in the frozen config before evaluate: {asset} -> {digest}")
    if digest.lower() != expected.lower():
        raise ValueError(f"MediaPipe asset SHA-256 mismatch: expected {expected}, got {digest}")
    return asset, digest


def _save_overlay(path, image, regions):
    import cv2
    out = np.asarray(image).copy()
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    for mask, color in zip((regions.mouth_core, regions.perioral_ring, regions.cheek_left, regions.cheek_right), colors):
        out[mask] = ((0.55 * out[mask]) + (0.45 * np.asarray(color))).astype(np.uint8)
    if not cv2.imwrite(str(path), cv2.cvtColor(out, cv2.COLOR_RGB2BGR)):
        raise OSError(f"Could not write R0 overlay: {path}")


def _record_reason(counter: Counter, reason: str) -> None:
    counter[reason] += 1


def _crop_geometry_match(fake: dict, real: dict, mode: str) -> tuple[bool, str]:
    """Check paired crop geometry in an explicitly selected coordinate space.

    ``exact_pixel_bbox`` preserves the original strict audit. 
    ``shared_source_space`` is the corrected protocol: fake and real may have
    different raw pixel bboxes after resolution mapping, but must carry the
    same source-space crop transform and source dimensions.
    """
    if mode == "exact_pixel_bbox":
        fields = ("crop_left", "crop_top", "crop_right", "crop_bottom", "crop_scale")
        ok = all(fake.get(field) is not None and fake.get(field) == real.get(field) for field in fields)
        return ok, "raw_target_pixel_bbox"
    if mode == "shared_source_space":
        fields = (
            "crop_source_left", "crop_source_top", "crop_source_right",
            "crop_source_bottom", "crop_source_width", "crop_source_height",
            "crop_scale",
        )
        missing = [field for field in fields if fake.get(field) is None or real.get(field) is None]
        if missing:
            return False, "missing_source_space_provenance:" + ",".join(missing)
        ok = all(fake.get(field) == real.get(field) for field in fields)
        return ok, "shared_source_space"
    raise ValueError(f"Unsupported face_crop.geometry_match_mode: {mode!r}")


def _full_video_coverage(frame_rows: list[dict], pooled_rows: list[dict], config) -> bool:
    """Require every pre-registered fraction once and all frames geometry-valid."""
    expected = sorted(float(value) for value in config.require("sampling.frame_fractions"))
    observed = sorted(float(row["frame_fraction"]) for row in frame_rows)
    return observed == expected and len(pooled_rows) >= int(config.require("quality.min_evaluable_frames_per_video"))


def evaluate(config_path):
    c, root = ctx(config_path)
    _assert_frozen(c, root)
    rows = [normalize_record(x) for x in read_manifest(root / "manifest.jsonl")]
    pairs = read_jsonl(root / "pairing_manifest.jsonl")
    byid = {row["sample_id"]: row for row in rows}
    feature_root = resolve_config_path(c, c.require("paths.features"))
    out = root / "r0_artifacts"
    out.mkdir(exist_ok=True)
    (out / "regions").mkdir(exist_ok=True)
    (out / "overlays").mkdir(exist_ok=True)
    asset_path, asset_digest = _verify_mediapipe_asset(c)
    mp, landmarker = _landmarker(asset_path, c)
    receipt = read_json(root / "freeze_receipt.json")
    perm = _context_derangement(int(c.require("protocol.seed")))
    import hashlib
    permutation_sha256 = hashlib.sha256(perm.astype("<i8").tobytes()).hexdigest()
    write_json(root / "input_provenance.json", {
        "config_digest": config_digest(c),
        "split": "train",
        "manipulation": "Face2Face",
        "compression": "c23",
        "dino_architecture": c.get("backbone.architecture"),
        "dino_block_one_based": c.get("backbone.block_one_based"),
        "dino_expected_shape": [784, 768],
        "mediapipe_package_version": getattr(mp, "__version__", "unknown"),
        "mediapipe_solution": "Tasks Face Landmarker",
        "mediapipe_asset": str(asset_path),
        "mediapipe_asset_sha256": asset_digest,
        "runner_file_sha256": sha256_file(Path(__file__)),
        "source_sha256": receipt["source_sha256"],
        "dinov3_checkpoint_sha256": receipt["dinov3_checkpoint_sha256"],
        "context_permutation_seed": int(c.require("protocol.seed")),
        "context_permutation_sha256": permutation_sha256,
        "context_permutation_fixed_points": int(np.count_nonzero(perm == np.arange(784))),
        "official_test_locked": True,
    })
    region_rows, metrics = [], []
    reason_counts = Counter()
    error_types = Counter()
    for idx, pair in enumerate(pairs, 1):
        fake = byid.get(pair["fake_sample_id"])
        real = byid.get(pair["real_sample_id"])
        base = {
            **pair,
            "video_id": fake["video_name"] if fake else None,
            "video_key": f"{fake['compression']}:{fake['video_name']}" if fake else None,
            "generated": True,
            "evaluable": False,
        }
        if not fake or not real:
            base["reason"] = "missing_pair_crop_or_mask"
            _record_reason(reason_counts, base["reason"])
            metrics.append(base)
            continue
        if pair.get("pairing_status") != "strict_frame_match":
            base["reason"] = "frame_pair_mismatch"
            _record_reason(reason_counts, base["reason"])
            metrics.append(base)
            continue
        geometry_mode = str(c.get("face_crop.geometry_match_mode", "exact_pixel_bbox"))
        base["crop_geometry_mode"] = geometry_mode
        base["crop_geometry_match"], base["crop_geometry_coordinate_space"] = _crop_geometry_match(fake, real, geometry_mode)
        if not base["crop_geometry_match"]:
            base["reason"] = "crop_geometry_mismatch"
            _record_reason(reason_counts, base["reason"])
            metrics.append(base)
            continue
        if fake.get("exclusion_reason") or real.get("exclusion_reason") or not fake.get("mask_cache_path") or not fake.get("crop_path") or not real.get("crop_path"):
            base["reason"] = "missing_pair_crop_or_mask"
            _record_reason(reason_counts, base["reason"])
            metrics.append(base)
            continue
        try:
            real_image = np.load(real["crop_path"], allow_pickle=False)
            fake_image = np.load(fake["crop_path"], allow_pickle=False)
            mask = np.load(fake["mask_cache_path"], allow_pickle=False).astype(np.uint8)
            if real_image.shape != (448, 448, 3) or fake_image.shape != (448, 448, 3):
                raise ValueError("expected paired 448x448 RGB crop arrays")
            if mask.shape != (448, 448):
                raise ValueError(f"expected 448x448 mask, got {mask.shape}")
            real_lm, real_status = detect_landmarks(mp, landmarker, real_image)
            fake_lm, fake_status = detect_landmarks(mp, landmarker, fake_image)
            base.update({"target_real_landmark_status": real_status, "fake_only_landmark_status": fake_status})
            if real_lm is None:
                base["reason"] = "target_real_no_face"
                _record_reason(reason_counts, base["reason"])
                metrics.append(base)
                continue
            if real_status.get("status") != "ok":
                base["reason"] = "target_real_multiple_faces"
                _record_reason(reason_counts, base["reason"])
                metrics.append(base)
                continue
            regions = build_semantic_regions(real_lm, size=448, token_grid=28, ring_radius=int(c.require("regions.perioral_ring_radius")))
            fake_geometry_iou = {}
            fake_regions = None
            if fake_lm is not None:
                try:
                    fake_regions = build_semantic_regions(fake_lm, size=448, token_grid=28, ring_radius=int(c.require("regions.perioral_ring_radius")))
                    for name in ("mouth_core", "perioral_ring", "cheek_left", "cheek_right", "face"):
                        a, b = getattr(regions, name), getattr(fake_regions, name)
                        fake_geometry_iou[name] = float((a & b).sum() / max(1, (a | b).sum()))
                except Exception as exc:
                    base["fake_only_geometry_error"] = f"{type(exc).__name__}: {exc}"
            rid = f"{idx:06d}"
            np.savez_compressed(out / "regions" / f"{rid}.npz", **{
                name: getattr(regions, name)
                for name in ("mouth_core", "perioral_ring", "cheek_left", "cheek_right", "face", "valid", "token_mouth_core", "token_perioral_ring", "token_cheek_left", "token_cheek_right", "token_face", "token_valid")
            })
            region_rows.append({
                "pair_index": idx,
                "fake_sample_id": fake["sample_id"],
                "real_sample_id": real["sample_id"],
                "partition": pair["partition"],
                "source_group": pair["source_group"],
                "component_id": pair["component_id"],
                "fake_only_geometry_iou": fake_geometry_iou,
                "fake_only_geometry_error": base.get("fake_only_geometry_error"),
                "target_real_landmark_status": real_status,
                "fake_only_landmark_status": fake_status,
                "asset_sha256": asset_digest,
                "size": 448,
                "token_grid": [28, 28],
                "ring_radius": int(c.require("regions.perioral_ring_radius")),
                "token_coverage": {
                    "mouth_core": float(regions.token_mouth_core.mean()),
                    "perioral_ring": float(regions.token_perioral_ring.mean()),
                    "cheek_left": float(regions.token_cheek_left.mean()),
                    "cheek_right": float(regions.token_cheek_right.mean()),
                    "face": float(regions.token_face.mean()),
                },
            })
            _save_overlay(out / "overlays" / f"{rid}_real.png", real_image, regions)
            if fake_regions is not None:
                _save_overlay(out / "overlays" / f"{rid}_fake_only.png", fake_image, fake_regions)
            real_feat = _load_verified_feature(real, feature_root, c, receipt["dinov3_checkpoint_sha256"])
            fake_feat = _load_verified_feature(fake, feature_root, c, receipt["dinov3_checkpoint_sha256"])
            rel, rel_domain, audit = relation_score(fake_feat, real_feat, regions)
            local, local_domain = local_difference_score(fake_feat, real_feat, regions)
            template, template_domain = position_template_score(regions)
            broken, broken_domain = broken_context_relation_score(fake_feat, real_feat, regions, permutation=perm)
            if not rel_domain.any() or not broken_domain.any():
                base["reason"] = "empty_relation_domain_or_context"
                _record_reason(reason_counts, base["reason"])
                metrics.append(base)
                continue
            maps = {"relation": flatten_map(rel), "local_difference": flatten_map(local), "position_template": flatten_map(template), "broken_context": flatten_map(broken)}
            base["region_artifact"] = str(out / "regions" / f"{rid}.npz")
            base["mask_path"] = fake["mask_cache_path"]
            base["score_paths"] = {}
            for name, value in maps.items():
                map_path = out / f"{rid}_{name}.npy"
                np.save(map_path, value.astype(np.float32))
                base["score_paths"][name] = str(map_path)
            # Every scorer is evaluated on the same full-face valid pixel domain.
            # Relation/template maps may be zero outside their pre-defined mouth
            # and perioral support, but they are not allowed to change the AP
            # denominator relative to the local-difference baseline.
            for name, score in (("relation", maps["relation"]), ("local_difference", maps["local_difference"]), ("position_template", maps["position_template"]), ("broken_context", maps["broken_context"])):
                masked_score = np.asarray(score)[regions.valid]
                masked_truth = mask[regions.valid]
                ap = average_precision(masked_score, masked_truth)
                prev = prevalence(masked_truth)
                base[f"{name}_ap"] = ap
                base[f"{name}_prevalence"] = prev
                base[f"{name}_ap_minus_prevalence"] = float(ap - prev) if ap is not None else None
            semantic = regions.mouth_core | regions.perioral_ring
            total_mask = int(mask.sum())
            base.update({
                "geometry_valid": True,
                "ap_defined": all(base[f"{name}_ap"] is not None for name in SCORES),
                "evaluable": all(base[f"{name}_ap"] is not None for name in SCORES),
                "region_pixels": int(semantic.sum()),
                "face_pixels": int(regions.face.sum()),
                "mask_fraction_in_mouth_perioral": float(mask[semantic].sum() / total_mask) if total_mask else None,
                "mask_positive_pixels": total_mask,
                "valid_mask_positive_pixels": int(mask[regions.valid].sum()),
                "valid_mask_pixels": int(regions.valid.sum()),
                "empty_valid_mask": not bool(mask[regions.valid].any()),
                "full_valid_mask": bool(mask[regions.valid].all()),
                "fake_only_geometry_iou": fake_geometry_iou,
                "target_real_landmark_status": real_status,
                "fake_only_landmark_status": fake_status,
                "relation_audit": audit,
                "relation_domain_token_count": int(rel_domain.sum()),
                "local_domain_token_count": int(local_domain.sum()),
                "template_domain_token_count": int(template_domain.sum()),
                "broken_domain_token_count": int(broken_domain.sum()),
            })
        except R0ProvenanceError:
            raise
        except Exception as exc:
            error_type = type(exc).__name__
            error_types[error_type] += 1
            base["reason"] = f"evaluation_error:{error_type}"
            base["error_message"] = str(exc)
            _record_reason(reason_counts, base["reason"])
        metrics.append(base)
        progress = int(c.get("runtime.progress_every", 250))
        if progress and idx % progress == 0:
            print(f"[r0 evaluate] {idx}/{len(pairs)}", flush=True)
    write_jsonl(root / "per_frame_metrics.jsonl", metrics)
    write_jsonl(root / "semantic_region_manifest.jsonl", region_rows)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in metrics:
        if row.get("video_key") is not None:
            grouped[row["video_key"]].append(row)
    video_rows = []
    for video_key, frame_rows in sorted(grouped.items()):
        # A single all-zero frame has undefined frame AP, but its pixels must
        # still contribute to pooled video AP if the other frames contain masks.
        pooled = [row for row in frame_rows if row.get("geometry_valid") and row.get("score_paths")]
        coverage_ok = _full_video_coverage(frame_rows, pooled, c)
        video = {
            "video_key": video_key,
            "video_id": frame_rows[0].get("video_id"),
            "target_id": frame_rows[0].get("target_id"),
            "source_id": frame_rows[0].get("source_id"),
            "source_group": frame_rows[0].get("source_group"),
            "component_id": frame_rows[0].get("component_id"),
            "partition": frame_rows[0].get("partition"),
            "generated_frame_count": len(frame_rows),
            "evaluable_frame_count": sum(bool(row.get("evaluable")) for row in frame_rows),
            "pooled_frame_count": len(pooled),
            "frame_coverage_ok": coverage_ok,
            "target_real_landmark_failures": sum(row.get("target_real_landmark_status", {}).get("status") != "ok" for row in frame_rows),
            "fake_only_landmark_failures": sum(row.get("fake_only_landmark_status") is not None and row.get("fake_only_landmark_status", {}).get("status") not in {"ok", "multi_face_first_used"} for row in frame_rows),
            "undefined_ap_frame_count": sum(bool(row.get("geometry_valid")) and row.get("relation_ap") is None for row in frame_rows),
            "empty_valid_mask_frame_count": sum(bool(row.get("empty_valid_mask")) for row in frame_rows),
            "full_valid_mask_frame_count": sum(bool(row.get("full_valid_mask")) for row in frame_rows),
        }
        for name in SCORES:
            scores, truths = [], []
            for row in pooled if coverage_ok else ():
                region = np.load(row["region_artifact"], allow_pickle=False)
                valid = region["valid"]
                scores.append(np.load(row["score_paths"][name], allow_pickle=False)[valid])
                truths.append(np.load(row["mask_path"], allow_pickle=False)[valid])
            if scores:
                score = np.concatenate(scores)
                truth = np.concatenate(truths)
                ap = average_precision(score, truth)
                prev = prevalence(truth)
                video[f"{name}_ap"] = ap
                video[f"{name}_prevalence"] = prev
                video[f"{name}_ap_minus_prevalence"] = float(ap - prev) if ap is not None else None
            else:
                video[f"{name}_ap"] = video[f"{name}_prevalence"] = video[f"{name}_ap_minus_prevalence"] = None
        video["evaluable"] = coverage_ok and all(video[f"{name}_ap"] is not None for name in SCORES)
        video_rows.append(video)
    write_jsonl(root / "per_video_metrics.jsonl", video_rows)
    strict_count = sum(pair.get("pairing_status") == "strict_frame_match" for pair in pairs)
    exception_count = sum(str(row.get("reason", "")).startswith("evaluation_error:") for row in metrics)
    exception_fraction = exception_count / max(1, strict_count)
    pair_fraction = strict_count / max(1, len(pairs))
    landmark_attempts = sum(row.get("target_real_landmark_status") is not None for row in metrics)
    landmark_success = sum(row.get("target_real_landmark_status", {}).get("status") == "ok" for row in metrics)
    landmark_fraction = landmark_success / max(1, landmark_attempts)
    quality = {
        "generated_pair_rows": len(pairs),
        "strict_frame_pair_rows": strict_count,
        "frame_pair_mismatch_rows": sum(pair.get("pairing_status") == "index_mismatch" for pair in pairs),
        "missing_pair_rows": sum(pair.get("pairing_status") == "missing_real_pair" for pair in pairs),
        "reason_counts": dict(reason_counts),
        "evaluation_error_types": dict(error_types),
        "evaluation_exception_rows": exception_count,
        "evaluation_exception_fraction_of_all_pairs": exception_count / max(1, len(pairs)),
        "strict_pair_fraction": pair_fraction,
        "minimum_strict_pair_fraction": float(c.require("quality.min_strict_pair_fraction")),
        "landmark_attempt_rows": landmark_attempts,
        "target_real_single_face_rows": landmark_success,
        "target_real_single_face_fraction": landmark_fraction,
        "minimum_target_real_single_face_fraction": float(c.require("quality.min_target_real_single_face_fraction")),
        "geometry_valid_frame_rows": sum(bool(row.get("geometry_valid")) for row in metrics),
        "undefined_ap_frame_rows": sum(bool(row.get("geometry_valid")) and row.get("relation_ap") is None for row in metrics),
        "video_frame_coverage": dict(Counter(f"{row['pooled_frame_count']}/{row['generated_frame_count']}" for row in video_rows)),
        "insufficient_coverage_video_rows": sum(not row["frame_coverage_ok"] for row in video_rows),
        "minimum_evaluable_frames_per_video": int(c.require("quality.min_evaluable_frames_per_video")),
        "empty_valid_mask_frame_rows": sum(bool(row.get("empty_valid_mask")) for row in metrics),
        "full_valid_mask_frame_rows": sum(bool(row.get("full_valid_mask")) for row in metrics),
        "undefined_ap_video_rows": sum(row.get("relation_ap") is None for row in video_rows),
        "crop_geometry_mismatch_rows": reason_counts.get("crop_geometry_mismatch", 0),
        "target_real_no_face": reason_counts.get("target_real_no_face", 0),
        "target_real_multiple_faces": reason_counts.get("target_real_multiple_faces", 0),
        "fake_only_no_face": sum(row.get("fake_only_landmark_status", {}).get("status") == "no_face" for row in metrics),
        "fake_only_multiple_faces": sum(row.get("fake_only_landmark_status", {}).get("status") == "multi_face_first_used" for row in metrics),
        "evaluable_frame_rows": sum(bool(row.get("evaluable")) for row in metrics),
        "evaluable_pair_fraction": sum(bool(row.get("evaluable")) for row in metrics) / max(1, len(pairs)),
        "minimum_evaluable_pair_fraction": float(c.require("quality.min_evaluable_pair_fraction")),
        "evaluable_video_rows": sum(bool(row.get("evaluable")) for row in video_rows),
        "unexpected_exception_fraction": exception_fraction,
        "max_evaluation_exception_fraction": float(c.get("quality.max_evaluation_exception_fraction", 0.05)),
        "official_test_locked": True,
    }
    write_json(root / "quality_audit.json", quality)
    if exception_fraction > quality["max_evaluation_exception_fraction"]:
        raise RuntimeError(f"R0 evaluation exception fraction {exception_fraction:.4f} exceeds limit {quality['max_evaluation_exception_fraction']:.4f}")
    return {"status": "completed", **quality}


def _metric_summary(rows: list[dict], name: str) -> dict:
    values = [row[f"{name}_ap"] for row in rows if row.get(f"{name}_ap") is not None]
    prevalence_values = [row[f"{name}_prevalence"] for row in rows if row.get(f"{name}_prevalence") is not None]
    deltas = [row[f"{name}_ap_minus_prevalence"] for row in rows if row.get(f"{name}_ap_minus_prevalence") is not None]
    return {
        "mean_ap": float(np.mean(values)) if values else None,
        "n_ap": len(values),
        "mean_prevalence": float(np.mean(prevalence_values)) if prevalence_values else None,
        "mean_ap_minus_prevalence": float(np.mean(deltas)) if deltas else None,
    }


def _bootstrap_or_unavailable(rows, first: str, second: str, c):
    try:
        return paired_group_bootstrap(
            rows,
            f"{first}_ap",
            f"{second}_ap",
            replicates=int(c.require("statistics.bootstrap_replicates")),
            seed=int(c.require("protocol.bootstrap_seed")),
            group_key="component_id",
        )
    except ValueError as exc:
        return {"available": False, "reason": str(exc), "groups": len({row.get("component_id") for row in rows}), "group_key": "component_id"}


def finalize(config_path):
    c, root = ctx(config_path)
    _assert_frozen(c, root)
    rows = read_jsonl(root / "per_video_metrics.jsonl")
    frame_rows = read_jsonl(root / "per_frame_metrics.jsonl")
    pair_rows = read_jsonl(root / "pairing_manifest.jsonl")
    usable = [row for row in rows if row.get("evaluable")]
    partitions = {partition: [row for row in usable if row.get("partition") == partition] for partition in ("exploration", "holdout", "all_train")}
    summary = {
        "status": "completed",
        "protocol": "R0-semantic-relation-v1",
        "official_test_locked": True,
        "oracle_paired_real": True,
        "secondary_domain_status": "planned_not_implemented",
        "generated_video_rows": len(rows),
        "evaluable_video_rows": len(usable),
        "quality_audit": read_json(root / "quality_audit.json") if (root / "quality_audit.json").is_file() else None,
        "partitions": {},
    }
    for partition, partition_rows in partitions.items():
        if not partition_rows:
            continue
        section = {
            "video_rows": len(partition_rows),
            "source_groups": len({row["source_group"] for row in partition_rows}),
            "independent_source_components": len({row["component_id"] for row in partition_rows}),
            "metrics": {name: _metric_summary(partition_rows, name) for name in SCORES},
            "bootstraps": {},
        }
        for first, second in (("relation", "local_difference"), ("relation", "position_template"), ("relation", "broken_context")):
            section["bootstraps"][f"{first}_minus_{second}"] = _bootstrap_or_unavailable(partition_rows, first, second, c)
        summary["partitions"][partition] = section
    holdout = partitions["holdout"]
    holdout_summary = summary["partitions"].get("holdout", {})
    b_local = holdout_summary.get("bootstraps", {}).get("relation_minus_local_difference", {})
    b_template = holdout_summary.get("bootstraps", {}).get("relation_minus_position_template", {})
    b_broken = holdout_summary.get("bootstraps", {}).get("relation_minus_broken_context", {})
    quality = summary.get("quality_audit")
    if not quality:
        raise ValueError("quality_audit.json missing; evaluate must complete before finalize")
    quality_ok = (
        quality["strict_frame_pair_rows"] > 0
        and quality["landmark_attempt_rows"] > 0
        and quality["unexpected_exception_fraction"] <= quality["max_evaluation_exception_fraction"]
        and quality["strict_pair_fraction"] >= quality["minimum_strict_pair_fraction"]
        and quality["evaluable_pair_fraction"] >= quality["minimum_evaluable_pair_fraction"]
        and quality["target_real_single_face_fraction"] >= quality["minimum_target_real_single_face_fraction"]
    )
    holdout_pairs = [row for row in pair_rows if row.get("partition") == "holdout"]
    holdout_frames = [row for row in frame_rows if row.get("partition") == "holdout"]
    holdout_strict = sum(bool(row.get("strict_frame_pair")) for row in holdout_pairs)
    holdout_attempts = sum(row.get("target_real_landmark_status") is not None for row in holdout_frames)
    holdout_single_faces = sum(row.get("target_real_landmark_status", {}).get("status") == "ok" for row in holdout_frames)
    holdout_exceptions = sum(str(row.get("reason", "")).startswith("evaluation_error:") for row in holdout_frames)
    holdout_quality = {
        "generated_pairs": len(holdout_pairs),
        "strict_pairs": holdout_strict,
        "strict_pair_fraction": holdout_strict / max(1, len(holdout_pairs)),
        "target_real_landmark_attempts": holdout_attempts,
        "target_real_single_faces": holdout_single_faces,
        "target_real_single_face_fraction": holdout_single_faces / max(1, holdout_attempts),
        "evaluable_pairs": sum(bool(row.get("evaluable")) for row in holdout_frames),
        "evaluable_pair_fraction": sum(bool(row.get("evaluable")) for row in holdout_frames) / max(1, len(holdout_pairs)),
        "evaluation_exceptions": holdout_exceptions,
        "evaluation_exception_fraction": holdout_exceptions / max(1, holdout_strict),
    }
    holdout_quality_ok = (
        holdout_strict > 0 and holdout_attempts > 0
        and holdout_quality["strict_pair_fraction"] >= quality["minimum_strict_pair_fraction"]
        and holdout_quality["evaluable_pair_fraction"] >= quality["minimum_evaluable_pair_fraction"]
        and holdout_quality["target_real_single_face_fraction"] >= quality["minimum_target_real_single_face_fraction"]
        and holdout_quality["evaluation_exception_fraction"] <= quality["max_evaluation_exception_fraction"]
    )
    summary["holdout_quality"] = holdout_quality
    quality_ok = quality_ok and holdout_quality_ok
    ci_positive = all(item.get("available") is not False and item.get("ci95", [float("nan")])[0] > 0 for item in (b_local, b_template))
    broken_changed = b_broken.get("available") is not False and b_broken.get("ci95", [float("nan")])[0] > 0
    required_components = int(c.require("quality.min_holdout_independent_components"))
    holdout_available = (len(holdout) > 0 and b_local.get("groups", 0) >= required_components
                         and b_template.get("groups", 0) >= required_components
                         and b_broken.get("groups", 0) >= required_components)
    if holdout_available and quality_ok and ci_positive and broken_changed:
        decision = "FURTHER_RESEARCH_RELATION_SIGNAL"
    elif not holdout_available or not quality_ok:
        decision = "INCONCLUSIVE_QUALITY_OR_INDEPENDENCE"
    else:
        decision = "NO_LOCATION_INFORMATION_BEYOND_BASELINES"
    summary["decision"] = decision
    summary["decision_basis"] = {
        "decision_partition": "holdout",
        "holdout_available": holdout_available,
        "required_holdout_independent_components": required_components,
        "holdout_quality": holdout_quality,
        "quality_ok": quality_ok,
        "relation_exceeds_local_ci_lower": b_local.get("ci95", [None])[0] if b_local else None,
        "relation_exceeds_template_ci_lower": b_template.get("ci95", [None])[0] if b_template else None,
        "relation_exceeds_broken_context_ci_lower": b_broken.get("ci95", [None])[0] if b_broken else None,
    }
    write_json(root / "summary.json", summary)
    write_json(root / "decision.json", {
        "decision": decision,
        "decision_partition": "holdout",
        "official_test_locked": True,
        "oracle_not_deployable": True,
        "quality_ok": quality_ok,
        "decision_basis": summary["decision_basis"],
        "reasons": "train-only paired oracle; no automatic official-test unlock",
    })
    lines = [
        "# R0 semantic relation oracle diagnostic",
        "",
        f"Decision: `{decision}`",
        "",
        "This is an oracle analysis using a strictly same-frame paired target-real image; it is not a deployable detector.",
        "The decision is based on the train-only source-component holdout, not on exploration rows.",
        "",
        "## Partition summaries",
    ]
    for partition, section in summary["partitions"].items():
        lines.extend(["", f"### {partition}", f"- videos: {section['video_rows']}", f"- undirected source pairs: {section['source_groups']}", f"- independent source components: {section['independent_source_components']}"])
        for name in SCORES:
            metric = section["metrics"][name]
            lines.append(f"- {name}: mean AP={metric['mean_ap']} (n={metric['n_ap']}), mean AP - prevalence={metric['mean_ap_minus_prevalence']}")
        for name, boot in section["bootstraps"].items():
            lines.append(f"- {name}: point={boot.get('point')}, CI95={boot.get('ci95')}, groups={boot.get('groups')}")
    lines.extend(["", "## Quality and scope", f"- holdout quality: {holdout_quality}; acceptable: {holdout_quality_ok}", f"- minimum independent holdout components for a positive decision: {required_components}", f"- generated/evaluable frame rows: {quality['generated_pair_rows']}/{quality['evaluable_frame_rows']}", f"- generated/evaluable video rows: {summary['generated_video_rows']}/{summary['evaluable_video_rows']}", f"- strict pairing: {quality['strict_frame_pair_rows']} ({quality['strict_pair_fraction']:.3f}); crop mismatches: {quality['crop_geometry_mismatch_rows']}", f"- target-real single face: {quality['target_real_single_face_rows']}/{quality['landmark_attempt_rows']}; no face: {quality['target_real_no_face']}", f"- empty/full valid masks: {quality['empty_valid_mask_frame_rows']}/{quality['full_valid_mask_frame_rows']}; undefined frame/video AP: {quality['undefined_ap_frame_rows']}/{quality['undefined_ap_video_rows']}; video frame coverage: {quality['video_frame_coverage']}", f"- evaluation exception types: {quality['evaluation_error_types']}", "- mask coverage of the scorable mouth/perioral region is in per_frame_metrics.jsonl; scores outside that region are zero (bilinear token interpolation can soften its boundary).", "- official test: locked; not read"])
    (root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"status": "completed", "decision": decision, "reports": ["summary.md", "summary.json", "decision.json"], "official_test_locked": True}


def run_stage(path, stage):
    return {"prepare": prepare, "crops": crops, "features": features, "evaluate": evaluate, "finalize": finalize}[stage](path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", required=True, choices=("prepare", "crops", "features", "evaluate", "finalize"))
    args = parser.parse_args()
    print(json.dumps(run_stage(args.config, args.stage), ensure_ascii=False, indent=2), flush=True)
