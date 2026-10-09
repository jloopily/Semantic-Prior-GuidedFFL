"""Preregistered, validation-only limited M1 static localization pilot.

The module deliberately lives beside (not inside) the v5.5 gates.  It never
opens the official test split, never trains a model, and never changes the
frozen scorer/threshold/mask definitions after the freeze receipt is written.
It reuses exact T0 crops/features/gallery artifacts whenever their provenance
matches and keeps all M1 artifacts in an independent result namespace.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .config import Config, config_digest, load_config, require_protocol, resolve_config_path
from .crops import cache_manifest_crops
from .features import feature_artifact_path, load_features, save_features
from .gallery import load_gallery
from .geometry import fractional_frame_index
from .heatmap import upsample_score_map
from .io import atomic_write_text, read_json, read_jsonl, sha256_file, write_json, write_jsonl
from .knn import ExactL2Index, numpy_exact_l2
from .manifest import normalize_record, read_manifest, write_manifest
from .scoring import cir_lid, arithmetic_multi_neighbor

SCORERS = ("r_k", "mean_r", "C2", "CIR_LID")
PRIMARY = "r_k"
SECONDARY = ("mean_r", "C2")
MANIPULATIONS = ("Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures")
COMPRESSIONS = ("c23", "c40")
GRID = (28, 28)
IMAGE_SIZE = (448, 448)
T3 = "T3_frozen"


def _hash_id(value: str) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:32]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_digest(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _git_state(path: Path) -> dict[str, Any]:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=path, check=True,
                                capture_output=True, text=True).stdout.strip()
        status = subprocess.run(["git", "status", "--short"], cwd=path, check=True,
                                capture_output=True, text=True).stdout.splitlines()
        return {"commit": commit, "status": status, "available": True}
    except Exception as exc:  # pragma: no cover - server checkout may be a copied directory
        return {"commit": None, "status": [], "available": False, "error": f"{type(exc).__name__}: {exc}"}


def _context(path: str | Path) -> tuple[Config, Config, Path]:
    m1 = load_config(path)
    base = load_config(resolve_config_path(m1, m1.require("base_config")))
    require_protocol(base)
    expected = {
        "seed": 20260927,
        "protocol.namespace": "m1_static_geometry_pilot_v1",
        "protocol.version": "M1-limited-v1.0",
        "protocol.query_budget": T3,
        "protocol.query_positions": [0.25, 0.50, 0.75],
        "protocol.gallery_real_only": True,
        "protocol.gallery_train_videos": 64,
        "protocol.gallery_tokens": 50176,
        "protocol.k": 20,
        "protocol.distance": "euclidean_float32_exact_indexflatl2",
        "protocol.score_map_grid": [28, 28],
        "protocol.score_map_size": [448, 448],
        "protocol.score_map_align_corners": False,
        "protocol.score_map_postprocess": "none",
        "protocol.threshold_quantile": 0.99,
        "protocol.threshold_method": "linear",
        "protocol.threshold_comparison": "greater_than",
        "protocol.bootstrap_replicates": 2000,
        "protocol.bootstrap_confidence_level": 0.95,
        "protocol.bootstrap_seed": 20260927,
        "scorers.primary": PRIMARY,
        "scorers.fixed_secondary": ["mean_r", "C2"],
        "scorers.historical": "CIR_LID",
        "compressions": list(COMPRESSIONS),
        "manipulations": list(MANIPULATIONS),
        "official_test_locked": True,
        "validation_previously_used_in_t0": True,
    }
    mismatch = [f"{key}={m1.get(key)!r} (expected {value!r})"
                for key, value in expected.items() if m1.get(key) != value]
    if mismatch:
        raise ValueError("M1 frozen configuration mismatch: " + "; ".join(mismatch))
    root = resolve_config_path(m1, m1.require("output_dir"))
    expected_root = resolve_config_path(base, base.require("paths.results")) / "m1_static_geometry_pilot_v1"
    if root != expected_root:
        raise ValueError(f"M1 output must be independent results/m1_static_geometry_pilot_v1: {root}")
    root.mkdir(parents=True, exist_ok=True)
    freeze_receipt = root / "freeze_receipt.json"
    preregistered = root / "preregistered_config.yaml"
    if freeze_receipt.is_file():
        if not preregistered.is_file():
            raise RuntimeError("Freeze receipt exists but preregistered M1 configuration is missing")
        frozen = load_config(preregistered)
        current_digest = config_digest(m1)
        if config_digest(frozen) != current_digest:
            raise RuntimeError("Current M1 configuration differs from the frozen preregistered configuration")
        receipt = read_json(freeze_receipt)
        if receipt.get("status") != "frozen":
            raise RuntimeError("M1 freeze receipt is not marked frozen")
        if receipt.get("config_digest") != current_digest:
            raise RuntimeError("Freeze receipt configuration digest does not match the current M1 configuration")
        if receipt.get("base_config_digest") != config_digest(base):
            raise RuntimeError("Base configuration differs from the configuration recorded at M1 freeze")
        if receipt.get("freeze_before_fake_metrics") is not True:
            raise RuntimeError("Freeze receipt does not certify freeze-before-metrics")
        if receipt.get("official_test_locked") is not True:
            raise RuntimeError("Freeze receipt does not certify that official test remains locked")
    return m1, base, root


def _m1_config(base: Config, root: Path) -> Config:
    data = copy.deepcopy(base.data)
    workspace = root / "workspace"
    for name in ("manifests", "crops", "masks", "features", "indices", "environment"):
        data["paths"][name] = str(workspace / name)
    data["paths"]["results"] = str(root)
    data.setdefault("experiment", {})["m1_namespace"] = "validation"
    return Config(data, base.source)


def _upstream(m1: Config) -> Path:
    path = resolve_config_path(m1, m1.require("upstream_t0_dir"))
    if not path.is_dir():
        raise FileNotFoundError(f"T0 result directory does not exist: {path}")
    return path


def _gallery_dir(base: Config, upstream: Path) -> Path:
    preferred = resolve_config_path(base, base.require("paths.indices"))
    if (preferred / "gallery.metadata.json").is_file():
        return preferred
    candidate = upstream / "gallery_workspace" / "indices"
    if (candidate / "gallery.metadata.json").is_file():
        return candidate
    raise FileNotFoundError("No verified gallery metadata in base indices or T0 gallery_workspace")


def _gallery_manifest_path(gallery_dir: Path, upstream: Path) -> Path:
    """Resolve the exact ordered gallery manifest recorded by the gallery."""
    metadata_path = gallery_dir / "gallery.metadata.json"
    metadata = read_json(metadata_path)
    recorded = metadata.get("manifest")
    candidates: list[Path] = []
    if recorded:
        recorded_path = Path(str(recorded))
        candidates.append(recorded_path)
        if not recorded_path.is_absolute():
            candidates.append(gallery_dir / recorded_path)
    candidates.extend([
        upstream / "gallery_workspace" / "manifests" / "gallery_manifest.jsonl",
        gallery_dir.parent.parent / "manifests" / "gallery_manifest.jsonl",
    ])
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("No ordered gallery manifest found for the verified gallery")


def _gallery_transform(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return the crop provenance fields that must agree for a pair.

    The fake mask is only meaningful on the exact crop used for the target-real
    comparison.  Keep this representation explicit in manifests instead of
    reducing it to a source-video/frame string.
    """
    return {
        "crop_source_video": None if row.get("crop_source_video") is None else str(row.get("crop_source_video")),
        "crop_source_frame": None if row.get("crop_source_frame") is None else int(row.get("crop_source_frame")),
        "crop_left": None if row.get("crop_left") is None else int(row.get("crop_left")),
        "crop_top": None if row.get("crop_top") is None else int(row.get("crop_top")),
        "crop_right": None if row.get("crop_right") is None else int(row.get("crop_right")),
        "crop_bottom": None if row.get("crop_bottom") is None else int(row.get("crop_bottom")),
        "crop_scale": None if row.get("crop_scale") is None else float(row.get("crop_scale")),
    }


def _pair_crop_transform_id(
    fake_transform: Mapping[str, Any],
    target_transform: Mapping[str, Any],
) -> str:
    """Digest the shared source crop specification for a fake/target pair.

    The source video/frame and crop scale are shared across the pair.  The
    pixel-space bbox is deliberately not included here: fake and target-real
    rows may come from different compression streams, so their decoded image
    sizes can differ and each row must retain its own mapped bbox.
    """
    return _json_digest({
        "fake_crop_source_video": fake_transform.get("crop_source_video"),
        "fake_crop_source_frame": fake_transform.get("crop_source_frame"),
        "fake_crop_scale": fake_transform.get("crop_scale"),
        "target_crop_source_video": target_transform.get("crop_source_video"),
        "target_crop_source_frame": target_transform.get("crop_source_frame"),
        "target_crop_scale": target_transform.get("crop_scale"),
    })


def _validate_pair_crop_transform(
    fake: Mapping[str, Any],
    target: Mapping[str, Any],
    *,
    require_materialized: bool,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    """Validate shared crop provenance without requiring equal mapped bboxes.

    Bboxes are pixel coordinates in each decoded compression stream.  They may
    therefore differ between fake and target-real rows even when both rows are
    correctly mapped from the same source video/frame and use the same crop
    scale.
    """
    fake_transform = _gallery_transform(fake)
    target_transform = _gallery_transform(target)
    for key in ("crop_source_video", "crop_source_frame"):
        if fake_transform[key] != target_transform[key]:
            raise ValueError(f"Pair crop transform mismatch for {fake['sample_id']}: {key}")
    if fake_transform["crop_scale"] != target_transform["crop_scale"]:
        raise ValueError(f"Pair crop scale mismatch for {fake['sample_id']}")
    bbox_keys = ("crop_left", "crop_top", "crop_right", "crop_bottom")
    fake_bbox = [fake_transform[key] for key in bbox_keys]
    target_bbox = [target_transform[key] for key in bbox_keys]
    if require_materialized and any(value is None for value in fake_bbox + target_bbox + [
        fake_transform["crop_scale"], target_transform["crop_scale"]
    ]):
        raise ValueError(f"Pair crop transform is not materialized for {fake['sample_id']}")
    if any(value is None for value in fake_bbox + target_bbox):
        status = "pending_materialization"
    else:
        status = "verified" if require_materialized else "verified_materialized"
    return fake_transform, target_transform, status


def _check_gallery(base: Config, gallery_dir: Path,
                   manifest_rows: Sequence[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    gallery = load_gallery(gallery_dir)
    if gallery.vectors.shape != (50176, 768) or gallery.vectors.dtype != np.float32:
        raise ValueError(f"M1 requires verified gallery [50176,768] float32, got {gallery.vectors.shape} {gallery.vectors.dtype}")
    metadata = dict(gallery.metadata)
    if metadata.get("ntotal") not in (None, 50176):
        raise ValueError("Gallery metadata ntotal mismatch")
    array_hash = sha256_file(gallery_dir / "gallery.npy")
    recorded_hash = metadata.get("gallery_array_sha256")
    if recorded_hash is not None and recorded_hash != array_hash:
        raise ValueError("Gallery array hash mismatch")

    manifest_value = metadata.get("manifest")
    if not manifest_value:
        raise ValueError("Gallery metadata must record its ordered manifest path")
    manifest_path = Path(str(manifest_value))
    if not manifest_path.is_absolute() and not manifest_path.is_file():
        for candidate in (gallery_dir / manifest_path, gallery_dir.parent.parent / manifest_path):
            if candidate.is_file():
                manifest_path = candidate
                break
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Gallery manifest referenced by metadata is missing: {manifest_path}")
    recorded_manifest_hash = metadata.get("manifest_sha256")
    if not recorded_manifest_hash or recorded_manifest_hash != sha256_file(manifest_path):
        raise ValueError("Gallery manifest provenance/hash mismatch")
    ordered_manifest = list(manifest_rows) if manifest_rows is not None else read_jsonl(manifest_path)
    gallery_video_ids = _gallery_video_ids(metadata, ordered_manifest)
    gallery_sample_ids = metadata.get("gallery_sample_ids")
    if not isinstance(gallery_sample_ids, Sequence) or isinstance(gallery_sample_ids, (str, bytes)):
        raise ValueError("Gallery metadata must contain explicit gallery_sample_ids")
    gallery_sample_ids = [str(value) for value in gallery_sample_ids]
    if len(gallery_sample_ids) != 64 or len(set(gallery_sample_ids)) != 64:
        raise ValueError("Gallery sample IDs must contain 64 unique ordered entries")
    source_ids = metadata.get("source_ids")
    if source_ids is not None and [str(value) for value in source_ids] != gallery_sample_ids:
        raise ValueError("Gallery source_ids disagree with gallery_sample_ids")
    if len(ordered_manifest) != 64:
        raise ValueError(f"Gallery manifest must contain exactly 64 ordered rows, got {len(ordered_manifest)}")
    manifest_sample_ids = [str(row.get("sample_id")) for row in ordered_manifest]
    manifest_video_ids = [str(row.get("video_name") or row.get("video_id")) for row in ordered_manifest]
    if manifest_sample_ids != gallery_sample_ids or manifest_video_ids != gallery_video_ids:
        raise ValueError("Gallery manifest order disagrees with gallery metadata order")
    return {"directory": str(gallery_dir), "array_sha256": array_hash,
            "metadata_sha256": sha256_file(gallery_dir / "gallery.metadata.json"),
            "manifest_path": str(manifest_path), "manifest_sha256": recorded_manifest_hash,
            "shape": list(gallery.vectors.shape), "dtype": str(gallery.vectors.dtype),
            "metadata": metadata, "gallery_sample_ids": gallery_sample_ids,
            "gallery_video_ids": gallery_video_ids}


def _gallery_video_ids(metadata: Mapping[str, Any], manifest_rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Return explicit, ordered gallery video IDs and reject inferred token order."""
    values = metadata.get("gallery_video_ids")
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ValueError("Gallery metadata must contain explicit gallery_video_ids")
    resolved = [str(value) for value in values]
    if len(resolved) != 64 or len(set(resolved)) != 64:
        raise ValueError("Gallery video IDs must contain 64 unique ordered entries")
    if len(manifest_rows) != 64:
        raise ValueError("Gallery manifest order cannot be verified without exactly 64 rows")
    manifest_ids = [str(row.get("video_name") or row.get("video_id")) for row in manifest_rows]
    if manifest_ids != resolved:
        raise ValueError("Gallery video metadata order disagrees with gallery manifest order")
    return resolved


def _t0_manifest_paths(upstream: Path) -> tuple[Path, Path, Path]:
    manifest = upstream / "workspace" / "manifests" / "manifest.jsonl"
    feature_index = upstream / "workspace" / "features" / "validation" / "index.jsonl"
    frame_manifest = upstream / "frame_manifest.jsonl"
    for path in (manifest, feature_index, frame_manifest):
        if not path.is_file():
            raise FileNotFoundError(f"Required T0 artifact missing: {path}")
    return manifest, feature_index, frame_manifest


def _source_groups(fake_rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, str], dict[str, Any]]:
    """Build conservative target/source components without inventing mappings.

    The target/source fields are metadata, not something that may be inferred
    from filenames.  Missing fields therefore make the grouping incomplete;
    affected rows receive a deterministic video-level placeholder group so
    that they cannot silently enter a falsely verified cluster bootstrap.
    """
    known_ids: set[str] = set()
    unresolved: list[dict[str, Any]] = []
    normalized_rows: list[dict[str, Any]] = []
    for row in fake_rows:
        target = str(row.get("target_id")).strip() if row.get("target_id") not in (None, "") else None
        source = str(row.get("source_id")).strip() if row.get("source_id") not in (None, "") else None
        fake_video = str(row.get("video_name") or row.get("video_id") or row.get("sample_id"))
        normalized = {"sample_id": str(row.get("sample_id")), "fake_video_id": fake_video,
                      "compression": str(row.get("compression")),
                      "manipulation": str(row.get("manipulation")),
                      "target_original_id": target, "source_original_id": source}
        normalized_rows.append(normalized)
        if target:
            known_ids.add(target)
        if source:
            known_ids.add(source)
        if not target or not source:
            unresolved.append({**normalized, "reason": "missing_target_or_source_original_id"})

    parent = {value: value for value in sorted(known_ids)}

    def find(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        a, b = find(left), find(right)
        if a != b:
            parent[max(a, b)] = min(a, b)

    for row in normalized_rows:
        if row["target_original_id"] and row["source_original_id"]:
            union(str(row["target_original_id"]), str(row["source_original_id"]))
    components: dict[str, list[str]] = defaultdict(list)
    for value in sorted(known_ids):
        components[find(value)].append(value)
    ordered = sorted((sorted(values) for values in components.values()), key=lambda values: values[0])
    mapping: dict[str, str] = {}
    groups = []
    for number, members in enumerate(ordered, start=1):
        group_id = f"source-group-{number:04d}"
        for member in members:
            mapping[member] = group_id
        groups.append({"source_group_id": group_id, "original_ids": members,
                       "n_original_ids": len(members)})
    # Add deterministic placeholders for rows whose original relation cannot
    # be resolved.  They are intentionally not labelled as original IDs.
    placeholder_groups: dict[str, dict[str, Any]] = {}
    row_group_ids: dict[str, str] = {}
    for row in normalized_rows:
        target = row["target_original_id"]
        if target and target in mapping:
            group_id = mapping[target]
        elif row["source_original_id"] and row["source_original_id"] in mapping:
            group_id = mapping[row["source_original_id"]]
        else:
            group_id = f"unresolved-video-{_hash_id(row['fake_video_id'])[:12]}"
            placeholder_groups.setdefault(group_id, {
                "source_group_id": group_id, "original_ids": [], "n_original_ids": 0,
                "status": "incomplete", "definition": "deterministic fake-video placeholder; original relation unresolved",
            })
        row_group_ids[row["sample_id"]] = group_id

    groups.extend(placeholder_groups.values())
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in normalized_rows:
        by_group[row_group_ids[row["sample_id"]]].append(row)
    unresolved_sample_ids = {str(item["sample_id"]) for item in unresolved}
    for item in groups:
        group_id = str(item["source_group_id"])
        members = by_group.get(group_id, [])
        item["status"] = "incomplete" if any(m["sample_id"] in unresolved_sample_ids for m in members) else "verified"
        item["definition"] = ("connected components of target/source original IDs; c23/c40 and all manipulations stay together"
                               if item["status"] == "verified" else
                               "connected component with unresolved target/source metadata; placeholder membership is not an identity claim")
        item["target_original_ids"] = sorted({m["target_original_id"] for m in members if m["target_original_id"]})
        item["source_original_ids"] = sorted({m["source_original_id"] for m in members if m["source_original_id"]})
        item["fake_video_ids"] = sorted({m["fake_video_id"] for m in members})
        item["fake_video_count"] = len(item["fake_video_ids"])
        item["fake_entry_count"] = len(members)
        item["c23_count"] = sum(m["compression"] == "c23" for m in members)
        item["c40_count"] = sum(m["compression"] == "c40" for m in members)
        item["manipulation_counts"] = {name: sum(m["manipulation"] == name for m in members) for name in MANIPULATIONS}
    payload = {
        "status": "verified" if not unresolved else "incomplete",
        "group_count": len(groups), "groups": groups,
        "max_original_ids_per_group": max((g["n_original_ids"] for g in groups), default=0),
        "definition": "connected components of known target/source original IDs; unresolved rows use conservative fake-video placeholders",
        "unresolved": unresolved,
        "row_group_ids": row_group_ids,
        "video_disjoint_is_not_identity_disjoint": True,
    }
    return mapping, payload


def _find_target_row(rows: Sequence[Mapping[str, Any]], fake: Mapping[str, Any]) -> dict[str, Any]:
    target = str(fake["target_id"])
    compression = str(fake["compression"])
    fraction = float(fake["frame_fraction"])
    candidates = [dict(row) for row in rows if row.get("is_real") and row.get("compression") == compression
                  and row.get("video_name") == target]
    if not candidates:
        raise ValueError(f"Missing target-real rows for {fake['sample_id']}: {compression}/{target}")
    candidates.sort(key=lambda row: (abs(float(row["frame_fraction"]) - fraction), int(row["frame_index"])))
    chosen = candidates[0]
    if abs(float(chosen["frame_fraction"]) - fraction) > 1e-6:
        raise ValueError(f"Target-real frame mapping is not exact for {fake['sample_id']}")
    return chosen


def _write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyYAML is required for frozen M1 receipt") from exc
    atomic_write_text(path, yaml.safe_dump(dict(payload), allow_unicode=True, sort_keys=False))


def _scorer_registry() -> dict[str, Any]:
    return {
        "primary": "r_k",
        "scorers": {
            "r_k": {"formula": "r_20(q)", "direction": "higher_more_suspicious", "role": "primary"},
            "mean_r": {"formula": "mean(r_1,...,r_20)", "direction": "higher_more_suspicious", "role": "fixed_secondary"},
            "C2": {"formula": "mean(r_1/r_20,...,r_19/r_20)", "direction": "higher_more_suspicious", "role": "fixed_secondary"},
            "CIR_LID": {"formula": "[mean(log(r_20/max(r_j,1e-12))) for j=1..19]^-1 with denominator floor 1e-6",
                         "direction": "higher_more_suspicious", "role": "historical_diagnostic"},
        },
        "distance": {"source": "faiss.IndexFlatL2", "returned_distance": "squared_l2",
                     "stored_distance": "sqrt(float32 squared_l2)", "search": "exact", "k": 20,
                     "distance_floor": 1e-12, "cir_denominator_floor": 1e-6},
        "score_maps": {"grid": [28, 28], "output": [448, 448], "mode": "bilinear", "align_corners": False,
                       "postprocess": "none"},
        "sanity_checks": ["distance ordering", "finite scores", "zero-distance count", "duplicate neighbor count",
                           "C2 is not mean_r", "fixed direction"],
    }


def _load_roles(root: Path) -> list[dict[str, Any]]:
    return read_jsonl(root / "query_manifest.jsonl")


def _write_query_rows(root: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    write_jsonl(root / "query_manifest.jsonl", rows)


def prepare(m1: Config, base: Config, root: Path) -> dict[str, Any]:
    """Freeze the protocol and construct T3/pairing/source-group manifests."""
    upstream = _upstream(m1)
    t0_manifest_path, t0_feature_index, t0_frame_manifest = _t0_manifest_paths(upstream)
    gallery_dir = _gallery_dir(base, upstream)
    gallery_manifest = _gallery_manifest_path(gallery_dir, upstream)
    gallery_manifest_rows = read_jsonl(gallery_manifest)
    gallery_provenance = _check_gallery(base, gallery_dir, gallery_manifest_rows)
    m1_digest = config_digest(m1)
    prereg = root / "preregistered_config.yaml"
    if prereg.exists():
        frozen = load_config(prereg)
        if config_digest(frozen) != m1_digest:
            raise RuntimeError("Existing preregistered_config.yaml differs; use a new M1 namespace")
    else:
        _write_yaml(prereg, m1.data)
    config_hash = sha256_file(prereg)

    workspace = root / "workspace"
    manifests = workspace / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    standard_manifest = manifests / "manifest.jsonl"
    t0_rows = read_manifest(t0_manifest_path)
    frame_entries = read_jsonl(t0_frame_manifest)
    t3_entries = [entry for entry in frame_entries if entry.get("budget") == T3]
    if not t3_entries:
        raise ValueError("T0 frame manifest contains no T3_frozen entries")
    t3_ids = {str(sample_id) for entry in t3_entries for sample_id in entry.get("sample_ids", [])}
    selected = [dict(row) for row in t0_rows if str(row["sample_id"]) in t3_ids]
    if not selected:
        raise ValueError("No T3 rows could be reused from T0")
    selected_ids = {str(row["sample_id"]) for row in selected}
    fake_selected = [row for row in selected if not row["is_real"]]
    if not fake_selected or {row["manipulation"] for row in fake_selected} != set(MANIPULATIONS):
        raise ValueError("T3 selection does not cover all four frozen manipulations")

    by_id = {str(row["sample_id"]): row for row in selected}
    all_t0_by_id = {str(row["sample_id"]): row for row in t0_rows}
    role_rows: dict[str, dict[str, Any]] = {}
    for row in selected:
        role_rows[str(row["sample_id"])] = {**row, "roles": ["validation_fake" if not row["is_real"] else "validation_real"]}

    pairing: list[dict[str, Any]] = []
    for fake in fake_selected:
        target = _find_target_row(t0_rows, fake)
        target_id = str(target["sample_id"])
        if target_id not in role_rows:
            role_rows[target_id] = {**target, "roles": ["target_real"]}
        elif "target_real" not in role_rows[target_id]["roles"]:
            role_rows[target_id]["roles"].append("target_real")
        fake_transform, target_transform, crop_status = _validate_pair_crop_transform(
            fake, target, require_materialized=False
        )
        pairing.append({
            "run_id": m1.require("protocol.namespace"),
            "compression": fake["compression"], "manipulation": fake["manipulation"],
            "fake_video_id": fake["video_name"], "fake_sample_id": fake["sample_id"],
            "target_original_id": fake.get("target_id"), "source_original_id": fake.get("source_id"),
            "target_real_sample_id": target_id, "fake_frame_index": fake["frame_index"],
            "target_real_frame_index": target["frame_index"],
            "frame_fraction": fake["frame_fraction"],
            "crop_transform_id": _pair_crop_transform_id(fake_transform, target_transform),
            "fake_crop_transform": fake_transform,
            "target_real_crop_transform": target_transform,
            "mask_crop_transform": fake_transform,
            "mask_crop_transform_source": "fake mask uses the fake row mapped from the shared source crop specification",
            "crop_transform_status": crop_status,
            "mask_provenance": _mask_provenance_payload(fake),
            "mask_provenance_status": "pending_materialization",
            "feature_cache_provenance": None,
        })
    fake_unique = [row for row in fake_selected]
    _, source_payload = _source_groups(fake_unique)
    row_group_ids = {str(key): str(value) for key, value in source_payload.get("row_group_ids", {}).items()}
    if any(str(row["sample_id"]) not in row_group_ids for row in fake_selected):
        raise ValueError("Source grouping did not assign every selected fake row")

    # Target-real rows are not fake rows and therefore do not occur in
    # ``row_group_ids``.  Attach them through the already frozen pair map,
    # rather than guessing from a real-row video name.  A target-real frame
    # receiving two different source groups is a provenance contradiction.
    target_sample_groups: dict[str, set[str]] = defaultdict(set)
    for pair in pairing:
        group_id = row_group_ids.get(str(pair["fake_sample_id"]))
        if not group_id:
            raise ValueError(f"Pairing row has no source group: {pair['fake_sample_id']}")
        pair["source_group_id"] = group_id
        pair["frame_indices"] = [pair["fake_frame_index"]]
        target_sample_groups[str(pair["target_real_sample_id"])].add(group_id)
    for sample_id, group_ids in target_sample_groups.items():
        if len(group_ids) != 1:
            raise ValueError(f"Target-real sample maps to multiple source groups: {sample_id}")

    for row in role_rows.values():
        sample_id = str(row["sample_id"])
        if not row["is_real"]:
            row["source_group_id"] = row_group_ids[sample_id]
        elif sample_id in target_sample_groups:
            row["source_group_id"] = next(iter(target_sample_groups[sample_id]))
        else:
            # Ordinary validation-real query rows are not source-group
            # observations; leaving them unassigned avoids an identity claim.
            row["source_group_id"] = None
    ordered_rows = [role_rows[key] for key in sorted(role_rows)]
    write_manifest(ordered_rows, standard_manifest)

    # Keep a compact frame manifest with the exact T0 frame IDs; no new frame selection is performed.
    write_jsonl(root / "frame_manifest.jsonl", [
        {**entry, "source": "T0_verified_frame_manifest", "budget": T3}
        for entry in t3_entries
    ])
    write_jsonl(root / "gallery_manifest.jsonl", gallery_manifest_rows)
    gallery_video_ids = _gallery_video_ids(gallery_provenance["metadata"], gallery_manifest_rows)

    query_rows = []
    for row in ordered_rows:
        query_rows.append({**row, "run_id": m1.require("protocol.namespace"),
                           "feature_cache_provenance": None, "eligibility_status": "pending"})
    _write_query_rows(root, query_rows)
    write_jsonl(root / "pairing_manifest.jsonl", pairing)
    write_json(root / "source_groups.json", source_payload)

    gallery_ids = [str(value) for value in gallery_video_ids]
    validation_original_ids = {
        str(value) for row in fake_selected for value in (row.get("target_id"), row.get("source_id"))
        if value not in (None, "")
    }
    overlap = sorted(set(gallery_ids) & validation_original_ids)
    if overlap:
        raise ValueError(f"Gallery/query source overlap detected: {overlap[:10]}")

    selected_groups = sorted({str(item["source_group_id"]) for item in source_payload["groups"]})
    diagnostic_seed = int(m1.require("conditional_diagnostics.seed"))
    rng = np.random.default_rng(diagnostic_seed)
    permutation = list(rng.permutation(selected_groups))
    diagnostic_groups = sorted(permutation[:min(int(m1.require("conditional_diagnostics.max_source_groups")), len(permutation))])
    gallery_permutation = [str(value) for value in rng.permutation(gallery_ids)]
    matched_token_indices = [int(value) for value in rng.integers(0, int(m1.require("protocol.gallery_tokens")) // len(gallery_ids), size=len(gallery_ids))]
    case_count = int(m1.get("reports.case_count", 4))
    case_groups = selected_groups[:case_count]
    write_json(root / "diagnostic_selection.json", {
        "seed": diagnostic_seed, "source_groups_all": selected_groups,
        "selected_source_groups": diagnostic_groups, "selection_rule": "one fixed RNG permutation before fake evaluation",
        "case_source_groups": case_groups, "case_rule": "lexicographic source-group order before seeing metrics",
        "gallery_video_ids": gallery_ids, "gallery_video_order": gallery_permutation,
        "gallery_order_rule": "one fixed seed permutation; nested prefixes used for 16/32/48/64",
        "matched_token_indices": matched_token_indices,
        "matched_token_rule": "one real-only seed draw per gallery video, shared by all queries",
    })

    upstream_hashes = {
        "t0_manifest_sha256": sha256_file(t0_manifest_path),
        "t0_frame_manifest_sha256": sha256_file(t0_frame_manifest),
        "t0_feature_index_sha256": sha256_file(t0_feature_index),
        "gallery_array_sha256": gallery_provenance["array_sha256"],
        "gallery_metadata_sha256": gallery_provenance["metadata_sha256"],
    }
    checkpoint = resolve_config_path(base, base.require("paths.dinov3_weights"))
    if checkpoint.is_file():
        upstream_hashes["checkpoint_sha256"] = sha256_file(checkpoint)
    write_json(root / "input_provenance.json", {
        "status": "verified", "base_config_digest": config_digest(base), "m1_config_digest": m1_digest,
        "preregistered_config_sha256": config_hash, "upstream_t0_dir": str(upstream),
        "gallery": {**gallery_provenance, "gallery_video_ids": gallery_ids}, "upstream_hashes": upstream_hashes,
        "gallery_query_overlap_ids": overlap,
        "valid_pixel_domain": "all_448x448_crop_pixels_inherited_from_t0",
        "official_test_data_read": False, "test_split_or_videos_read": False,
        "validation_previously_used_in_t0": True,
    })
    receipt = {
        "status": "frozen", "created_utc": _utc_now(), "run_id": m1.require("protocol.namespace"),
        "config_sha256": config_hash, "config_digest": m1_digest,
        "base_config_digest": config_digest(base), "git_state": _git_state(Path.cwd()),
        "upstream_artifact_hashes": upstream_hashes,
        "freeze_before_fake_metrics": True, "official_test_locked": True,
    }
    if (root / "freeze_receipt.json").exists():
        existing = read_json(root / "freeze_receipt.json")
        if (existing.get("status") != "frozen" or
                existing.get("config_digest") != m1_digest or
                existing.get("base_config_digest") != config_digest(base) or
                existing.get("freeze_before_fake_metrics") is not True or
                existing.get("official_test_locked") is not True):
            raise RuntimeError("Freeze receipt already exists for a different configuration")
    else:
        write_json(root / "freeze_receipt.json", receipt)
    write_json(root / "scorer_registry.json", _scorer_registry())
    if not (root / "amendments.md").exists():
        atomic_write_text(root / "amendments.md", "# M1 amendments\n\nNone recorded at freeze.\n")
    write_jsonl(root / "eligibility_and_exclusions.jsonl", [])
    return {"status": "completed", "frozen": True, "t3_rows": len(ordered_rows),
            "fake_rows": len(fake_selected), "source_groups": len(selected_groups),
            "diagnostic_source_groups": diagnostic_groups, "official_test_locked": True}


def _role_rows(root: Path) -> list[dict[str, Any]]:
    rows = _load_roles(root)
    if not rows:
        raise RuntimeError("M1 query manifest missing; run prepare first")
    return rows


def _row_lookup(root: Path) -> dict[str, dict[str, Any]]:
    return {str(row["sample_id"]): row for row in _role_rows(root)}


def _valid_array(path: str | Path, shape: tuple[int, ...], dtype: np.dtype[Any], *, binary: bool = False) -> bool:
    try:
        value = np.load(path, allow_pickle=False)
        if value.shape != shape or value.dtype != dtype or not np.isfinite(value).all():
            return False
        if binary and np.any((value != 0) & (value != 1)):
            return False
        return True
    except (OSError, ValueError):
        return False


def _validated_mask(row: Mapping[str, Any]) -> np.ndarray:
    """Load one frozen mask and enforce the M1 mask support contract."""
    value = row.get("mask_cache_path")
    if not value:
        raise ValueError(f"Mask cache missing for {row.get('sample_id')}")
    path = Path(str(value))
    if not path.is_file():
        raise FileNotFoundError(f"Mask cache does not exist: {path}")
    mask = np.load(path, allow_pickle=False)
    if (mask.shape != IMAGE_SIZE or mask.dtype != np.uint8 or
            not np.isfinite(mask).all() or not np.isin(mask, [0, 1]).all()):
        raise ValueError(f"Invalid frozen mask cache for {row.get('sample_id')}: {path}")
    return np.asarray(mask, dtype=np.uint8)


def _mask_provenance_payload(row: Mapping[str, Any], mask: np.ndarray | None = None) -> dict[str, Any]:
    """Describe the exact cached mask consumed by evaluation and figures."""
    path_value = row.get("mask_cache_path")
    path = Path(str(path_value)) if path_value else None
    return {
        "source_path": None if row.get("mask_path") is None else str(row.get("mask_path")),
        "source_frame_index": row.get("mask_frame_index"),
        "source_frame_count": row.get("mask_frame_count"),
        "source_frame_remapped": bool(row.get("mask_frame_remapped", False)),
        "cache_path": None if path is None else str(path),
        "cache_sha256": sha256_file(path) if path is not None and path.is_file() else None,
        "shape": None if mask is None else list(mask.shape),
        "dtype": None if mask is None else str(mask.dtype),
        "binary": None if mask is None else bool(np.isin(mask, [0, 1]).all()),
        "crop_transform": _gallery_transform(row),
    }


def _validate_pair_mask_provenance(pair: Mapping[str, Any], fake: Mapping[str, Any],
                                   target: Mapping[str, Any]) -> np.ndarray:
    """Require a cached mask to belong to the exact fake/target crop pair."""
    fake_transform = _gallery_transform(fake)
    target_transform = _gallery_transform(target)
    if pair.get("fake_crop_transform") != fake_transform:
        raise ValueError(f"Fake crop provenance mismatch for {fake['sample_id']}")
    if pair.get("target_real_crop_transform") != target_transform:
        raise ValueError(f"Target-real crop provenance mismatch for {fake['sample_id']}")
    if pair.get("mask_crop_transform") != fake_transform:
        raise ValueError(f"Mask crop provenance mismatch for {fake['sample_id']}")
    if pair.get("crop_transform_id") != _pair_crop_transform_id(fake_transform, target_transform):
        raise ValueError(f"Pair crop transform ID mismatch for {fake['sample_id']}")
    if pair.get("crop_transform_status") != "verified":
        raise ValueError(f"Pair crop transform is not verified for {fake['sample_id']}")
    mask = _validated_mask(fake)
    expected = _mask_provenance_payload(fake, mask)
    if pair.get("mask_provenance_status") != "verified":
        raise ValueError(f"Mask provenance is not verified for {fake['sample_id']}")
    if pair.get("mask_provenance") != expected:
        raise ValueError(f"Mask provenance mismatch for {fake['sample_id']}")
    return mask


def _pairing_integrity_status(root: Path) -> str:
    """Return a conservative status for all frozen fake/target pair rows."""
    try:
        pairing = read_jsonl(root / "pairing_manifest.jsonl")
        rows = _role_rows(root)
    except (OSError, ValueError, RuntimeError):
        return "missing"
    if not pairing:
        return "missing"
    row_by_id = {str(row["sample_id"]): row for row in rows}
    try:
        fake_ids = {str(row["sample_id"]) for row in rows if not bool(row.get("is_real"))}
        pairing_ids = [str(pair["fake_sample_id"]) for pair in pairing]
        if (len(pairing_ids) != len(set(pairing_ids)) or
                set(pairing_ids) != fake_ids or len(pairing_ids) != len(fake_ids)):
            return "partial"
        for pair in pairing:
            fake = row_by_id[str(pair["fake_sample_id"])]
            target = row_by_id[str(pair["target_real_sample_id"])]
            fake_group = str(fake.get("source_group_id"))
            target_group = str(target.get("source_group_id"))
            pair_group = str(pair.get("source_group_id"))
            if fake_group == "None" or target_group == "None" or pair_group != fake_group or pair_group != target_group:
                return "partial"
            _validate_pair_mask_provenance(pair, fake, target)
    except (KeyError, FileNotFoundError, OSError, ValueError, TypeError):
        return "partial"
    return "verified"


def crops_stage(m1: Config, base: Config, root: Path) -> dict[str, Any]:
    """Reuse exact T0 crops/masks; only materialize missing artifacts after freeze."""
    rows = _role_rows(root)
    missing = []
    for row in rows:
        crop = row.get("crop_path")
        mask = row.get("mask_cache_path")
        ok_crop = crop is not None and _valid_array(crop, (448, 448, 3), np.dtype(np.uint8))
        ok_mask = row["is_real"] or (mask is not None and _valid_array(mask, (448, 448), np.dtype(np.uint8), binary=True))
        if not (ok_crop and ok_mask):
            missing.append(dict(row))
    if missing:
        cfg = _m1_config(base, root)
        missing_path = root / "workspace" / "manifests" / "missing_manifest.jsonl"
        write_manifest(missing, missing_path)
        summary = cache_manifest_crops(
            missing_path,
            crops_dir=resolve_config_path(cfg, cfg.require("paths.crops")),
            masks_dir=resolve_config_path(cfg, cfg.require("paths.masks")),
            input_size=448, detector_upsample_primary=1, detector_upsample_retry=2,
            square_scale=1.3, padding="reflection", mask_policy="nonzero",
            exclude_empty_fake_masks=False, metadata_path=None,
        )
        updated = {str(row["sample_id"]): row for row in read_manifest(missing_path)}
        for row in rows:
            if str(row["sample_id"]) in updated:
                for key in ("crop_path", "mask_cache_path", "crop_left", "crop_top", "crop_right", "crop_bottom", "crop_scale", "exclusion_reason"):
                    if key in updated[str(row["sample_id"])]:
                        row[key] = updated[str(row["sample_id"])].get(key)
        write_manifest(rows, root / "workspace" / "manifests" / "manifest.jsonl")
        _write_query_rows(root, rows)
        result = {"status": summary.status, "reused": len(rows) - len(missing), "materialized": len(missing),
                  "summary": summary.__dict__}
    else:
        result = {"status": "reused", "reused": len(rows), "materialized": 0}
    row_by_id = {str(row["sample_id"]): row for row in rows}
    pairing_path = root / "pairing_manifest.jsonl"
    pairing = read_jsonl(pairing_path)
    for pair in pairing:
        fake = row_by_id.get(str(pair["fake_sample_id"]))
        target = row_by_id.get(str(pair["target_real_sample_id"]))
        if fake is None or target is None:
            raise ValueError(f"Pairing row cannot be materialized: {pair.get('fake_sample_id')}")
        fake_transform, target_transform, crop_status = _validate_pair_crop_transform(
            fake, target, require_materialized=True
        )
        pair["crop_transform_id"] = _pair_crop_transform_id(fake_transform, target_transform)
        pair["fake_crop_transform"] = fake_transform
        pair["target_real_crop_transform"] = target_transform
        pair["mask_crop_transform"] = fake_transform
        pair["crop_transform_status"] = crop_status
        mask = _validated_mask(fake)
        pair["mask_provenance"] = _mask_provenance_payload(fake, mask)
        pair["mask_provenance_status"] = "verified"
    write_jsonl(pairing_path, pairing)
    result["pairing_crop_transform_status"] = "verified"
    result["pairing_mask_provenance_status"] = "verified"
    write_json(root / "crop_summary.json", result)
    return result


def _feature_provenance_ok(path: Path, row: Mapping[str, Any], base: Config,
                           checkpoint_hash: str | None, manifest_hash: str | None) -> bool:
    try:
        value, metadata = load_features(path)
    except (OSError, ValueError, TypeError):
        return False
    if value.shape != (1, 784, 768) or value.dtype != np.float32:
        return False
    expected = {
        "sample_id": str(row["sample_id"]), "architecture": "dinov3_vitb16", "block_one_based": 6,
        "api_n": [5], "norm": True, "reshape": False,
        "config_digest": config_digest(base), "manifest_sha256": manifest_hash,
        "video_name": str(row["video_name"]), "frame_index": int(row["frame_index"]),
        "frame_fraction": float(row["frame_fraction"]),
        "token_grid": list(GRID), "token_count": 784, "feature_dimension": 768,
        "token_order": "row_major_patch_grid_28x28_excluding_cls_storage_register",
    }
    if checkpoint_hash is None or manifest_hash is None:
        return False
    expected["checkpoint_sha256"] = checkpoint_hash
    for key, value_expected in expected.items():
        if key == "manifest_sha256" and value_expected is None:
            return False
        if metadata.get(key) != value_expected:
            return False
    crop_path = row.get("crop_path")
    if not crop_path or not Path(str(crop_path)).is_file():
        return False
    if metadata.get("crop_path") != str(crop_path):
        return False
    if metadata.get("crop_sha256") != sha256_file(crop_path):
        return False
    for key, value_expected in _gallery_transform(row).items():
        if metadata.get(key) != value_expected:
            return False
    return True


def _extract_missing_features(m1: Config, base: Config, root: Path, pending: list[dict[str, Any]],
                              feature_root: Path, manifest_hash: str) -> list[dict[str, Any]]:
    if not pending:
        return []
    from .dinov3 import extract_patch_tokens, load_official_model, preprocess_rgb, spec_from_config
    import torch
    repo = resolve_config_path(base, base.require("paths.dinov3_repo"))
    checkpoint = resolve_config_path(base, base.require("paths.dinov3_weights"))
    spec = spec_from_config(base)
    model = load_official_model(repo, checkpoint, spec, "cuda")
    checkpoint_hash = sha256_file(checkpoint)
    out = []
    for start in range(0, len(pending), 8):
        batch_rows = pending[start:start + 8]
        images = [np.load(row["crop_path"], allow_pickle=False) for row in batch_rows]
        batch = preprocess_rgb(images, base).to(torch.device("cuda"), non_blocking=True)
        tokens = extract_patch_tokens(model, batch, spec)
        for offset, row in enumerate(batch_rows):
            destination = feature_root / f"{_hash_id(row['sample_id'])}.npy"
            metadata = save_features(destination, tokens[offset:offset + 1], config=base, metadata={
                "sample_id": row["sample_id"], "subset": "m1_validation", "crop_path": row["crop_path"],
                "crop_sha256": sha256_file(row["crop_path"]), "checkpoint_path": str(checkpoint),
                "checkpoint_sha256": checkpoint_hash, "architecture": spec.architecture,
                "block_one_based": spec.block_one_based, "api_n": list(spec.api_n), "norm": spec.norm,
                "reshape": spec.reshape, "device": "cuda", "m1_supplement": True,
                "manifest_sha256": manifest_hash,
                "video_name": str(row["video_name"]), "frame_index": int(row["frame_index"]),
                "frame_fraction": float(row["frame_fraction"]),
                **_gallery_transform(row),
                "token_grid": list(GRID), "token_count": 784, "feature_dimension": 768,
                "token_order": "row_major_patch_grid_28x28_excluding_cls_storage_register",
            })
            out.append({"sample_id": row["sample_id"], "feature_path": str(destination),
                        "status": "extracted", "source": "m1_supplement", "array_sha256": metadata["array_sha256"]})
        print(f"[m1 features] extracted={min(start + 8, len(pending))}/{len(pending)}", flush=True)
    return out


def features_stage(m1: Config, base: Config, root: Path) -> dict[str, Any]:
    rows = _role_rows(root)
    upstream = _upstream(m1)
    t0_manifest_path, t0_feature_index, _ = _t0_manifest_paths(upstream)
    t0_index = {str(row["sample_id"]): row for row in read_jsonl(t0_feature_index)}
    checkpoint = resolve_config_path(base, base.require("paths.dinov3_weights"))
    checkpoint_hash = sha256_file(checkpoint) if checkpoint.is_file() else None
    manifest_hash = sha256_file(t0_manifest_path)
    feature_root = root / "workspace" / "features" / "validation"
    feature_root.mkdir(parents=True, exist_ok=True)
    index_rows: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    reused = 0
    for row in rows:
        source = t0_index.get(str(row["sample_id"]), {})
        candidate = Path(str(source.get("feature_path", ""))) if source.get("feature_path") else Path()
        if candidate.is_file() and _feature_provenance_ok(candidate, row, base, checkpoint_hash, manifest_hash):
            index_rows.append({"sample_id": row["sample_id"], "feature_path": str(candidate),
                               "status": "reused", "source": "T0_verified", "array_sha256": sha256_file(candidate)})
            row["feature_path"] = str(candidate)
            row["feature_cache_provenance"] = "T0_verified_exact_match"
            reused += 1
        else:
            if not row.get("crop_path") or not _valid_array(row["crop_path"], (448, 448, 3), np.dtype(np.uint8)):
                raise RuntimeError(f"Missing exact crop for feature extraction: {row['sample_id']}")
            pending.append(row)
    extracted_rows = _extract_missing_features(m1, base, root, pending, feature_root, manifest_hash)
    for item in extracted_rows:
        row = next(row for row in rows if str(row["sample_id"]) == item["sample_id"])
        row["feature_path"] = item["feature_path"]
        row["feature_cache_provenance"] = "M1_supplement_exact_config"
    index_rows.extend(extracted_rows)
    index_rows.sort(key=lambda item: str(item["sample_id"]))
    _write_query_rows(root, rows)
    write_jsonl(root / "feature_index.jsonl", index_rows)
    result = {"status": "completed", "rows": len(rows), "reused": reused,
              "extracted": len(extracted_rows), "index": str(root / "feature_index.jsonl"),
              "feature_config": {"architecture": "dinov3_vitb16", "block_one_based": 6,
                                  "api_n": [5], "norm": True, "reshape": False,
                                  "extra_l2_normalization": False, "shape": [1, 784, 768]}}
    write_json(root / "feature_summary.json", result)
    return result


def _load_feature_index(root: Path) -> dict[str, Path]:
    rows = read_jsonl(root / "feature_index.jsonl")
    result = {str(row["sample_id"]): Path(str(row["feature_path"])) for row in rows}
    if len(result) != len(rows):
        raise ValueError("Duplicate sample IDs in M1 feature index")
    return result


def _load_gallery(base: Config, m1: Config, root: Path) -> tuple[Any, Path]:
    directory = Path(read_json(root / "input_provenance.json")["gallery"]["directory"])
    gallery = load_gallery(directory)
    _check_gallery(base, directory)
    return gallery, directory


def _knn_scores(distances: np.ndarray, neighbor_indices: np.ndarray | None = None, *,
                           distance_floor: float = 1e-12, cir_floor: float = 1e-6) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    d = np.asarray(distances, dtype=np.float32)
    if d.ndim != 2 or d.shape[1] != 20 or not np.isfinite(d).all() or np.any(d < 0):
        raise ValueError("M1 distances must be finite [N,20] nonnegative float32")
    if np.any(d[:, 1:] + 1e-6 * np.maximum(1.0, d[:, :-1]) < d[:, :-1]):
        raise ValueError("M1 neighbor distances are not sorted")
    if neighbor_indices is not None:
        indices = np.asarray(neighbor_indices, dtype=np.int64)
        if indices.shape != d.shape:
            raise ValueError("M1 neighbor indices must have the same [N,20] shape as distances")
        duplicate_count = int(sum(np.unique(row).size < row.size for row in indices))
    else:
        # Distance ties are not a substitute for duplicate-index checks, but keep
        # a conservative fallback for small standalone formula checks.
        duplicate_count = int(np.sum(np.any(np.diff(d, axis=1) == 0.0, axis=1)))
    zero_count = int(np.sum(d == 0.0))
    distance_floor_count = int(np.sum(d < distance_floor))
    r = np.maximum(d, distance_floor)
    cir_denominator = np.mean(np.log(r[:, -1:] / r[:, :-1]), axis=1)
    cir_denominator_floor_count = int(np.sum(cir_denominator < cir_floor))
    values = {
        "r_k": d[:, -1].astype(np.float32),
        "mean_r": np.mean(d, axis=1).astype(np.float32),
        "C2": np.mean(r[:, :-1] / r[:, -1:,], axis=1).astype(np.float32),
        "CIR_LID": cir_lid(d, distance_floor=distance_floor, denominator_floor=cir_floor),
    }
    for key, value in values.items():
        if not np.isfinite(value).all():
            raise ValueError(f"Non-finite {key} score")
    return values, {
        "zero_distance_count": zero_count,
        "neighbor_duplicate_count": duplicate_count,
        "distance_floor_applied_count": distance_floor_count,
        "cir_denominator_floor_applied_count": cir_denominator_floor_count,
        "distance_floor": float(distance_floor),
        "cir_denominator_floor": float(cir_floor),
        "token_count": int(d.shape[0]),
    }


def _map_path(root: Path, scorer: str, sample_id: str) -> Path:
    return root / "score_maps" / scorer / f"{_hash_id(sample_id)}.npy"


def _read_map(path: Path) -> np.ndarray:
    value = np.load(path, allow_pickle=False)
    if value.shape != GRID or value.dtype != np.float32 or not np.isfinite(value).all():
        raise ValueError(f"Invalid M1 score map: {path}")
    return value


def scores_stage(m1: Config, base: Config, root: Path) -> dict[str, Any]:
    rows = _role_rows(root)
    feature_paths = _load_feature_index(root)
    gallery, gallery_dir = _load_gallery(base, m1, root)
    k = int(m1.require("protocol.k"))
    score_dir = root / "score_maps"
    token_dir = root / "per_token_scores"
    for scorer in SCORERS:
        (score_dir / scorer).mkdir(parents=True, exist_ok=True)
    token_dir.mkdir(parents=True, exist_ok=True)
    diagnostics = {"distance_min": float("inf"), "distance_max": 0.0, "zero_distance_count": 0,
                   "neighbor_duplicate_count": 0, "distance_floor_applied_count": 0,
                   "cir_denominator_floor_applied_count": 0, "total_token_count": 0,
                   "distance_floor": 1e-12, "cir_denominator_floor": 1e-6,
                   "score_nan_inf_count": 0, "rows": 0,
                   "distance_search": "faiss.IndexFlatL2 exact; returned distances already square-rooted by ExactL2Index"}
    index_rows = []
    index = gallery.index
    for number, row in enumerate(rows, start=1):
        sample_id = str(row["sample_id"])
        if sample_id not in feature_paths:
            raise RuntimeError(f"Feature missing for {sample_id}")
        feature, _ = load_features(feature_paths[sample_id])
        vectors = feature[0]
        result = index.search(vectors, k, batch_size=4096)
        distances = result.distances.astype(np.float32, copy=False)
        diagnostics["distance_min"] = min(diagnostics["distance_min"], float(np.min(distances)))
        diagnostics["distance_max"] = max(diagnostics["distance_max"], float(np.max(distances)))
        values, counts = _knn_scores(distances, result.indices)
        diagnostics["zero_distance_count"] += counts["zero_distance_count"]
        diagnostics["neighbor_duplicate_count"] += counts["neighbor_duplicate_count"]
        diagnostics["distance_floor_applied_count"] += counts["distance_floor_applied_count"]
        diagnostics["cir_denominator_floor_applied_count"] += counts["cir_denominator_floor_applied_count"]
        diagnostics["total_token_count"] += counts["token_count"]
        diagnostics["rows"] += 1
        maps = {}
        for scorer, scores in values.items():
            path = _map_path(root, scorer, sample_id)
            np.save(path, scores.reshape(GRID).astype(np.float32, copy=False), allow_pickle=False)
            maps[scorer] = {"path": str(path), "sha256": sha256_file(path), "token_min": float(np.min(scores)),
                            "token_max": float(np.max(scores))}
        token_path = token_dir / f"{_hash_id(sample_id)}.npz"
        np.savez_compressed(token_path, distances=distances, neighbor_indices=result.indices,
                            **{scorer: scores for scorer, scores in values.items()})
        index_rows.append({"sample_id": sample_id, "maps": maps,
                           "per_token_path": str(token_path), "per_token_sha256": sha256_file(token_path),
                           "feature_path": str(feature_paths[sample_id]),
                           "status": "completed", "frame_index": row["frame_index"], "compression": row["compression"],
                           "is_real": row["is_real"], "manipulation": row["manipulation"]})
        if number % 100 == 0 or number == len(rows):
            print(f"[m1 scores] {number}/{len(rows)}", flush=True)
    write_jsonl(root / "score_index.jsonl", index_rows)
    diagnostics["distance_min"] = None if diagnostics["rows"] == 0 else diagnostics["distance_min"]
    total_distances = max(1, diagnostics["total_token_count"] * k)
    diagnostics["distance_floor_applied_fraction"] = float(diagnostics["distance_floor_applied_count"] / total_distances)
    diagnostics["cir_denominator_floor_applied_fraction"] = float(
        diagnostics["cir_denominator_floor_applied_count"] / max(1, diagnostics["total_token_count"])
    )
    write_json(root / "score_diagnostics.json", diagnostics)
    return {"status": "completed", "rows": len(rows), "gallery": str(gallery_dir), "scorers": list(SCORERS),
            "diagnostics": diagnostics}


def _score_index(root: Path) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(root / "score_index.jsonl")
    return {str(row["sample_id"]): row for row in rows}


def _map_from_index(index: Mapping[str, Mapping[str, Any]], sample_id: str, scorer: str) -> np.ndarray:
    item = index[str(sample_id)]["maps"][scorer]
    path = Path(str(item["path"]))
    if sha256_file(path) != item["sha256"]:
        raise ValueError(f"Score-map hash mismatch for {sample_id}/{scorer}")
    return upsample_score_map(_read_map(path), grid=GRID, output_size=IMAGE_SIZE)


def _linear_quantile(values: np.ndarray, q: float) -> float:
    value = np.asarray(values, dtype=np.float64).reshape(-1)
    if value.size == 0 or not np.isfinite(value).all():
        raise ValueError("Empty/non-finite threshold pool")
    return float(np.quantile(value, q, method="linear"))


def _binary_metrics(scores: np.ndarray, mask: np.ndarray, threshold: float) -> dict[str, Any]:
    from sklearn.metrics import average_precision_score, roc_auc_score
    values = np.asarray(scores, dtype=np.float32).reshape(-1)
    labels = np.asarray(mask, dtype=np.uint8).reshape(-1)
    if values.shape != labels.shape or not np.isfinite(values).all() or not np.isin(labels, [0, 1]).all():
        raise ValueError("M1 score/mask shape or values invalid")
    positive = int(labels.sum())
    total = int(labels.size)
    pred = values > float(threshold)
    tp = int(np.sum(pred & (labels == 1)))
    fp = int(np.sum(pred & (labels == 0)))
    fn = int(np.sum(~pred & (labels == 1)))
    union = tp + fp + fn
    denom = 2 * tp + fp + fn
    return {
        "pixel_ap": float(average_precision_score(labels, values)) if 0 < positive < total else None,
        "pixel_auroc": float(roc_auc_score(labels, values)) if 0 < positive < total else None,
        "fixed_threshold": float(threshold),
        "f1": float(2 * tp / denom) if denom else 1.0,
        "iou": float(tp / union) if union else 1.0,
        "foreground_tpr": float(tp / positive) if positive else None,
        "fake_background_fpr": float(fp / (total - positive)) if positive < total else None,
        "positive_pixels": positive, "n_valid_pixels": total,
        "mask_positive_fraction": float(positive / total) if total else None,
        "undefined_reason": "LOCALIZATION_UNDEFINED" if positive in (0, total) else None,
    }


def _pooled_maps(index: Mapping[str, Mapping[str, Any]], rows: Mapping[str, Mapping[str, Any]],
                 sample_ids: Sequence[str], scorer: str, masks: bool = False) -> tuple[np.ndarray, np.ndarray | None]:
    maps = [_map_from_index(index, sample_id, scorer).reshape(-1) for sample_id in sample_ids]
    scores = np.concatenate(maps)
    if not masks:
        return scores, None
    labels = []
    for sample_id in sample_ids:
        mask = _validated_mask(rows[sample_id])
        labels.append(mask.reshape(-1))
    return scores, np.concatenate(labels)


def _frame_groups(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str, str, bool], list[str]]:
    result: dict[tuple[str, str, str, bool], list[str]] = defaultdict(list)
    for row in rows:
        # T3 has one entry per row/position; videos are pooled across the three frames.
        key = (str(row["compression"]), str(row["manipulation"]), str(row["video_name"]), bool(row["is_real"]))
        result[key].append(str(row["sample_id"]))
    return result


def _macro(values_by_manip: Mapping[str, Sequence[float]]) -> float | None:
    means = [float(np.mean(values)) for values in values_by_manip.values() if len(values)]
    return float(np.mean(means)) if means else None


def _template_from_real(real_ids: Sequence[str], index: Mapping[str, Mapping[str, Any]], scorer: str) -> np.ndarray:
    maps = [_map_from_index(index, sid, scorer).reshape(448, 448) for sid in real_ids]
    if not maps:
        raise ValueError("No real maps for spatial template")
    return np.mean(np.stack(maps, axis=0), axis=0).astype(np.float32)


def calibrate_stage(m1: Config, base: Config, root: Path) -> dict[str, Any]:
    rows = _role_rows(root)
    index = _score_index(root)
    groups = _frame_groups(rows)
    real_ids = [sid for (compression, _, _, is_real), ids in groups.items() if compression == "c23" and is_real for sid in ids]
    thresholds = {}
    template_paths = {}
    real_checks = {}
    for scorer in SCORERS:
        scores, _ = _pooled_maps(index, {str(r["sample_id"]): r for r in rows}, real_ids, scorer)
        threshold = _linear_quantile(scores, float(m1.require("protocol.threshold_quantile")))
        thresholds[scorer] = threshold
        template = _template_from_real(real_ids, index, scorer)
        path = root / "spatial_templates" / f"{scorer}.npy"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, template.astype(np.float32), allow_pickle=False)
        template_paths[scorer] = {"path": str(path), "sha256": sha256_file(path)}
        real_checks[scorer] = {"calibration_pixels": int(scores.size),
                               "real_c23_calibration_fpr": float(np.mean(scores > threshold))}
    payload = {"thresholds": thresholds, "source": "validation_real_c23_T3_pooled_pixels",
               "quantile": 0.99, "numpy_method": "linear", "comparison": ">", "scorers": list(SCORERS),
               "calibration_real_sample_ids": real_ids, "real_checks": real_checks,
               "template_paths": template_paths, "fixed_across": ["fake_c23", "real_c23", "fake_c40", "real_c40"]}
    write_json(root / "threshold_summary.json", payload)
    return {"status": "completed", "thresholds": thresholds, "real_checks": real_checks}


def _group_fake_metrics(root: Path) -> list[dict[str, Any]]:
    rows = read_jsonl(root / "per_video_metrics.jsonl")
    return [row for row in rows if not bool(row.get("is_real"))]


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def _aggregate_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    metric_names = (
        "pixel_ap", "pixel_auroc", "mask_positive_fraction", "ap_minus_prevalence", "paired_real_virtual_mask_ap",
        "ap_minus_paired_real", "delta_score_inside", "delta_score_outside", "delta_score_all",
        "region_difference_of_differences", "real_spatial_template_ap", "f1", "iou",
        "fake_background_fpr",
    )
    tables: list[dict[str, Any]] = []
    for compression in COMPRESSIONS:
        for scorer in SCORERS:
            selected = [row for row in rows if row.get("compression") == compression and row.get("scorer") == scorer
                        and row.get("eligibility_status") in {"eligible", "LOCALIZATION_UNDEFINED"}]
            for metric in metric_names:
                per_manipulation: dict[str, Any] = {}
                for manipulation in MANIPULATIONS:
                    values = [float(row[metric]) for row in selected
                              if row.get("manipulation") == manipulation and row.get(metric) is not None]
                    per_manipulation[manipulation] = {
                        "mean": float(np.mean(values)) if values else None,
                        "n": len(values),
                    }
                means = [value["mean"] for value in per_manipulation.values() if value["mean"] is not None]
                tables.append({"compression": compression, "scorer": scorer, "metric": metric,
                               "per_manipulation": per_manipulation,
                               "macro": float(np.mean(means)) if means else None,
                               "status": "completed" if means else "undefined"})

    real_tables: list[dict[str, Any]] = []
    # ``rows`` already contains both fake and real per-video metric rows.
    # Re-reading via ``root`` here was invalid because this helper intentionally
    # receives only the metric sequence and has no ``root`` parameter.
    all_rows = rows
    for compression in COMPRESSIONS:
        for scorer in SCORERS:
            real = [row for row in all_rows if row.get("is_real") and row.get("compression") == compression
                    and row.get("scorer") == scorer and row.get("real_fpr") is not None]
            values = [float(row["real_fpr"]) for row in real]
            real_tables.append({"compression": compression, "scorer": scorer,
                                "real_fpr_mean": float(np.mean(values)) if values else None,
                                "real_fpr_pooled": (
                                    float(np.sum([row["positive_predictions"] for row in real]) /
                                          max(1, np.sum([row["n_valid_pixels"] for row in real])))
                                    if real else None),
                                "n_videos": len(real),
                                "role": "real_c23_calibration_FPR" if compression == "c23" else "real_c40_transfer_FPR"})
    return {"status": "completed", "tables": tables, "real_tables": real_tables,
            "macro_definition": "equal-weight manipulation mean after per-video pooled-T3 metrics",
            "main_scorer": PRIMARY, "main_condition": "c23"}


def evaluate_stage(m1: Config, base: Config, root: Path) -> dict[str, Any]:
    rows = _role_rows(root)
    row_by_id = {str(row["sample_id"]): row for row in rows}
    score_index = _score_index(root)
    threshold_payload = read_json(root / "threshold_summary.json")
    thresholds = {str(key): float(value) for key, value in threshold_payload["thresholds"].items()}
    templates = {}
    for scorer, item in threshold_payload["template_paths"].items():
        path = Path(str(item["path"]))
        if sha256_file(path) != item["sha256"]:
            raise ValueError(f"Spatial template hash mismatch: {path}")
        templates[scorer] = np.load(path, allow_pickle=False).astype(np.float32)
    pairing = read_jsonl(root / "pairing_manifest.jsonl")
    pairing_by_fake: dict[str, dict[str, Any]] = {str(row["fake_sample_id"]): row for row in pairing}
    groups = _frame_groups(rows)
    metric_rows: list[dict[str, Any]] = []
    frame_diagnostic_rows: list[dict[str, Any]] = []
    paired_region_rows: list[dict[str, Any]] = []
    exclusion_rows: list[dict[str, Any]] = []

    # Real FPR is recorded separately and never used as a fake localization label.
    for (compression, _, video_name, is_real), sample_ids in sorted(groups.items()):
        if not is_real:
            continue
        for scorer in SCORERS:
            scores, _ = _pooled_maps(score_index, row_by_id, sample_ids, scorer)
            threshold = thresholds[scorer]
            positive_predictions = int(np.sum(scores > threshold))
            metric_rows.append({
                "run_id": m1.require("protocol.namespace"), "scorer": scorer, "compression": compression,
                "manipulation": "original", "fake_video_id": None, "video_id": video_name,
                "target_original_id": video_name, "source_original_id": None, "source_group_id": None,
                "frame_indices": [int(row_by_id[sid]["frame_index"]) for sid in sample_ids],
                "sample_ids": list(sample_ids), "n_valid_pixels": int(scores.size),
                "real_fpr": float(np.mean(scores > threshold)), "positive_predictions": positive_predictions,
                "threshold_role": "real_c23_calibration_FPR" if compression == "c23" else "real_c40_transfer_FPR",
                "is_real": True, "eligibility_status": "eligible",
            })

    for compression in COMPRESSIONS:
        for manipulation in MANIPULATIONS:
            fake_groups = sorted((key, ids) for key, ids in groups.items()
                                 if key[0] == compression and key[1] == manipulation and not key[3])
            for (_, _, fake_video_id, _), fake_sample_ids in fake_groups:
                fake_sample_ids = sorted(fake_sample_ids, key=lambda sid: (float(row_by_id[sid]["frame_fraction"]), int(row_by_id[sid]["frame_index"])))
                pair_rows = [pairing_by_fake.get(sid) for sid in fake_sample_ids]
                missing_pairs = [sid for sid, pair in zip(fake_sample_ids, pair_rows) if pair is None]
                if missing_pairs:
                    exclusion_rows.append({"compression": compression, "manipulation": manipulation,
                                           "fake_video_id": fake_video_id, "reason": "missing_pairing",
                                           "sample_ids": missing_pairs})
                    continue
                target_sample_ids = [str(pair["target_real_sample_id"]) for pair in pair_rows if pair]
                if any(sid not in row_by_id for sid in target_sample_ids):
                    exclusion_rows.append({"compression": compression, "manipulation": manipulation,
                                           "fake_video_id": fake_video_id, "reason": "missing_target_real_rows"})
                    continue
                pair_group_ids = {str(pair["source_group_id"]) for pair in pair_rows}
                if len(pair_group_ids) != 1 or "None" in pair_group_ids:
                    exclusion_rows.append({"compression": compression, "manipulation": manipulation,
                                           "fake_video_id": fake_video_id, "reason": "inconsistent_pair_source_groups",
                                           "source_group_ids": sorted(pair_group_ids)})
                    continue
                try:
                    for fake_sid, real_sid, pair in zip(fake_sample_ids, target_sample_ids, pair_rows):
                        _validate_pair_mask_provenance(pair, row_by_id[fake_sid], row_by_id[real_sid])
                except (FileNotFoundError, OSError, ValueError, TypeError) as exc:
                    exclusion_rows.append({"compression": compression, "manipulation": manipulation,
                                           "fake_video_id": fake_video_id, "reason": "blocked_mask_provenance",
                                           "detail": f"{type(exc).__name__}: {exc}",
                                           "sample_ids": list(fake_sample_ids)})
                    continue
                base = {
                    "run_id": m1.require("protocol.namespace"), "compression": compression,
                    "manipulation": manipulation, "fake_video_id": fake_video_id,
                    "video_id": fake_video_id,
                    "target_original_id": row_by_id[fake_sample_ids[0]].get("target_id"),
                    "source_original_id": row_by_id[fake_sample_ids[0]].get("source_id"),
                    "source_group_id": next(iter(pair_group_ids)),
                    "frame_indices": [int(row_by_id[sid]["frame_index"]) for sid in fake_sample_ids],
                    "sample_ids": list(fake_sample_ids), "target_real_sample_ids": target_sample_ids,
                    "is_real": False,
                }
                for scorer in SCORERS:
                    fake_scores, mask = _pooled_maps(score_index, row_by_id, fake_sample_ids, scorer, masks=True)
                    real_scores = np.concatenate([_map_from_index(score_index, sid, scorer).reshape(-1)
                                                  for sid in target_sample_ids])
                    template_scores = np.tile(templates[scorer].reshape(-1), len(fake_sample_ids))
                    if mask is None or real_scores.shape != fake_scores.shape:
                        metric_rows.append({**base, "scorer": scorer, "eligibility_status": "blocked_pair_support",
                                            "undefined_reason": "pair_support_shape_mismatch"})
                        continue
                    metrics = _binary_metrics(fake_scores, mask, thresholds[scorer])
                    real_virtual = _binary_metrics(real_scores, mask, thresholds[scorer])
                    template_metric = _binary_metrics(template_scores, mask, thresholds[scorer])
                    delta = fake_scores.astype(np.float64) - real_scores.astype(np.float64)
                    inside = mask.astype(bool)
                    outside = ~inside
                    delta_inside = float(np.mean(delta[inside])) if np.any(inside) else None
                    delta_outside = float(np.mean(delta[outside])) if np.any(outside) else None
                    delta_all = float(np.mean(delta)) if delta.size else None
                    region_diff = (delta_inside - delta_outside) if delta_inside is not None and delta_outside is not None else None
                    row = {**base, "scorer": scorer, **metrics,
                           "paired_real_virtual_mask_ap": real_virtual["pixel_ap"],
                           "ap_minus_prevalence": (metrics["pixel_ap"] - metrics["mask_positive_fraction"]
                                                    if metrics["pixel_ap"] is not None else None),
                           "ap_minus_paired_real": (metrics["pixel_ap"] - real_virtual["pixel_ap"]
                                                     if metrics["pixel_ap"] is not None and real_virtual["pixel_ap"] is not None else None),
                           "delta_score_inside": delta_inside, "delta_score_outside": delta_outside,
                           "delta_score_all": delta_all,
                           "region_difference_of_differences": region_diff,
                           "real_spatial_template_ap": template_metric["pixel_ap"],
                           "paired_real_f1": real_virtual["f1"], "paired_real_iou": real_virtual["iou"],
                           "eligibility_status": "eligible" if metrics["undefined_reason"] is None else "LOCALIZATION_UNDEFINED",
                           "paired_real_undefined_reason": real_virtual["undefined_reason"],
                           "template_undefined_reason": template_metric["undefined_reason"]}
                    metric_rows.append(row)
                    paired_region_rows.append({
                        "run_id": row["run_id"], "compression": compression, "manipulation": manipulation,
                        "fake_video_id": fake_video_id, "scorer": scorer,
                        "source_group_id": row["source_group_id"],
                        "mask_positive_pixels": metrics["positive_pixels"],
                        "delta_score_inside": delta_inside, "delta_score_outside": delta_outside,
                        "delta_score_all": delta_all,
                        "region_difference_of_differences": region_diff,
                    })
                    for fake_sid, real_sid in zip(fake_sample_ids, target_sample_ids):
                        frame_scores = _map_from_index(score_index, fake_sid, scorer).reshape(-1)
                        frame_real_scores = _map_from_index(score_index, real_sid, scorer).reshape(-1)
                        frame_mask = _validated_mask(row_by_id[fake_sid]).reshape(-1)
                        frame_fake_metric = _binary_metrics(frame_scores, frame_mask, thresholds[scorer])
                        frame_real_metric = _binary_metrics(frame_real_scores, frame_mask, thresholds[scorer])
                        frame_delta = frame_scores.astype(np.float64) - frame_real_scores.astype(np.float64)
                        frame_inside = frame_mask.astype(bool)
                        frame_outside = ~frame_inside
                        frame_diagnostic_rows.append({
                            "run_id": row["run_id"], "compression": compression, "manipulation": manipulation,
                            "fake_video_id": fake_video_id, "fake_sample_id": fake_sid,
                            "target_real_sample_id": real_sid, "scorer": scorer,
                            "frame_index": int(row_by_id[fake_sid]["frame_index"]),
                            "n_valid_pixels": int(frame_mask.size),
                            "mask_positive_fraction": frame_fake_metric["mask_positive_fraction"],
                            "fake_pixel_ap": frame_fake_metric["pixel_ap"],
                            "paired_real_virtual_mask_ap": frame_real_metric["pixel_ap"],
                            "ap_minus_paired_real": (frame_fake_metric["pixel_ap"] - frame_real_metric["pixel_ap"]
                                                      if frame_fake_metric["pixel_ap"] is not None and frame_real_metric["pixel_ap"] is not None else None),
                            "delta_score_inside": float(np.mean(frame_delta[frame_inside])) if np.any(frame_inside) else None,
                            "delta_score_outside": float(np.mean(frame_delta[frame_outside])) if np.any(frame_outside) else None,
                            "delta_score_all": float(np.mean(frame_delta)) if frame_delta.size else None,
                        })
    write_jsonl(root / "per_frame_diagnostics.jsonl", frame_diagnostic_rows)
    write_jsonl(root / "paired_region_statistics.jsonl", paired_region_rows)
    write_jsonl(root / "per_video_metrics.jsonl", metric_rows)
    write_jsonl(root / "eligibility_and_exclusions.jsonl", exclusion_rows + [
        {"compression": row.get("compression"), "manipulation": row.get("manipulation"),
         "fake_video_id": row.get("fake_video_id"), "scorer": row.get("scorer"),
         "eligibility_status": row.get("eligibility_status"), "undefined_reason": row.get("undefined_reason")}
        for row in metric_rows if not row.get("is_real")
    ])
    aggregate = _aggregate_metrics(metric_rows)
    write_json(root / "aggregate_metrics.json", aggregate)
    result = {"status": "completed", "fake_video_metric_rows": sum(not row.get("is_real") for row in metric_rows),
              "real_metric_rows": sum(bool(row.get("is_real")) for row in metric_rows),
              "exclusions": len(exclusion_rows), "aggregate": str(root / "aggregate_metrics.json")}
    write_json(root / "evaluation_summary.json", result)
    return result


def _bootstrap_draws(values: Sequence[Mapping[str, Any]], metric: str, *, group_draws: np.ndarray,
                     group_ids: Sequence[str], replicates: int = 2000) -> tuple[np.ndarray, int]:
    by_group_manip: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in values:
        value = row.get(metric)
        if value is None or row.get("eligibility_status") not in {"eligible", "LOCALIZATION_UNDEFINED"}:
            continue
        group = str(row.get("source_group_id"))
        manip = str(row.get("manipulation"))
        by_group_manip[group][manip].append(float(value))
    arrays = np.full(replicates, np.nan, dtype=np.float64)
    valid = 0
    for replicate in range(replicates):
        selected_groups = [str(group_ids[int(index)]) for index in group_draws[replicate]]
        means = []
        for manipulation in MANIPULATIONS:
            selected_values: list[float] = []
            for group in selected_groups:
                selected_values.extend(by_group_manip.get(group, {}).get(manipulation, []))
            if not selected_values:
                means = []
                break
            means.append(float(np.mean(selected_values)))
        if len(means) == len(MANIPULATIONS):
            arrays[replicate] = float(np.mean(means))
            valid += 1
    return arrays, valid


def _historical_draws(values: Sequence[Mapping[str, Any]], metric: str, *, rng: np.random.Generator,
                      replicates: int = 2000) -> tuple[np.ndarray, int]:
    by_manip: dict[str, list[float]] = defaultdict(list)
    for row in values:
        value = row.get(metric)
        if value is None or row.get("eligibility_status") not in {"eligible", "LOCALIZATION_UNDEFINED"}:
            continue
        by_manip[str(row["manipulation"])].append(float(value))
    arrays = np.full(replicates, np.nan, dtype=np.float64)
    valid = 0
    for replicate in range(replicates):
        means = []
        for manipulation in MANIPULATIONS:
            values_for_manip = np.asarray(by_manip.get(manipulation, []), dtype=np.float64)
            if values_for_manip.size == 0:
                means = []
                break
            draw = rng.integers(0, values_for_manip.size, size=values_for_manip.size)
            means.append(float(np.mean(values_for_manip[draw])))
        if len(means) == len(MANIPULATIONS):
            arrays[replicate] = float(np.mean(means))
            valid += 1
    return arrays, valid


def _interval(values: np.ndarray, confidence: float) -> list[float | None]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return [None, None]
    alpha = (1.0 - confidence) / 2.0
    return [float(v) for v in np.quantile(finite, [alpha, 1.0 - alpha], method="linear")]


def _bootstrap_entry(metric_rows: Sequence[Mapping[str, Any]], scorer: str, compression: str,
                     metric: str, *, group_draws: np.ndarray, group_ids: Sequence[str],
                     cluster_draws: np.ndarray, cluster_valid: int,
                     historical_draws: np.ndarray, historical_valid: int) -> dict[str, Any]:
    values = [row for row in metric_rows if not row.get("is_real") and row.get("scorer") == scorer
              and row.get("compression") == compression]
    per_manip: dict[str, list[float]] = {}
    for manipulation in MANIPULATIONS:
        per_manip[manipulation] = [float(row[metric]) for row in values if row.get("manipulation") == manipulation
                                   and row.get(metric) is not None and row.get("eligibility_status") in {"eligible", "LOCALIZATION_UNDEFINED"}]
    means = [float(np.mean(item)) for item in per_manip.values() if item]
    point = float(np.mean(means)) if len(means) == len(MANIPULATIONS) else None
    return {"scorer": scorer, "compression": compression, "metric": metric, "point": point,
            "ci95": _interval(cluster_draws, 0.95), "ci97_5": _interval(cluster_draws, 0.975),
            "cluster_valid_replicates": cluster_valid, "cluster_total_replicates": int(group_draws.shape[0]),
            "historical_ci95": _interval(historical_draws, 0.95), "historical_valid_replicates": historical_valid,
            "n_source_groups": len(group_ids), "n_per_manipulation": {key: len(value) for key, value in per_manip.items()},
            "estimand": "equal-weight macro over four manipulations; within each manipulation, pooled T3 fake-video rows",
            "cluster_bootstrap_estimand": "source-group cluster resampling with replacement; all fake videos in a drawn group move together",
            "historical_bootstrap_note": "auxiliary manipulation-stratified video bootstrap; shared target/source originals may make its CI optimistic"}


def _decision_from_results(results: Sequence[Mapping[str, Any]], aggregate: Mapping[str, Any], *,
                           source_grouping_status: str = "incomplete",
                           pairing_status: str = "incomplete") -> dict[str, Any]:
    def find(scorer: str, compression: str, metric: str) -> Mapping[str, Any] | None:
        return next((row for row in results if row["scorer"] == scorer and row["compression"] == compression and row["metric"] == metric), None)
    u = find(PRIMARY, "c23", "ap_minus_prevalence")
    d = find(PRIMARY, "c23", "ap_minus_paired_real")
    r = find(PRIMARY, "c23", "region_difference_of_differences")
    d_subgroup = [row for row in _metric_rows_for_decision(aggregate, PRIMARY, "c23", "ap_minus_paired_real")]
    main_available = all(item is not None for item in (u, d, r))
    primary_d_positive_count = sum(value is not None and value > 0 for value in d_subgroup)
    secondary_signals = [
        row for row in results
        if row.get("scorer") in SECONDARY and row.get("compression") == "c23"
        and row.get("metric") == "ap_minus_paired_real"
        and row.get("point") is not None and row.get("point") >= 0.02
        and row.get("ci97_5", [None])[0] is not None and row.get("ci97_5", [None])[0] > 0
    ]
    if source_grouping_status != "verified" or pairing_status != "verified":
        status = "INCONCLUSIVE_DEPENDENCE"
        reason = "Source/target association groups or fake/target pairing provenance are incomplete; continuation and conditional diagnostics are blocked."
    elif not main_available:
        status = "INCONCLUSIVE"
        reason = "Primary c23 r_k endpoint unavailable or undefined."
    else:
        u_ok = u["point"] is not None and u["point"] >= 0.02 and u["ci97_5"][0] is not None and u["ci97_5"][0] > 0
        d_ok = d["point"] is not None and d["point"] >= 0.02 and d["ci97_5"][0] is not None and d["ci97_5"][0] > 0
        r_positive = r["point"] is not None and r["point"] > 0
        if u_ok and d_ok and primary_d_positive_count >= 2 and r_positive:
            c40_u = find(PRIMARY, "c40", "ap_minus_prevalence")
            c40_d = find(PRIMARY, "c40", "ap_minus_paired_real")
            real_c40 = next((row for row in aggregate.get("real_tables", [])
                             if row.get("scorer") == PRIMARY and row.get("compression") == "c40"), {})
            risk = ((c40_u and c40_u.get("point") is not None and u["point"] is not None and c40_u["point"] < 0.5 * u["point"]) or
                    (c40_d and c40_d.get("point") is not None and d["point"] is not None and c40_d["point"] < 0.5 * d["point"]) or
                    (real_c40.get("real_fpr_mean") is not None and real_c40["real_fpr_mean"] > 0.05))
            status = "LOCALIZATION_WITH_COMPRESSION_RISK" if risk else "CONTINUE_STATIC_LIMITED"
            reason = "Primary c23 r_k U/D/R criteria met; c40 transfer risk evaluated separately."
        elif secondary_signals:
            status = "PRIMARY_NOT_SUPPORTED_SECONDARY_SIGNAL"
            reason = "Primary r_k criteria not jointly supported; a preregistered fixed secondary has a positive c23 paired signal."
        elif primary_d_positive_count == 1:
            status = "METHOD_SPECIFIC_SIGNAL"
            reason = "At most one manipulation has a positive r_k paired localization increment."
        elif u.get("point") is not None and u["point"] >= 0.02 and d.get("point") is not None and d["point"] >= 0:
            status = "DISTRIBUTION_ONLY"
            reason = "Fake score distribution changes do not establish a paired local increment."
        else:
            status = "NO_SUPPORT"
            reason = "Frozen primary and fixed secondary endpoints do not meet continuation criteria."
    continuation_statuses = {"CONTINUE_STATIC_LIMITED", "LOCALIZATION_WITH_COMPRESSION_RISK",
                             "METHOD_SPECIFIC_SIGNAL", "PRIMARY_NOT_SUPPORTED_SECONDARY_SIGNAL"}
    return {"status": status, "reason": reason, "primary_scorer": PRIMARY,
            "source_grouping_status": source_grouping_status, "pairing_status": pairing_status,
            "official_test_locked": True,
            "main_comparisons": {"U": u, "D": d, "R": r},
            "primary_d_positive_count": primary_d_positive_count,
            "secondary_signal_rows": secondary_signals,
            "conditional_diagnostics_triggered": (source_grouping_status == "verified" and
                                                   pairing_status == "verified" and
                                                   status in continuation_statuses)}


def _metric_rows_for_decision(aggregate: Mapping[str, Any], scorer: str, compression: str, metric: str) -> list[float | None]:
    for table in aggregate.get("tables", []):
        if table.get("scorer") == scorer and table.get("compression") == compression and table.get("metric") == metric:
            return [value.get("mean") for value in table.get("per_manipulation", {}).values()]
    return []


def bootstrap_stage(m1: Config, base: Config, root: Path) -> dict[str, Any]:
    metric_rows = _group_fake_metrics(root)
    aggregate = read_json(root / "aggregate_metrics.json")
    source_groups = read_json(root / "source_groups.json")
    group_ids = [str(item["source_group_id"]) for item in source_groups.get("groups", [])]
    if not group_ids:
        raise ValueError("No source groups available for bootstrap")
    replicates = int(m1.require("protocol.bootstrap_replicates"))
    rng = np.random.default_rng(int(m1.require("protocol.bootstrap_seed")))
    group_draws = rng.integers(0, len(group_ids), size=(replicates, len(group_ids)))
    historical_rng = np.random.default_rng(int(m1.require("protocol.bootstrap_seed")) + 1)
    metrics = ("ap_minus_prevalence", "ap_minus_paired_real", "region_difference_of_differences", "delta_score_all",
               "pixel_ap", "paired_real_virtual_mask_ap", "real_spatial_template_ap", "f1", "iou", "fake_background_fpr")
    result_rows = []
    draws: dict[str, np.ndarray] = {}
    valid_counts: dict[str, int] = {}
    for scorer in SCORERS:
        for compression in COMPRESSIONS:
            values = [row for row in metric_rows if row.get("scorer") == scorer and row.get("compression") == compression]
            for metric in metrics:
                historical, hist_valid = _historical_draws(values, metric, rng=historical_rng, replicates=replicates)
                key = f"{compression}__{scorer}__{metric}"
                cluster, cluster_valid = _bootstrap_draws(
                    values, metric, group_draws=group_draws, group_ids=group_ids,
                    replicates=replicates
                )
                draws[f"{key}__cluster"] = cluster
                draws[f"{key}__historical"] = historical
                valid_counts[f"{key}__cluster"] = cluster_valid
                valid_counts[f"{key}__historical"] = hist_valid
                result_rows.append(_bootstrap_entry(metric_rows, scorer, compression, metric,
                                                    group_draws=group_draws, group_ids=group_ids,
                                                    cluster_draws=cluster, cluster_valid=cluster_valid,
                                                    historical_draws=historical, historical_valid=hist_valid))
    np.savez_compressed(root / "bootstrap_cluster_results.npz", group_draws=group_draws,
                        group_ids=np.asarray(group_ids, dtype="<U128"), **draws)
    source_grouping_status = str(source_groups.get("status", "incomplete"))
    pairing_status = _pairing_integrity_status(root)
    payload = {"status": "completed", "seed": int(m1.require("protocol.bootstrap_seed")),
               "replicates": replicates, "source_group_ids": group_ids,
               "source_grouping_status": source_grouping_status, "pairing_status": pairing_status,
               "results": result_rows,
               "main_comparisons": ["c23/r_k/ap_minus_prevalence", "c23/r_k/ap_minus_paired_real"],
               "estimand": "equal-weight macro over four manipulations; within each manipulation, pooled T3 fake-video rows",
               "cluster_bootstrap": "source-group cluster resampling with replacement; all fake videos in a drawn group move together",
               "historical_bootstrap": "manipulation-stratified video bootstrap; auxiliary only and may understate shared target/source dependence"}
    write_json(root / "bootstrap_differences.json", payload)
    decision = _decision_from_results(result_rows, aggregate,
                                      source_grouping_status=source_grouping_status,
                                      pairing_status=pairing_status)
    decision["bootstrap_results_path"] = str(root / "bootstrap_differences.json")
    write_json(root / "decision.json", decision)
    return {"status": "completed", "decision": decision["status"], "results": len(result_rows),
            "bootstrap": str(root / "bootstrap_cluster_results.npz")}

# ---------------------------------------------------------------------------
# Conditional, preregistered confound diagnostics and report assembly.
# These functions are deliberately downstream of the frozen main analysis.


def _token_arrays(root: Path, score_index: Mapping[str, Mapping[str, Any]], sample_id: str) -> dict[str, np.ndarray]:
    item = score_index[str(sample_id)]
    path = Path(str(item["per_token_path"]))
    if sha256_file(path) != item["per_token_sha256"]:
        raise ValueError(f"Per-token score hash mismatch: {path}")
    with np.load(path, allow_pickle=False) as payload:
        arrays = {scorer: np.asarray(payload[scorer], dtype=np.float32) for scorer in SCORERS}
    if any(value.shape != (784,) or not np.isfinite(value).all() for value in arrays.values()):
        raise ValueError(f"Invalid per-token scores for {sample_id}")
    return arrays


def _rank_average(values: np.ndarray) -> np.ndarray:
    value = np.asarray(values, dtype=np.float64).reshape(-1)
    order = np.argsort(value, kind="mergesort")
    sorted_value = value[order]
    ranks = np.empty(value.size, dtype=np.float64)
    start = 0
    while start < value.size:
        end = start + 1
        while end < value.size and sorted_value[end] == sorted_value[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def _spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    a = np.asarray(left, dtype=np.float64).reshape(-1)
    b = np.asarray(right, dtype=np.float64).reshape(-1)
    if a.size != b.size or a.size < 3 or not np.isfinite(a).all() or not np.isfinite(b).all():
        return None
    ar, br = _rank_average(a), _rank_average(b)
    ar -= ar.mean()
    br -= br.mean()
    denominator = float(np.sqrt(np.sum(ar * ar) * np.sum(br * br)))
    return None if denominator == 0 else float(np.sum(ar * br) / denominator)


def _threshold_free_metrics(scores: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    from sklearn.metrics import average_precision_score
    value = np.asarray(scores, dtype=np.float32).reshape(-1)
    labels = np.asarray(mask, dtype=np.uint8).reshape(-1)
    if value.shape != labels.shape or not np.isfinite(value).all() or not np.isin(labels, [0, 1]).all():
        raise ValueError("Invalid threshold-free diagnostic arrays")
    positive = int(labels.sum())
    total = int(labels.size)
    ap = float(average_precision_score(labels, value)) if 0 < positive < total else None
    return {"pixel_ap": ap, "prevalence": float(positive / total) if total else None,
            "ap_minus_prevalence": (ap - positive / total) if ap is not None and total else None,
            "positive_pixels": positive, "n_valid_pixels": total,
            "undefined_reason": "LOCALIZATION_UNDEFINED" if positive in (0, total) else None}


def _diagnostic_records(root: Path, selected_groups: set[str]) -> list[dict[str, Any]]:
    rows = _role_rows(root)
    row_by_id = {str(row["sample_id"]): row for row in rows}
    pairing = {str(row["fake_sample_id"]): row for row in read_jsonl(root / "pairing_manifest.jsonl")}
    fake_rows = [row for row in rows if not row["is_real"] and str(row.get("source_group_id")) in selected_groups]
    groups = _frame_groups(fake_rows)
    records: list[dict[str, Any]] = []
    for (compression, manipulation, video_name, is_real), sample_ids in sorted(groups.items()):
        if is_real:
            continue
        fake_ids = sorted(sample_ids, key=lambda sid: (float(row_by_id[sid]["frame_fraction"]), int(row_by_id[sid]["frame_index"])))
        pair_rows = [pairing.get(sid) for sid in fake_ids]
        if any(item is None for item in pair_rows):
            records.append({"status": "blocked", "compression": compression, "manipulation": manipulation,
                            "fake_video_id": video_name, "fake_sample_ids": fake_ids, "reason": "missing_pairing"})
            continue
        target_ids = [str(item["target_real_sample_id"]) for item in pair_rows if item is not None]
        if any(sid not in row_by_id for sid in target_ids):
            records.append({"status": "blocked", "compression": compression, "manipulation": manipulation,
                            "fake_video_id": video_name, "fake_sample_ids": fake_ids, "reason": "missing_target_real"})
            continue
        pair_group_ids = {str(item["source_group_id"]) for item in pair_rows if item is not None}
        if len(pair_group_ids) != 1 or "None" in pair_group_ids:
            records.append({"status": "blocked", "compression": compression, "manipulation": manipulation,
                            "fake_video_id": video_name, "fake_sample_ids": fake_ids,
                            "reason": "inconsistent_pair_source_groups"})
            continue
        try:
            for fake_sid, target_sid, pair in zip(fake_ids, target_ids, pair_rows):
                _validate_pair_mask_provenance(pair, row_by_id[fake_sid], row_by_id[target_sid])
        except (FileNotFoundError, OSError, ValueError, TypeError) as exc:
            records.append({"status": "blocked", "compression": compression, "manipulation": manipulation,
                            "fake_video_id": video_name, "fake_sample_ids": fake_ids,
                            "reason": "blocked_mask_provenance",
                            "detail": f"{type(exc).__name__}: {exc}"})
            continue
        records.append({"status": "eligible", "compression": compression, "manipulation": manipulation,
                        "fake_video_id": video_name, "fake_sample_ids": fake_ids,
                        "target_real_sample_ids": target_ids,
                        "source_group_id": str(pair_rows[0]["source_group_id"])})
    return records


def _feature_vectors(feature_index: Mapping[str, Path], sample_id: str) -> np.ndarray:
    value, _ = load_features(feature_index[str(sample_id)])
    if value.shape != (1, 784, 768):
        raise ValueError(f"Unexpected feature shape in diagnostic: {sample_id}: {value.shape}")
    return value[0].astype(np.float32, copy=False)


def _unrestricted_diagnostic_scores(feature: np.ndarray, gallery_vectors: np.ndarray) -> dict[str, np.ndarray]:
    result = numpy_exact_l2(gallery_vectors, feature, 20, batch_size=4096)
    values, _ = _knn_scores(result.distances, result.indices)
    return values


def _position_diagnostic_scores(feature: np.ndarray, gallery_by_video: np.ndarray,
                               token_indices: np.ndarray | None) -> dict[str, np.ndarray]:
    if gallery_by_video.shape != (64, 784, 768):
        raise ValueError(f"Expected gallery grouped as [64,784,768], got {gallery_by_video.shape}")
    if token_indices is not None and token_indices.shape != (64,):
        raise ValueError("Expected one fixed token index per gallery video")
    outputs = {scorer: np.empty(784, dtype=np.float32) for scorer in SCORERS}
    video_ids = np.arange(64)
    for token in range(784):
        candidates = gallery_by_video[:, token, :] if token_indices is None else gallery_by_video[video_ids, token_indices, :]
        result = numpy_exact_l2(candidates, feature[token:token + 1], 20)
        values, _ = _knn_scores(result.distances, result.indices)
        for scorer in SCORERS:
            outputs[scorer][token] = values[scorer][0]
    return outputs


def _pooled_diagnostic_metrics(record: Mapping[str, Any], scorer: str,
                               fake_cache: Mapping[str, dict[str, np.ndarray]],
                               real_cache: Mapping[str, dict[str, np.ndarray]],
                               row_by_id: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    def flatten_score(value: np.ndarray) -> np.ndarray:
        array = np.asarray(value, dtype=np.float32)
        if array.shape == (784,):
            return upsample_score_map(array, grid=GRID, output_size=IMAGE_SIZE).reshape(-1)
        if array.shape == IMAGE_SIZE:
            return array.reshape(-1)
        raise ValueError(f"Diagnostic score must be token vector or score map, got {array.shape}")
    fake_scores = np.concatenate([flatten_score(fake_cache[sid][scorer]) for sid in record["fake_sample_ids"]])
    real_scores = np.concatenate([flatten_score(real_cache[sid][scorer]) for sid in record["target_real_sample_ids"]])
    mask = np.concatenate([_validated_mask(row_by_id[sid]).reshape(-1)
                           for sid in record["fake_sample_ids"]])
    fake = _threshold_free_metrics(fake_scores, mask)
    real = _threshold_free_metrics(real_scores, mask)
    return {"pixel_ap": fake["pixel_ap"], "prevalence": fake["prevalence"],
            "ap_minus_prevalence": fake["ap_minus_prevalence"],
            "paired_real_virtual_mask_ap": real["pixel_ap"],
            "ap_minus_paired_real": (fake["pixel_ap"] - real["pixel_ap"]
                                      if fake["pixel_ap"] is not None and real["pixel_ap"] is not None else None),
            "undefined_reason": fake["undefined_reason"]}


def _summarize_diagnostic_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for variant in sorted({str(row["variant"]) for row in rows}):
        for compression in COMPRESSIONS:
            for scorer in SCORERS:
                subset = [row for row in rows if row.get("variant") == variant and row.get("compression") == compression
                          and row.get("scorer") == scorer and row.get("status") == "eligible"]
                per_manipulation = {}
                for manipulation in MANIPULATIONS:
                    values = [row.get("ap_minus_prevalence") for row in subset
                              if row.get("manipulation") == manipulation and row.get("ap_minus_prevalence") is not None]
                    d_values = [row.get("ap_minus_paired_real") for row in subset
                                if row.get("manipulation") == manipulation and row.get("ap_minus_paired_real") is not None]
                    per_manipulation[manipulation] = {"n": len(values),
                                                       "U_mean": float(np.mean(values)) if values else None,
                                                       "D_mean": float(np.mean(d_values)) if d_values else None}
                u = [v["U_mean"] for v in per_manipulation.values() if v["U_mean"] is not None]
                d = [v["D_mean"] for v in per_manipulation.values() if v["D_mean"] is not None]
                output.append({"variant": variant, "compression": compression, "scorer": scorer,
                               "per_manipulation": per_manipulation,
                               "U_macro": float(np.mean(u)) if len(u) == len(MANIPULATIONS) else None,
                               "D_macro": float(np.mean(d)) if len(d) == len(MANIPULATIONS) else None,
                               "n_rows": len(subset)})
    return output


def diagnostics_stage(m1: Config, base: Config, root: Path) -> dict[str, Any]:
    decision = read_json(root / "decision.json")
    source_groups = read_json(root / "source_groups.json")
    source_grouping_status = str(source_groups.get("status", "incomplete"))
    pairing_status = _pairing_integrity_status(root)
    diagnostic_dir = root / "confound_diagnostics"
    diagnostic_dir.mkdir(parents=True, exist_ok=True)
    trigger_statuses = {"CONTINUE_STATIC_LIMITED", "LOCALIZATION_WITH_COMPRESSION_RISK",
                        "METHOD_SPECIFIC_SIGNAL", "PRIMARY_NOT_SUPPORTED_SECONDARY_SIGNAL"}
    triggered = bool(decision.get("status") in trigger_statuses and
                     source_grouping_status == "verified" and pairing_status == "verified")
    common = {"official_test_locked": True, "official_test_data_read": False,
              "validation_previously_used_in_t0": True, "fake_masks_used_for_fitting_or_tuning": False,
              "primary_scorer_changed_after_results": False, "feature_config_mismatch": [],
              "main_threshold_recalibrated_on_c40": False, "conditional_diagnostics_triggered": triggered,
              "source_grouping_status": source_grouping_status, "pairing_status": pairing_status}
    if not triggered or not bool(m1.get("conditional_diagnostics.enabled", True)):
        if source_grouping_status != "verified":
            reason = "Blocked because source/target association groups are incomplete."
        elif pairing_status != "verified":
            reason = "Blocked because fake/target crop and mask pairing provenance is incomplete."
        else:
            reason = "Skipped because the preregistered decision did not trigger conditional diagnostics." if not triggered else "Disabled in frozen configuration."
        payload = {**common, "status": "completed", "conditional_diagnostics_executed": False,
                   "skip_reason": reason,
                   "pairing_status": pairing_status,
                   "gallery_query_overlap_ids": read_json(root / "input_provenance.json").get("gallery_query_overlap_ids", []),
                   "score_nan_inf_count": int(read_json(root / "score_diagnostics.json").get("score_nan_inf_count", 0))}
        write_json(diagnostic_dir / "diagnostics.json", payload)
        write_json(root / "diagnostics.json", payload)
        atomic_write_text(diagnostic_dir / "README.md", f"# Conditional diagnostics\n\n{reason}\n")
        return {"status": "completed", "executed": False, "reason": reason}

    if source_grouping_status != "verified" or pairing_status != "verified":
        raise RuntimeError("Conditional diagnostics cannot run with incomplete source grouping or pairing provenance")
    selection = read_json(root / "diagnostic_selection.json")
    selected_groups = {str(value) for value in selection.get("selected_source_groups", [])}
    records = _diagnostic_records(root, selected_groups)
    eligible_records = [record for record in records if record.get("status") == "eligible"]
    errors: list[str] = [str(record.get("reason")) for record in records if record.get("status") != "eligible"]
    score_index = _score_index(root)
    feature_index = _load_feature_index(root)
    row_by_id = {str(row["sample_id"]): row for row in _role_rows(root)}
    gallery, _ = _load_gallery(base, m1, root)
    gallery_provenance = read_json(root / "input_provenance.json")["gallery"]
    gallery_video_ids = [str(value) for value in gallery_provenance["gallery_video_ids"]]
    gallery_sample_ids = [str(value) for value in gallery_provenance.get("gallery_sample_ids", [])]
    gallery_manifest_rows = read_jsonl(root / "gallery_manifest.jsonl")
    manifest_video_ids = [str(row.get("video_name") or row.get("video_id")) for row in gallery_manifest_rows]
    manifest_sample_ids = [str(row.get("sample_id")) for row in gallery_manifest_rows]
    if (len(gallery_video_ids) != 64 or len(set(gallery_video_ids)) != 64 or
            len(gallery_sample_ids) != 64 or len(set(gallery_sample_ids)) != 64 or
            gallery.vectors.shape != (50176, 768) or len(gallery_manifest_rows) != 64 or
            manifest_video_ids != gallery_video_ids or manifest_sample_ids != gallery_sample_ids):
        raise ValueError("Conditional diagnostics require the verified 64-video gallery")
    if any(int(row.get("token_count", 784)) != 784 for row in gallery_manifest_rows):
        raise ValueError("Gallery manifest must explicitly describe 784 tokens per video")
    gallery_by_video = gallery.vectors.reshape(64, 784, 768)

    # 10.1 Norm and normalized-feature diagnosis.
    norm_rows: list[dict[str, Any]] = []
    raw_cache: dict[str, dict[str, np.ndarray]] = {}
    for record in eligible_records:
        for sid in list(record["fake_sample_ids"]) + list(record["target_real_sample_ids"]):
            if sid not in raw_cache:
                raw_cache[sid] = _token_arrays(root, score_index, sid)
        for sid in record["fake_sample_ids"]:
            feature = _feature_vectors(feature_index, sid)
            norm = np.linalg.norm(feature, axis=1).astype(np.float32)
            for scorer in SCORERS:
                norm_rows.append({"variant": "raw_norm_token", "compression": record["compression"],
                                  "manipulation": record["manipulation"], "fake_video_id": record["fake_video_id"],
                                  "source_group_id": record["source_group_id"], "scorer": scorer,
                                  "sample_id": sid, "spearman": _spearman(raw_cache[sid][scorer], norm)})
    norm_summary = []
    for scorer in SCORERS:
        token_values = [row["spearman"] for row in norm_rows if row["scorer"] == scorer and row["spearman"] is not None]
        by_video = []
        for fake_video in sorted({row["fake_video_id"] for row in norm_rows}):
            vals = [row["spearman"] for row in norm_rows if row["fake_video_id"] == fake_video and row["scorer"] == scorer and row["spearman"] is not None]
            if vals:
                by_video.append(float(np.mean(vals)))
        norm_summary.append({"scorer": scorer, "token_level_mean": float(np.mean(token_values)) if token_values else None,
                             "token_level_median": float(np.median(token_values)) if token_values else None,
                             "video_level_mean": float(np.mean(by_video)) if by_video else None,
                             "video_level_median": float(np.median(by_video)) if by_video else None,
                             "n_token_rows": len(token_values), "n_videos": len(by_video)})
    write_jsonl(diagnostic_dir / "feature_norm_correlations.jsonl", norm_rows)
    write_json(diagnostic_dir / "feature_norm_summary.json", {"status": "completed", "scorers": norm_summary})

    # Recompute only the fixed small query subset with L2-normalized features.
    normalized_cache: dict[str, dict[str, np.ndarray]] = {}
    gallery_normalized = gallery.vectors / np.maximum(np.linalg.norm(gallery.vectors, axis=1, keepdims=True), 1e-12)
    for record in eligible_records:
        for sid in list(record["fake_sample_ids"]) + list(record["target_real_sample_ids"]):
            if sid not in normalized_cache:
                feature = _feature_vectors(feature_index, sid)
                feature = feature / np.maximum(np.linalg.norm(feature, axis=1, keepdims=True), 1e-12)
                normalized_cache[sid] = _unrestricted_diagnostic_scores(feature, gallery_normalized)
    normalized_rows = []
    for record in eligible_records:
        for scorer in SCORERS:
            metrics = _pooled_diagnostic_metrics(record, scorer, normalized_cache, normalized_cache, row_by_id)
            normalized_rows.append({"variant": "normalized_features", **record, "scorer": scorer, **metrics})
    write_jsonl(diagnostic_dir / "normalized_feature_metrics.jsonl", normalized_rows)

    # 10.2 Nested gallery sizes, using the freeze-time gallery order.
    gallery_order = [str(value) for value in selection["gallery_video_order"]]
    gallery_index = {video_id: index for index, video_id in enumerate(gallery_video_ids)}
    gallery_size_rows = []
    gallery_sizes = [int(value) for value in m1.require("conditional_diagnostics.gallery_sizes")]
    size_cache: dict[tuple[int, str], dict[str, np.ndarray]] = {}
    for size in gallery_sizes:
        if size <= 0 or size > 64:
            raise ValueError(f"Invalid diagnostic gallery size: {size}")
        indices = [gallery_index[value] for value in gallery_order[:size]]
        subset_gallery = np.concatenate([gallery_by_video[index] for index in indices], axis=0)
        for record in eligible_records:
            for sid in list(record["fake_sample_ids"]) + list(record["target_real_sample_ids"]):
                key = (size, sid)
                if key not in size_cache:
                    size_cache[key] = _unrestricted_diagnostic_scores(_feature_vectors(feature_index, sid), subset_gallery)
            fake_cache = {sid: size_cache[(size, sid)] for sid in record["fake_sample_ids"]}
            real_cache = {sid: size_cache[(size, sid)] for sid in record["target_real_sample_ids"]}
            for scorer in SCORERS:
                metrics = _pooled_diagnostic_metrics(record, scorer, fake_cache, real_cache, row_by_id)
                gallery_size_rows.append({"variant": f"gallery_size_{size}", "gallery_size": size,
                                          **record, "scorer": scorer, **metrics})
    write_jsonl(diagnostic_dir / "gallery_size_metrics.jsonl", gallery_size_rows)

    # 10.3 Fixed spatial candidate controls.
    matched_indices = np.asarray(selection["matched_token_indices"], dtype=np.int64)
    spatial_rows = []
    spatial_cache: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for record in eligible_records:
        for sid in list(record["fake_sample_ids"]) + list(record["target_real_sample_ids"]):
            feature = _feature_vectors(feature_index, sid)
            spatial_cache[("same_token_index", sid)] = _position_diagnostic_scores(feature, gallery_by_video, None)
            spatial_cache[("matched_count_unrestricted", sid)] = _position_diagnostic_scores(feature, gallery_by_video, matched_indices)
        for variant in ("same_token_index", "matched_count_unrestricted"):
            fake_cache = {sid: spatial_cache[(variant, sid)] for sid in record["fake_sample_ids"]}
            real_cache = {sid: spatial_cache[(variant, sid)] for sid in record["target_real_sample_ids"]}
            for scorer in SCORERS:
                metrics = _pooled_diagnostic_metrics(record, scorer, fake_cache, real_cache, row_by_id)
                spatial_rows.append({"variant": variant, **record, "scorer": scorer, **metrics})
    write_jsonl(diagnostic_dir / "spatial_control_metrics.jsonl", spatial_rows)

    all_variant_rows = normalized_rows + gallery_size_rows + spatial_rows
    write_json(diagnostic_dir / "aggregate.json", {"status": "completed", "norm_summary": norm_summary,
                                                     "threshold_free_aggregate": _summarize_diagnostic_rows(all_variant_rows),
                                                     "selected_source_groups": sorted(selected_groups),
                                                     "fixed_gallery_order": gallery_order,
                                                     "matched_token_indices": matched_indices.tolist()})
    payload = {**common, "status": "completed" if not errors else "partial",
               "conditional_diagnostics_executed": True, "errors": errors,
               "pairing_status": pairing_status if not errors else "partial",
               "gallery_query_overlap_ids": read_json(root / "input_provenance.json").get("gallery_query_overlap_ids", []),
               "score_nan_inf_count": int(read_json(root / "score_diagnostics.json").get("score_nan_inf_count", 0)),
               "diagnostic_source_groups": sorted(selected_groups), "diagnostic_record_count": len(eligible_records)}
    write_json(diagnostic_dir / "diagnostics.json", payload)
    write_json(root / "diagnostics.json", payload)
    atomic_write_text(diagnostic_dir / "README.md", "# Conditional diagnostics\n\nExecuted only after the frozen M1 decision triggered a registered continuation signal. All outputs are threshold-free except the unchanged raw-score main analysis.\n")
    return {"status": payload["status"], "executed": True, "records": len(eligible_records), "errors": errors}


def _aggregate_lookup(aggregate: Mapping[str, Any], scorer: str, compression: str, metric: str) -> Mapping[str, Any]:
    return next((row for row in aggregate.get("tables", []) if row.get("scorer") == scorer and row.get("compression") == compression and row.get("metric") == metric), {})


def _bootstrap_lookup(bootstrap: Mapping[str, Any], scorer: str, compression: str, metric: str) -> Mapping[str, Any]:
    return next((row for row in bootstrap.get("results", []) if row.get("scorer") == scorer and row.get("compression") == compression and row.get("metric") == metric), {})


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float) and not np.isfinite(value):
        return "NA"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def _markdown_effect_table(aggregate: Mapping[str, Any], compression: str, scorer: str) -> str:
    ap = _aggregate_lookup(aggregate, scorer, compression, "pixel_ap")
    prev = _aggregate_lookup(aggregate, scorer, compression, "mask_positive_fraction")
    u = _aggregate_lookup(aggregate, scorer, compression, "ap_minus_prevalence")
    real = _aggregate_lookup(aggregate, scorer, compression, "paired_real_virtual_mask_ap")
    d = _aggregate_lookup(aggregate, scorer, compression, "ap_minus_paired_real")
    f1 = _aggregate_lookup(aggregate, scorer, compression, "f1")
    iou = _aggregate_lookup(aggregate, scorer, compression, "iou")
    bg = _aggregate_lookup(aggregate, scorer, compression, "fake_background_fpr")
    lines = ["| manipulation | AP | π | AP−π | paired-real AP | D | F1 | IoU | background FPR |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for manipulation in MANIPULATIONS:
        values = [table.get("per_manipulation", {}).get(manipulation, {}).get("mean") for table in (ap, prev, u, real, d, f1, iou, bg)]
        lines.append("| " + manipulation + " | " + " | ".join(_fmt(value) for value in values) + " |")
    macro = [table.get("macro") for table in (ap, prev, u, real, d, f1, iou, bg)]
    lines.append("| **macro** | **" + "** | **".join(_fmt(value) for value in macro) + "** |")
    return "\n".join(lines)


def _write_figures(root: Path, aggregate: Mapping[str, Any], bootstrap: Mapping[str, Any]) -> list[dict[str, Any]]:
    figures = root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    catalog: list[dict[str, Any]] = []
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        atomic_write_text(figures / "FIGURE_GENERATION_BLOCKED.txt", f"matplotlib unavailable: {exc}\n")
        return [{"path": "figures/FIGURE_GENERATION_BLOCKED.txt", "status": "blocked"}]

    # Figure 1: manipulation-wise primary localization increments.
    x = np.arange(len(MANIPULATIONS))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10, 5))
    for offset, metric, label, color in ((-width / 2, "ap_minus_prevalence", "AP−π", "#3366cc"),
                                          (width / 2, "ap_minus_paired_real", "fake−paired-real AP", "#dc3912")):
        table = _aggregate_lookup(aggregate, PRIMARY, "c23", metric)
        vals = [table.get("per_manipulation", {}).get(name, {}).get("mean") for name in MANIPULATIONS]
        ax.bar(x + offset, [0 if value is None else value for value in vals], width, label=label, color=color)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x, MANIPULATIONS, rotation=20)
    ax.set_ylabel("macro per-video effect")
    ax.set_title("M1 c23 primary r_k localization effects")
    ax.legend()
    fig.tight_layout()
    path = figures / "main_localization_effects.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    catalog.append({"path": "figures/main_localization_effects.png", "description": "c23 r_k AP−π and fake−paired-real AP by manipulation."})

    # Figure 2: paired-region score change.
    fig, ax = plt.subplots(figsize=(10, 5))
    table = _aggregate_lookup(aggregate, PRIMARY, "c23", "region_difference_of_differences")
    vals = [table.get("per_manipulation", {}).get(name, {}).get("mean") for name in MANIPULATIONS]
    ax.bar(x, [0 if value is None else value for value in vals], color="#109618")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x, MANIPULATIONS, rotation=20)
    ax.set_ylabel("inside Δscore − outside Δscore")
    ax.set_title("M1 c23 r_k paired region difference")
    fig.tight_layout()
    path = figures / "region_delta_rk.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    catalog.append({"path": "figures/region_delta_rk.png", "description": "c23 r_k mask-inside versus mask-outside fake−real score change."})

    # Figure 3: c23/c40 transfer and real FPR.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for compression, color in (("c23", "#3366cc"), ("c40", "#ff9900")):
        vals = []
        for metric in ("pixel_ap", "ap_minus_paired_real"):
            vals.append(_aggregate_lookup(aggregate, PRIMARY, compression, metric).get("macro"))
        axes[0].plot(["AP", "D"], [0 if value is None else value for value in vals], marker="o", label=compression, color=color)
        real = next((row for row in aggregate.get("real_tables", []) if row.get("scorer") == PRIMARY and row.get("compression") == compression), {})
        axes[1].bar(compression, 0 if real.get("real_fpr_pooled") is None else real.get("real_fpr_pooled"), color=color)
    axes[0].set_title("r_k AP and paired D")
    axes[0].set_ylabel("macro value")
    axes[0].legend()
    axes[1].axhline(0.05, color="red", linestyle="--", linewidth=0.8)
    axes[1].set_title("real FPR: c23 calibration vs c40 transfer")
    axes[1].set_ylabel("pooled pixel FPR")
    fig.tight_layout()
    path = figures / "compression_transfer.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    catalog.append({"path": "figures/compression_transfer.png", "description": "c23/c40 r_k AP/D and real FPR transfer diagnostic."})

    # Figure 4+: fixed case groups selected before seeing metrics.
    selection = read_json(root / "diagnostic_selection.json")
    rows = _role_rows(root)
    pairing = {str(row["fake_sample_id"]): row for row in read_jsonl(root / "pairing_manifest.jsonl")}
    score_index = _score_index(root)
    row_by_id = {str(row["sample_id"]): row for row in rows}
    for group_id in selection.get("case_source_groups", []):
        candidates = sorted([row for row in rows if not row["is_real"] and str(row.get("source_group_id")) == str(group_id)
                             and row.get("compression") == "c23"], key=lambda row: (row.get("manipulation"), row.get("video_name"), row.get("frame_index")))
        if not candidates:
            continue
        fake = candidates[0]
        pair = pairing.get(str(fake["sample_id"]))
        if pair is None:
            continue
        real = row_by_id.get(str(pair["target_real_sample_id"]))
        if real is None:
            continue
        mask = _validate_pair_mask_provenance(pair, fake, real)
        fake_crop = np.load(fake["crop_path"], allow_pickle=False)
        real_crop = np.load(real["crop_path"], allow_pickle=False)
        maps = {scorer: _map_from_index(score_index, str(fake["sample_id"]), scorer) for scorer in SCORERS}
        vmin = min(float(np.min(value)) for value in maps.values())
        vmax = max(float(np.max(value)) for value in maps.values())
        fig, axes = plt.subplots(2, 4, figsize=(14, 7))
        axes[0, 0].imshow(real_crop); axes[0, 0].set_title("target-real")
        axes[0, 1].imshow(fake_crop); axes[0, 1].set_title(f"fake: {fake['manipulation']}")
        axes[0, 2].imshow(mask, cmap="gray"); axes[0, 2].set_title("frozen mask")
        axes[0, 3].axis("off")
        for axis, scorer in zip(axes[1], SCORERS):
            image = axis.imshow(maps[scorer], cmap="magma", vmin=vmin, vmax=vmax)
            axis.contour(mask, levels=[0.5], colors="cyan", linewidths=0.5)
            axis.set_title(scorer)
            fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        for axis in axes.reshape(-1):
            axis.axis("off") if not axis.has_data() else None
        fig.suptitle(f"Fixed case {group_id}; shared color range across scorers")
        fig.tight_layout()
        filename = f"case_{_hash_id(str(group_id))[:12]}.png"
        fig.savefig(figures / filename, dpi=160)
        plt.close(fig)
        catalog.append({"path": f"figures/{filename}", "description": f"Preselected case group {group_id}; not selected by result."})
    write_json(root / "figure-catalog.json", catalog)
    return catalog


def finalize_stage(m1: Config, base: Config, root: Path) -> dict[str, Any]:
    aggregate = read_json(root / "aggregate_metrics.json")
    bootstrap = read_json(root / "bootstrap_differences.json")
    decision = read_json(root / "decision.json")
    freeze_receipt = read_json(root / "freeze_receipt.json")
    source_groups = read_json(root / "source_groups.json")
    source_grouping_status = str(source_groups.get("status", "incomplete"))
    pairing_status = _pairing_integrity_status(root)
    frozen_config_status = {
        "status": freeze_receipt.get("status"),
        "config_digest": freeze_receipt.get("config_digest"),
        "base_config_digest": freeze_receipt.get("base_config_digest"),
        "freeze_before_fake_metrics": freeze_receipt.get("freeze_before_fake_metrics"),
        "official_test_locked": freeze_receipt.get("official_test_locked"),
    }
    decision.update({
        "frozen_config_status": frozen_config_status,
        "source_grouping_status": source_grouping_status,
        "pairing_status": pairing_status,
        "official_test_locked": True,
    })
    write_json(root / "decision.json", decision)
    diagnostics_path = root / "diagnostics.json"
    if diagnostics_path.is_file():
        diagnostics = read_json(diagnostics_path)
    else:
        source_grouping_status = str(read_json(root / "source_groups.json").get("status", "incomplete"))
        pairing_status = _pairing_integrity_status(root)
        diagnostics = {"status": "partial", "official_test_locked": True,
                       "official_test_data_read": False, "validation_previously_used_in_t0": True,
                       "fake_masks_used_for_fitting_or_tuning": False,
                       "primary_scorer_changed_after_results": False,
                       "feature_config_mismatch": [], "gallery_query_overlap_ids": [],
                       "source_grouping_status": source_grouping_status,
                       "pairing_status": pairing_status,
                       "score_nan_inf_count": int(read_json(root / "score_diagnostics.json").get("score_nan_inf_count", 0)),
                       "main_threshold_recalibrated_on_c40": False,
                       "conditional_diagnostics_executed": False,
                       "skip_reason": "diagnostics stage was not run before finalize"}
        write_json(diagnostics_path, diagnostics)
    figures = _write_figures(root, aggregate, bootstrap)
    query_rows = _role_rows(root)
    fake_rows = [row for row in query_rows if not row["is_real"]]
    metric_rows = read_jsonl(root / "per_video_metrics.jsonl")
    eligible = [row for row in metric_rows if not row.get("is_real") and row.get("eligibility_status") in {"eligible", "LOCALIZATION_UNDEFINED"}]
    readme = f"""# M1 static geometry pilot\n\n- Namespace: `{m1.require('protocol.namespace')}`\n- Validation-only; official test remains locked.\n- Run order: `prepare → crops → features → scores → calibrate → evaluate → bootstrap → diagnostics (conditional) → finalize`.\n- Query rows: {len(query_rows)}; fake rows: {len(fake_rows)}; source groups: {source_groups.get('group_count')}; per-video scorer rows: {len(metric_rows)}; eligible fake rows: {len(eligible)}.\n- Frozen configuration: **{frozen_config_status.get('status')}**; freeze-before-metrics: **{frozen_config_status.get('freeze_before_fake_metrics')}**; M1/base digests are recorded in `freeze_receipt.json`.\n- Source grouping status: **{source_grouping_status}**; pairing status: **{pairing_status}**.\n- Main reproducible inputs are listed in `freeze_receipt.json` and `input_provenance.json`.\n- Raw per-token distances/neighbor indices are in `per_token_scores/`; score maps are in `score_maps/`.\n- `summary.md` is the short conclusion; `analysis-report.md` and `stats-appendix.md` contain the detailed tables.\n\nOriginal Synthetic Gate B remains FAIL. M0 remains a provisional structural audit. This M1 is a validation-only exploratory pilot, not an independent official-test confirmation. Official test remains locked.\n"""
    atomic_write_text(root / "README.md", readme)
    main_table = _markdown_effect_table(aggregate, "c23", PRIMARY)
    transfer_table = _markdown_effect_table(aggregate, "c40", PRIMARY)
    analysis = f"""# M1 analysis report\n\n## Protocol and support\n\nThis is the preregistered M1-limited-v1.0 validation-only pilot. It reuses T0 T3 frame IDs, a real-only 64-video c23 gallery, frozen DINOv3 ViT-B/16 block 6 features, k=20 exact Euclidean neighbors, and no postprocessing. Validation was previously used in T0; this is not untouched validation.\n\n- Query rows: **{len(query_rows)}**; fake rows: **{len(fake_rows)}**.\n- Source groups: **{source_groups.get('group_count')}**; maximum original IDs per group: **{source_groups.get('max_original_ids_per_group')}**.\n- Eligible fake scorer rows: **{len(eligible)}**.\n- Conditional diagnostics: **{diagnostics.get('conditional_diagnostics_executed')}**.\n- Frozen configuration status: **{frozen_config_status.get('status')}**; freeze-before-fake-metrics: **{frozen_config_status.get('freeze_before_fake_metrics')}**.\n- Source grouping status: **{source_grouping_status}**; pairing status: **{pairing_status}**.\n- Official test locked: **{frozen_config_status.get('official_test_locked')}**.\n\n## c23 primary r_k table\n\n{main_table}\n\n## c40 fixed-threshold transfer table\n\n{transfer_table}\n\n## Decision\n\n**{decision.get('status')}** — {decision.get('reason')}\n\nThe primary comparisons are U = AP−π and D = AP−paired-target-real virtual-mask AP. R is a paired region-difference diagnostic, not a deployment score. c40 thresholds were not recalibrated.\n\n`delta_score_all` is the full-image mean fake−target-real score shift. It is a diagnostic for a global common offset; it is not evidence of localization and is not a deployment score. The mask-inside/outside difference (`region_difference_of_differences`) is the localization-oriented contrast.\n\n## Integrity\n\nOfficial test data were not read by this pipeline. Fake masks were used only after freeze for evaluation, controls, and fixed visualizations; they were not used for fitting, frame selection, gallery selection, or threshold tuning.\n\nOriginal Synthetic Gate B remains FAIL. M0 remains a provisional structural audit. This M1 is a validation-only exploratory pilot, not an independent official-test confirmation. Official test remains locked.\n"""
    atomic_write_text(root / "analysis-report.md", analysis)
    main_boot = [_bootstrap_lookup(bootstrap, PRIMARY, "c23", metric) for metric in ("ap_minus_prevalence", "ap_minus_paired_real", "region_difference_of_differences")]
    stats_lines = ["# Statistical appendix", "", f"Source-group cluster bootstrap: {bootstrap.get('replicates')} fixed-seed replicates; ordinary 95% percentile CIs for all metrics; 97.5% two-sided percentile CIs for U and D as the preregistered two-comparison coverage control.", f"Estimand: {bootstrap.get('estimand', 'equal-weight macro over four manipulations')}.", f"Cluster resampling: {bootstrap.get('cluster_bootstrap', 'source-group cluster resampling with replacement') }.", "", "| scorer | compression | metric | point | 95% CI | 97.5% CI | cluster valid | historical 95% CI |", "|---|---|---|---:|---|---|---:|---|"]
    for row in bootstrap.get("results", []):
        ci = row.get("ci95", [None, None]); ci97 = row.get("ci97_5", [None, None]); hci = row.get("historical_ci95", [None, None])
        stats_lines.append(f"| {row.get('scorer')} | {row.get('compression')} | {row.get('metric')} | {_fmt(row.get('point'))} | [{_fmt(ci[0])}, {_fmt(ci[1])}] | [{_fmt(ci97[0])}, {_fmt(ci97[1])}] | {row.get('cluster_valid_replicates')} / {row.get('cluster_total_replicates')} | [{_fmt(hci[0])}, {_fmt(hci[1])}] |")
    stats_lines.extend(["", "`delta_score_all` measures a full-image common score shift; `region_difference_of_differences` is the mask-inside/outside contrast. Neither is a replacement deployment score.", "The historical manipulation-stratified video bootstrap is auxiliary and may understate dependence from shared target/source originals. No bootstrap tail proportion is reported as a standard p-value."])
    atomic_write_text(root / "stats-appendix.md", "\n".join(stats_lines) + "\n")
    catalog = read_json(root / "figure-catalog.json") if (root / "figure-catalog.json").is_file() else figures
    atomic_write_text(root / "figure-catalog.md", "# Figure catalog\n\n" + "\n".join(f"- `{item.get('path')}` — {item.get('description', item.get('status', ''))}" for item in catalog) + "\n")
    diagnostic_note = "Conditional diagnostics were executed because the frozen decision triggered a registered continuation signal." if diagnostics.get("conditional_diagnostics_executed") else f"Conditional diagnostics were not executed: {diagnostics.get('skip_reason', 'not triggered')}."
    summary = f"""# M1 summary\n\n## Observed\n\n- The run contains {len(fake_rows)} validation fake frame rows and {source_groups.get('group_count')} source-group components; exact counts and exclusions are in `query_manifest.jsonl`, `source_groups.json`, and `eligibility_and_exclusions.jsonl`.\n- The frozen main scorer is `r_k`; fixed secondaries are `mean_r` and `C2`; `CIR_LID` is historical diagnostic only.\n- The main c23 decision is **{decision.get('status')}**.\n- Frozen configuration status: **{frozen_config_status.get('status')}**; freeze-before-fake-metrics: **{frozen_config_status.get('freeze_before_fake_metrics')}**.\n- Source grouping status: **{source_grouping_status}**; pairing status: **{pairing_status}**.\n- Official test locked: **{frozen_config_status.get('official_test_locked')}**.\n\n## Supported\n\n- The only continuation endpoints are the preregistered c23 U and D with source-group bootstrap uncertainty; AP above 0.5 alone is not evidence of localization.\n- `target-real` virtual-mask AP, mask-inside/outside Δscore, c40 transfer FPR, and spatial-template results must be read together; normal mouth/eye or center bias can produce high real virtual-mask AP.\n- `delta_score_all` is a whole-image common-shift diagnostic; the localization-oriented contrast is `region_difference_of_differences`, and neither replaces the frozen deployment scorer.\n- {diagnostic_note}\n\n## Speculative\n\nNo strict manifold, normal/tangent decomposition, identity-disjoint generalization, or official-test claim is established by this pilot. Any apparent dependence on feature norm, gallery size, or token position is a confound diagnostic, not a replacement main method.\n\n## Required stop conditions\n\nDo not unlock official test or automatically launch another experiment. Original Synthetic Gate B remains FAIL. M0 remains a provisional structural audit. This M1 is a validation-only exploratory pilot, not an independent official-test confirmation. Official test remains locked.\n"""
    atomic_write_text(root / "summary.md", summary)
    return {"status": "completed", "decision": decision.get("status"), "figures": len(figures),
            "reports": ["README.md", "analysis-report.md", "stats-appendix.md", "figure-catalog.md", "summary.md"]}


def run_stage(config_path: str | Path, stage: str) -> dict[str, Any]:
    m1, base, root = _context(config_path)
    dispatch = {"prepare": prepare, "crops": crops_stage, "features": features_stage,
                "scores": scores_stage, "calibrate": calibrate_stage, "evaluate": evaluate_stage,
                "bootstrap": bootstrap_stage, "diagnostics": diagnostics_stage, "finalize": finalize_stage}
    if stage not in dispatch:
        raise ValueError(f"Unknown M1 stage: {stage}")
    return dispatch[stage](m1, base, root)
