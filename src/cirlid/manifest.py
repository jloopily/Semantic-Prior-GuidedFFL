"""Manifest construction primitives that do not assume a particular FF++ layout."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .geometry import fractional_frame_index, remap_frame_index
from .io import read_json, read_jsonl, write_csv, write_json, write_jsonl
from .types import ManifestRecord


REQUIRED_FIELDS = {
    "sample_id",
    "split",
    "compression",
    "manipulation",
    "is_real",
    "target_id",
    "video_name",
    "video_path",
    "frame_count",
    "frame_fraction",
    "frame_index",
    "crop_source_video",
    "crop_source_frame",
}


def select_gallery_ids(train_real_ids: Sequence[str], count: int = 64, seed: int = 20260919) -> list[str]:
    ids = sorted({str(value) for value in train_real_ids})
    if len(ids) < count:
        raise ValueError(f"Need at least {count} train-real IDs, got {len(ids)}")
    selected = np.random.default_rng(seed).choice(ids, size=count, replace=False)
    return sorted(str(value) for value in selected)


def split_validation_directed_names(names: Sequence[str], count: int = 64, seed: int = 20260919) -> tuple[list[str], list[str]]:
    unique = sorted({str(value) for value in names})
    if len(unique) < count:
        raise ValueError(f"Need at least {count} validation names, got {len(unique)}")
    permutation = np.random.default_rng(seed).permutation(unique)
    pilot = sorted(str(value) for value in permutation[:count])
    reference_bank = sorted(str(value) for value in permutation[count:])
    return pilot, reference_bank


def make_frame_records(
    *,
    video_name: str,
    split: str,
    compression: str,
    manipulation: str,
    is_real: bool,
    target_id: str,
    source_id: str | None,
    video_path: str,
    mask_path: str | None,
    frame_count: int,
    fractions: Sequence[float],
    crop_source_video: str,
    crop_source_frame: int,
    mask_frame_count: int | None = None,
    seed: int = 20260919,
) -> list[ManifestRecord]:
    records: list[ManifestRecord] = []
    for fraction in fractions:
        frame_index = fractional_frame_index(frame_count, float(fraction))
        if mask_path is None:
            resolved_mask_count = None
            mask_frame_index = None
        else:
            resolved_mask_count = frame_count if mask_frame_count is None else int(mask_frame_count)
            if resolved_mask_count <= 0:
                raise ValueError("mask_frame_count must be positive")
            mask_frame_index = remap_frame_index(frame_index, frame_count, resolved_mask_count)
        sample_id = f"{split}:{compression}:{manipulation}:{video_name}:{frame_index}"
        records.append(
            ManifestRecord(
                sample_id=sample_id,
                split=split,
                compression=compression,
                manipulation=manipulation,
                is_real=is_real,
                target_id=target_id,
                source_id=source_id,
                video_name=video_name,
                video_path=video_path,
                mask_path=mask_path,
                frame_count=frame_count,
                frame_fraction=float(fraction),
                frame_index=frame_index,
                mask_frame_index=mask_frame_index,
                crop_source_video=crop_source_video,
                crop_source_frame=crop_source_frame,
                mask_frame_remapped=bool(mask_path is not None and resolved_mask_count != frame_count),
                mask_frame_count=resolved_mask_count,
                seed=seed,
            )
        )
    return records


def _blank_to_none(value: Any) -> Any:
    return None if value is None or (isinstance(value, str) and not value.strip()) else value


def _coerce_bool(value: Any, *, field: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
    if isinstance(value, (int, np.integer)) and value in (0, 1):
        return bool(value)
    raise ValueError(f"Manifest field {field!r} must be boolean, got {value!r}")


def normalize_record(row: Mapping[str, Any] | ManifestRecord) -> dict[str, Any]:
    """Normalize JSON/CSV scalar fields before validation or downstream use."""

    value = row.to_dict() if isinstance(row, ManifestRecord) else dict(row)
    value["sample_id"] = str(value.get("sample_id", ""))
    value["split"] = str(value.get("split", ""))
    value["compression"] = str(value.get("compression", ""))
    value["manipulation"] = str(value.get("manipulation", ""))
    value["is_real"] = _coerce_bool(value.get("is_real"), field="is_real")
    for field in ("target_id", "video_name", "video_path", "crop_source_video"):
        value[field] = str(value.get(field, ""))
    for field in ("source_id", "mask_path", "exclusion_reason"):
        value[field] = _blank_to_none(value.get(field))
        if value[field] is not None:
            value[field] = str(value[field])
    # Crop artifacts are written after the initial manifest is built.  Keep
    # them when a JSONL manifest is rewritten; silently dropping these fields
    # would make downstream feature extraction unable to find its inputs.
    for field in ("crop_path", "mask_cache_path"):
        value[field] = _blank_to_none(value.get(field))
        if value[field] is not None:
            value[field] = str(value[field])
    for field in ("frame_count", "frame_index", "crop_source_frame"):
        value[field] = int(value.get(field))
    seed_value = _blank_to_none(value.get("seed", 20260919))
    value["seed"] = 20260919 if seed_value is None else int(seed_value)
    value["frame_fraction"] = float(value.get("frame_fraction"))
    value["mask_frame_index"] = _blank_to_none(value.get("mask_frame_index"))
    if value["mask_frame_index"] is not None:
        value["mask_frame_index"] = int(value["mask_frame_index"])
    value["mask_frame_remapped"] = _coerce_bool(value.get("mask_frame_remapped", False), field="mask_frame_remapped")
    value["mask_frame_count"] = _blank_to_none(value.get("mask_frame_count"))
    if value["mask_frame_count"] is not None:
        value["mask_frame_count"] = int(value["mask_frame_count"])
    for field in ("crop_left", "crop_top", "crop_right", "crop_bottom",
                  "crop_source_left", "crop_source_top",
                  "crop_source_right", "crop_source_bottom",
                  "crop_source_width", "crop_source_height",
                  "crop_image_width", "crop_image_height"):
        value[field] = _blank_to_none(value.get(field))
        if value[field] is not None:
            value[field] = int(value[field])
    value["crop_scale"] = _blank_to_none(value.get("crop_scale"))
    if value["crop_scale"] is not None:
        value["crop_scale"] = float(value["crop_scale"])
    return value


def validate_records(rows: Sequence[Mapping[str, Any] | ManifestRecord], *, allow_excluded: bool = True) -> None:
    seen: set[str] = set()
    for row in rows:
        raw = row.to_dict() if isinstance(row, ManifestRecord) else dict(row)
        missing = REQUIRED_FIELDS - raw.keys()
        if missing:
            raise ValueError(f"Manifest row is missing fields: {sorted(missing)}")
        value = normalize_record(raw)
        sample_id = value["sample_id"]
        if not sample_id:
            raise ValueError("Manifest sample_id must not be empty")
        if sample_id in seen:
            raise ValueError(f"Duplicate sample_id: {sample_id}")
        seen.add(sample_id)
        if value["split"] not in {"train", "validation", "test"}:
            raise ValueError(f"Invalid split in {sample_id}: {value['split']!r}")
        if value["compression"] not in {"c23", "c40"}:
            raise ValueError(f"Invalid compression in {sample_id}: {value['compression']!r}")
        if (
            not value["target_id"]
            or not value["video_name"]
            or not value["video_path"]
            or not value["crop_source_video"]
        ):
            raise ValueError(f"Manifest identity/path fields must not be empty in {sample_id}")
        if not np.isfinite(value["frame_fraction"]) or not 0.0 <= value["frame_fraction"] <= 1.0:
            raise ValueError(f"Invalid frame_fraction in {sample_id}")
        count = value["frame_count"]
        index = value["frame_index"]
        if count <= 0 or not 0 <= index < count:
            raise ValueError(f"Invalid frame index/count in {sample_id}")
        expected_index = fractional_frame_index(count, value["frame_fraction"])
        if index != expected_index:
            raise ValueError(
                f"frame_index/frame_fraction disagree in {sample_id}: "
                f"index={index}, expected={expected_index}"
            )
        if value["crop_source_frame"] < 0:
            raise ValueError(f"Invalid crop_source_frame in {sample_id}")
        if value["mask_frame_index"] is None:
            if value["mask_frame_count"] is not None or value["mask_frame_remapped"]:
                raise ValueError(f"Mask metadata requires mask_frame_index in {sample_id}")
        else:
            if value["mask_frame_index"] < 0:
                raise ValueError(f"Invalid mask_frame_index in {sample_id}")
            mask_count = value["mask_frame_count"]
            if value["mask_frame_remapped"] and mask_count is None:
                raise ValueError(
                    f"Remapped mask row must record mask_frame_count in {sample_id}"
                )
            effective_mask_count = count if mask_count is None else mask_count
            if effective_mask_count <= 0 or value["mask_frame_index"] >= effective_mask_count:
                raise ValueError(f"Invalid mask_frame_index/mask_frame_count in {sample_id}")
            if mask_count is not None:
                expected_remapped = mask_count != count
                if value["mask_frame_remapped"] != expected_remapped:
                    raise ValueError(
                        f"mask_frame_remapped disagrees with frame counts in {sample_id}"
                    )
                expected_mask_index = remap_frame_index(index, count, mask_count)
                if value["mask_frame_index"] != expected_mask_index:
                    raise ValueError(
                        f"mask_frame_index/frame_count disagree in {sample_id}: "
                        f"index={value['mask_frame_index']}, expected={expected_mask_index}"
                    )
        if value["is_real"]:
            if (
                value["source_id"] is not None
                or value["mask_path"] is not None
                or value["mask_frame_index"] is not None
                or value["mask_frame_count"] is not None
                or value["mask_frame_remapped"]
                or value["mask_cache_path"] is not None
            ):
                raise ValueError(f"Real manifest row must not have a mask in {sample_id}")
        elif value["exclusion_reason"] is None:
            if value["mask_path"] is None or value["mask_frame_index"] is None:
                raise ValueError(f"Non-excluded fake row must have a mask in {sample_id}")
            if value["source_id"] is None:
                raise ValueError(f"Non-excluded fake row must have source_id in {sample_id}")
            if value["mask_frame_count"] is None:
                raise ValueError(f"Non-excluded fake row must have mask_frame_count in {sample_id}")
        crop_fields = [value[field] for field in ("crop_left", "crop_top", "crop_right", "crop_bottom")]
        if any(field is not None for field in crop_fields):
            if any(field is None for field in crop_fields):
                raise ValueError(f"Crop bbox must be complete in {sample_id}")
            left, top, right, bottom = crop_fields
            if right <= left or bottom <= top:
                raise ValueError(f"Invalid crop bbox in {sample_id}")
        if value["crop_scale"] is not None and value["crop_scale"] <= 0:
            raise ValueError(f"crop_scale must be positive in {sample_id}")
        if not allow_excluded and value["exclusion_reason"] is not None:
            raise ValueError(f"Excluded row encountered: {sample_id}")


def write_manifest(rows: Sequence[Mapping[str, Any] | ManifestRecord], path: str | Path) -> None:
    # Validate the raw mapping first so missing fields produce a useful error
    # instead of an incidental ``int(None)``/``float(None)`` conversion error.
    validate_records(rows)
    normalized = [normalize_record(row) for row in rows]
    destination = Path(path)
    if destination.suffix.lower() == ".jsonl":
        write_jsonl(destination, normalized)
    elif destination.suffix.lower() == ".csv":
        fields = sorted({key for row in normalized for key in row})
        write_csv(destination, normalized, fields)
    elif destination.suffix.lower() == ".json":
        write_json(destination, normalized)
    else:
        raise ValueError("Manifest path must end in .jsonl, .csv, or .json")


def read_manifest(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if source.suffix.lower() == ".jsonl":
        rows = read_jsonl(source)
    elif source.suffix.lower() == ".json":
        rows = read_json(source)
        if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
            raise ValueError("JSON manifest must be a list of objects")
    elif source.suffix.lower() == ".csv":
        import csv

        with source.open("r", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    else:
        raise ValueError("Manifest path must end in .jsonl, .csv, or .json")
    validate_records(rows)
    normalized = [normalize_record(row) for row in rows]
    return normalized
