"""Deterministic FF++ face-crop and mask-cache stage."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from .ffpp import decode_frame_rgb, decode_mask
from .geometry import BBox, crop_and_resize, map_bbox, square_bbox
from .io import read_json, sha256_file, write_json
from .manifest import normalize_record, read_manifest, write_manifest


@dataclass(frozen=True)
class CropCacheSummary:
    status: str
    manifest: str
    rows: int
    processed_rows: int
    cached_images: int
    cached_masks: int
    reused_images: int
    reused_masks: int
    failed_rows: int
    excluded_rows: int
    exclusion_reasons: dict[str, int]
    bbox_count: int


def _dlib_detector():
    try:
        import dlib
    except ImportError as exc:  # pragma: no cover - server dependency
        raise RuntimeError("dlib is required for the frozen HOG face detector") from exc
    return dlib.get_frontal_face_detector()


def detect_largest_face_box(
    image_rgb: np.ndarray,
    *,
    detector: Any,
    upsample_primary: int = 1,
    upsample_retry: int = 2,
) -> BBox:
    """Detect the largest face using only the frozen dlib HOG detector."""

    image = np.asarray(image_rgb)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError(f"Expected uint8 RGB image [H,W,3], got {image.shape} {image.dtype}")
    detections = detector(image, int(upsample_primary))
    if not detections and int(upsample_retry) != int(upsample_primary):
        detections = detector(image, int(upsample_retry))
    if not detections:
        raise ValueError("No face detected by frozen dlib HOG detector")
    rectangles = []
    for rectangle in detections:
        left = int(rectangle.left())
        top = int(rectangle.top())
        right = int(rectangle.right())
        bottom = int(rectangle.bottom())
        if right > left and bottom > top:
            rectangles.append(BBox(left, top, right, bottom))
    if not rectangles:
        raise ValueError("dlib returned no positive-area face box")
    return max(rectangles, key=lambda box: box.width * box.height)


def _artifact_key(sample_id: str) -> str:
    return hashlib.sha256(sample_id.encode("utf-8")).hexdigest()[:24]


def _save_array(path: Path, array: np.ndarray) -> None:
    """Write a numpy array atomically so an interrupted job leaves no .npy."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, np.asarray(array), allow_pickle=False)
    temporary.replace(path)


def _load_cached_array(
    path: Path,
    *,
    shape: tuple[int, ...],
    dtype: np.dtype[Any],
    binary: bool = False,
) -> np.ndarray | None:
    """Return a validated cached array, or ``None`` when it must be rebuilt."""

    if not path.is_file():
        return None
    try:
        value = np.load(path, allow_pickle=False)
    except (OSError, ValueError):
        return None
    if value.shape != shape or value.dtype != dtype:
        return None
    if not np.isfinite(value).all():
        return None
    # RGB crops are also uint8, but are not binary masks. Applying the binary
    # check to every uint8 artifact would invalidate normal RGB crops on every
    # resume and force an unnecessary re-decode.
    if binary and np.any((value != 0) & (value != 1)):
        return None
    return np.asarray(value)


def _set_exclusion(row: dict[str, Any], reason: str) -> None:
    previous = row.get("exclusion_reason")
    row["exclusion_reason"] = reason if not previous else f"{previous};{reason}"


def _bbox_from_row(row: dict[str, Any]) -> BBox | None:
    fields = tuple(row.get(name) for name in ("crop_left", "crop_top", "crop_right", "crop_bottom"))
    if any(value is None for value in fields):
        return None
    left, top, right, bottom = (int(value) for value in fields)
    if right <= left or bottom <= top:
        return None
    return BBox(left, top, right, bottom)


def cache_manifest_crops(
    manifest_path: str | Path,
    *,
    crops_dir: str | Path,
    masks_dir: str | Path,
    input_size: int = 448,
    detector_upsample_primary: int = 1,
    detector_upsample_retry: int = 2,
    square_scale: float = 1.3,
    padding: str = "reflection",
    mask_policy: str = "nonzero",
    mask_threshold: float = 0.5,
    detector: Any | None = None,
    metadata_path: str | Path | None = None,
    progress_every: int = 250,
    exclude_empty_fake_masks: bool = True,
) -> CropCacheSummary:
    """Decode, crop, and cache all manifest rows.

    Face detection is performed only on the original c23 crop-source frame.
    Existing artifacts are reused only after shape/dtype/finite checks; invalid
    or partial files are rebuilt atomically. The function is safe to resume
    after an interrupted server job. T0 disables empty-mask exclusion so that
    label content cannot select its evaluation cohort; the frozen default is
    unchanged.
    """

    if padding != "reflection":
        raise ValueError(f"Frozen crop stage requires reflection padding, got {padding!r}")
    if input_size <= 0 or square_scale <= 0:
        raise ValueError("input_size and square_scale must be positive")
    if progress_every < 0:
        raise ValueError("progress_every must be non-negative")
    rows = [normalize_record(row) for row in read_manifest(manifest_path)]
    crop_root = Path(crops_dir)
    mask_root = Path(masks_dir)
    crop_root.mkdir(parents=True, exist_ok=True)
    mask_root.mkdir(parents=True, exist_ok=True)
    detector = detector or _dlib_detector()

    bbox_cache: dict[tuple[str, int], tuple[BBox, tuple[int, int]]] = {}
    bbox_errors: dict[tuple[str, int], str] = {}
    mask_positive: defaultdict[tuple[str, str, str, str], int] = defaultdict(int)
    mask_expected: defaultdict[tuple[str, str, str, str], int] = defaultdict(int)
    cached_images = 0
    cached_masks = 0
    reused_images = 0
    reused_masks = 0
    failed_rows = 0
    total = len(rows)

    for row_number, row in enumerate(rows, start=1):
        if row.get("exclusion_reason"):
            if progress_every and (row_number == total or row_number % progress_every == 0):
                excluded = sum(bool(item.get("exclusion_reason")) for item in rows[:row_number])
                print(f"[cache_crops] processed={row_number}/{total} excluded={excluded}", flush=True)
            continue
        source_key = (str(row["crop_source_video"]), int(row["crop_source_frame"]))
        try:
            key = _artifact_key(str(row["sample_id"]))
            crop_path = crop_root / f"{key}.npy"
            mask_path = mask_root / f"{key}.npy"
            cached_crop = _load_cached_array(
                crop_path,
                shape=(int(input_size), int(input_size), 3),
                dtype=np.dtype(np.uint8),
                binary=False,
            )
            box = _bbox_from_row(row)
            if cached_crop is not None and box is not None:
                reused_images += 1
                row["crop_path"] = str(crop_path)
                cached_images += 1
            else:
                if source_key in bbox_errors:
                    raise ValueError(bbox_errors[source_key])
                if source_key not in bbox_cache:
                    source_image = decode_frame_rgb(*source_key)
                    detected = detect_largest_face_box(
                        source_image,
                        detector=detector,
                        upsample_primary=detector_upsample_primary,
                        upsample_retry=detector_upsample_retry,
                    )
                    bbox_cache[source_key] = (
                        square_bbox(detected, scale=square_scale),
                        (int(source_image.shape[1]), int(source_image.shape[0])),
                    )
                box, source_size = bbox_cache[source_key]
                image = decode_frame_rgb(row["video_path"], int(row["frame_index"]))
                image_size = (int(image.shape[1]), int(image.shape[0]))
                image_box = box if image_size == source_size else map_bbox(box, source_size, image_size)
                crop = crop_and_resize(image, image_box, input_size, is_mask=False)
                if not np.isfinite(crop).all():
                    raise ValueError("crop contains non-finite values")
                _save_array(crop_path, crop.astype(np.uint8, copy=False))
                row["crop_path"] = str(crop_path)
                row["crop_left"], row["crop_top"], row["crop_right"], row["crop_bottom"] = image_box.as_tuple()
                row["crop_scale"] = float(square_scale)
                # Preserve the common source-space transform.  The target
                # image bbox is allowed to differ when fake/real videos have
                # different encoded resolutions.
                row["crop_source_left"], row["crop_source_top"], row["crop_source_right"], row["crop_source_bottom"] = box.as_tuple()
                row["crop_source_width"], row["crop_source_height"] = (int(source_size[0]), int(source_size[1]))
                row["crop_image_width"], row["crop_image_height"] = (int(image_size[0]), int(image_size[1]))
                cached_images += 1

            if not row["is_real"]:
                group = (
                    str(row["split"]),
                    str(row["compression"]),
                    str(row["manipulation"]),
                    str(row["video_name"]),
                )
                mask_expected[group] += 1
                cached_mask = _load_cached_array(
                    mask_path,
                    shape=(int(input_size), int(input_size)),
                    dtype=np.dtype(np.uint8),
                    binary=True,
                )
                if cached_mask is not None:
                    reused_masks += 1
                    row["mask_cache_path"] = str(mask_path)
                    cached_masks += 1
                    if int(np.count_nonzero(cached_mask)) > 0:
                        mask_positive[group] += 1
                else:
                    if source_key not in bbox_cache:
                        source_image = decode_frame_rgb(*source_key)
                        detected = detect_largest_face_box(
                            source_image,
                            detector=detector,
                            upsample_primary=detector_upsample_primary,
                            upsample_retry=detector_upsample_retry,
                        )
                        bbox_cache[source_key] = (
                            square_bbox(detected, scale=square_scale),
                            (int(source_image.shape[1]), int(source_image.shape[0])),
                        )
                    box, source_size = bbox_cache[source_key]
                    mask = decode_mask(row["mask_path"], int(row["mask_frame_index"]))
                    mask_size = (int(mask.shape[1]), int(mask.shape[0]))
                    mask_box = box if mask_size == source_size else map_bbox(box, source_size, mask_size)
                    mask_crop = crop_and_resize(
                        mask,
                        mask_box,
                        input_size,
                        is_mask=True,
                        mask_policy=mask_policy,
                        mask_threshold=mask_threshold,
                    )
                    if not np.isfinite(mask_crop).all():
                        raise ValueError("mask crop contains non-finite values")
                    _save_array(mask_path, mask_crop.astype(np.uint8, copy=False))
                    row["mask_cache_path"] = str(mask_path)
                    if int(np.count_nonzero(mask_crop)) > 0:
                        mask_positive[group] += 1
                    cached_masks += 1
        except Exception as exc:  # exclusions are recorded before any scoring stage
            failed_rows += 1
            if source_key not in bbox_cache and source_key not in bbox_errors:
                bbox_errors[source_key] = f"{type(exc).__name__}: {exc}"
            _set_exclusion(row, f"crop_or_decode_error:{type(exc).__name__}:{exc}")

        if progress_every and (row_number == total or row_number % progress_every == 0):
            exclusions = sum(bool(item.get("exclusion_reason")) for item in rows[:row_number])
            print(
                f"[cache_crops] processed={row_number}/{total} cached_images={cached_images} "
                f"cached_masks={cached_masks} exclusions={exclusions}",
                flush=True,
            )

    # A fake video with no positive pixels in any of its sampled masks is an
    # exclusion at video level, not a negative example.
    for row in rows:
        if row["is_real"]:
            continue
        group = (str(row["split"]), str(row["compression"]), str(row["manipulation"]), str(row["video_name"]))
        if exclude_empty_fake_masks and mask_expected[group] and mask_positive[group] == 0:
            _set_exclusion(row, "mask_empty_after_crop")

    write_manifest(rows, manifest_path)
    counts = Counter(str(row["exclusion_reason"]) for row in rows if row.get("exclusion_reason"))
    summary = CropCacheSummary(
        status="completed",
        manifest=str(manifest_path),
        rows=len(rows),
        processed_rows=total,
        cached_images=cached_images,
        cached_masks=cached_masks,
        reused_images=reused_images,
        reused_masks=reused_masks,
        failed_rows=failed_rows,
        excluded_rows=sum(counts.values()),
        exclusion_reasons=dict(sorted(counts.items())),
        bbox_count=len(bbox_cache),
    )
    if metadata_path is not None:
        path = Path(metadata_path)
        payload = read_json(path) if path.is_file() else {}
        payload["manifest_sha256"] = sha256_file(manifest_path)
        payload["crop_cache"] = {
            "status": summary.status,
            "rows": summary.rows,
            "processed_rows": summary.processed_rows,
            "cached_images": summary.cached_images,
            "cached_masks": summary.cached_masks,
            "reused_images": summary.reused_images,
            "reused_masks": summary.reused_masks,
            "failed_rows": summary.failed_rows,
            "excluded_rows": summary.excluded_rows,
            "exclusion_reasons": summary.exclusion_reasons,
            "bbox_count": summary.bbox_count,
            "mask_binarization": mask_policy,
        }
        write_json(path, payload)
    return summary
