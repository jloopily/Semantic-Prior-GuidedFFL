"""FaceForensics++ filesystem, split, and video decoding integration.

The code in this module only reads the locally authorised FF++ release.  It
never downloads data and it does not modify the raw dataset.  Path layout is
kept explicit because a silent fallback to a different FF++ release would
invalidate the frozen protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

from .geometry import fractional_frame_index, middle_frame_index
from .io import read_json
from .types import ManifestRecord


class FFPPIntegrationUnavailable(NotImplementedError):
    """Raised when the local FF++ integration cannot safely proceed."""


@dataclass(frozen=True)
class FFPPPaths:
    root: Path
    split_train: Path
    split_validation: Path
    split_test: Path

    @classmethod
    def from_root(cls, root: str | Path, split_files: Mapping[str, str | Path]) -> "FFPPPaths":
        root_path = Path(root).expanduser().resolve()
        required = {"train", "validation", "test"}
        missing = required - set(split_files)
        if missing:
            raise ValueError(f"Missing FF++ split paths: {sorted(missing)}")
        return cls(
            root=root_path,
            split_train=Path(split_files["train"]).expanduser().resolve(),
            split_validation=Path(split_files["validation"]).expanduser().resolve(),
            split_test=Path(split_files["test"]).expanduser().resolve(),
        )

    def split_path(self, split: str) -> Path:
        if split == "train":
            return self.split_train
        if split == "validation":
            return self.split_validation
        if split == "test":
            return self.split_test
        raise ValueError(f"Unknown split: {split!r}")


def _require_file(path: Path, description: str) -> Path:
    if not path.is_file():
        raise FFPPIntegrationUnavailable(f"Missing {description}: {path}")
    return path


def _id(value: Any) -> str:
    text = str(value).strip()
    if text.isdigit():
        return text.zfill(3)
    raise ValueError(f"FF++ split ID must be numeric, got {value!r}")


def read_split_pairs(path: str | Path) -> list[tuple[str, str]]:
    """Read an official FaceForensics split as normalized directed pairs."""

    payload = read_json(_require_file(Path(path), "split JSON"))
    if not isinstance(payload, list):
        raise ValueError(f"FF++ split must be a JSON list: {path}")
    pairs: list[tuple[str, str]] = []
    for position, pair in enumerate(payload):
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError(f"Invalid pair at {path}:{position + 1}: {pair!r}")
        target, source = _id(pair[0]), _id(pair[1])
        if target == source:
            raise ValueError(f"Self-pair at {path}:{position + 1}: {pair!r}")
        pairs.append((target, source))
    return pairs


def split_original_ids(path: str | Path) -> list[str]:
    pairs = read_split_pairs(path)
    return sorted({value for pair in pairs for value in pair})


def directed_fake_names(path: str | Path) -> list[tuple[str, str, str]]:
    """Return ``(video_name, target_id, source_id)`` for both directions.

    In the official FF++ manipulated-video naming convention ``A_B`` denotes
    a video generated with ``B`` as the target and ``A`` as the source.  The
    original implementation treated the filename's first identifier as the
    target, which paired ``A_B`` with the wrong original video whenever the
    two source videos had different durations.
    """

    values: set[tuple[str, str, str]] = set()
    for first, second in read_split_pairs(path):
        values.add((f"{first}_{second}", second, first))
        values.add((f"{second}_{first}", first, second))
    return sorted(values)


def _video_path(root: Path, *, manipulation: str, compression: str, video_name: str) -> Path:
    if manipulation == "original":
        return root / "original_sequences" / "youtube" / compression / "videos" / f"{video_name}.mp4"
    return root / "manipulated_sequences" / manipulation / compression / "videos" / f"{video_name}.mp4"


def _mask_path(root: Path, *, manipulation: str, video_name: str) -> Path:
    return root / "manipulated_sequences" / manipulation / "masks" / "videos" / f"{video_name}.mp4"


def _normalized_mask_index(video_index: int, video_count: int, mask_count: int) -> int:
    if video_count <= 0 or mask_count <= 0:
        raise ValueError("video and mask frame counts must be positive")
    if video_count == mask_count:
        return video_index
    if video_count == 1 or mask_count == 1:
        return 0
    return int(math.floor(video_index * (mask_count - 1) / (video_count - 1) + 0.5))


def probe_video(path: str | Path) -> tuple[int, int, int, float]:
    """Return `(frame_count, width, height, fps)` for an FF++ video."""

    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - exercised on server
        raise FFPPIntegrationUnavailable("opencv-python-headless is required to probe FF++ videos") from exc

    source = _require_file(Path(path), "FF++ video")
    capture = cv2.VideoCapture(str(source))
    try:
        if not capture.isOpened():
            raise FFPPIntegrationUnavailable(f"OpenCV cannot open FF++ video: {source}")
        frame_count = int(round(float(capture.get(cv2.CAP_PROP_FRAME_COUNT))))
        width = int(round(float(capture.get(cv2.CAP_PROP_FRAME_WIDTH))))
        height = int(round(float(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()
    if frame_count <= 0 or width <= 0 or height <= 0:
        raise FFPPIntegrationUnavailable(
            f"Invalid FF++ video metadata for {source}: frames={frame_count}, size={width}x{height}"
        )
    if not np.isfinite(fps) or fps <= 0:
        fps = 0.0
    return frame_count, width, height, fps


def _decode_bgr(path: str | Path, frame_index: int) -> np.ndarray:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover
        raise FFPPIntegrationUnavailable("opencv-python-headless is required to decode FF++ videos") from exc

    source = _require_file(Path(path), "FF++ video")
    capture = cv2.VideoCapture(str(source))
    try:
        if not capture.isOpened():
            raise FFPPIntegrationUnavailable(f"OpenCV cannot open FF++ video: {source}")
        count = int(round(float(capture.get(cv2.CAP_PROP_FRAME_COUNT))))
        if count <= 0 or not 0 <= int(frame_index) < count:
            raise IndexError(f"Frame index {frame_index} outside {source} with {count} frames")
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok or frame is None:
        raise FFPPIntegrationUnavailable(f"Cannot decode frame {frame_index} from {source}")
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise FFPPIntegrationUnavailable(f"Expected BGR color frame from {source}, got {frame.shape}")
    return np.asarray(frame)


def decode_frame_rgb(path: str | Path, frame_index: int) -> np.ndarray:
    """Decode one frame and explicitly convert BGR to RGB."""

    try:
        import cv2
    except ImportError as exc:  # pragma: no cover
        raise FFPPIntegrationUnavailable("opencv-python-headless is required to decode FF++ videos") from exc
    return cv2.cvtColor(_decode_bgr(path, frame_index), cv2.COLOR_BGR2RGB)


def decode_mask(path: str | Path, frame_index: int) -> np.ndarray:
    """Decode a raw FF++ mask frame while preserving non-black regions.

    Some FF++ manipulation masks are RGB/grayscale images rather than strict
    0/255 masks.  Taking the channel-wise maximum preserves the frozen
    `mask != 0` semantics without accidentally discarding colored pixels.
    """

    frame = _decode_bgr(path, frame_index)
    if frame.ndim == 2:
        return frame
    return np.max(frame, axis=2)


def iter_authorized_videos(paths: FFPPPaths, split: str) -> Iterator[Path]:
    """Yield original c23 paths for the official split, in ID order."""

    for video_id in split_original_ids(paths.split_path(split)):
        yield _video_path(paths.root, manipulation="original", compression="c23", video_name=video_id)


def _probe_cached(cache: dict[Path, tuple[int, int, int, float]], path: Path) -> tuple[int, int, int, float]:
    if path not in cache:
        cache[path] = probe_video(path)
    return cache[path]


def enumerate_records(
    paths: FFPPPaths,
    *,
    split: str,
    compressions: Sequence[str],
    manipulations: Sequence[str] = ("Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"),
    frame_fractions: Sequence[float] | None = None,
    real_frame_fractions: Sequence[float] | None = None,
    fake_frame_fractions: Sequence[float] | None = None,
    seed: int = 20260919,
) -> list[ManifestRecord]:
    """Enumerate real/fake/mask records from the verified FF++ layout.

    The official split pairs define the original-video partition.  Each pair
    contributes both directed fake names (`a_b` and `b_a`), matching the
    locally verified FF++ download.  Crop detection is intentionally not done
    here: every row points back to the target original c23 frame, and the crop
    stage later computes one bbox from that source only.
    """

    if split not in {"train", "validation", "test"}:
        raise ValueError(f"Unknown split: {split!r}")
    compressions = tuple(str(value) for value in compressions)
    if any(value not in {"c23", "c40"} for value in compressions):
        raise ValueError(f"Unsupported FF++ compression: {compressions!r}")
    if not compressions:
        raise ValueError("At least one compression is required")
    # ``frame_fractions`` remains as a compatibility shorthand.  The frozen
    # config has separate validation-real and validation-fake positions, so
    # callers must be able to express those independently rather than silently
    # applying the test positions to every row type.
    if frame_fractions is not None:
        if real_frame_fractions is None:
            real_frame_fractions = frame_fractions
        if fake_frame_fractions is None:
            fake_frame_fractions = frame_fractions
    if real_frame_fractions is None:
        real_frame_fractions = (0.25, 0.50, 0.75)
    if fake_frame_fractions is None:
        fake_frame_fractions = real_frame_fractions

    def _fractions(values: Sequence[float], label: str) -> tuple[float, ...]:
        result = tuple(float(value) for value in values)
        if not result or any(not np.isfinite(value) or not 0.0 <= value <= 1.0 for value in result):
            raise ValueError(f"{label} must be finite values in [0, 1]")
        return result

    real_fractions = _fractions(real_frame_fractions, "real_frame_fractions")
    fake_fractions = _fractions(fake_frame_fractions, "fake_frame_fractions")
    manipulations = tuple(str(value) for value in manipulations)
    if not manipulations:
        raise ValueError("At least one manipulation is required")

    split_path = paths.split_path(split)
    original_ids = sorted({value for pair in read_split_pairs(split_path) for value in pair})
    fake_names = directed_fake_names(split_path)
    probe_cache: dict[Path, tuple[int, int, int, float]] = {}
    records: list[ManifestRecord] = []

    # Crop detection is always sourced from original c23.  Probe it once per
    # target and reuse the count for both compression streams.
    source_paths = {
        video_id: _video_path(paths.root, manipulation="original", compression="c23", video_name=video_id)
        for video_id in original_ids
    }
    source_meta = {video_id: _probe_cached(probe_cache, path) for video_id, path in source_paths.items()}

    for compression in compressions:
        for video_id in original_ids:
            original_path = _video_path(paths.root, manipulation="original", compression=compression, video_name=video_id)
            frame_count, _, _, _ = _probe_cached(probe_cache, original_path)
            source_count = source_meta[video_id][0]
            for fraction in real_fractions:
                frame_index = fractional_frame_index(frame_count, fraction)
                crop_source_frame = fractional_frame_index(source_count, fraction)
                records.append(
                    ManifestRecord(
                        sample_id=f"{split}:{compression}:original:{video_id}:{frame_index}",
                        split=split,
                        compression=compression,
                        manipulation="original",
                        is_real=True,
                        target_id=video_id,
                        source_id=None,
                        video_name=video_id,
                        video_path=str(original_path),
                        mask_path=None,
                        frame_count=frame_count,
                        frame_fraction=fraction,
                        frame_index=frame_index,
                        mask_frame_index=None,
                        crop_source_video=str(source_paths[video_id]),
                        crop_source_frame=crop_source_frame,
                        seed=seed,
                    )
                )

    for manipulation in manipulations:
        for video_name, target_id, source_id in fake_names:
            mask = _mask_path(paths.root, manipulation=manipulation, video_name=video_name)
            mask_count, _, _, _ = _probe_cached(probe_cache, mask)
            source_count = source_meta[target_id][0]
            for compression in compressions:
                video = _video_path(paths.root, manipulation=manipulation, compression=compression, video_name=video_name)
                frame_count, _, _, _ = _probe_cached(probe_cache, video)
                for fraction in fake_fractions:
                    frame_index = fractional_frame_index(frame_count, fraction)
                    mask_frame_index = _normalized_mask_index(frame_index, frame_count, mask_count)
                    crop_source_frame = fractional_frame_index(source_count, fraction)
                    records.append(
                        ManifestRecord(
                            sample_id=f"{split}:{compression}:{manipulation}:{video_name}:{frame_index}",
                            split=split,
                            compression=compression,
                            manipulation=manipulation,
                            is_real=False,
                            target_id=target_id,
                            source_id=source_id,
                            video_name=video_name,
                            video_path=str(video),
                            mask_path=str(mask),
                            frame_count=frame_count,
                            frame_fraction=fraction,
                            frame_index=frame_index,
                            mask_frame_index=mask_frame_index,
                            crop_source_video=str(source_paths[target_id]),
                            crop_source_frame=crop_source_frame,
                            mask_frame_remapped=(mask_count != frame_count),
                            mask_frame_count=mask_count,
                            seed=seed,
                        )
                    )
    return records



def enumerate_real_records(
    paths: FFPPPaths,
    *,
    split: str,
    compressions: Sequence[str] = ("c23", "c40"),
    frame_fractions: Sequence[float] = (0.50,),
    seed: int = 20260919,
) -> list[ManifestRecord]:
    """Enumerate only original/real frames without touching fake media.

    This deliberately does not call :func:`enumerate_records`: that function
    also enumerates manipulation videos and masks. Real-only audits use this
    helper so that a structural audit cannot accidentally read fake manifests,
    fake videos, or fake masks.
    """

    if split not in {"train", "validation"}:
        raise ValueError("real-only enumeration is restricted to train/validation")
    compressions = tuple(str(value) for value in compressions)
    fractions = tuple(float(value) for value in frame_fractions)
    if not compressions or not fractions:
        raise ValueError("compressions and frame_fractions must not be empty")
    if any(not np.isfinite(value) or not 0.0 <= value <= 1.0 for value in fractions):
        raise ValueError("frame_fractions must be finite values in [0, 1]")

    original_ids = sorted({value for pair in read_split_pairs(paths.split_path(split)) for value in pair})
    probe_cache: dict[Path, tuple[int, int, int, float]] = {}
    source_paths = {
        video_id: _video_path(paths.root, manipulation="original", compression="c23", video_name=video_id)
        for video_id in original_ids
    }
    source_meta = {video_id: _probe_cached(probe_cache, path) for video_id, path in source_paths.items()}
    records: list[ManifestRecord] = []
    for compression in compressions:
        for video_id in original_ids:
            video_path = _video_path(
                paths.root, manipulation="original", compression=compression, video_name=video_id
            )
            frame_count, _, _, _ = _probe_cached(probe_cache, video_path)
            source_count = source_meta[video_id][0]
            for fraction in fractions:
                frame_index = fractional_frame_index(frame_count, fraction)
                crop_source_frame = fractional_frame_index(source_count, fraction)
                records.append(
                    ManifestRecord(
                        sample_id=f"{split}:{compression}:original:{video_id}:{frame_index}",
                        split=split,
                        compression=compression,
                        manipulation="original",
                        is_real=True,
                        target_id=video_id,
                        source_id=None,
                        video_name=video_id,
                        video_path=str(video_path),
                        mask_path=None,
                        frame_count=frame_count,
                        frame_fraction=fraction,
                        frame_index=frame_index,
                        mask_frame_index=None,
                        crop_source_video=str(source_paths[video_id]),
                        crop_source_frame=crop_source_frame,
                        seed=seed,
                    )
                )
    return records

def enumerate_gallery_records(
    paths: FFPPPaths,
    *,
    gallery_ids: Sequence[str],
    seed: int = 20260919,
) -> list[ManifestRecord]:
    """Create the separate train-real/c23 middle-frame gallery manifest.

    The ordinary manifest stores validation/test sample positions.  Gallery
    construction has a different frozen rule (the exact middle frame), and
    for even-length videos that is not always the same as the 50% sample.
    Keeping a dedicated manifest prevents an accidental off-by-one reuse of a
    validation frame and gives the crop stage a concrete artifact to cache.
    """

    records: list[ManifestRecord] = []
    for video_id in sorted({str(value) for value in gallery_ids}):
        path = _video_path(paths.root, manipulation="original", compression="c23", video_name=video_id)
        frame_count, _, _, _ = probe_video(path)
        frame_index = middle_frame_index(frame_count)
        fraction = 0.0 if frame_count == 1 else frame_index / (frame_count - 1)
        records.append(
            ManifestRecord(
                sample_id=f"gallery:c23:original:{video_id}:{frame_index}",
                split="train",
                compression="c23",
                manipulation="original",
                is_real=True,
                target_id=video_id,
                source_id=None,
                video_name=video_id,
                video_path=str(path),
                mask_path=None,
                frame_count=frame_count,
                frame_fraction=fraction,
                frame_index=frame_index,
                mask_frame_index=None,
                crop_source_video=str(path),
                crop_source_frame=frame_index,
                seed=seed,
            )
        )
    return records


def audit_ffpp(
    paths: FFPPPaths,
    *,
    compressions: Sequence[str] = ("c23", "c40"),
    manipulations: Sequence[str] = ("Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"),
) -> dict[str, Any]:
    """Audit expected paths and official split coverage without decoding frames."""

    split_summary: dict[str, Any] = {}
    missing: list[str] = []
    split_ids: dict[str, set[str]] = {}
    split_fake_names: dict[str, set[str]] = {}
    split_issues: list[str] = []
    for split in ("train", "validation", "test"):
        split_path = paths.split_path(split)
        if not split_path.is_file():
            missing.append(str(split_path))
            continue
        pairs = read_split_pairs(split_path)
        exact_duplicate_count = len(pairs) - len(set(pairs))
        undirected_pairs = {tuple(sorted(pair)) for pair in pairs}
        undirected_duplicate_count = len(pairs) - len(undirected_pairs)
        if exact_duplicate_count:
            split_issues.append(f"{split}: {exact_duplicate_count} duplicate pair entries")
        if undirected_duplicate_count:
            split_issues.append(
                f"{split}: {undirected_duplicate_count} duplicate undirected pair entries"
            )
        ids = sorted({value for pair in pairs for value in pair})
        fake_names = [name for name, _, _ in directed_fake_names(split_path)]
        endpoint_reuse_count = 2 * len(pairs) - len(ids)
        if endpoint_reuse_count:
            split_issues.append(
                f"{split}: {endpoint_reuse_count} reused original IDs across split pairs"
            )
        if len(fake_names) != 2 * len(pairs):
            split_issues.append(
                f"{split}: directed fake-name expansion is not two directions per pair"
            )
        split_ids[split] = set(ids)
        split_fake_names[split] = set(fake_names)
        split_summary[split] = {
            "pair_count": len(pairs),
            "original_video_count": len(ids),
            "directed_fake_name_count": len(fake_names),
            "exact_duplicate_pair_count": exact_duplicate_count,
            "undirected_duplicate_pair_count": undirected_duplicate_count,
            "reused_original_id_count": endpoint_reuse_count,
            "split_path": str(split_path),
        }
        for compression in compressions:
            for video_id in ids:
                path = _video_path(paths.root, manipulation="original", compression=compression, video_name=video_id)
                if not path.is_file():
                    missing.append(str(path))
            for manipulation in manipulations:
                for video_name in fake_names:
                    video = _video_path(paths.root, manipulation=manipulation, compression=compression, video_name=video_name)
                    mask = _mask_path(paths.root, manipulation=manipulation, video_name=video_name)
                    if not video.is_file():
                        missing.append(str(video))
                    if not mask.is_file():
                        missing.append(str(mask))
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = sorted(split_ids.get(left, set()) & split_ids.get(right, set()))
        if overlap:
            split_issues.append(f"{left}/{right}: overlapping original IDs ({len(overlap)})")
        fake_overlap = sorted(split_fake_names.get(left, set()) & split_fake_names.get(right, set()))
        if fake_overlap:
            split_issues.append(f"{left}/{right}: overlapping directed fake names ({len(fake_overlap)})")

    union_ids = set().union(*split_ids.values()) if split_ids else set()
    status = "invalid_splits" if split_issues else ("ok" if not missing else "missing_files")
    return {
        "status": status,
        "root": str(paths.root),
        "splits": split_summary,
        "split_issues": split_issues,
        "original_video_union_count": len(union_ids),
        "expected_compressions": list(compressions),
        "expected_manipulations": list(manipulations),
        "missing_count": len(missing),
        "missing_examples": missing[:20],
    }
