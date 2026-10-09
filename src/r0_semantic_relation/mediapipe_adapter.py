"""Official MediaPipe Tasks Face Landmarker adapter for R0."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import hashlib


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""): h.update(chunk)
    return h.hexdigest()


def create_landmarker(
    asset_path: str | Path, *, num_faces: int = 2,
    min_face_detection_confidence: float = 0.5,
    min_face_presence_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
):
    path = Path(asset_path).expanduser().resolve()
    if not path.is_file(): raise FileNotFoundError(f"MediaPipe Face Landmarker asset is missing: {path}")
    try:
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
    except ImportError as exc:
        raise RuntimeError("R0 requires the official mediapipe package with Tasks Face Landmarker") from exc
    options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(path)),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=int(num_faces),
        min_face_detection_confidence=min_face_detection_confidence,
        min_face_presence_confidence=min_face_presence_confidence,
        min_tracking_confidence=min_tracking_confidence,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    return mp, vision.FaceLandmarker.create_from_options(options)


def detect_landmarks(mp: Any, landmarker: Any, image_rgb):
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    result = landmarker.detect(image)
    faces = getattr(result, "face_landmarks", [])
    if len(faces) == 0: return None, {"status": "no_face", "face_count": 0}
    if len(faces) != 1: return faces[0], {"status": "multi_face_first_used", "face_count": len(faces)}
    return faces[0], {"status": "ok", "face_count": 1}
