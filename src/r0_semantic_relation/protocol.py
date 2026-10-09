"""Explicit R0-only protocol checks and immutable algorithm provenance."""
from __future__ import annotations

from pathlib import Path

from src.cirlid.config import Config, ConfigError
from src.cirlid.io import sha256_file


R0_SOURCE_FILES = (
    "src/run_r0.py",
    "src/r0_semantic_relation/protocol.py",
    "src/r0_semantic_relation/scoring.py",
    "src/r0_semantic_relation/geometry.py",
    "src/r0_semantic_relation/metrics.py",
    "src/r0_semantic_relation/bootstrap.py",
    "src/r0_semantic_relation/mediapipe_adapter.py",
    "src/cirlid/config.py",
    "src/cirlid/io.py",
    "src/cirlid/ffpp.py",
    "src/cirlid/crops.py",
    "src/cirlid/dinov3.py",
    "src/cirlid/features.py",
    "src/cirlid/manifest.py",
)


R0_FIXED_VALUES = {
    "protocol.namespace": "r0_semantic_relation_v1",
    "protocol.version": "R0-semantic-relation-v1.0",
    "protocol.split": "train",
    "protocol.compression": "c23",
    "protocol.manipulation": "Face2Face",
    "protocol.official_test_locked": True,
    "protocol.oracle_paired_target_real": True,
    "protocol.deployable_detector": False,
    "holdout.enabled": True,
    "holdout.fraction": 0.30,
    "holdout.decision_partition": "holdout",
    "input.size": 448,
    "input.color": "RGB",
    "input.imagenet_mean": [0.485, 0.456, 0.406],
    "input.imagenet_std": [0.229, 0.224, 0.225],
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
    "backbone.weights": "LVD-1689M",
    "backbone.frozen": True,
    "backbone.block_one_based": 6,
    "backbone.api_n": [5],
    "backbone.norm": True,
    "backbone.reshape": False,
    "backbone.extra_l2_normalization": False,
    "backbone.feature_dtype_on_disk": "float32",
    "features.expected_patch_grid": [28, 28],
    "features.expected_tokens_per_frame": 784,
    "features.expected_dimension": 768,
    "sampling.frame_fractions": [0.25, 0.50, 0.75],
    "mediapipe.package": "mediapipe",
    "mediapipe.solution": "Tasks Face Landmarker",
    "mediapipe.running_mode": "IMAGE",
    "mediapipe.asset_source": "official_mediapipe_tasks_asset_only",
    "mediapipe.num_faces": 2,
    "mediapipe.min_face_detection_confidence": 0.5,
    "mediapipe.min_face_presence_confidence": 0.5,
    "mediapipe.min_tracking_confidence": 0.5,
    "regions.mouth_polygon": "official_outer_lip_ordered_loop",
    "regions.perioral_ring_radius": 12,
    "regions.cheek_definition": "fixed_face_clipped_side_quadrilaterals_excluding_eyes_mouth_ring",
    "regions.token_mapping": "area_fraction_448_to_28",
    "regions.empty_region": "record_failure_do_not_substitute_full_face",
    "statistics.primary_domain": "full_face_valid_domain",
    "statistics.planned_secondary_domain": "mouth_and_perioral_conditioned",
    "statistics.bootstrap_replicates": 2000,
    "statistics.confidence_level": 0.95,
    "statistics.grouping": "source_video_connected_component",
    "statistics.ap_all_zero_mask": "undefined",
    "statistics.ap_all_one_mask": "undefined",
    "statistics.score_threshold": "none",
    "quality.max_evaluation_exception_fraction": 0.05,
    "quality.min_strict_pair_fraction": 0.90,
    "quality.min_evaluable_pair_fraction": 0.80,
    "quality.min_target_real_single_face_fraction": 0.90,
    "quality.min_holdout_independent_components": 5,
    "quality.min_evaluable_frames_per_video": 3,
}


def require_r0_protocol(config: Config) -> None:
    problems = [f"{key}: expected {expected!r}, got {config.get(key)!r}"
                for key, expected in R0_FIXED_VALUES.items()
                if type(config.get(key)) is not type(expected) or config.get(key) != expected]
    for key in ("protocol.seed", "protocol.bootstrap_seed", "holdout.seed"):
        value = config.get(key)
        if type(value) is not int or value < 0:
            problems.append(f"{key}: expected a nonnegative integer, got {value!r}")
    for key in ("paths.ffpp_root", "paths.split_files.train", "paths.dinov3_repo",
                "paths.dinov3_weights", "paths.crops", "paths.masks", "paths.features",
                "paths.output_dir", "mediapipe.asset_path", "mediapipe.asset_sha256"):
        if not isinstance(config.get(key), str) or not config.get(key):
            problems.append(f"{key}: required nonempty string")
    if problems:
        raise ConfigError("R0 frozen protocol mismatch:\n" + "\n".join(problems))


def r0_source_hashes() -> dict[str, str]:
    project_root = Path(__file__).resolve().parents[2]
    return {name: sha256_file(project_root / name) for name in R0_SOURCE_FILES}
