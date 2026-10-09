import numpy as np
import pytest
from types import SimpleNamespace

from src.r0_semantic_relation.geometry import SemanticRegions, build_semantic_regions, polygon_is_simple, OUTER_LIPS, FACE_OVAL, LEFT_EYE, RIGHT_EYE
from src.r0_semantic_relation.scoring import relation_score, local_difference_score, position_template_score, broken_context_relation_score
from src.r0_semantic_relation.metrics import average_precision
from src.r0_semantic_relation.bootstrap import paired_group_bootstrap


def _regions():
    z = np.zeros(784, dtype=np.float32)
    face = np.ones(784, dtype=np.float32)
    mouth = z.copy(); mouth[:100] = 1
    ring = z.copy(); ring[100:180] = 1
    cheek_l = z.copy(); cheek_l[300:500] = 1
    cheek_r = z.copy(); cheek_r[500:700] = 1
    return SemanticRegions(
        mouth_core=np.ones((448,448), bool), perioral_ring=np.zeros((448,448), bool),
        cheek_left=np.ones((448,448), bool), cheek_right=np.ones((448,448), bool),
        face=np.ones((448,448), bool), valid=np.ones((448,448), bool),
        token_mouth_core=mouth, token_perioral_ring=ring, token_cheek_left=cheek_l,
        token_cheek_right=cheek_r, token_face=face, token_valid=face, metadata={})


def _landmarks_fixture():
    landmarks = [SimpleNamespace(x=0.5, y=0.5) for _ in range(468)]
    for i in range(468):
        landmarks[i].x = 0.1 + 0.8 * ((i * 37) % 101) / 100
        landmarks[i].y = 0.1 + 0.8 * ((i * 53) % 101) / 100
    for indices, cx, cy, rx, ry in (
        (FACE_OVAL, 0.5, 0.5, 0.40, 0.48),
        (OUTER_LIPS, 0.5, 0.62, 0.12, 0.06),
        (LEFT_EYE, 0.34, 0.38, 0.07, 0.025),
        (RIGHT_EYE, 0.66, 0.38, 0.07, 0.025),
    ):
        for j, idx in enumerate(indices):
            angle = 2 * np.pi * j / len(indices)
            landmarks[idx].x = cx + rx * np.cos(angle)
            landmarks[idx].y = cy + ry * np.sin(angle)
    return landmarks


def test_polygon_and_production_regions_and_mapping():
    points = np.array([[0,0],[4,0],[4,4],[0,4]], dtype=float)
    assert polygon_is_simple(points)
    assert not polygon_is_simple(np.array([[0,0],[4,4],[0,4],[4,0]], dtype=float))
    regions = build_semantic_regions(_landmarks_fixture(), size=448, token_grid=28, ring_radius=12)
    assert regions.mouth_core.shape == (448, 448)
    assert regions.token_mouth_core.shape == (784,)
    assert regions.token_valid.shape == (784,)
    assert regions.mouth_core.any() and regions.face.any()
    assert regions.cheek_left.any() and regions.cheek_right.any()
    assert 0 < regions.token_mouth_core.sum() < 784
    assert np.isclose(regions.token_mouth_core.sum() * 16 * 16, regions.mouth_core.sum())


def test_zero_vectors_and_empty_context_are_safe():
    r = _regions()
    fake = np.zeros((784, 8), dtype=np.float32)
    real = np.zeros((784, 8), dtype=np.float32)
    score, domain, audit = relation_score(fake, real, r)
    assert not domain.any()
    assert not audit["fake_context_valid"]
    assert np.isfinite(score).all()


def test_all_zero_and_all_one_masks_have_undefined_ap():
    scores = np.arange(9, dtype=float)
    assert average_precision(scores, np.zeros(9, dtype=np.uint8)) is None
    assert average_precision(scores, np.ones(9, dtype=np.uint8)) is None


def test_tied_scores_are_order_invariant_and_binary_template_is_defined():
    scores = np.array([1.0, 1.0, 0.0, 0.0])
    mask_a = np.array([1, 0, 0, 1], dtype=np.uint8)
    mask_b = np.array([0, 1, 1, 0], dtype=np.uint8)
    assert average_precision(scores, mask_a) == average_precision(scores, mask_b)
    assert average_precision(scores, np.array([1, 1, 0, 0], dtype=np.uint8)) == 1.0
    assert average_precision(np.zeros(4), mask_a) == .5


def test_template_relation_heatmaps_differ_and_broken_context_preserves_query_control():
    r = _regions()
    rng = np.random.default_rng(3)
    real = rng.normal(size=(784, 8)).astype(np.float32)
    fake = real + 0.05 * rng.normal(size=(784, 8)).astype(np.float32)
    local_a, _ = local_difference_score(fake, real, r)
    local_b, _ = local_difference_score(fake, real, r)
    template, _ = position_template_score(r)
    relation, _domain, _audit = relation_score(fake, real, r)
    broken, _ = broken_context_relation_score(fake, real, r, permutation=np.random.default_rng(9).permutation(784))
    assert np.array_equal(local_a, local_b)
    assert not np.array_equal(local_a, template)
    assert relation.shape == template.shape == broken.shape == (784,)
    assert not np.array_equal(relation, broken)


def test_group_bootstrap_counts_source_groups_not_frames_as_independent_groups():
    rows = [
        {"source_group":"g1", "relation_ap":.6, "local_ap":.5},
        {"source_group":"g1", "relation_ap":.7, "local_ap":.5},
        {"source_group":"g2", "relation_ap":.4, "local_ap":.3},
    ]
    result = paired_group_bootstrap(rows, "relation_ap", "local_ap", replicates=20, seed=4)
    assert result["groups"] == 2
    assert result["rows"] == 3
    assert result["resampling_unit"] == "source_group_with_replacement"


def test_cheek_only_change_affects_relation_not_local_mouth_difference():
    r = _regions()
    real = np.zeros((784, 3), dtype=np.float32)
    real[:] = (1, 0, 0)
    fake = real.copy()
    fake[300:500] = (0, 1, 0)
    relation, domain, audit = relation_score(fake, real, r)
    local, _ = local_difference_score(fake, real, r)
    assert audit["fake_context_valid"] and audit["real_context_valid"]
    assert domain[:180].all()
    assert np.all(relation[:180] > 0)
    assert np.all(local[:180] == 0)


def test_component_bootstrap_uses_shared_video_identity_as_one_unit():
    rows = [
        {"component_id": "cc1", "relation_ap": .8, "local_ap": .2},
        {"component_id": "cc1", "relation_ap": .7, "local_ap": .3},
        {"component_id": "cc2", "relation_ap": .4, "local_ap": .5},
    ]
    result = paired_group_bootstrap(rows, "relation_ap", "local_ap", group_key="component_id", replicates=2000, seed=1)
    assert result["groups"] == 2
    assert result["resampling_unit"] == "component_id_with_replacement"
    assert result["group_sizes"] == {"cc1": 2, "cc2": 1}
    assert result["ci95"][0] < result["ci95"][1]
    with pytest.raises(ValueError, match="at least two"):
        paired_group_bootstrap(rows[:2], "relation_ap", "local_ap", group_key="component_id")


def test_pairing_requires_actual_frame_mask_and_crop_source_agreement():
    from src.run_r0 import _pair_metadata
    fake = dict(sample_id="f", target_id="001", source_id="002", compression="c23",
                manipulation="Face2Face", frame_fraction=.5, frame_index=10, frame_count=21,
                crop_source_frame=10, crop_source_video="original001.mp4",
                mask_frame_index=10, mask_frame_count=21, mask_frame_remapped=False)
    real = dict(sample_id="r", frame_index=10, frame_count=21, crop_source_frame=10,
                crop_source_video="original001.mp4")
    assert _pair_metadata(fake, real, "g", "cc", "exploration")["strict_frame_pair"]
    bad = dict(real, frame_index=11)
    assert not _pair_metadata(fake, bad, "g", "cc", "exploration")["strict_frame_pair"]
    bad = dict(real, crop_source_video="other.mp4")
    assert not _pair_metadata(fake, bad, "g", "cc", "exploration")["strict_frame_pair"]
    assert not _pair_metadata(dict(fake, mask_frame_remapped=True), real, "g", "cc", "exploration")["strict_frame_pair"]
