"""R1 tests for the experiment environment. Not executed during local editing."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from src.cirlid.config import Config, ConfigError, load_config
from src.cirlid.io import read_json, sha256_file, write_json
from src.r0_semantic_relation.geometry import SemanticRegions
from src.r0_semantic_relation.scoring import (
    broken_context_relation_score, local_difference_score, relation_score,
)
from src.r1_non_oracle_semantic_relation import METHODS, SEMANTIC_CONTROL, SEMANTIC_CONTROL_METHODS
from src.r1_non_oracle_semantic_relation.dataset import assert_disjoint, component_map, select_roles
from src.r1_non_oracle_semantic_relation.metrics import pixel_map, pixel_metrics, warp_real_to_fake
from src.r1_non_oracle_semantic_relation.preprocessing import read_regions, resource_guard, save_regions
from src.r1_non_oracle_semantic_relation.protocol import ProtocolError, Run, validate
from src.r1_non_oracle_semantic_relation.reporting import average_seeds, bootstrap, statistics
from src.r1_non_oracle_semantic_relation.scoring import (
    Descriptor, context_derangement, describe, score_gallery,
)


PROJECT = Path(__file__).resolve().parents[1]


def fixture_regions() -> SemanticRegions:
    zero = np.zeros(784, dtype=np.float32)
    mouth = zero.copy()
    mouth[:100] = 1
    ring = zero.copy()
    ring[100:180] = 1
    left = zero.copy()
    left[300:450] = 1
    right = zero.copy()
    right[500:700] = 1

    def pixels(tokens):
        return np.repeat(np.repeat(tokens.reshape(28, 28), 16, axis=0), 16, axis=1).astype(bool)

    full = np.ones((448, 448), bool)
    return SemanticRegions(pixels(mouth), pixels(ring), pixels(left), pixels(right), full, full,
                           mouth, ring, left, right, np.ones(784, np.float32), np.ones(784, np.float32), {})


def fixture_config(tmp_path) -> Config:
    payload = deepcopy(load_config(PROJECT / 'configs/r1_non_oracle_semantic_relation_v1.yaml').data)
    split = tmp_path / 'train.json'
    write_json(split, [['000', '001'], ['002', '003']])
    payload['paths']['split_files']['train'] = str(split)
    payload['paths']['output_dir'] = str(tmp_path / 'r1_non_oracle_test')
    payload['paths']['ffpp_root'] = str(tmp_path / 'raw')
    return Config(payload, tmp_path / 'config.yaml')


def frozen_run(tmp_path) -> Run:
    config = fixture_config(tmp_path)
    run = Run(config, Path(config.require('paths.output_dir')))
    run.freeze({'train_split_sha256': sha256_file(config.require('paths.split_files.train')),
                'roles': {}, 'mediapipe_asset_sha256': 'test-only', 'dinov3_checkpoint_sha256': 'test-only'})
    return run


def test_default_protocol_and_requested_fifty_percent_gate():
    config = load_config(PROJECT / 'configs/r1_non_oracle_semantic_relation_v1.yaml')
    validate(config)
    assert config.require('quality.min_evaluable_frame_fraction') == 0.50
    assert config.require('protocol.oracle_paired_target_real') is False
    assert config.require('protocol.official_test_locked') is True


@pytest.mark.parametrize('field,value', [
    ('oracle_paired_target_real', True), ('official_test_locked', False), ('split', 'test')])
def test_oracle_and_official_test_rejected(tmp_path, field, value):
    config = fixture_config(tmp_path)
    config.data['protocol'][field] = value
    with pytest.raises(ConfigError):
        validate(config)


def test_crop_reference_or_mask_adaptation_rejected(tmp_path):
    config = fixture_config(tmp_path)
    config.data['face_crop']['detector_source'] = 'original_target_c23_only'
    with pytest.raises(ConfigError):
        validate(config)


def test_component_disjoint_roles_deterministic_and_all_seeds_reported(tmp_path):
    config = fixture_config(tmp_path)
    pairs = [(f'{2*i:03d}', f'{2*i+1:03d}') for i in range(80)]
    first = select_roles(pairs, config)
    second = select_roles(pairs, config)
    assert first == second
    assert set(first['galleries']) == {'11', '23', '37'}
    assert len(first['calibration_ids']) == 32
    assert all(len(v) == 64 for v in first['galleries'].values())
    assert_disjoint(first)
    # Reproduce the R0 component holdout exactly, without importing its runner.
    names = sorted(set(first['components'].values()))
    expected = np.random.default_rng(20260930).permutation(names)[:round(len(names) * .30)]
    assert first['evaluation_components'] == sorted(expected)


def test_connected_sources_cannot_cross_roles():
    assert component_map([('000', '001'), ('001', '002'), ('004', '005')]) == {
        '000': 'component-0001', '001': 'component-0001', '002': 'component-0001',
        '004': 'component-0002', '005': 'component-0002'}


def test_gallery_overlap_rejected(tmp_path):
    config = fixture_config(tmp_path)
    roles = select_roles([(f'{2*i:03d}', f'{2*i+1:03d}') for i in range(80)], config)
    roles['galleries']['11'][0] = roles['evaluation_ids'][0]
    with pytest.raises(ProtocolError):
        assert_disjoint(roles)


def test_single_reference_reproduces_R0_pairwise_mathematics():
    regions = fixture_regions()
    rng = np.random.default_rng(101)
    fake = rng.normal(size=(784, 768)).astype(np.float32)
    real = rng.normal(size=(784, 768)).astype(np.float32)
    permutation = context_derangement(20260930)
    maps, audit = score_gallery(describe(fake, regions, permutation), regions,
                               [('gallery-video', [describe(real, regions, permutation)])])
    expected, _, _ = relation_score(fake, real, regions)
    local, _ = local_difference_score(fake, real, regions)
    broken, _ = broken_context_relation_score(fake, real, regions, permutation=permutation)
    np.testing.assert_allclose(maps['relation'], expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(maps['local_difference'], local, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(maps['broken_context'], broken, rtol=1e-6, atol=1e-6)
    assert audit['gallery_videos'] == 1


def test_reference_uses_its_own_cheek_geometry():
    query_regions = fixture_regions()
    reference_regions = fixture_regions()
    # Same token features, different independent region weights.
    reference_regions.token_cheek_left[:] = 0
    reference_regions.token_cheek_left[5:80] = 1
    tokens = np.random.default_rng(102).normal(size=(784, 768)).astype(np.float32)
    permutation = context_derangement(20260930)
    query = describe(tokens, query_regions, permutation)
    reference = describe(tokens, reference_regions, permutation)
    assert not np.allclose(query.relation, reference.relation)
    maps, _ = score_gallery(query, query_regions, [('real', [reference])])
    assert np.any(maps['relation'][:100] > 0)


def test_equal_video_aggregation_is_not_frame_weighted_or_distance_to_mean():
    unit = np.zeros((784, 768), np.float32)

    def descriptor(value):
        relation = np.zeros((784, 2), np.float32)
        relation[:, 0] = value
        return Descriptor(unit, relation, relation)

    query = descriptor(0)
    maps, audit = score_gallery(query, fixture_regions(),
                               [('A', [descriptor(1)]), ('B', [descriptor(-.5)] * 3)])
    assert maps['relation'][0] == pytest.approx(.75)
    assert audit['frames_per_video'] == {'A': 1, 'B': 3}
    assert maps['relation'][0] != pytest.approx(.625)  # all frames equally weighted
    assert maps['relation'][0] != pytest.approx(.25)   # distance to equally averaged video descriptors


def test_template_preserves_query_support_and_other_scores_share_full_face_domain():
    regions = fixture_regions()
    tokens = np.ones((784, 768), np.float32)
    descriptor = describe(tokens, regions, context_derangement(20260930))
    maps, audit = score_gallery(descriptor, regions, [('A', [descriptor])])
    assert set(maps) == {'relation', 'local_difference', 'position_template', 'broken_context'}
    np.testing.assert_array_equal(maps['position_template'][:180], 1)
    np.testing.assert_array_equal(maps['position_template'][180:], 0)
    assert audit['full_face_valid_tokens'] == 784
    assert audit['score_support_tokens'] == 180
    assert audit['zero_pairwise_distances']['relation'] == 784


@pytest.mark.parametrize('bad_shape', [(783, 768), (784, 8), (1, 784, 768)])
def test_wrong_features_rejected(bad_shape):
    with pytest.raises(ValueError):
        describe(np.ones(bad_shape, np.float32), fixture_regions(), context_derangement(20260930))


def test_zero_context_and_nan_do_not_silently_score():
    permutation = context_derangement(20260930)
    with pytest.raises(ValueError):
        describe(np.zeros((784, 768), np.float32), fixture_regions(), permutation)
    tokens = np.ones((784, 768), np.float32)
    tokens[0, 0] = np.nan
    with pytest.raises(ValueError):
        describe(tokens, fixture_regions(), permutation)


def test_duplicate_gallery_video_rejected():
    tokens = np.ones((784, 768), np.float32)
    regions = fixture_regions()
    descriptor = describe(tokens, regions, context_derangement(20260930))
    with pytest.raises(ValueError):
        score_gallery(descriptor, regions, [('A', [descriptor]), ('A', [descriptor])])


def test_derangement_reproducible_no_fixed_indices():
    first = context_derangement(20260930)
    np.testing.assert_array_equal(first, context_derangement(20260930))
    assert not np.any(first == np.arange(784))
    np.testing.assert_array_equal(np.sort(first), np.arange(784))


def test_fixed_bilinear_projection_matches_torch():
    torch = pytest.importorskip('torch')
    grid = np.random.default_rng(55).normal(size=784).astype(np.float32)
    expected = torch.nn.functional.interpolate(torch.from_numpy(grid.reshape(1, 1, 28, 28)),
                                               size=(448, 448), mode='bilinear', align_corners=False)[0, 0].numpy()
    np.testing.assert_allclose(pixel_map(grid), expected, atol=1e-6, rtol=1e-6)


def test_threshold_ties_and_undefined_AP_are_explicit():
    result = pixel_metrics(np.array([0, 0, 1, 1]), np.array([0, 0, 1, 1]), 1)
    assert result['ap'] == 1
    assert result['tp'] == 0  # strict > threshold; ties are not predicted positive
    assert result['f1'] == 0
    zero = pixel_metrics(np.ones(3), np.zeros(3, np.uint8), .5)
    assert zero['ap'] is None and zero['fpr'] == 1
    full = pixel_metrics(np.ones(3), np.ones(3, np.uint8), .5)
    assert full['ap'] is None and full['fpr'] is None
    empty = pixel_metrics(np.empty(0), np.empty(0, np.uint8), .5)
    assert empty['ap'] is None and empty['f1'] is None and empty['fpr'] is None


def test_independent_crop_warp_identity_and_shift():
    pytest.importorskip('cv2')
    y, x = np.mgrid[:448, :448]
    values = (x + 1000 * y).astype(np.float32)
    valid = np.ones((448, 448), bool)
    row = {'width': 448, 'height': 448, 'crop_bbox': [0, 0, 448, 448]}
    warped, overlap = warp_real_to_fake(values, valid, row, row)
    np.testing.assert_array_equal(warped, values)
    assert overlap.all()
    shifted = {**row, 'crop_bbox': [16, 0, 464, 448]}
    warped, overlap = warp_real_to_fake(values, valid, shifted, row)
    assert warped[100, 50] == values[100, 66]
    assert not overlap[:, 432:].any()


def test_region_serialization_has_no_pickled_metadata(tmp_path):
    run = frozen_run(tmp_path)
    destination = run.path('regions.npz')
    original = fixture_regions()
    save_regions(destination, original)
    restored = read_regions(run, 'regions.npz')
    np.testing.assert_array_equal(original.token_cheek_left, restored.token_cheek_left)
    np.testing.assert_array_equal(original.valid, restored.valid)


def test_seed_average_requires_every_seed_and_does_not_inflate_sample_size():
    rows = []
    for seed, ap in [(11, .6), (23, .4), (37, .5)]:
        row = {'gallery_seed': seed, 'compression': 'c23', 'role': 'evaluation_fake',
               'video_id': '001_000', 'component_id': 'c1',
               **{f'{m}_ap': ap for m in ('relation', 'local_difference', 'position_template', 'broken_context')}}
        rows.append(row)
    outputs = average_seeds(rows, [11, 23, 37])
    assert len(outputs) == 1
    assert outputs[0]['relation_ap'] == .5
    assert average_seeds(rows[:2], [11, 23, 37]) == []
    rows[0]['relation_ap'] = None
    assert average_seeds(rows, [11, 23, 37])[0]['relation_ap'] is None


def test_component_bootstrap_estimand_not_ordinary_video_average(tmp_path):
    run = frozen_run(tmp_path)
    rows = [{'component_id': 'A', 'first': 1.0, 'second': 0.0}] * 4
    rows += [{'component_id': 'B', 'first': 0.0, 'second': 0.0}]
    result = bootstrap(rows, 'first', 'second', run)
    assert result['point'] == .5
    assert result['groups'] == 2 and result['rows'] == 5
    assert result['group_key'] == 'component_id'
    assert result['replicates'] == 2000


def test_no_components_statistics_are_reported_as_undefined(tmp_path):
    run = frozen_run(tmp_path)
    report = statistics([], run)
    assert report['methods']['relation']['ap']['ordinary_video_mean'] is None
    assert report['differences']['relation_minus_local']['status'] == 'unavailable'


def test_stage_integrity_resume_and_failed_rows_preserved(tmp_path):
    run = frozen_run(tmp_path)
    with run.stage('example', ()) as stage:
        stage.record({'sample_id': 'one', 'status': 'failed', 'reason': 'no_face'})
        stage.finish([stage.cached('one')])
    assert run.read_stage('example')[0]['reason'] == 'no_face'
    with pytest.raises(ProtocolError):
        with run.stage('example', ()):
            pass
    run.resume = True
    with run.stage('example', ()) as stage:
        assert stage.complete
    destination = run.path('example/rows.jsonl')
    destination.write_text('{}\n', encoding='utf-8')
    with pytest.raises(ProtocolError):
        run.read_stage('example')


def test_upstream_artifact_hash_and_config_change_stop_the_run(tmp_path):
    run = frozen_run(tmp_path)
    with run.stage('example', ()) as stage:
        artifact = run.path('example/data.txt')
        artifact.write_text('original', encoding='utf-8')
        stage.record({'sample_id': 'one', 'status': 'ok'}, [artifact])
        stage.finish([stage.cached('one')])
    artifact.write_text('changed', encoding='utf-8')
    with pytest.raises(ProtocolError):
        run.read_stage('example')
    run.config.data['quality']['min_evaluable_frame_fraction'] = .8
    with pytest.raises(ProtocolError):
        run.frozen()


def test_path_traversal_and_low_resource_execution_rejected(tmp_path, monkeypatch):
    run = frozen_run(tmp_path)
    with pytest.raises(ProtocolError):
        run.path('../outside.txt')
    with pytest.raises(ProtocolError):
        resource_guard(run, allow_heavy=False)
    monkeypatch.setattr('src.r1_non_oracle_semantic_relation.preprocessing.os.cpu_count', lambda: 1)
    with pytest.raises(ProtocolError, match='CPU quota'):
        resource_guard(run, allow_heavy=True)


def test_unknown_evaluation_or_calibration_roles_rejected(tmp_path):
    config = fixture_config(tmp_path)
    roles = select_roles([(f'{2*i:03d}', f'{2*i+1:03d}') for i in range(80)], config)
    invalid = deepcopy(roles)
    invalid['evaluation_ids'].pop()
    with pytest.raises(ProtocolError, match='evaluation video'):
        assert_disjoint(invalid)
    invalid = deepcopy(roles)
    invalid['calibration_ids'].append(invalid['calibration_ids'][0])
    with pytest.raises(ProtocolError, match='duplicate calibration'):
        assert_disjoint(invalid)
    invalid = deepcopy(roles)
    invalid['gallery_pool_ids'].append(invalid['evaluation_ids'][0])
    with pytest.raises(ProtocolError, match='gallery pool'):
        assert_disjoint(invalid)


def test_validation_or_test_split_filename_cannot_be_used_as_train(tmp_path):
    config = fixture_config(tmp_path)
    config.data['paths']['split_files']['train'] = str(tmp_path / 'test.json')
    with pytest.raises(ConfigError, match='train.json'):
        validate(config)


def test_resume_checks_upstream_payload_not_just_upstream_receipt(tmp_path):
    run = frozen_run(tmp_path)
    with run.stage('first', ()) as stage:
        artifact = run.path('first/payload.txt')
        artifact.write_text('intact', encoding='utf-8')
        stage.record({'sample_id': 'first', 'status': 'ok'}, [artifact])
        stage.finish([stage.cached('first')])
    with run.stage('second', ('first',)) as stage:
        stage.finish([])
    artifact.write_text('modified without changing complete.json', encoding='utf-8')
    with pytest.raises(ProtocolError, match='modified'):
        run.read_stage('second')


def test_package_version_change_cannot_mix_one_run(tmp_path, monkeypatch):
    run = frozen_run(tmp_path)
    monkeypatch.setattr('src.r1_non_oracle_semantic_relation.protocol.environment_versions',
                        lambda: {'python': 'changed-runtime'})
    with pytest.raises(ProtocolError, match='package versions changed'):
        run.frozen()


def test_duplicate_crop_content_is_audited_not_silently_removed():
    from src.r1_non_oracle_semantic_relation.reference import duplicate_crops
    rows = [{'sample_id': 'A/one', 'crop_sha256': 'same'},
            {'sample_id': 'B/two', 'crop_sha256': 'same'},
            {'sample_id': 'C/three', 'crop_sha256': 'unique'}]
    assert duplicate_crops(rows) == {'same': ['A/one', 'B/two']}
    assert len(rows) == 3


def test_report_does_not_call_small_uncertain_difference_random_error():
    from src.r1_non_oracle_semantic_relation.reporting import comparison_description
    text = comparison_description({'status': 'ok', 'point': .001,
                                   'ci95': [-.004, .006], 'rows': 10, 'groups': 5})
    assert '跨越/触及 0' in text
    assert '0.1000 pp' in text
    assert '优于所有' not in text


def semantic_config(tmp_path) -> Config:
    config = fixture_config(tmp_path)
    supplement = load_config(PROJECT / 'configs/r1_non_oracle_semantic_relation_semantic_control_v1.yaml')
    for field in ('namespace', 'version'):
        config.data['protocol'][field] = supplement.require('protocol.' + field)
    config.data['scoring'] = deepcopy(supplement.require('scoring'))
    config.data['paths']['output_dir'] = str(tmp_path / 'r1_non_oracle_semantic_relation_semantic_control_test')
    return config


def test_semantic_profile_is_opt_in_and_requires_separate_output(tmp_path):
    old = fixture_config(tmp_path)
    new = semantic_config(tmp_path)
    validate(old)
    validate(new)
    assert Run(old, Path(old.require('paths.output_dir'))).methods == METHODS
    assert Run(new, Path(new.require('paths.output_dir'))).methods == SEMANTIC_CONTROL_METHODS
    assert new.require('quality.min_evaluable_frame_fraction') == .50
    new.data['paths']['output_dir'] = old.require('paths.output_dir')
    with pytest.raises(ConfigError, match='own output namespace'):
        validate(new)


@pytest.mark.parametrize('change', ['missing_control', 'wrong_version', 'wrong_support'])
def test_semantic_profile_cannot_silently_change_method_or_support(tmp_path, change):
    config = semantic_config(tmp_path)
    if change == 'missing_control':
        config.data['scoring']['methods'] = list(METHODS)
    elif change == 'wrong_version':
        config.data['protocol']['version'] = 'R1-non-oracle-v1.0'
    else:
        config.data['scoring']['semantic_control_definition'] = 'fake_mask_support'
    with pytest.raises(ConfigError):
        validate(config)


def test_semantic_local_preserves_old_maps_and_exact_equal_video_aggregation():
    rng = np.random.default_rng(110)
    regions = fixture_regions()
    regions.token_valid[10] = 0  # Inside mouth, but excluded by relation's valid-token check.
    permutation = context_derangement(20260930)
    descriptors = [describe(rng.normal(size=(784, 768)).astype(np.float32), regions, permutation)
                   for _ in range(4)]
    query, a, b, c = descriptors
    references = [('A', [a]), ('B', [b, c])]
    old, old_audit = score_gallery(query, regions, references)
    new, audit = score_gallery(query, regions, references, methods=SEMANTIC_CONTROL_METHODS)
    assert tuple(old) == METHODS and tuple(new) == SEMANTIC_CONTROL_METHODS
    for method in METHODS:
        np.testing.assert_array_equal(new[method], old[method])
    support = ((regions.token_mouth_core + regions.token_perioral_ring) > 0) & (regions.token_valid > 0)
    np.testing.assert_array_equal(new[SEMANTIC_CONTROL][support], old['local_difference'][support])
    assert np.all(new[SEMANTIC_CONTROL][~support] == 0)
    assert new[SEMANTIC_CONTROL][10] == 0
    assert not np.shares_memory(new[SEMANTIC_CONTROL], new['local_difference'])
    assert np.any(new['local_difference'][~support] > 0)
    assert audit['frames_per_video'] == old_audit['frames_per_video'] == {'A': 1, 'B': 2}
    expected_local = (np.linalg.norm(query.unit_tokens - a.unit_tokens, axis=1).astype(np.float64)
                      + (np.linalg.norm(query.unit_tokens - b.unit_tokens, axis=1).astype(np.float64)
                         + np.linalg.norm(query.unit_tokens - c.unit_tokens, axis=1).astype(np.float64)) / 2) / 2
    np.testing.assert_array_equal(new[SEMANTIC_CONTROL][support], expected_local.astype(np.float32)[support])
    assert audit['semantic_local_control']['evaluation_domain'] == 'full_face_valid_domain'


def test_semantic_comparisons_use_all_seeds_and_component_units(tmp_path):
    config = semantic_config(tmp_path)
    run = Run(config, Path(config.require('paths.output_dir')))
    rows = []
    for video, component, relation, semantic, local in [('A', 'c1', .70, .60, .50),
                                                       ('B', 'c2', .80, .70, .60)]:
        for seed in (11, 23, 37):
            row = {'gallery_seed': seed, 'compression': 'c23', 'role': 'evaluation_fake',
                   'video_id': video, 'component_id': component,
                   **{f'{m}_ap': .55 for m in run.methods}}
            row.update(relation_ap=relation, local_difference_semantic_ap=semantic, local_difference_ap=local)
            rows.append(row)
    averaged = average_seeds(rows, [11, 23, 37], run.methods)
    report = statistics(averaged, run)
    for name in ('relation_minus_local_semantic', 'local_semantic_minus_local'):
        value = report['differences'][name]
        assert value['point'] == pytest.approx(.1)
        assert value['rows'] == 2 and value['groups'] == 2
    rows[0]['local_difference_semantic_ap'] = None
    incomplete = average_seeds(rows, [11, 23, 37], run.methods)
    assert incomplete[0]['local_difference_semantic_ap'] is None
    assert not incomplete[0]['evaluable']
    assert statistics(incomplete, run)['differences']['relation_minus_local_semantic']['status'] == 'unavailable'


def test_semantic_evaluation_rejects_legacy_score_archive(tmp_path):
    from src.r1_non_oracle_semantic_relation.evaluation import load_maps
    config = semantic_config(tmp_path)
    run = Run(config, Path(config.require('paths.output_dir')))
    path = run.path('maps/legacy.npz')
    path.parent.mkdir(parents=True)
    np.savez(path, **{m: np.zeros(784, np.float32) for m in METHODS})
    with pytest.raises(ProtocolError, match='score-map methods'):
        load_maps(run, {'score_path': run.relative(path)})


@pytest.mark.parametrize('supplement', [False, True])
def test_registered_methods_flow_through_calibration_scoring_evaluation_and_reports(tmp_path, monkeypatch, supplement):
    """Synthetic cached inputs exercise stage wiring without models or raw videos."""
    from src.cirlid.io import read_jsonl, save_array
    from src.r1_non_oracle_semantic_relation import reference, evaluation, reporting

    config = semantic_config(tmp_path) if supplement else fixture_config(tmp_path)
    # Tiny synthetic roles only for this integration test; production defaults stay unchanged.
    config.data['gallery'].update(videos=1, min_valid_videos=1)
    config.data['calibration'].update(videos=1, min_valid_videos=1)
    validate(config)
    run = Run(config, Path(config.require('paths.output_dir')))
    roles = {'components': {'100': 'gallery', '200': 'calibration', '001': 'evaluation', '002': 'evaluation'},
             'evaluation_components': ['evaluation'], 'evaluation_ids': ['001', '002'],
             'calibration_components': ['calibration'], 'calibration_ids': ['200'],
             'gallery_pool_ids': ['100'], 'galleries': {str(seed): ['100'] for seed in (11, 23, 37)}}
    run.freeze({'train_split_sha256': sha256_file(config.require('paths.split_files.train')), 'roles': roles})
    regions = fixture_regions()
    regions_path = run.path('geometry/synthetic.npz')
    features_path = run.path('features/synthetic.npy')
    rows = []
    for role, video, component, compressions in (
        ('gallery', '100', 'gallery', ('c23', 'c40')),
        ('calibration', '200', 'calibration', ('c23',)),
        ('evaluation_fake', '002_001', 'evaluation', ('c23', 'c40')),
        ('evaluation_real', '001', 'evaluation', ('c23', 'c40')),
    ):
        for compression in compressions:
            for index, fraction in enumerate((.25, .50, .75)):
                row = {'sample_id': f'{role}/{compression}/{video}/p{fraction:.2f}',
                       'status': 'ok', 'role': role, 'video_id': video, 'component_id': component,
                       'compression': compression, 'fraction': fraction, 'frame_index': index,
                       'label': int(role == 'evaluation_fake'),
                       'regions_path': run.relative(regions_path), 'features_path': run.relative(features_path),
                       'landmark_status': {'status': 'ok'},
                       'region_areas': {n: int(getattr(regions, n).sum()) for n in ('mouth_core', 'perioral_ring', 'cheek_left', 'cheek_right', 'valid')},
                       'region_token_weights': {n: float(getattr(regions, 'token_' + n).sum()) for n in ('mouth_core', 'perioral_ring', 'cheek_left', 'cheek_right', 'valid')}}
                if role == 'evaluation_fake':
                    row.update(paired_real_id=f'evaluation_real/{compression}/001/p{fraction:.2f}',
                               strict_real_frame_match=True, source_id='002', target_id='001', source_group='pair')
                rows.append(row)
    with run.stage('geometry', ()) as stage:
        save_regions(regions_path, regions)
        for row in rows:
            stage.record(row, [regions_path])
        stage.finish(rows)
    with run.stage('features', ('geometry',)) as stage:
        save_array(features_path, np.ones((784, 768), np.float32))
        for row in rows:
            stage.record(row, [features_path])
        stage.finish(rows)
    mask_calls = []

    def fake_truth(active_run, row):
        assert active_run.path('score/complete.json').is_file()
        mask_calls.append(row['sample_id'])
        truth = np.zeros((448, 448), np.uint8)
        truth[:224] = 1
        return truth, {'mask_used_only_for_evaluation': True}

    def paired_maps(active_run, fake, real, fake_maps, truth):
        # Test the new method's paired-diagnostic wiring without requiring cv2.
        # Independent-crop coordinate mapping has its own existing tests.
        real_maps = evaluation.load_maps(active_run, real)
        return ({m: fake_maps[m][regions.valid] for m in active_run.methods},
                {m: real_maps[m][regions.valid] for m in active_run.methods}, truth[regions.valid])

    monkeypatch.setattr(evaluation, 'load_fake_truth', fake_truth)
    monkeypatch.setattr(evaluation, 'paired_maps', paired_maps)
    reference.gallery(run)
    reference.calibrate(run)
    assert not mask_calls
    threshold = read_json(run.path('calibrate/thresholds.json'))
    assert all(t['status'] == 'ok' and set(t['thresholds']) == set(run.methods) for t in threshold.values())
    reference.score(run)
    assert not mask_calls
    scored = run.read_stage('score')
    assert all(r['status'] == 'ok' for r in scored)
    for row in scored:
        assert set(evaluation.load_maps(run, row)) == set(run.methods)
    evaluation.evaluate(run)
    assert len(mask_calls) == 18  # 3 frames × 2 compressions × 3 registered seeds.
    for row in run.read_stage('evaluate'):
        assert set(row['metrics']) == set(run.methods)
        assert all(m['pixels'] == 448 * 448 for m in row['metrics'].values())
        if row['role'] == 'evaluation_fake':
            assert set(row['paired_metrics']) == set(run.methods)
    for video in read_jsonl(run.path('evaluate/per_video_metrics.jsonl')):
        assert set(video['metrics']) == set(run.methods)
        assert all(m['pixels'] == 3 * 448 * 448 for m in video['metrics'].values())
        assert all(m['fpr'] is not None for m in video['metrics'].values())
        if video['role'] == 'evaluation_fake':
            assert set(video['paired_metrics']) == set(run.methods)
    reporting.finalize(run)
    summary = read_json(run.path('finalize/summary.json'))
    assert summary['methods'] == list(run.methods)
    assert summary['decision']['decision'] == 'PARTIAL_PASS_DIAGNOSTIC'
    csv_header = run.path('finalize/all_seed_video_metrics.csv').read_text().splitlines()[0]
    markdown = run.path('finalize/analysis_report.md').read_text()
    if supplement:
        assert 'local_difference_semantic_ap' in csv_header
        assert 'relation_minus_local_semantic' in markdown and 'local_semantic_minus_local' in markdown
        diagnostic = read_json(run.path('finalize/semantic_control_diagnostic.json'))
        assert diagnostic['enabled'] and not diagnostic['c23_relation_increment_supported']
        assert 'local_difference_semantic' in summary['c40_minus_c23_paired']
    else:
        assert not run.path('finalize/semantic_control_diagnostic.json').exists()
        assert 'local_difference_semantic' not in csv_header
