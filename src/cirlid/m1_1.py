"""M1.1 C2-only, preregistered gallery-size/position deconfounding audit.

No official-test path is opened. M1 inputs are read-only. Run stages in order;
failed provenance is a block, not permission to substitute a different cache.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np

from .config import config_digest, load_config, resolve_config_path
from .features import load_features
from .heatmap import upsample_score_map
from .io import atomic_write_text, read_json, read_jsonl, sha256_file, write_json, write_jsonl
from .knn import ExactL2Index
from . import m1

CONDITIONS = ('full_unrestricted', 'same_token_index')
SEEDS = (20260929, 20260930, 20260931, 20260932, 20260933)
MANIPS = m1.MANIPULATIONS
COMPRESSIONS = m1.COMPRESSIONS
METRICS = ('pixel_ap', 'pixel_auroc', 'ap_minus_prevalence', 'ap_minus_paired_real',
           'ap_minus_spatial_template', 'region_difference_of_differences',
           'paired_real_virtual_mask_ap', 'spatial_template_ap', 'f1', 'iou',
           'foreground_tpr', 'fake_background_fpr', 'delta_score_inside', 'delta_score_outside')


def _conditions():
    return [(CONDITIONS[0], None), (CONDITIONS[1], None)] + [('matched_count_unrestricted', seed) for seed in SEEDS]


def _label(condition, seed):
    return condition if seed is None else f'{condition}_seed{seed}'


def _context(config_path, *, frozen=True):
    config = load_config(config_path)
    expected = {
        'seed': 20260929, 'official_test_locked': True,
        'protocol.namespace': 'm1_1_c2_deconfounding_v1', 'protocol.version': 'M1.1-v1.0',
        'protocol.scorer': 'C2', 'protocol.k': 20, 'protocol.grid': [28, 28],
        'protocol.image_size': [448, 448], 'protocol.distance': 'euclidean_float32_exact_indexflatl2',
        'protocol.full_candidates': 50176, 'protocol.restricted_candidates': 64,
        'protocol.gallery_videos': 64, 'protocol.random_gallery_seeds': list(SEEDS),
        'protocol.compressions': list(COMPRESSIONS), 'protocol.manipulations': list(MANIPS),
        'protocol.threshold_quantile': 0.99, 'protocol.threshold_method': 'linear',
        'protocol.threshold_comparison': 'greater_than', 'protocol.bootstrap_unit': 'source_group',
        'protocol.bootstrap_replicates': 2000, 'protocol.bootstrap_seed': 20260929,
        'protocol.official_test_locked': True,
    }
    errors = [key for key, value in expected.items() if config.get(key) != value]
    if errors:
        raise ValueError('M1.1 frozen protocol mismatch: ' + ', '.join(errors))
    root = resolve_config_path(config, config.require('output_dir'))
    base = load_config(resolve_config_path(config, config.require('base_config')))
    expected_root = resolve_config_path(base, base.require('paths.results')) / 'm1_1_c2_deconfounding_v1' / str(config.require('run_id'))
    if not str(config.require('run_id')).startswith('r') or not str(config.require('run_id'))[1:].isdigit():
        raise ValueError('run_id must be an explicit rNN revision')
    if root != expected_root:
        raise ValueError('M1.1 output must be the independent r01 namespace')
    m1_path = resolve_config_path(config, config.require('m1_config'))
    old, old_base, old_root = m1._context(m1_path)
    if resolve_config_path(config, config.require('m1_output_dir')) != old_root:
        raise ValueError('M1 output path differs from verified M1 config')
    if config_digest(base) != config_digest(old_base):
        raise ValueError('M1/base configuration mismatch')
    if frozen:
        receipt = read_json(root / 'freeze_receipt.json')
        if receipt.get('status') != 'frozen' or not receipt.get('official_test_locked'):
            raise RuntimeError('M1.1 prepare/freeze required')
        if receipt['config_digest'] != config_digest(config) or receipt['base_config_digest'] != config_digest(base):
            raise RuntimeError('M1.1 frozen config changed')
        if sha256_file(root / 'preregistered_config.yaml') != receipt['config_sha256']:
            raise RuntimeError('M1.1 saved config changed')
        for name, digest in receipt['m1_artifact_hashes'].items():
            if sha256_file(old_root / name) != digest:
                raise RuntimeError(f'M1 upstream artifact changed: {name}')
    return config, base, root, old, old_root


def _m1_files(old_root):
    names = ('freeze_receipt.json', 'preregistered_config.yaml', 'query_manifest.jsonl',
             'frame_manifest.jsonl', 'gallery_manifest.jsonl', 'pairing_manifest.jsonl',
             'source_groups.json', 'feature_index.jsonl', 'input_provenance.json',
             'per_video_metrics.jsonl', 'score_index.jsonl', 'threshold_summary.json',
             'scorer_registry.json', 'decision.json')
    return {name: sha256_file(old_root / name) for name in names}


def _feature_paths(old_root):
    rows = read_jsonl(old_root / 'feature_index.jsonl')
    result = {str(row['sample_id']): Path(str(row['feature_path'])) for row in rows}
    if len(result) != len(rows):
        raise ValueError('M1 feature index contains duplicate sample IDs')
    return result


def _sanity():
    d = np.arange(1, 21, dtype=np.float32)[None, :]
    value = float(m1._knn_scores(d)[0]['C2'][0])
    expected = float(np.mean(d[0, :-1] / d[0, -1]))
    scaled = float(m1._knn_scores(d * 7.0)[0]['C2'][0])
    if abs(value - expected) >= 1e-5 or abs(value - scaled) >= 1e-5 or abs(value - float(d.mean())) < 1e-5:
        raise ValueError('C2 formula/scaling sanity check failed')
    return {'distances': d[0].tolist(), 'C2': value, 'expected': expected,
            'scaled_C2': scaled, 'mean_r': float(d.mean()), 'tolerance': 1e-5, 'passed': True}


def prepare(config_path):
    cfg, base, root, old, old_root = _context(config_path, frozen=False)
    if (root / 'freeze_receipt.json').exists():
        _context(config_path)
        return {'status': 'reused', 'frozen': True, 'output': str(root)}
    if root.exists() and any(root.iterdir()):
        raise RuntimeError('Nonempty unfrozen run directory: use a new run ID for repairs')
    rows = read_jsonl(old_root / 'query_manifest.jsonl')
    pairing = read_jsonl(old_root / 'pairing_manifest.jsonl')
    original = read_jsonl(old_root / 'per_video_metrics.jsonl')
    fakes = [r for r in original if not r.get('is_real')]
    counts = Counter(r.get('eligibility_status') for r in fakes)
    if (len(rows), len(pairing), len(fakes), counts['eligible'], counts['LOCALIZATION_UNDEFINED']) != (4200, 3360, 4480, 4448, 32):
        raise ValueError('M1 original 4200/3360/4480/4448/32 accounting mismatch')
    if len(read_jsonl(old_root / 'gallery_manifest.jsonl')) != 64:
        raise ValueError('M1 gallery manifest is not 64 videos')
    if len({str(r['sample_id']) for r in rows}) != len(rows):
        raise ValueError('Duplicate M1 query IDs')
    if len({str(r['fake_sample_id']) for r in pairing}) != len(pairing):
        raise ValueError('Duplicate M1 fake pairing IDs')
    if {r['sample_id'] for r in rows if not r['is_real']} != {r['fake_sample_id'] for r in pairing}:
        raise ValueError('Pairing does not cover every fake T3 frame')
    if read_json(old_root / 'source_groups.json').get('status') != 'verified' or m1._pairing_integrity_status(old_root) != 'verified':
        raise ValueError('M1 source-group/pairing integrity not verified')
    row_by_id = {str(r['sample_id']): r for r in rows}
    for pair in pairing:
        fake = row_by_id[pair['fake_sample_id']]
        real = row_by_id[pair['target_real_sample_id']]
        if pair['source_group_id'] != fake['source_group_id']:
            raise ValueError('Pair source-group mismatch')
        m1._validate_pair_mask_provenance(pair, fake, real)
    gallery, gallery_dir = m1._load_gallery(base, old, old_root)
    ordered_gallery = read_jsonl(old_root / 'gallery_manifest.jsonl')
    if gallery.vectors.shape != (64 * 784, 768) or len(ordered_gallery) != 64:
        raise ValueError('Gallery shape/order mismatch')
    features = _feature_paths(old_root)
    index_rows = {r['sample_id']: r for r in read_jsonl(old_root / 'feature_index.jsonl')}
    if set(features) != set(row_by_id):
        raise ValueError('M1 feature index does not exactly cover query manifest')
    # Validate every reused query feature's feature/crop provenance before freeze.
    checkpoint = sha256_file(resolve_config_path(base, base.require('paths.dinov3_weights')))
    manifest_hash = sha256_file(m1._t0_manifest_paths(m1._upstream(old))[0])
    for number, row in enumerate(rows, 1):
        sid = row['sample_id']
        f = features[sid]
        entry = index_rows[sid]
        if not f.is_file() or sha256_file(f) != entry['array_sha256']:
            raise ValueError(f'Missing or changed M1 feature: {sid}')
        feature, metadata = load_features(f)
        if feature.shape != (1, 784, 768) or metadata.get('token_order') != 'row_major_patch_grid_28x28_excluding_cls_storage_register':
            raise ValueError(f'Query token shape/order mismatch: {sid}')
        # M1 supplementary and reused T0 rows have different original manifest
        # digests; the sidecar must be self-consistent and identical to M1's index.
        if metadata.get('checkpoint_sha256') != checkpoint or metadata.get('config_digest') != config_digest(base) or metadata.get('manifest_sha256') != manifest_hash:
            raise ValueError(f'Backbone/weight/config mismatch: {sid}')
        if metadata.get('sample_id') != sid or metadata.get('frame_index') != row['frame_index']:
            raise ValueError(f'Feature frame identity mismatch: {sid}')
        if not m1._feature_provenance_ok(f, row, base, checkpoint, manifest_hash):
            raise ValueError(f'Exact M1 feature/crop/token configuration mismatch: {sid}')
        if number % 100 == 0:
            print(f'[m1.1 prepare] feature {number}/{len(rows)}', flush=True)
    old_threshold = read_json(old_root / 'threshold_summary.json')
    if old_threshold.get('quantile') != 0.99 or old_threshold.get('numpy_method') != 'linear' or 'C2' not in old_threshold['thresholds']:
        raise ValueError('M1 C2 threshold provenance mismatch')
    old_index = read_jsonl(old_root / 'score_index.jsonl')
    if {r['sample_id'] for r in old_index} != set(row_by_id):
        raise ValueError('M1 raw score index incomplete')
    for entry in old_index:
        if sha256_file(entry['per_token_path']) != entry['per_token_sha256']:
            raise ValueError(f'M1 raw distances/indices missing or changed: {entry["sample_id"]}')
        item = entry['maps']['C2']
        if not Path(item['path']).is_file() or sha256_file(item['path']) != item['sha256']:
            raise ValueError(f'M1 C2 raw map missing/hash mismatch: {entry["sample_id"]}')
    sanity = _sanity()
    rng_manifests = {}
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        indices = rng.integers(0, 784, size=64).tolist()
        rng_manifests[str(seed)] = [dict(gallery_video_id=r['video_name'], gallery_sample_id=r['sample_id'],
                                         token_index=int(i), flattened_index=video * 784 + int(i))
                                    for video, (r, i) in enumerate(zip(ordered_gallery, indices))]
    root.mkdir(parents=True)
    shutil.copyfile(resolve_config_path(cfg, cfg.require('base_config')), root / 'base_config_reference.yaml')
    shutil.copyfile(Path(cfg.source), root / 'preregistered_config.yaml')
    for name in ('query_manifest.jsonl', 'pairing_manifest.jsonl', 'source_groups.json', 'gallery_manifest.jsonl'):
        shutil.copyfile(old_root / name, root / name)
    write_json(root / 'random_gallery_seed_manifest.json', rng_manifests)
    write_json(root / 'condition_manifest.json', {'conditions': [dict(condition=c, random_gallery_seed=s,
        candidates=50176 if c == CONDITIONS[0] else 64, k=20) for c, s in _conditions()],
        'spatial_control': 'crop-grid position matching, not anatomical correspondence',
        'fair_comparison': 'B vs C have 64 candidates; A has 50176 and is descriptive only'})
    gm = root / 'gallery_manifests'
    gm.mkdir()
    write_jsonl(gm / 'full_unrestricted.jsonl', [dict(gallery_sample_id=r['sample_id'], token_indices='all_784') for r in ordered_gallery])
    write_jsonl(gm / 'same_token_index.jsonl', [dict(gallery_sample_id=r['sample_id'], token_index='query_grid_index') for r in ordered_gallery])
    for seed in SEEDS:
        write_jsonl(gm / f'matched_count_unrestricted_seed{seed}.jsonl', rng_manifests[str(seed)])
    write_json(root / 'scorer_registry.json', {'only_scorer': 'C2', 'formula': 'mean(r[1:19]/r[20])',
        'rank_convention': 'one-based; numerator ranks 1..19; denominator rank 20', 'direction': 'greater_is_more_anomalous',
        'distance_floor': 1e-12, 'k': 20})
    write_json(root / 'formula_sanity.json', sanity)
    write_json(root / 'input_provenance.json', {'m1_root': str(old_root), 'm1_artifact_hashes': _m1_files(old_root),
        'gallery_directory': str(gallery_dir), 'gallery_sha256': sha256_file(gallery_dir / 'gallery.npy'),
        'feature_count': len(features), 'mask_pair_count': len(pairing), 'm1_row_accounting': dict(
            generated_fake_scorer_rows=4480, eligible=4448, localization_undefined=32,
            original_fake_video_entries=1120, original_real_video_entries=280)})
    atomic_write_text(root / 'amendments.md', '# M1.1 amendments\n\nNone recorded. Repairs require a new run ID; do not overwrite r01.\n')
    receipt = {'status': 'frozen', 'run_id': cfg.require('run_id'), 'config_digest': config_digest(cfg),
               'config_sha256': sha256_file(root / 'preregistered_config.yaml'),
               'base_config_digest': config_digest(base), 'm1_artifact_hashes': _m1_files(old_root),
               'freeze_before_fake_metrics': True, 'official_test_locked': True}
    write_json(root / 'freeze_receipt.json', receipt)
    return {'status': 'completed', 'frozen': True, 'query_frames': len(rows), 'm1_fake_rows': len(fakes),
            'm1_defined_rows': 4448, 'm1_undefined_rows': 32, 'official_test_locked': True}


def _index_rows(old_root):
    rows = read_jsonl(old_root / 'score_index.jsonl')
    return {r['sample_id']: r for r in rows}


def _write_token(root, label, sid, values, *, reused=None, distances=None, indices=None, raw_reused=None):
    if reused is not None:
        return {'sample_id': sid, 'path': str(reused), 'sha256': sha256_file(reused),
                'raw_knn_path': str(raw_reused), 'raw_knn_sha256': sha256_file(raw_reused), 'reused_m1': True}
    path = root / 'per_token_scores' / label / (m1._hash_id(sid) + '.npy')
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(values, dtype=np.float32).reshape(28, 28), allow_pickle=False)
    if distances is None or indices is None or distances.shape != (784, 20) or indices.shape != (784, 20):
        raise ValueError('Raw kNN arrays must match 784×20')
    raw = root / 'raw_knn' / label / (m1._hash_id(sid) + '.npz')
    raw.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(raw, distances=np.asarray(distances, dtype=np.float32),
                        neighbor_indices=np.asarray(indices, dtype=np.int64))
    return {'sample_id': sid, 'path': str(path), 'sha256': sha256_file(path),
            'raw_knn_path': str(raw), 'raw_knn_sha256': sha256_file(raw), 'reused_m1': False}


def _score_from_knn(result):
    return m1._knn_scores(result.distances, result.indices)[0]['C2']


def scores_stage(config_path):
    _, base, root, old, old_root = _context(config_path)
    rows = read_jsonl(old_root / 'query_manifest.jsonl')
    old_index = _index_rows(old_root)
    paths = _feature_paths(old_root)
    gallery, _ = m1._load_gallery(base, old, old_root)
    gallery_vectors = gallery.vectors.reshape(64, 784, 768)
    manifests = read_json(root / 'random_gallery_seed_manifest.json')
    write_json(root / 'score_protocol.json', {'formula_sanity': read_json(root / 'formula_sanity.json'),
        'C2_direction': 'greater_is_more_anomalous', 'distance': 'FAISS IndexFlatL2 squared distances square-rooted by ExactL2Index',
        'k': 20, 'distance_floor': 1e-12})
    for condition, seed in _conditions():
        label = _label(condition, seed)
        destination = root / 'score_indices' / f'{label}.jsonl'
        if destination.exists():
            raise RuntimeError(f'{label} score output exists; do not silently replace an old run')
        items = []
        if condition == CONDITIONS[0]:
            for row in rows:
                sid = row['sample_id']
                previous = Path(old_index[sid]['maps']['C2']['path'])
                if sha256_file(previous) != old_index[sid]['maps']['C2']['sha256']:
                    raise ValueError(f'M1 C2 score changed: {sid}')
                items.append(_write_token(root, label, sid, None, reused=previous,
                    raw_reused=Path(old_index[sid]['per_token_path'])))
        elif condition == CONDITIONS[1]:
            indices = [ExactL2Index.build(gallery_vectors[:, i, :].copy()) for i in range(784)]
            # Position-wise batched exact search avoids 784 FAISS calls per frame.
            for start in range(0, len(rows), 32):
                batch = rows[start:start + 32]
                vectors = np.stack([load_features(paths[r['sample_id']])[0][0] for r in batch])
                scores = np.empty((len(batch), 784), dtype=np.float32)
                raw_distances = np.empty((len(batch), 784, 20), dtype=np.float32)
                raw_indices = np.empty((len(batch), 784, 20), dtype=np.int64)
                for i, index in enumerate(indices):
                    result = index.search(vectors[:, i, :].copy(), 20)
                    scores[:, i] = _score_from_knn(result)
                    raw_distances[:, i] = result.distances
                    raw_indices[:, i] = result.indices
                for j, row in enumerate(batch):
                    items.append(_write_token(root, label, row['sample_id'], scores[j],
                                              distances=raw_distances[j], indices=raw_indices[j]))
                print(f'[m1.1 scores] {label} {start + len(batch)}/{len(rows)}', flush=True)
        else:
            chosen = manifests[str(seed)]
            if len(chosen) != 64 or [r['flattened_index'] // 784 for r in chosen] != list(range(64)):
                raise ValueError('Random gallery token manifest changed')
            selected = np.asarray([r['flattened_index'] for r in chosen], dtype=np.int64)
            index = ExactL2Index.build(gallery.vectors[selected].copy())
            for number, row in enumerate(rows, 1):
                sid = row['sample_id']
                query = load_features(paths[sid])[0][0]
                result = index.search(query, 20)
                values = _score_from_knn(result)
                items.append(_write_token(root, label, sid, values,
                                          distances=result.distances, indices=result.indices))
                if number % 100 == 0 or number == len(rows):
                    print(f'[m1.1 scores] {label} {number}/{len(rows)}', flush=True)
        write_jsonl(destination, items)
        write_jsonl(root / 'score_maps' / f'{label}.jsonl',
                    [dict(sample_id=item['sample_id'], token_grid=[28, 28],
                          map_path=item['path'], map_sha256=item['sha256'],
                          bilinear_output=[448, 448], align_corners=False) for item in items])
        print(f'[m1.1 scores] completed {label}', flush=True)
    return {'status': 'completed', 'conditions': len(_conditions()), 'frames_per_condition': len(rows)}


def _load_scores(root, label, expected_ids):
    entries = read_jsonl(root / 'score_indices' / f'{label}.jsonl')
    by_id = {entry['sample_id']: entry for entry in entries}
    if len(entries) != len(by_id) or set(by_id) != set(expected_ids):
        raise ValueError(f'Incomplete/duplicate score index for {label}')
    for sid, entry in by_id.items():
        if sha256_file(entry['path']) != entry['sha256'] or sha256_file(entry['raw_knn_path']) != entry['raw_knn_sha256']:
            raise ValueError(f'Changed score or raw kNN array {label}/{sid}')
    return by_id


def _token(entry):
    value = np.load(entry['path'], allow_pickle=False)
    if value.shape != (28, 28) or value.dtype != np.float32 or not np.isfinite(value).all():
        raise ValueError(f'Invalid C2 tokens: {entry["path"]}')
    return value


def _pixels(entry):
    return upsample_score_map(_token(entry)).reshape(-1)


def _macro(rows, metric):
    means = []
    for manip in MANIPS:
        values = [float(r[metric]) for r in rows if r.get('manipulation') == manip and r.get(metric) is not None
                  and r.get('eligibility_status') == 'eligible']
        if not values:
            return None
        means.append(float(np.mean(values)))
    return float(np.mean(means))


def _condition_metrics(root, old_root, condition, seed, rows, pairing):
    label = _label(condition, seed)
    by_id = {r['sample_id']: r for r in rows}
    scores = _load_scores(root, label, by_id)
    c23_real = [r['sample_id'] for r in rows if r['is_real'] and r['compression'] == 'c23']
    if len(c23_real) != 420:
        raise ValueError('M1 c23 real calibration support mismatch')
    threshold = m1._linear_quantile(np.concatenate([_pixels(scores[sid]) for sid in c23_real]), .99)
    if condition == CONDITIONS[0]:
        previous = float(read_json(old_root / 'threshold_summary.json')['thresholds']['C2'])
        if abs(previous - threshold) > 1e-5:
            raise ValueError(f'Condition A threshold differs from M1: {previous} != {threshold}')
        threshold = previous
    template = upsample_score_map(np.mean(np.stack([_token(scores[sid]) for sid in c23_real]), axis=0)).reshape(-1)
    path = root / 'spatial_templates' / f'{label}.npy'
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, template.reshape(448, 448), allow_pickle=False)
    groups = m1._frame_groups(rows)
    fakes = []
    reals = []
    frames = []
    regions = []
    spatial = []
    for (compression, manipulation, video, is_real), ids in sorted(groups.items()):
        if not is_real:
            continue
        values = np.concatenate([_pixels(scores[sid]) for sid in ids])
        reals.append({'run_id': 'r01', 'condition': condition, 'random_gallery_seed': seed,
                      'compression': compression, 'manipulation': 'original', 'video_id': video,
                      'frame_indices': [int(by_id[sid]['frame_index']) for sid in ids],
                      'is_real': True, 'n_valid_pixels': int(values.size),
                      'real_fpr': float(np.mean(values > threshold)),
                      'positive_predictions': int(np.sum(values > threshold)), 'fixed_threshold': threshold,
                      'threshold_role': 'c23_real_calibration' if compression == 'c23' else 'c40_fixed_transfer'})
    pairs = {p['fake_sample_id']: p for p in pairing}
    for (compression, manipulation, video, is_real), ids in sorted(groups.items()):
        if is_real:
            continue
        ids = sorted(ids, key=lambda sid: (by_id[sid]['frame_fraction'], by_id[sid]['frame_index']))
        if len(ids) != 3 or any(sid not in pairs for sid in ids):
            raise ValueError(f'Missing T3 pairing: {video}')
        paired = [pairs[sid]['target_real_sample_id'] for sid in ids]
        source_groups = {pairs[sid]['source_group_id'] for sid in ids}
        if len(source_groups) != 1 or any(sid not in by_id for sid in paired):
            raise ValueError(f'Incomplete source/paired real for {video}')
        masks = [m1._validated_mask(by_id[sid]).reshape(-1) for sid in ids]
        fake_maps = [_pixels(scores[sid]) for sid in ids]
        real_maps = [_pixels(scores[sid]) for sid in paired]
        mask = np.concatenate(masks)
        fake = np.concatenate(fake_maps)
        real = np.concatenate(real_maps)
        template_pixels = np.tile(template, 3)
        metric = m1._binary_metrics(fake, mask, threshold)
        paired_metric = m1._binary_metrics(real, mask, threshold)
        template_metric = m1._binary_metrics(template_pixels, mask, threshold)
        defined = metric['pixel_ap'] is not None
        if not defined:
            for item in (metric, paired_metric, template_metric):
                item['f1'] = None
                item['iou'] = None
                item['foreground_tpr'] = None
                item['fake_background_fpr'] = None
        delta = fake.astype(np.float64) - real.astype(np.float64)
        inside = mask.astype(bool)
        mean_in = float(delta[inside].mean()) if inside.any() else None
        mean_out = float(delta[~inside].mean()) if (~inside).any() else None
        r = mean_in - mean_out if mean_in is not None and mean_out is not None else None
        base = {'run_id': 'r01', 'condition': condition, 'random_gallery_seed': seed,
                'compression': compression, 'manipulation': manipulation,
                'fake_video_id': video, 'target_original_id': by_id[ids[0]]['target_id'],
                'source_group_id': next(iter(source_groups)), 'frame_indices': [by_id[sid]['frame_index'] for sid in ids],
                'fake_sample_ids': ids, 'target_real_sample_ids': paired, 'is_real': False,
                'eligibility_status': 'eligible' if defined else 'LOCALIZATION_UNDEFINED',
                'undefined_reason': None if defined else 'empty_or_full_mask'}
        entry = {**base, **metric, 'paired_real_virtual_mask_ap': paired_metric['pixel_ap'],
                 'spatial_template_ap': template_metric['pixel_ap'],
                 'ap_minus_prevalence': metric['pixel_ap'] - metric['mask_positive_fraction'] if defined else None,
                 'ap_minus_paired_real': metric['pixel_ap'] - paired_metric['pixel_ap'] if defined else None,
                 'ap_minus_spatial_template': metric['pixel_ap'] - template_metric['pixel_ap'] if defined else None,
                 'delta_score_inside': mean_in, 'delta_score_outside': mean_out,
                 'region_difference_of_differences': r}
        fakes.append(entry)
        regions.append({**base, 'delta_score_inside': mean_in, 'delta_score_outside': mean_out,
                        'region_difference_of_differences': r})
        spatial.append({**base, 'pixel_ap': metric['pixel_ap'], 'spatial_template_ap': template_metric['pixel_ap'],
                        'ap_minus_spatial_template': entry['ap_minus_spatial_template']})
        for sid, tid, fm, rm, mk in zip(ids, paired, fake_maps, real_maps, masks):
            fmetric = m1._binary_metrics(fm, mk, threshold)
            rmetric = m1._binary_metrics(rm, mk, threshold)
            frames.append({**base, 'fake_sample_id': sid, 'target_real_sample_id': tid,
                           'frame_index': by_id[sid]['frame_index'], 'pixel_ap': fmetric['pixel_ap'],
                           'paired_real_virtual_mask_ap': rmetric['pixel_ap'],
                           'mask_positive_fraction': fmetric['mask_positive_fraction'],
                           'eligibility_status': 'eligible' if fmetric['pixel_ap'] is not None else 'LOCALIZATION_UNDEFINED'})
        if len(fakes) % 50 == 0:
            print(f'[m1.1 evaluate] {label} {len(fakes)}/1120 fake videos', flush=True)
    if len(fakes) != 1120 or Counter(r['eligibility_status'] for r in fakes) != {'eligible': 1112, 'LOCALIZATION_UNDEFINED': 8}:
        raise ValueError(f'M1.1 undefined/defined accounting changed in {label}')
    return fakes, reals, frames, regions, spatial, threshold, path


def evaluate_stage(config_path):
    _, _, root, _, old_root = _context(config_path)
    rows = read_jsonl(old_root / 'query_manifest.jsonl')
    pairing = read_jsonl(old_root / 'pairing_manifest.jsonl')
    all_fake, all_real, all_frame, all_region, all_spatial, all_eligibility = [], [], [], [], [], []
    thresholds = {}
    for condition, seed in _conditions():
        label = _label(condition, seed)
        out = root / 'condition_results' / label
        if out.exists():
            raise RuntimeError(f'{label} evaluation exists; do not overwrite a previous run')
        fake, real, frame, region, spatial, threshold, template_path = _condition_metrics(
            root, old_root, condition, seed, rows, pairing)
        out.mkdir(parents=True)
        write_jsonl(out / 'per_video_metrics.jsonl', fake + real)
        write_jsonl(out / 'per_frame_metrics.jsonl', frame)
        write_jsonl(out / 'paired_region_statistics.jsonl', region)
        write_jsonl(out / 'spatial_template_metrics.jsonl', spatial)
        write_jsonl(out / 'eligibility_and_exclusions.jsonl', [dict(condition=condition, random_gallery_seed=seed,
            compression=r['compression'], manipulation=r['manipulation'], fake_video_id=r['fake_video_id'],
            eligibility_status=r['eligibility_status'], undefined_reason=r['undefined_reason']) for r in fake])
        write_json(out / 'threshold_summary.json', {'threshold': threshold, 'source': 'real_c23_T3_pooled_99th_percentile',
            'numpy_method': 'linear', 'comparison': '>', 'c40_recalibrated': False,
            'spatial_template': str(template_path), 'template_source': 'validation_real_c23_T3_token_position_mean'})
        all_fake.extend(fake)
        all_real.extend(real)
        all_frame.extend(frame)
        all_region.extend(region)
        all_spatial.extend(spatial)
        all_eligibility.extend([dict(condition=condition, random_gallery_seed=seed,
            compression=r['compression'], manipulation=r['manipulation'], fake_video_id=r['fake_video_id'],
            eligibility_status=r['eligibility_status'], undefined_reason=r['undefined_reason']) for r in fake])
        thresholds[label] = {'threshold': threshold, 'real_c23_calibration_only': True,
                             'fixed_c40_transfer': True, 'spatial_template': str(template_path)}
        print(f'[m1.1 evaluate] completed {label}', flush=True)
    write_jsonl(root / 'per_video_metrics.jsonl', all_fake + all_real)
    write_jsonl(root / 'per_frame_metrics.jsonl', all_frame)
    write_jsonl(root / 'paired_region_statistics.jsonl', all_region)
    write_jsonl(root / 'spatial_template_metrics.jsonl', all_spatial)
    write_jsonl(root / 'eligibility_and_exclusions.jsonl', all_eligibility)
    write_json(root / 'threshold_summary.json', thresholds)
    return {'status': 'completed', 'conditions': len(_conditions())}


def _summarize(rows, metric):
    selected = [r for r in rows if not r.get('is_real') and r.get('eligibility_status') == 'eligible']
    per_manip = {}
    for manip in MANIPS:
        values = [float(r[metric]) for r in selected if r['manipulation'] == manip and r.get(metric) is not None]
        per_manip[manip] = {'mean': float(np.mean(values)) if values else None, 'n': len(values)}
    mean = [per_manip[m]['mean'] for m in MANIPS]
    return {'macro': float(np.mean(mean)) if all(v is not None for v in mean) else None,
            'per_manipulation': per_manip}


def _bootstrap(rows, metric, groups, draws):
    group_index = {group: i for i, group in enumerate(groups)}
    manip_index = {manip: i for i, manip in enumerate(MANIPS)}
    totals = np.zeros((len(groups), len(MANIPS)), dtype=np.float64)
    counts = np.zeros_like(totals)
    for row in rows:
        value = row.get(metric)
        if row.get('eligibility_status') == 'eligible' and value is not None:
            i = group_index[row['source_group_id']]
            j = manip_index[row['manipulation']]
            totals[i, j] += float(value)
            counts[i, j] += 1
    sampled_totals = totals[draws].sum(axis=1)
    sampled_counts = counts[draws].sum(axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        per_manip = sampled_totals / sampled_counts
    return np.mean(per_manip, axis=1)

def _average_seeds(seed_rows):
    # Same exact video support across all five seeds; average each endpoint
    # within video before either manipulation macro or group resampling.
    per_seed = []
    for seed in SEEDS:
        rows = [r for r in seed_rows if r['random_gallery_seed'] == seed and not r['is_real']]
        per_seed.append({(r['compression'], r['manipulation'], r['fake_video_id']): r for r in rows})
    keys = set(per_seed[0])
    if any(set(s) != keys for s in per_seed[1:]):
        raise ValueError('Condition C seeds do not have identical video support')
    averaged = []
    for key in sorted(keys):
        items = [s[key] for s in per_seed]
        if any(item['source_group_id'] != items[0]['source_group_id'] or
               item['eligibility_status'] != items[0]['eligibility_status'] for item in items):
            raise ValueError('Condition C seed grouping/eligibility mismatch')
        row = dict(items[0])
        row['random_gallery_seed'] = 'five_seed_video_mean'
        for metric in METRICS:
            vals = [item.get(metric) for item in items]
            row[metric] = float(np.mean(vals)) if all(v is not None for v in vals) else None
        averaged.append(row)
    return averaged


def _ci(draws, confidence):
    valid = draws[np.isfinite(draws)]
    if valid.size != draws.size:
        raise ValueError('Incomplete source-group bootstrap support; INCONCLUSIVE_DEPENDENCE')
    q = (1 - confidence) / 2
    return [float(x) for x in np.quantile(valid, [q, 1 - q], method='linear')]


def bootstrap_stage(config_path):
    _, _, root, _, old_root = _context(config_path)
    groups_manifest = read_json(old_root / 'source_groups.json')
    if groups_manifest.get('status') != 'verified':
        raise ValueError('INCONCLUSIVE_DEPENDENCE: source groups not verified')
    groups = [r['source_group_id'] for r in groups_manifest['groups']]
    if len(groups) != 70 or len(set(groups)) != len(groups):
        raise ValueError('INCONCLUSIVE_DEPENDENCE: unexpected source groups')
    rng = np.random.default_rng(20260929)
    group_draws = rng.integers(0, len(groups), size=(2000, len(groups)))
    all_rows = {}
    c_rows = []
    for condition, seed in _conditions():
        label = _label(condition, seed)
        rows = read_jsonl(root / 'condition_results' / label / 'per_video_metrics.jsonl')
        if len([r for r in rows if not r['is_real']]) != 1120:
            raise ValueError(f'Incomplete {label} video table')
        all_rows[label] = rows
        if seed is not None:
            c_rows.extend(rows)
    all_rows['matched_count_unrestricted_five_seed_video_mean'] = _average_seeds(c_rows)
    if (root / 'bootstrap_differences.json').exists():
        raise RuntimeError('Bootstrap output exists; repair via a new run ID')
    result = []
    arrays = {'group_draws': group_draws, 'group_ids': np.asarray(groups, dtype='<U128')}
    for label, rows in all_rows.items():
        for compression in COMPRESSIONS:
            fake = [r for r in rows if not r.get('is_real') and r['compression'] == compression]
            if len(fake) != 560 or any(r['source_group_id'] not in groups for r in fake):
                raise ValueError(f'Invalid fake/group support {label}/{compression}')
            for metric in METRICS:
                point = _summarize(fake, metric)
                draw = _bootstrap(fake, metric, groups, group_draws)
                key = f'{label}__{compression}__{metric}'
                arrays[key] = draw
                result.append({'condition_label': label, 'compression': compression, 'metric': metric,
                               'point': point, 'ci95': _ci(draw, .95),
                               'ci97_5': _ci(draw, .975) if metric in ('ap_minus_prevalence', 'ap_minus_paired_real') else None,
                               'valid_bootstrap_replicates': int(np.isfinite(draw).sum())})
    # Paired B-C difference under identical candidate count and identical group draws.
    comparisons = []
    for compression in COMPRESSIONS:
        for metric in ('ap_minus_paired_real', 'ap_minus_prevalence', 'region_difference_of_differences'):
            b = f'same_token_index__{compression}__{metric}'
            c = f'matched_count_unrestricted_five_seed_video_mean__{compression}__{metric}'
            comparison = arrays[b] - arrays[c]
            comparisons.append({'compression': compression, 'metric': metric, 'contrast': 'B_minus_five_seed_C',
                                'point': next(x['point']['macro'] for x in result if x['condition_label'] == 'same_token_index'
                                              and x['compression'] == compression and x['metric'] == metric) -
                                         next(x['point']['macro'] for x in result if x['condition_label'] == 'matched_count_unrestricted_five_seed_video_mean'
                                              and x['compression'] == compression and x['metric'] == metric),
                                'ci95': _ci(comparison, .95)})
    np.savez_compressed(root / 'bootstrap_cluster_results.npz', **arrays)
    write_json(root / 'bootstrap_differences.json', {'status': 'completed', 'unit': 'source_group',
        'seed': 20260929, 'replicates': 2000, 'shared_indices_all_conditions': True,
        'estimand': 'pooled T3 per-video metrics, manipulation-equal-weight macro; C first averaged per-video over five seeds',
        'results': result, 'comparisons': comparisons})
    write_json(root / 'condition_comparison.json', {'results': result, 'matched_candidate_count_B_C': 64,
        'condition_A_candidate_count': 50176, 'paired_B_minus_C': comparisons})
    return {'status': 'completed', 'bootstrap_replicates': 2000, 'result_rows': len(result)}


def _get(results, label, compression, metric):
    return next(r for r in results if r['condition_label'] == label and r['compression'] == compression and r['metric'] == metric)


def finalize_stage(config_path):
    _, _, root, _, old_root = _context(config_path)
    if (root / 'decision.json').exists():
        raise RuntimeError('Decision already saved; do not silently overwrite')
    bootstrap = read_json(root / 'bootstrap_differences.json')
    results = bootstrap['results']
    a = 'full_unrestricted'
    b = 'same_token_index'
    c = 'matched_count_unrestricted_five_seed_video_mean'
    U = _get(results, a, 'c23', 'ap_minus_prevalence')
    D = _get(results, a, 'c23', 'ap_minus_paired_real')
    R = _get(results, a, 'c23', 'region_difference_of_differences')
    template = _get(results, a, 'c23', 'ap_minus_spatial_template')
    db = _get(results, b, 'c23', 'ap_minus_paired_real')
    dc = _get(results, c, 'c23', 'ap_minus_paired_real')
    d40 = _get(results, a, 'c40', 'ap_minus_paired_real')
    seed_d = [_get(results, f'matched_count_unrestricted_seed{seed}', 'c23', 'ap_minus_paired_real') for seed in SEEDS]
    reals40 = read_jsonl(root / 'condition_results' / a / 'per_video_metrics.jsonl')
    reals40 = [r for r in reals40 if r['is_real'] and r['compression'] == 'c40']
    fpr40 = sum(r['positive_predictions'] for r in reals40) / sum(r['n_valid_pixels'] for r in reals40)
    reasons = []
    def value(x):
        return x['point']['macro']
    if value(U) is None or value(D) is None or value(R) is None:
        status = 'INCONCLUSIVE'
    else:
        if value(U) < .02 or U['ci97_5'][0] <= 0:
            reasons.append('A U fails preregistered margin/97.5% CI')
        if value(D) < .02 or D['ci97_5'][0] <= 0:
            reasons.append('A D fails preregistered margin/97.5% CI')
        if value(db) < 0 or value(dc) < 0 or sum(value(x) >= 0 for x in seed_d) < 4:
            reasons.append('B/C matched-count direction or five-seed stability fails')
        if value(template) < 0:
            reasons.append('A fake AP below real-only spatial template AP')
        if value(R) <= 0:
            reasons.append('A paired region difference not positive')
        if sum(v['mean'] > 0 for v in D['point']['per_manipulation'].values()) < 2:
            reasons.append('Positive D not present in at least two manipulations')
        if fpr40 > .05 or value(d40) < 0:
            reasons.append('c40 fixed-threshold transfer FPR/direction fails')
        if not reasons:
            status = 'CONTINUE_C2_AS_CANDIDATE'
        elif value(D) > 0 and (value(db) < 0 or value(dc) < 0 or value(template) < 0 or
                             value(R) <= 0 or U['ci95'][0] <= 0 or sum(value(x) >= 0 for x in seed_d) < 4):
            status = 'C2_SIGNAL_BUT_SPATIAL_OR_GALLERY_CONFUNDED'
        else:
            status = 'NO_SUPPORT'
    decision = {'status': status, 'reasons': reasons, 'official_test_locked': True,
                'official_test_data_read': False, 'exploratory_validation_previously_used': True,
                'preregistered_main_scorer': 'C2', 'c23_U': U, 'c23_D': D, 'c23_R': R,
                'c23_template_contrast': template, 'c23_B_D': db, 'c23_C_D': dc,
                'c23_C_single_seed_D': seed_d, 'c40_A_D': d40, 'c40_real_fpr_fixed': fpr40}
    diagnostic = {'status': 'completed', 'official_test_locked': True, 'official_test_data_read': False,
                  'validation_previously_used_in_t0_and_m1': True,
                  'fake_masks_used_for_fitting_or_tuning': False, 'c2_formula_changed_after_freeze': False,
                  'gallery_original_manifest_reused': True, 'feature_config_mismatch': [],
                  'source_grouping_status': 'verified', 'pairing_status': 'verified',
                  'all_generated_fake_scorer_rows': 4480, 'localization_defined_rows': 4448,
                  'localization_undefined_rows': 32, 'condition_a_candidate_count': 50176,
                  'condition_b_candidate_count': 64, 'condition_c_candidate_count': 64,
                  'condition_c_seed_count': 5, 'threshold_recalibrated_on_c40': False,
                  'primary_scorer_changed_after_results': False, 'official_test_unlocked': False,
                  'mask_placement_control': 'skipped_existing_implementation_unavailable'}
    write_json(root / 'decision.json', decision)
    write_json(root / 'diagnostics.json', diagnostic)
    def fmt(r):
        return f"{value(r):+.5f} [95% {r['ci95'][0]:+.5f}, {r['ci95'][1]:+.5f}]"
    lines = ['# M1.1 C2 定向去混杂审查', '', f'**Decision:** `{status}` — exploratory only, official test locked.',
             '', '## Observed', '', '| Compression | Condition | U | D | R | AP − real spatial template AP |',
             '|---|---|---:|---:|---:|---:|']
    for compression in COMPRESSIONS:
        for label in (a, b, c):
            lines.append('| ' + ' | '.join([compression, label] +
                [fmt(_get(results, label, compression, metric)) for metric in
                 ('ap_minus_prevalence', 'ap_minus_paired_real', 'region_difference_of_differences', 'ap_minus_spatial_template')]) + ' |')
    lines += ['', '### Manipulation-level D (pooled T3 per-video, equal-weight in macro)',
              '', '| Compression | Condition | Deepfakes | Face2Face | FaceSwap | NeuralTextures |',
              '|---|---|---:|---:|---:|---:|']
    for compression in COMPRESSIONS:
        for label in (a, b, c):
            by_manip = _get(results, label, compression, 'ap_minus_paired_real')['point']['per_manipulation']
            lines.append('| ' + ' | '.join([compression, label] +
                [f"{by_manip[manip]['mean']:+.5f} (n={by_manip[manip]['n']})" for manip in MANIPS]) + ' |')
    lines += ['', f"Condition A c23 U 97.5% CI: {U['ci97_5']}; D 97.5% CI: {D['ci97_5']}.",
              'B-C 64-candidate paired group-bootstrap contrasts are in condition_comparison.json; '
              'A cannot be interpreted as an equal-candidate spatial contrast.']
    historical = read_json(old_root / 'bootstrap_differences.json')['results']
    past = next(row for row in historical if row['compression'] == 'c23' and row['scorer'] == 'C2'
                and row['metric'] == 'ap_minus_paired_real')
    lines += ['', f"M1 historical video-bootstrap c23 C2 D CI95: {past['historical_ci95']}; "
              f"M1 source-group CI95: {past['ci95']}. M1.1 uses source-group resampling; "
              'video bootstrap is auxiliary and may understate target/source dependence.']
    lines += ['', 'Single-seed C23 D (all five; no seed selection):']
    for seed, row in zip(SEEDS, seed_d):
        lines.append(f'- {seed}: {fmt(row)}')
    lines += ['', f'c40 full-gallery fixed-c23-threshold pooled real FPR: {fpr40:.5f}.',
              'M1 raw: 1,120 unique fake-video×compression×manipulation entries; 4,480 fake×scorer rows, '
              '4,448 eligible and 32 LOCALIZATION_UNDEFINED (8 unique empty-mask entries per condition).',
              '', '## Supported', '', f'Decision reasons: {"; ".join(reasons) if reasons else "all preregistered continuation gates met (candidate only)"}.',
              'B and C each have 64 candidates; A has 50,176. A vs B/C cannot isolate spatial effects.',
              '', '## Speculative (not established)', '',
              'A real-face manifold, tangent/normal departure, forgery-specific anomaly mechanism, and cross-dataset transfer are not established.',
              '', 'Original Synthetic Gate B remains FAIL. M0 remains a provisional structural audit. '
              'M1 `r_k` remains unsupported. This M1.1 is an exploratory deconfounding audit on previously used '
              'validation data, not an independent confirmation. Official test remains locked.', '']
    atomic_write_text(root / 'summary.md', '\n'.join(lines))
    atomic_write_text(root / 'README.md', '# M1.1 outputs\n\nFrozen r01. A, B and all five C conditions have separate '
        'condition_results directories. Raw 28×28 C2 score maps live in per_token_scores/ and raw kNN '
        'distances/indices in raw_knn/; score_maps/ indexes the frozen 28×28 maps (bilinear 448×448 at evaluation). Full A reuses M1 read-only arrays by path. '
        'bootstrap_cluster_results.npz preserves shared group draws. Do not substitute this exploratory validation audit for independent confirmation.\n')
    atomic_write_text(root / 'analysis-report.md', '\n'.join(lines))
    atomic_write_text(root / 'stats-appendix.md', '# Source-group statistics\n\n70 verified source groups; 2,000 '
        'shared replacement draws. Within each draw: pool eligible per-video T3 metrics in each manipulation, '
        'then average four manipulation means. Five C seeds are first averaged within each identical video. '
        '95% and (U/D only) 97.5% percentile intervals are in bootstrap_differences.json. '
        'No ordinary video-bootstrap interval is presented as independent.\n')
    atomic_write_text(root / 'figure-catalog.md', '# Figure catalog\n\nNo figures generated; raw tabulations and '
        'paired comparisons are retained in machine-readable files.\n')
    return {'status': 'completed', 'decision': status, 'official_test_locked': True, 'output': str(root)}


def run_stage(config_path, stage):
    return {'prepare': prepare, 'scores': scores_stage, 'evaluate': evaluate_stage,
            'bootstrap': bootstrap_stage, 'finalize': finalize_stage}[stage](config_path)
