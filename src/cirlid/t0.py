"""Validation-only exploratory multi-frame evaluation, separate from all gates.

T0-A retains independent frame maps. No temporal token correspondence is
assumed. Technical exclusions use a common video cohort across all budgets
and compressions; mask content never determines frame selection or thresholds.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
import copy
import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from .config import Config, config_digest, load_config, require_protocol, resolve_config_path
from .crops import cache_manifest_crops, _save_array
from .features import feature_artifact_path, load_features
from .ffpp import FFPPPaths, enumerate_records, enumerate_gallery_records, read_split_pairs
from .gallery import load_gallery
from .geometry import fractional_frame_index
from .heatmap import upsample_score_map
from .io import read_json, read_jsonl, sha256_file, write_json, write_jsonl, atomic_write_text
from .manifest import read_manifest, write_manifest, select_gallery_ids
from .metrics import false_positive_rate, _binary_overlap
from .scoring import score_from_knn

BUDGETS = {
    'T1_center': [0.50], 'T3_frozen': [0.25, 0.50, 0.75],
    'T5_uniform': [0.10, 0.30, 0.50, 0.70, 0.90],
    'T9_uniform': [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90],
}
METHODS = ('arithmetic_multi_neighbor', 'CIR_LID')
MANIPULATIONS = ('Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures')
METRICS = ('pixel_ap', 'pixel_auroc', 'f1', 'iou')
COMPARISONS = (('T3_frozen', 'T1_center'), ('T5_uniform', 'T3_frozen'),
               ('T9_uniform', 'T3_frozen'), ('T9_uniform', 'T5_uniform'))
SKIP_REASON = ('Temporal mean/variance maps were not evaluated because token-wise temporal '
               'correspondence was not guaranteed; only frame-pooled evaluation was considered valid.')


def _context(path: str | Path) -> tuple[Config, Config, Path]:
    t0 = load_config(path)
    base = load_config(resolve_config_path(t0, t0.require('base_config')))
    require_protocol(base)
    expected = {
        'frame_rounding': 'half_up',
        'frame_budgets': BUDGETS, 'methods': list(METHODS), 'compressions': ['c23', 'c40'],
        'threshold.calibration_budget': 'T3_frozen', 'threshold.compression': 'c23',
        'threshold.real_only': True, 'threshold.quantile': 0.99, 'threshold.numpy_method': 'linear',
        'cohort_policy': 'complete_union_all_budgets_and_compressions',
        'bootstrap.replicates': 2000, 'bootstrap.confidence_level': 0.95,
        'temporal_maps': 'skip_unverified_correspondence',
    }
    for key, value in expected.items():
        if t0.get(key) != value:
            raise ValueError(f'T0-v1 protocol mismatch: {key}')
    if base.require('ffpp.manipulations') != list(MANIPULATIONS):
        raise ValueError('T0 requires the four frozen manipulations in their original order')
    root = resolve_config_path(t0, t0.require('output_dir'))
    if root != resolve_config_path(base, base.require('paths.results')) / 't0_multiframe_v1':
        raise ValueError('T0-v1 output must be the independent results/t0_multiframe_v1 directory')
    root.mkdir(parents=True, exist_ok=True)
    snapshot = {'t0': t0.data, 'base': base.data,
                't0_digest': config_digest(t0), 'base_digest': config_digest(base)}
    lock = root / 'protocol.json'
    if lock.exists() and read_json(lock) != snapshot:
        raise RuntimeError('Protocol changed after preparation; preserve this run and use a new version')
    write_json(lock, snapshot)
    import yaml
    atomic_write_text(root / 'config_t0_v1.yaml', yaml.safe_dump(snapshot, allow_unicode=True, sort_keys=False))
    return t0, base, root


def _isolated(base: Config, root: Path, *, gallery: bool = False) -> Config:
    data = copy.deepcopy(base.data)
    workspace = root / ('gallery_workspace' if gallery else 'workspace')
    for name in ('manifests', 'crops', 'masks', 'features', 'indices', 'environment'):
        data['paths'][name] = str(workspace / name)
    data['paths']['results'] = str(root)
    data['experiment']['t0_namespace'] = 'gallery' if gallery else 'validation'
    return Config(data, base.source)


def _paths(base: Config) -> FFPPPaths:
    # Construct paths only; never open test split or enumerate/probe test videos.
    return FFPPPaths.from_root(resolve_config_path(base, base.require('paths.ffpp_root')),
        {key: resolve_config_path(base, value) for key, value in base.require('ffpp.split_files').items()})


def _key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row['manipulation']), str(row['video_name'])


def _crop(config: Config, manifest: Path, *, empty_exclusion: bool = False) -> dict[str, Any]:
    return asdict(cache_manifest_crops(
        manifest, crops_dir=resolve_config_path(config, config.require('paths.crops')),
        masks_dir=resolve_config_path(config, config.require('paths.masks')),
        input_size=int(config.require('input.size')),
        detector_upsample_primary=int(config.require('face_crop.upsample_primary')),
        detector_upsample_retry=int(config.require('face_crop.upsample_retry')),
        square_scale=float(config.require('face_crop.square_scale')),
        padding=str(config.require('face_crop.padding')),
        mask_policy=str(config.require('face_crop.mask_binarization')),
        exclude_empty_fake_masks=empty_exclusion))


def prepare(t0: Config, base: Config, root: Path) -> dict[str, Any]:
    cfg = _isolated(base, root)
    manifest = resolve_config_path(cfg, cfg.require('paths.manifests')) / 'manifest.jsonl'
    if manifest.exists():
        raise RuntimeError('T0 manifest already exists; resume at crops/features, do not rewrite provenance')
    paths = _paths(base)
    train_ids = {v for pair in read_split_pairs(paths.split_train) for v in pair}
    val_ids = {v for pair in read_split_pairs(paths.split_validation) for v in pair}
    if train_ids & val_ids:
        raise ValueError('Train/validation original identity overlap')
    positions = sorted({p for values in BUDGETS.values() for p in values})
    records = enumerate_records(paths, split='validation', compressions=('c23', 'c40'),
                               manipulations=MANIPULATIONS, frame_fractions=positions,
                               seed=int(t0.require('seed')))
    by_sample: dict[str, dict[str, Any]] = {}
    groups: dict[tuple[str, str, str], dict[str, Any]] = {}
    for record in records:
        row = record.to_dict()
        # Union positions can collide for short videos. Use one deterministic
        # source mapping per actual frame, shared by all budgets.
        by_sample.setdefault(row['sample_id'], row)
        group = (row['compression'], row['manipulation'], row['video_name'])
        groups.setdefault(group, row)
    entries = []
    for group, row in sorted(groups.items()):
        for budget, fractions in BUDGETS.items():
            indices, duplicates = [], []
            for position in fractions:
                index = fractional_frame_index(row['frame_count'], position)
                if index in indices:
                    duplicates.append({'position': position, 'frame_index': index})
                else:
                    indices.append(index)
            entries.append({'split': 'validation', 'compression': group[0], 'manipulation': group[1],
                'video_name': group[2], 'is_real': row['is_real'], 'budget': budget,
                'requested_positions': fractions, 'requested_frames': len(fractions),
                'frame_rounding': 'half_up',
                'final_frame_indices': indices, 'actual_frames': len(indices),
                'duplicate_positions': duplicates, 'duplicate_count': len(duplicates),
                'sample_ids': [f'validation:{group[0]}:{group[1]}:{group[2]}:{i}' for i in indices]})
    write_manifest(list(by_sample.values()), manifest)
    write_jsonl(root / 'frame_manifest.jsonl', entries)
    write_json(root / 'split_audit.json', {'train_validation_overlap': [],
        'test_split_read': False, 'test_videos_read': False,
        'test_cross_split_overlap': 'not_reaudited; rely on preserved original preflight audit',
        'frame_manifest_sha256': sha256_file(root / 'frame_manifest.jsonl'),
        'split_hashes': {'train': sha256_file(paths.split_train), 'validation': sha256_file(paths.split_validation)}})
    return {'status': 'completed', 'unique_frames': len(by_sample), 'video_budget_rows': len(entries)}


def gallery_stage(base: Config, root: Path) -> dict[str, Any]:
    """Reuse a verified gallery; otherwise build a separate identical train gallery."""
    original = resolve_config_path(base, base.require('paths.indices'))
    if (original / 'gallery.metadata.json').exists():
        gallery = load_gallery(original)
        _verify_gallery(base, root, gallery, original)
        return {'status': 'reused', 'gallery': str(original)}
    cfg = _isolated(base, root, gallery=True)
    paths = _paths(base)
    train_ids = sorted({v for pair in read_split_pairs(paths.split_train) for v in pair})
    val_ids = {v for pair in read_split_pairs(paths.split_validation) for v in pair}
    if set(train_ids) & val_ids:
        raise ValueError('Train/validation overlap')
    ids = select_gallery_ids(train_ids, int(base.require('gallery.videos')), int(base.require('experiment.seed')))
    manifest = resolve_config_path(cfg, cfg.require('paths.manifests')) / 'gallery_manifest.jsonl'
    destination = resolve_config_path(cfg, cfg.require('paths.indices'))
    if (destination / 'gallery.metadata.json').exists():
        _verify_gallery(base, root, load_gallery(destination), destination)
        return {'status': 'reused', 'gallery': str(destination)}
    if not manifest.exists():
        write_manifest(enumerate_gallery_records(paths, gallery_ids=ids,
                       seed=int(base.require('experiment.seed'))), manifest)
    rows = read_manifest(manifest)
    if sorted(row['video_name'] for row in rows) != ids or any(
            row['split'] != 'train' or not row['is_real'] or row['compression'] != 'c23' for row in rows):
        raise ValueError('Isolated gallery manifest does not match frozen selection')
    _crop(cfg, manifest)
    from .commands import extract_features, build_gallery_command
    extract_features(cfg, subset='gallery')
    result = build_gallery_command(cfg)
    _verify_gallery(base, root, load_gallery(destination), destination)
    return result


def _verify_gallery(base: Config, root: Path, gallery: Any, directory: Path) -> None:
    cfg = base if directory == resolve_config_path(base, base.require('paths.indices')) else _isolated(base, root, gallery=True)
    meta = gallery.metadata
    if meta.get('config_digest') != config_digest(cfg):
        raise ValueError('Gallery does not match frozen base / isolated gallery config')
    if gallery.vectors.shape != (50176, 768) or gallery.vectors.dtype != np.float32:
        raise ValueError('Gallery must be float32 [50176,768]')
    if meta.get('gallery_array_sha256') != sha256_file(directory / 'gallery.npy'):
        raise ValueError('Gallery hash mismatch')
    paths = _paths(base)
    train_ids = {v for pair in read_split_pairs(paths.split_train) for v in pair}
    validation_ids = {v for pair in read_split_pairs(paths.split_validation) for v in pair}
    if train_ids & validation_ids:
        raise ValueError('Gallery train identities overlap validation')
    ids = select_gallery_ids(sorted(train_ids), 64, int(base.require('experiment.seed')))
    if meta.get('gallery_video_ids') != ids:
        raise ValueError('Gallery IDs do not match the frozen train-only selection')
    manifest = Path(str(meta.get('manifest', '')))
    if not manifest.is_file() or meta.get('manifest_sha256') != sha256_file(manifest):
        raise ValueError('Gallery manifest provenance mismatch')
    checkpoint = resolve_config_path(base, base.require('paths.dinov3_weights'))
    checkpoint_hash = sha256_file(checkpoint)
    feature_paths = meta.get('feature_paths', [])
    if len(feature_paths) != 64:
        raise ValueError('Gallery must reference all 64 frozen feature artifacts')
    for feature_path in feature_paths:
        sidecar = read_json(Path(feature_path).with_suffix('.npy.metadata.json'))
        if sidecar.get('checkpoint_sha256') != checkpoint_hash or sidecar.get('config_digest') != config_digest(cfg):
            raise ValueError('Gallery feature checkpoint/config provenance mismatch')
    # Validate index contents as well as dimensions, without reading query/test data.
    reconstructed = gallery.index._index.reconstruct_n(0, gallery.index.ntotal)
    if not np.array_equal(reconstructed, gallery.vectors):
        raise ValueError('Gallery index vectors disagree with gallery.npy')


def _gallery(base: Config, root: Path) -> tuple[Any, Path]:
    original = resolve_config_path(base, base.require('paths.indices'))
    directory = original if (original / 'gallery.metadata.json').exists() else root / 'gallery_workspace/indices'
    gallery = load_gallery(directory)
    _verify_gallery(base, root, gallery, directory)
    return gallery, directory


def _rows(base: Config, root: Path) -> tuple[Config, Path, list[dict[str, Any]]]:
    cfg = _isolated(base, root)
    manifest = resolve_config_path(cfg, cfg.require('paths.manifests')) / 'manifest.jsonl'
    audit = read_json(root / 'split_audit.json')
    if audit.get('frame_manifest_sha256') != sha256_file(root / 'frame_manifest.jsonl'):
        raise ValueError('Frame sampling manifest changed after preparation')
    paths = _paths(base)
    for split, path in (('train', paths.split_train), ('validation', paths.split_validation)):
        if audit['split_hashes'][split] != sha256_file(path):
            raise ValueError('Official train/validation split changed after preparation')
    rows = read_manifest(manifest)
    if any(row['split'] != 'validation' for row in rows):
        raise ValueError('T0 manifest contains a forbidden split')
    return cfg, manifest, rows


def score_stage(base: Config, root: Path) -> dict[str, Any]:
    cfg, manifest, rows = _rows(base, root)
    gallery, directory = _gallery(base, root)
    feature_root = resolve_config_path(cfg, cfg.require('paths.features')) / 'validation'
    manifest_hash = sha256_file(manifest)
    checkpoint_hash = sha256_file(resolve_config_path(base, base.require('paths.dinov3_weights')))
    provenance = {'config_digest': config_digest(cfg), 'manifest_sha256': manifest_hash,
                  'gallery_sha256': sha256_file(directory / 'gallery.npy'), 'checkpoint_sha256': checkpoint_hash}
    index_rows = []
    for number, row in enumerate(rows, 1):
        if row.get('exclusion_reason'):
            index_rows.append({'sample_id': row['sample_id'], 'status': 'excluded',
                               'reason': row['exclusion_reason']})
            continue
        feature_path = feature_artifact_path(feature_root, row['sample_id'])
        tokens, meta = load_features(feature_path)
        if tokens.shape != (1, 784, 768):
            raise ValueError(f'Expected one frozen feature frame: {feature_path}')
        for key, value in {'config_digest': config_digest(cfg), 'manifest_sha256': manifest_hash,
                           'checkpoint_sha256': checkpoint_hash, 'subset': 'validation',
                           'sample_id': row['sample_id'], 'device': 'cuda'}.items():
            if meta.get(key) != value:
                raise ValueError(f'Feature provenance mismatch: {key}: {feature_path}')
        if meta.get('array_sha256') != sha256_file(feature_path):
            raise ValueError('Feature array hash missing or mismatched')
        if meta.get('crop_sha256') != sha256_file(row['crop_path']):
            raise ValueError(f'Crop changed after feature extraction: {row["sample_id"]}')
        distances = gallery.index.search(tokens[0], 20,
                     batch_size=int(base.require('knn.query_batch_size'))).distances
        scores = score_from_knn(distances, distance_floor=float(base.require('scores.distance_floor')),
                               cir_denominator_floor=float(base.require('scores.cir_denominator_floor')))
        paths = {}
        for method in METHODS:
            score = scores[method].reshape(28, 28)
            if not np.isfinite(score).all():
                raise ValueError('Nonfinite score map')
            name = hashlib.sha256(row['sample_id'].encode()).hexdigest()[:24]
            path = root / 'score_maps' / method / f'{name}.npy'
            _save_array(path, score)
            paths[method] = {'path': str(path), 'sha256': sha256_file(path)}
        index_rows.append({'sample_id': row['sample_id'], 'status': 'completed', 'maps': paths,
                           'mask_cache_path': row.get('mask_cache_path'), 'frame_index': row['frame_index'],
                           'valid_pixels': 448 * 448,
                           'mask_sha256': sha256_file(row['mask_cache_path']) if not row['is_real'] else None})
        if number % 100 == 0 or number == len(rows):
            print(f'[t0 scores] {number}/{len(rows)}', flush=True)
    write_jsonl(root / 'score_index.jsonl', index_rows)
    write_json(root / 'scores.provenance.json', provenance)
    return {'status': 'completed', 'frames': len(index_rows)}


def _map(item: dict[str, Any], method: str) -> np.ndarray:
    artifact = item['maps'][method]
    if sha256_file(artifact['path']) != artifact['sha256']:
        raise ValueError('Score artifact hash mismatch')
    values = np.load(artifact['path'], allow_pickle=False)
    if values.shape != (28, 28) or values.dtype != np.float32 or not np.isfinite(values).all():
        raise ValueError('Invalid score map')
    return upsample_score_map(values)


def _linear_quantile_inplace(pool: np.ndarray, quantile: float) -> float:
    """Exact NumPy-linear quantile for finite float32 arrays/memmaps.

    The caller owns the array; this helper changes its order in-place.
    """
    if pool.ndim != 1 or pool.size == 0 or not 0 < quantile < 1:
        raise ValueError('Invalid calibration pool/quantile')
    # Scores were finite-checked frame-by-frame before writing the memmap.
    location = (pool.size - 1) * quantile
    lo, hi = int(np.floor(location)), int(np.ceil(location))
    pool.partition((lo, hi))
    a, b = float(pool[lo]), float(pool[hi])
    return a + (b - a) * (location - lo)


def _pixel(scores: np.ndarray, mask: np.ndarray, threshold: float) -> dict[str, Any]:
    from sklearn.metrics import average_precision_score, roc_auc_score
    if not np.isin(mask, [0, 1]).all():
        raise ValueError('Mask is not binary')
    positive = int(mask.sum())
    iou, f1 = _binary_overlap(mask, scores > threshold)
    return {'pixel_ap': float(average_precision_score(mask, scores)) if positive else None,
            'pixel_auroc': float(roc_auc_score(mask, scores)) if 0 < positive < len(mask) else None,
            'f1': f1, 'iou': iou, 'positive_pixels': positive, 'total_pixels': len(mask),
            'undefined_reason': 'single_class_mask' if positive in (0, len(mask)) else None}


def evaluate(t0: Config, base: Config, root: Path) -> dict[str, Any]:
    cfg, manifest, rows = _rows(base, root)
    gallery, gallery_dir = _gallery(base, root)
    del gallery
    expected = {'config_digest': config_digest(cfg), 'manifest_sha256': sha256_file(manifest),
                'gallery_sha256': sha256_file(gallery_dir / 'gallery.npy'),
                'checkpoint_sha256': sha256_file(resolve_config_path(base, base.require('paths.dinov3_weights')))}
    if read_json(root / 'scores.provenance.json') != expected:
        raise ValueError('Scores have stale provenance; rerun scores')
    frames = read_jsonl(root / 'frame_manifest.jsonl')
    by_id = {row['sample_id']: row for row in rows}
    score_rows = read_jsonl(root / 'score_index.jsonl')
    index = {row['sample_id']: row for row in score_rows}
    if len(index) != len(score_rows) or set(index) != set(by_id):
        raise ValueError('Score index and manifest sample IDs disagree')
    failures = defaultdict(list)
    for row in rows:
        if row.get('exclusion_reason') or index[row['sample_id']]['status'] != 'completed':
            failures[_key(row)].append({'sample_id': row['sample_id'],
                                      'reason': row.get('exclusion_reason') or 'missing_score'})
    valid_keys = {_key(row) for row in rows} - set(failures)
    if not valid_keys:
        raise ValueError('No complete-union videos remain')
    # Label-free selection: reject a video only for technical failure at any
    # union frame, on either compression. Never exclude by mask foreground.
    expected_ids = {sid for entry in frames for sid in entry['sample_ids']}
    if expected_ids != set(by_id):
        raise ValueError('Sampling entries and crop manifest disagree')
    for entry in frames:
        if len(entry['sample_ids']) != entry['actual_frames']:
            raise ValueError('Frame budget count mismatch')
        for sid, frame_index in zip(entry['sample_ids'], entry['final_frame_indices']):
            row = by_id[sid]
            if (_key(row) != _key(entry) or row['compression'] != entry['compression']
                    or row['frame_index'] != frame_index):
                raise ValueError('Frame sampling identity mismatch')
    valid_frames = [entry for entry in frames if _key(entry) in valid_keys]
    calibration = [entry for entry in valid_frames if entry['is_real'] and
                   entry['compression'] == 'c23' and entry['budget'] == 'T3_frozen']
    if not calibration:
        raise ValueError('No complete real c23 calibration videos')
    thresholds = {}
    for method in METHODS:
        ids = [sid for entry in calibration for sid in entry['sample_ids']]
        count = len(ids) * 448 * 448
        # Exact pooled quantile without a multi-GB float64 concatenation.
        mmap_path = root / f'.calibration_{method}.f32'
        pool = np.memmap(mmap_path, mode='w+', dtype=np.float32, shape=(count,))
        try:
            for number, sid in enumerate(ids):
                pool[number * 448 * 448:(number + 1) * 448 * 448] = _map(index[sid], method).ravel()
                if (number + 1) % 100 == 0 or number + 1 == len(ids):
                    print(f'[t0 threshold] {method} {number + 1}/{len(ids)}', flush=True)
            # NumPy's linear quantile promotes endpoints to float64; partition
            # in-place in the file-backed array and compute that interpolation.
            thresholds[method] = _linear_quantile_inplace(pool, 0.99)
        finally:
            del pool
            mmap_path.unlink(missing_ok=True)
    write_json(root / 'threshold_summary.json', {'thresholds': thresholds, 'budget': 'T3_frozen',
        'source': 'validation_real_c23_pooled_pixels', 'quantile': 0.99, 'numpy_method': 'linear',
        'shared_across_all_budgets_and_compressions': True, 'calibration_videos': len(calibration),
        'calibration_frames': sum(e['actual_frames'] for e in calibration),
        'calibration_sample_ids': [sid for e in calibration for sid in e['sample_ids']],
        'common_technical_cohort': True})
    metrics = []
    for number, entry in enumerate(valid_frames, 1):
        masks = []
        if not entry['is_real']:
            for sid in entry['sample_ids']:
                item = index[sid]
                path = by_id[sid]['mask_cache_path']
                if sha256_file(path) != item['mask_sha256']:
                    raise ValueError('Mask changed after scoring')
                mask = np.load(path, allow_pickle=False)
                if mask.shape != (448, 448) or not np.isin(mask, [0, 1]).all():
                    raise ValueError('Invalid cached mask')
                masks.append(mask.ravel())
        for method in METHODS:
            scores = np.concatenate([_map(index[sid], method).ravel() for sid in entry['sample_ids']])
            values = ({'fpr': false_positive_rate(scores, thresholds[method]), 'total_pixels': len(scores)}
                      if entry['is_real'] else _pixel(scores, np.concatenate(masks), thresholds[method]))
            metrics.append({**entry, 'method': method, 'video_id': entry['video_name'], **values})
        if number % 100 == 0 or number == len(valid_frames):
            print(f'[t0 evaluate] video-budget={number}/{len(valid_frames)}', flush=True)
    write_jsonl(root / 'per_video_metrics.jsonl', metrics)
    aggregate, differences = _statistics(metrics, int(t0.require('seed')))
    write_json(root / 'aggregate_metrics.json', aggregate)
    write_json(root / 'bootstrap_differences.json', differences)
    diagnostics = _diagnostics(rows, frames, valid_keys, failures, metrics)
    write_json(root / 'diagnostics.json', diagnostics)
    _report(root, aggregate, differences, diagnostics)
    return {'status': 'completed', 'output': str(root), 'official_test_unlocked': False,
            'synthetic_gate_b': 'FAIL', 't0_b_executed': False}


def _statistics(rows: list[dict[str, Any]], seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """One shared stratified video draw for all budgets within each endpoint.

    Undefined AP/AUROC are reported separately; each endpoint uses the
    intersection of mathematically defined videos, not different cohorts per T.
    """
    aggregates, differences = [], []
    for compression in ('c23', 'c40'):
        for method in METHODS:
            for metric in (*METRICS, 'fpr'):
                real = metric == 'fpr'
                selected = [r for r in rows if r['compression'] == compression and
                            r['method'] == method and r['is_real'] == real]
                lookup = {(r['budget'], r['manipulation'], r['video_name']): r for r in selected}
                groups = ('original',) if real else MANIPULATIONS
                group_arrays, counts, undefined = [], {}, {}
                for group in groups:
                    all_ids = sorted({r['video_name'] for r in selected if r['manipulation'] == group})
                    ids = [video for video in all_ids if all(
                        lookup.get((budget, group, video), {}).get(metric) is not None for budget in BUDGETS)]
                    counts[group] = len(ids)
                    undefined[group] = [v for v in all_ids if v not in set(ids)]
                    group_arrays.append(np.asarray([[lookup[(budget, group, v)][metric]
                                                     for budget in BUDGETS] for v in ids], dtype=np.float64)
                                        .reshape(len(ids), len(BUDGETS)))
                descriptor = {'compression': compression, 'method': method, 'metric': metric}
                if any(len(a) == 0 for a in group_arrays):
                    aggregates.append({**descriptor, 'status': 'undefined_missing_stratum',
                                       'n_videos': counts, 'undefined_videos': undefined})
                    continue
                means = np.mean([a.mean(axis=0) for a in group_arrays], axis=0)
                # Reset to the same seed: sorted video identities are shared
                # for all methods/endpoints whenever their eligibility agrees.
                rng = np.random.default_rng(seed)
                samples = np.zeros((2000, len(BUDGETS)), dtype=np.float64)
                stratum_samples = []
                for array in group_arrays:
                    draw = rng.integers(0, len(array), size=(2000, len(array)))
                    boot = array[draw].mean(axis=1)
                    stratum_samples.append(boot)
                    samples += boot / len(groups)
                pooled = np.concatenate(group_arrays)
                for column, budget in enumerate(BUDGETS):
                    record = {**descriptor, 'budget': budget, 'status': 'completed',
                        'n_videos': int(sum(counts.values())), 'mean': float(means[column]),
                        'std': float(np.std(pooled[:, column], ddof=1)) if len(pooled) > 1 else 0.0,
                        'ci95': np.quantile(samples[:, column], [0.025, 0.975]).tolist(),
                        'per_manipulation': {g: {'n_videos': counts[g],
                            'mean': float(a[:, column].mean()),
                            'std': float(a[:, column].std(ddof=1)) if len(a) > 1 else 0.0,
                            'ci95': np.quantile(b[:, column], [0.025, 0.975]).tolist()}
                            for g, a, b in zip(groups, group_arrays, stratum_samples)},
                        'undefined_videos': undefined}
                    if real:
                        subset = [r for r in selected if r['budget'] == budget]
                        record['pooled_pixel_fpr'] = float(sum(r['fpr'] * r['total_pixels'] for r in subset)
                                                          / sum(r['total_pixels'] for r in subset))
                    aggregates.append(record)
                names = list(BUDGETS)
                for first, second in COMPARISONS:
                    a, b = names.index(first), names.index(second)
                    absolute = samples[:, a] - samples[:, b]
                    relative = np.divide(absolute, samples[:, b], out=np.full(2000, np.nan),
                                         where=samples[:, b] != 0)
                    finite = relative[np.isfinite(relative)]
                    differences.append({**descriptor, 'comparison': f'{first} - {second}',
                        'n_videos': counts, 'replicates': 2000, 'seed': seed,
                        'absolute_delta': float(means[a] - means[b]),
                        'absolute_ci95': np.quantile(absolute, [0.025, 0.975]).tolist(),
                        'relative_delta': float((means[a] - means[b]) / means[b]) if means[b] else None,
                        'relative_ci95': np.quantile(finite, [0.025, 0.975]).tolist() if len(finite) == 2000 else None,
                        'undefined_relative_replicates': int(2000 - len(finite)),
                        'per_manipulation_delta': {g: float(x[:, a].mean() - x[:, b].mean())
                                                   for g, x in zip(groups, group_arrays)},
                        'shared_video_draws_all_budgets': True})
    return aggregates, differences


def _diagnostics(rows: list[dict[str, Any]], frames: list[dict[str, Any]], valid: set,
                 failures: dict, metrics: list[dict[str, Any]]) -> dict[str, Any]:
    import cv2
    quality = {}
    for row in rows:
        if row.get('exclusion_reason') or not row.get('crop_path'):
            continue
        crop = np.load(row['crop_path'], allow_pickle=False)
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        # Diagnostic only: never used to select frames or exclude videos.
        quality[row['sample_id']] = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    summary = []
    for compression in ('c23', 'c40'):
        for budget in BUDGETS:
            entries = [e for e in frames if e['compression'] == compression and e['budget'] == budget]
            q = [quality[sid] for e in entries for sid in e['sample_ids'] if sid in quality]
            summary.append({'compression': compression, 'budget': budget,
                'requested_videos': len(entries), 'valid_videos': sum(_key(e) in valid for e in entries),
                'actual_frame_count_distribution': {str(n): sum(e['actual_frames'] == n for e in entries)
                                                   for n in sorted({e['actual_frames'] for e in entries})},
                'duplicate_frames': sum(e['duplicate_count'] for e in entries),
                'invalid_frames': sum(sid not in quality for e in entries for sid in e['sample_ids']),
                'laplacian_variance_quantiles': np.quantile(q, [0.1, 0.5, 0.9]).tolist() if q else None})
    return {'cohort_policy': 'complete_union_all_budgets_and_compressions',
        'same_video_set_all_budgets_and_compressions': True,
        'valid_videos_per_manipulation': {g: sum(k[0] == g for k in valid) for g in ('original', *MANIPULATIONS)},
        'technical_exclusions': [{'manipulation': k[0], 'video_name': k[1], 'failures': v}
                                for k, v in sorted(failures.items())],
        'frame_budget_diagnostics': summary, 'per_frame_blur_proxy': quality,
        'mask_remapped_frames': sum(bool(r.get('mask_frame_remapped')) for r in rows),
        'undefined_pixel_metric_rows': [{k: r.get(k) for k in ('method', 'compression', 'budget',
                    'video_name', 'manipulation', 'undefined_reason')} for r in metrics if r.get('undefined_reason')],
        'anatomical_high_score_regions': 'not_evaluated: no validated landmarks/region annotations; manual review required',
        'blur_interpretation': 'Laplacian variance is descriptive only, not a quality exclusion threshold',
        't0_b': {'executed': False, 'reason': SKIP_REASON},
        'crop_jitter_causal_attribution': 'not_evaluated; detector boxes do not guarantee temporal alignment',
        'official_test_read': False}


def _report(root: Path, aggregates: list, differences: list, diagnostics: dict) -> None:
    lines = ['# Exploratory T0-A multi-frame report', '',
        'Original Synthetic Gate B remains **FAIL**. T0 is not a retest and official test remains locked.',
        '', '## T0-A results', '', '| Method | Compression | Budget | Pixel AP | 95% CI |',
        '|---|---|---|---:|---|']
    for r in aggregates:
        if r['metric'] == 'pixel_ap' and r.get('status') == 'completed':
            lines.append(f"| {r['method']} | {r['compression']} | {r['budget']} | {r['mean']:.6f} | {r['ci95']} |")
    lines += ['', '## Real FPR (common frozen thresholds)', '',
              '| Method | Compression | Budget | Video macro FPR | Pixel pooled FPR | 95% CI |',
              '|---|---|---|---:|---:|---|']
    for r in aggregates:
        if r['metric'] == 'fpr' and r.get('status') == 'completed':
            lines.append(f"| {r['method']} | {r['compression']} | {r['budget']} | {r['mean']:.6f} | {r['pooled_pixel_fpr']:.6f} | {r['ci95']} |")
    lines += ['', '## Paired AP differences', '']
    for r in differences:
        if r['metric'] == 'pixel_ap':
            lines.append(f"- {r['method']} / {r['compression']} / {r['comparison']}: "
                         f"delta={r['absolute_delta']:.6f}, CI={r['absolute_ci95']}; "
                         f"per-manipulation delta={r['per_manipulation_delta']}")
    evidence = [r for r in differences if r['metric'] == 'pixel_ap' and r['compression'] == 'c23'
                and r['comparison'] in ('T5_uniform - T3_frozen', 'T9_uniform - T3_frozen')
                and r['absolute_ci95'][0] > 0]
    concordance = {}
    for compression in ('c23', 'c40'):
        for comparison in ('T5_uniform - T3_frozen', 'T9_uniform - T3_frozen'):
            pair = [r for r in differences if r['metric'] == 'pixel_ap' and
                    r['compression'] == compression and r['comparison'] == comparison]
            concordance[f'{compression}: {comparison}'] = (
                'same point-estimate direction' if len(pair) == 2 and
                np.sign(pair[0]['absolute_delta']) == np.sign(pair[1]['absolute_delta']) else
                'different direction or missing endpoint')
    lines += ['', '## Interpretation and limits',
        '- T0-A: ' + ('some comparisons have strictly positive paired AP intervals; check individual methods above.'
                     if evidence else 'no evaluated T5/T9 versus T3 comparison has a strictly positive paired AP interval.'),
        '- T0-B: skipped. ' + SKIP_REASON,
        '- No evidence beyond static sampling gains for temporal inconsistency is established by this experiment.',
        f'- C2/CIR_LID point-estimate directional agreement: {concordance}. This does not imply equal effect sizes or CI support.',
        f"- Valid video counts per manipulation: {diagnostics['valid_videos_per_manipulation']}; technical exclusions and undefined endpoints: diagnostics.json.",
        '- T1: not automatically approved. Review c40 deterioration, real FPR, manipulation dependence, '
        'blur and anatomical confounds first; anatomical region diagnostics are not yet available.',
        '- Stable observations may motivate a separately preregistered next experiment; they are not confirmatory evidence for the failed CIR hypothesis.',
        '- Evidence: frame_manifest.jsonl; threshold_summary.json; per_video_metrics.jsonl; '
        'aggregate_metrics.json; bootstrap_differences.json; diagnostics.json; split_audit.json.',
        '- Undefined endpoint videos are listed explicitly; each endpoint uses the intersection of defined videos across budgets.',
        '- No fake-mask tuning, temporal fusion, official-test access or Gate decision generation occurred.']
    atomic_write_text(root / 'summary.md', '\n'.join(lines) + '\n')
    atomic_write_text(root / 'README.md', '# T0-v1 artifacts\n\n'
        'Independent validation-only exploratory run. Read summary.md for interpretations and limits.\n'
        'protocol.json locks semantic configs; scores.provenance.json ties scores to manifest/checkpoint/gallery.\n'
        'score_maps/ stores per-frame 28x28 maps; workspace/ contains isolated crops/masks/features.\n'
        'thresholds are common across T budgets, calibrated only on T3 real c23.\n'
        'The strict complete-union technical cohort applies to both compressions. No official test unlock.\n')


def run_stage(path: str | Path, stage: str) -> dict[str, Any]:
    t0, base, root = _context(path)
    print(f'[t0] stage={stage} output={root}; official test locked', flush=True)
    if stage == 'prepare':
        return prepare(t0, base, root)
    if stage == 'gallery':
        return gallery_stage(base, root)
    if stage == 'crops':
        cfg, manifest, _ = _rows(base, root)
        result = _crop(cfg, manifest)
        write_json(root / 'crop_summary.json', result)
        return result
    if stage == 'features':
        cfg, _, _ = _rows(base, root)
        from .commands import extract_features
        return extract_features(cfg, subset='validation')
    if stage == 'scores':
        return score_stage(base, root)
    if stage == 'evaluate':
        return evaluate(t0, base, root)
    raise ValueError(f'Unknown stage: {stage}')
