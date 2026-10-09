"""M0-v1: isolated, real-only static feature geometry audit.

No fake/test enumeration, thresholds, scoring, or gate decisions occur here.
The T0 mixed validation manifest is NEVER opened, even when reusing a verified
real-frame feature artifact from that run.
"""
from __future__ import annotations

from collections import defaultdict
import copy
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np

from .config import Config, config_digest, load_config, require_protocol, resolve_config_path
from .features import feature_artifact_path, load_features
from .ffpp import FFPPPaths, enumerate_real_records, read_split_pairs
from .gallery import GalleryArtifact
from .io import atomic_write_text, read_json, sha256_file, write_json, write_jsonl
from .knn import ExactL2Index
from .manifest import read_manifest, write_manifest

METRICS = ('r_1', 'r_k', 'mean_r', 'median_r', 'distance_cv', 'log_radius_ratio',
           'unique_neighbor_videos_fraction', 'same_video_neighbor_fraction',
           'neighbor_video_count', 'neighbor_concentration', 'local_anisotropy',
           'participation_ratio', 'effective_dim_90', 'effective_dim_95', 'lid_hat',
           'feature_norm', 'zero_distance_count', 'neighbor_duplicate_count')
STABILITY = ('r_k', 'median_r', 'participation_ratio', 'effective_dim_90',
             'lid_hat', 'unique_neighbor_videos_fraction')
NULL_METRICS = METRICS  # Full preregistered metric set, not selected after inspection.
SPECTRUM_N = 10


def _context(path: str | Path) -> tuple[Config, Config, Path]:
    m0 = load_config(path)
    base = load_config(resolve_config_path(m0, m0.require('base_config')))
    root = resolve_config_path(m0, m0.require('output_dir'))
    if root != resolve_config_path(base, base.require('paths.results')) / 'm0_manifold_audit_v1':
        raise ValueError('M0 must have its own results/m0_manifold_audit_v1 directory')
    root.mkdir(parents=True, exist_ok=True)
    try:
        require_protocol(base)
        # ``require_protocol`` covers the shared v5.5 contract.  M0 also
        # freezes the backbone label and the real-only FF++ scope explicitly so
        # a relabeled or partially edited base YAML cannot silently enter the
        # audit.
        frozen_base = {
            'backbone.weights': 'LVD-1689M',
            'ffpp.manipulations': ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures'],
            'ffpp.compressions': ['c23', 'c40'],
        }
        for key, value in frozen_base.items():
            if base.get(key) != value:
                raise ValueError(f'Frozen feature/data setting mismatch: {key}: '
                                 f'{base.get(key)!r} != {value!r}')
    except Exception as exc:
        write_json(root / 'diagnostics.json', {'status': 'BLOCKED',
            'reason': 'Frozen feature/config mismatch: ' + str(exc),
            'feature_config_mismatch': [str(exc)], 'null_control_skipped': True,
            'official_test_locked': True, 'fake_data_read': False, 'test_data_read': False})
        atomic_write_text(root / 'summary.md', '# M0 BLOCKED\n\nFrozen configuration mismatch; '
            'no real/fake analysis conducted, no official test unlock.\n')
        raise
    expected = {
        'split': 'validation', 'query_frame_position': 0.50,
        'compressions': ['c23', 'c40'], 'spatial_grid': [28, 28],
        'num_tokens_per_video_for_audit': 128,
        'sampling_rule': 'linspace_0_783_round_deduplicate', 'k_primary': 20,
        'k_sensitivity': [10, 20, 40], 'eps': 1e-12,
        'gallery_resampling.seeds': [11, 23, 37, 41, 53],
        'gallery_resampling.fraction': 0.80,
        'null_control.seeds': [101, 103, 107, 109, 113],
        'bootstrap.replicates': 2000, 'bootstrap.confidence_level': 0.95,
    }
    for key, value in expected.items():
        if m0.get(key) != value:
            raise ValueError(f'M0-v1 frozen setting mismatch: {key}: {m0.get(key)!r} != {value!r}')
    if m0.get('query_batch_size', 0) <= 0:
        raise ValueError('query_batch_size must be positive')
    protocol = {'m0': m0.data, 'base': base.data,
                'm0_digest': config_digest(m0), 'base_digest': config_digest(base)}
    lock = root / 'protocol.json'
    if lock.exists() and read_json(lock) != protocol:
        raise RuntimeError('M0 protocol changed; preserve old result and create a new version')
    write_json(lock, protocol)
    import yaml
    atomic_write_text(root / 'config_m0_v1.yaml', yaml.safe_dump(m0.data, sort_keys=False))
    return m0, base, root


def _isolated(base: Config, root: Path) -> Config:
    data = copy.deepcopy(base.data)
    for name in ('manifests', 'crops', 'masks', 'features', 'indices', 'environment'):
        data['paths'][name] = str(root / 'workspace' / name)
    data['paths']['results'] = str(root)
    data['experiment']['m0_namespace'] = 'validation_real_only'
    return Config(data, base.source)


def _paths(base: Config) -> FFPPPaths:
    return FFPPPaths.from_root(resolve_config_path(base, base.require('paths.ffpp_root')),
        {key: resolve_config_path(base, value) for key, value in base.require('ffpp.split_files').items()})


def _gallery(base: Config) -> tuple[GalleryArtifact, Path, list[dict[str, Any]]]:
    # Reuse the frozen gallery, or T0's independently rebuilt exact gallery.
    # T0's mixed validation manifest is not consulted by the gallery verifier.
    from .t0 import _gallery as verified_t0_gallery
    t0root = resolve_config_path(base, base.require('paths.results')) / 't0_multiframe_v1'
    gallery, directory = verified_t0_gallery(base, t0root)
    meta = gallery.metadata
    manifest = Path(str(meta['manifest']))
    rows = read_manifest(manifest)
    ids = meta['gallery_video_ids']
    if len(rows) != 64 or [r['video_name'] for r in rows] != ids or any(
        r['split'] != 'train' or not r['is_real'] or r['manipulation'] != 'original'
        or r['compression'] != 'c23' or r['mask_path'] is not None for r in rows
    ):
        raise RuntimeError('BLOCKED: gallery source mapping or real-only provenance invalid')
    return gallery, directory, rows


def _indices(m0: Config) -> np.ndarray:
    count = int(m0.require('num_tokens_per_video_for_audit'))
    # Round-to-nearest as frozen; for these 128 positions there are no ties.
    return np.unique(np.rint(np.linspace(0, 783, count)).astype(np.int32))


def _manifest(base: Config, root: Path) -> tuple[Config, Path, list[dict[str, Any]]]:
    cfg = _isolated(base, root)
    path = resolve_config_path(cfg, cfg.require('paths.manifests')) / 'manifest.jsonl'
    audit = read_json(root / 'split_audit.json')
    paths = _paths(base)
    inputs = read_json(root / 'input_manifest.json')
    if (audit['split_hashes'] != {
        'train': sha256_file(paths.split_train), 'validation': sha256_file(paths.split_validation)
    } or audit.get('manifest_sha256') != sha256_file(path) or
        inputs.get('query_manifest_sha256') != sha256_file(path)):
        raise RuntimeError('BLOCKED: real-only manifest/split provenance changed')
    rows = read_manifest(path)
    if not rows or len({r['sample_id'] for r in rows}) != len(rows) or any(
        r['split'] != 'validation' or not r['is_real'] or r['manipulation'] != 'original'
        or r['compression'] not in ('c23', 'c40') or r['mask_path'] is not None
        or r['frame_fraction'] != 0.50 for r in rows
    ):
        raise RuntimeError('BLOCKED: M0 manifest is not unique, center-frame, real-only validation')
    return cfg, path, rows


def prepare(m0: Config, base: Config, root: Path) -> dict[str, Any]:
    cfg = _isolated(base, root)
    manifest = resolve_config_path(cfg, cfg.require('paths.manifests')) / 'manifest.jsonl'
    if manifest.exists():
        raise RuntimeError('M0 manifest already exists; do not replace frozen frame provenance')
    gallery, directory, grows = _gallery(base)
    paths = _paths(base)
    train = {v for pair in read_split_pairs(paths.split_train) for v in pair}
    validation = {v for pair in read_split_pairs(paths.split_validation) for v in pair}
    if train & validation or set(g['video_name'] for g in grows) & validation:
        raise RuntimeError('BLOCKED: gallery and validation identities overlap')
    records = enumerate_real_records(paths, split='validation', compressions=('c23', 'c40'),
                                     frame_fractions=(0.50,), seed=int(m0.require('seed')))
    rows = [record.to_dict() for record in records]
    if {r['video_name'] for r in rows if r['compression'] == 'c23'} != validation or {
        r['video_name'] for r in rows if r['compression'] == 'c40'} != validation:
        raise RuntimeError('BLOCKED: missing real validation video(s)')
    write_manifest(rows, manifest)
    write_json(root / 'split_audit.json', {'split_hashes': {
        'train': sha256_file(paths.split_train), 'validation': sha256_file(paths.split_validation)},
        'manifest_sha256': sha256_file(manifest), 'train_validation_overlap': [],
        'test_split_read': False, 'test_videos_read': False,
        'fake_manifest_read': False, 'fake_images_or_masks_read': False})
    write_json(root / 'input_manifest.json', {'gallery_manifest': str(gallery.metadata['manifest']),
        'gallery_metadata': str(directory / 'gallery.metadata.json'),
        'gallery_manifest_sha256': sha256_file(Path(str(gallery.metadata['manifest']))),
        'gallery_array_sha256': sha256_file(directory / 'gallery.npy'),
        'query_real_only_manifest': str(manifest), 'query_manifest_sha256': sha256_file(manifest),
        'gallery_videos': len(grows), 'validation_videos_per_compression': len(validation),
        'selection': 'gallery: frozen middle frame; validation: T1_center=0.50, half-up',
        'test_read': False, 'fake_read': False})
    write_jsonl(root / 'sampled_token_manifest.jsonl', [
        {'sample_id': r['sample_id'], 'video_id': r['video_name'], 'compression': r['compression'],
         'frame_index': r['frame_index'], 'frame_position': r['frame_fraction'],
         'token_indices': _indices(m0).tolist()} for r in rows
    ] + [
        {'sample_id': r['sample_id'], 'video_id': r['video_name'], 'compression': 'c23',
         'frame_index': r['frame_index'], 'frame_position': r['frame_fraction'],
         'token_indices': _indices(m0).tolist(), 'gallery': True} for r in grows
    ])
    return {'status': 'prepared', 'gallery_videos': len(grows), 'validation_real_videos': len(validation),
            'query_frames': len(rows), 'gallery_source': str(directory), 'test_and_fake_locked': True}


def _feature(path: Path, sample_id: str, expected: dict[str, Any], dimension: int) -> np.ndarray:
    """Load one feature artifact only after strict on-disk provenance checks.

    ``load_features`` canonicalizes arrays to float32 for normal pipeline use.
    That is useful downstream, but it would hide an on-disk dtype mismatch from
    this audit.  M0 therefore checks the raw ``.npy`` header first, then checks
    the sidecar and crop hash, and only then loads the canonical value.
    """
    path = Path(path)
    expected_shape = (1, 784, int(dimension))
    if not path.is_file():
        raise RuntimeError(f'BLOCKED: missing feature artifact at {path}')
    try:
        raw = np.load(path, allow_pickle=False, mmap_mode='r')
        raw_shape = tuple(raw.shape)
        raw_dtype = raw.dtype
        del raw
    except (OSError, ValueError) as exc:
        raise RuntimeError(f'BLOCKED: unreadable feature artifact at {path}: {exc}') from exc
    if raw_shape != expected_shape or raw_dtype != np.dtype(np.float32):
        raise RuntimeError(
            f'BLOCKED: raw feature array mismatch at {path}: '
            f'shape={raw_shape}, dtype={raw_dtype}; expected {expected_shape}, float32')

    value, meta = load_features(path)
    checks = {
        'sample_id': sample_id,
        'path': str(path),
        'array_sha256': sha256_file(path),
        'shape': list(expected_shape),
        'dtype': 'float32',
        **expected,
    }
    mismatches = [key for key, wanted in checks.items() if meta.get(key) != wanted]
    if mismatches or value.shape != expected_shape or value.dtype != np.dtype(np.float32):
        raise RuntimeError(f'BLOCKED: exact feature provenance mismatch {mismatches} at {path}')

    crop_value = meta.get('crop_path')
    crop_hash = meta.get('crop_sha256')
    if not crop_value or not crop_hash:
        raise RuntimeError(f'BLOCKED: feature crop provenance is incomplete at {path}')
    crop = Path(str(crop_value))
    if not crop.is_file() or crop_hash != sha256_file(crop):
        raise RuntimeError(f'BLOCKED: crop provenance mismatch at {path}')
    return value[0]


def _query_features(base: Config, cfg: Config, manifest: Path,
                    rows: list[dict[str, Any]]) -> tuple[dict[tuple[str, str], np.ndarray], list[dict[str, Any]]]:
    checkpoint = resolve_config_path(base, base.require('paths.dinov3_weights'))
    checkpoint_sha = sha256_file(checkpoint)
    t0root = resolve_config_path(base, base.require('paths.results')) / 't0_multiframe_v1'
    t0_protocol_path = t0root / 'protocol.json'
    t0_cfg: Config | None = None
    t0_summary: dict[str, Any] | None = None
    if t0_protocol_path.is_file():
        protocol = read_json(t0_protocol_path)
        from .t0 import _isolated as t0_isolated
        t0_cfg = t0_isolated(base, t0root)
        summary_path = t0root / 'workspace/features/validation/summary.json'
        if protocol.get('base_digest') == config_digest(base) and summary_path.is_file():
            candidate = read_json(summary_path)
            if (candidate.get('config_digest') == config_digest(t0_cfg) and
                    candidate.get('checkpoint_sha256') == checkpoint_sha and
                    candidate.get('subset') == 'validation'):
                t0_summary = candidate
            else:
                t0_cfg = None
        else:
            t0_cfg = None
    own_hash = sha256_file(manifest)
    queries: dict[tuple[str, str], np.ndarray] = {}
    sources: list[dict[str, Any]] = []
    dimension = int(base.require('features.expected_dimension'))
    for row in rows:
        sample = str(row['sample_id'])
        if row.get('exclusion_reason'):
            raise RuntimeError(f'BLOCKED: preexisting real-only technical exclusion: {sample}')
        selected: tuple[Path, dict[str, Any], str] | None = None
        if t0_cfg is not None and t0_summary is not None:
            t0_path = feature_artifact_path(t0root / 'workspace/features/validation', sample)
            if t0_path.is_file():
                selected = (t0_path, {'config_digest': config_digest(t0_cfg),
                    'manifest_sha256': t0_summary.get('manifest_sha256'),
                    'manifest_path': t0_summary.get('manifest'), 'subset': 'validation',
                    'checkpoint_sha256': checkpoint_sha, 'architecture': 'dinov3_vitb16',
                    'block_one_based': 6, 'api_n': [5], 'norm': True, 'reshape': False,
                    'device': 'cuda'}, 'T0_verified_center_cache')
        if selected is None:
            path = feature_artifact_path(root / 'workspace/features/validation', sample)
            if not path.is_file():
                raise RuntimeError('BLOCKED: no exact real-only center-frame feature for ' + sample +
                    '; explicitly run --stage crops and --stage features. No automatic re-extraction.')
            selected = (path, {'config_digest': config_digest(cfg),
                'manifest_sha256': own_hash, 'manifest_path': str(manifest),
                'subset': 'validation', 'checkpoint_sha256': checkpoint_sha,
                'architecture': 'dinov3_vitb16', 'block_one_based': 6,
                'api_n': [5], 'norm': True, 'reshape': False, 'device': 'cuda'}, 'M0_real_only_cache')
        path, expected, origin = selected
        # Even a present but mismatched T0 artifact is fatal, not a license to fallback.
        # For an M0-owned artifact, compare the sidecar crop path with the
        # frozen M0 manifest row.  A reused T0 artifact is checked against its
        # own sidecar/crop hash without opening T0's mixed manifest.
        if origin == 'M0_real_only_cache' and row.get('crop_path'):
            expected = {**expected, 'crop_path': str(row['crop_path'])}
        array = _feature(path, sample, expected, dimension)
        queries[(row['compression'], row['video_name'])] = array
        sources.append({'sample_id': sample, 'origin': origin, 'feature_path': str(path),
                        'feature_sha256': sha256_file(path), 'frame_index': row['frame_index']})
    return queries, sources


def _gallery_features(base: Config, gallery: GalleryArtifact, grows: list[dict[str, Any]]) -> None:
    meta = gallery.metadata
    checkpoint_sha = sha256_file(resolve_config_path(base, base.require('paths.dinov3_weights')))
    for n, row in enumerate(grows):
        path = Path(meta['feature_paths'][n])
        expected = {'config_digest': meta['config_digest'],
            'manifest_sha256': meta['manifest_sha256'], 'checkpoint_sha256': checkpoint_sha,
            'subset': 'gallery', 'architecture': 'dinov3_vitb16', 'block_one_based': 6,
            'api_n': [5], 'norm': True, 'reshape': False, 'device': 'cuda'}
        if row.get('crop_path'):
            expected['crop_path'] = str(row['crop_path'])
        value = _feature(path, row['sample_id'], expected, gallery.vectors.shape[1])
        if not np.array_equal(value, gallery.vectors[n * 784:(n + 1) * 784]):
            raise RuntimeError(f'BLOCKED: gallery vector/feature mismatch for {row["sample_id"]}')


def _geometry(dist: np.ndarray, ids: np.ndarray, neighbors: np.ndarray,
              query_video: str, k: int, eps: float) -> tuple[dict[str, float], np.ndarray, int]:
    d = np.asarray(dist[:k], dtype=np.float64)
    if not np.isfinite(d).all() or (d < 0).any():
        raise RuntimeError('BLOCKED: invalid Euclidean neighbor distances')
    source = ids[:k]
    unique, counts = np.unique(source, return_counts=True)
    local = np.asarray(neighbors[:k], dtype=np.float64)
    local -= local.mean(axis=0, keepdims=True)
    # k x k Gram matrix is equivalent to PCA of the centered k x d matrix.
    eig = np.linalg.eigvalsh(local @ local.T)
    unstable = int((eig < -1e-8 * max(1.0, float(np.max(np.abs(eig))))).any())
    eig = np.maximum(eig[::-1], 0)
    numerical_zero_count = int(np.count_nonzero(eig <= max(eps, float(eig[0]) * 1e-10)))
    total = float(eig.sum())
    spectrum = eig[:SPECTRUM_N] / total if total > eps else np.zeros(SPECTRUM_N)
    pr = total * total / float(np.dot(eig, eig)) if total > eps else 0.0
    cum = np.cumsum(eig) / total if total > eps else np.zeros_like(eig)
    safe = np.maximum(d, eps)
    logs = np.log(safe / max(float(safe[-1]), eps))
    lid = -1.0 / float(np.mean(logs)) if float(np.mean(logs)) < -eps else 0.0
    row = {'r_1': float(d[0]), 'r_k': float(d[-1]), 'mean_r': float(np.mean(d)),
        'median_r': float(np.median(d)), 'distance_cv': float(np.std(d) / max(float(np.mean(d)), eps)),
        'log_radius_ratio': float(np.log(max(float(d[-1]), eps) / max(float(d[0]), eps))),
        'unique_neighbor_videos_fraction': float(len(unique) / k),
        'same_video_neighbor_fraction': float(np.mean(source == query_video)),
        'neighbor_video_count': float(len(unique)),
        'neighbor_video_counts': {str(v): int(c) for v, c in zip(unique, counts)},
        'neighbor_concentration': float(np.max(counts) / k),
        'local_anisotropy': float(eig[0] / total) if total > eps else 0.0,
        'participation_ratio': float(pr),
        'effective_dim_90': float(np.searchsorted(cum, .9) + 1) if total > eps else 0.0,
        'effective_dim_95': float(np.searchsorted(cum, .95) + 1) if total > eps else 0.0,
        'lid_hat': float(lid), 'zero_distance_count': float(np.count_nonzero(d == 0)),
        'neighbor_duplicate_count': float(k - len(np.unique(neighbors[:k], axis=0))),
        'pca_degenerate': int(total <= eps), 'pca_zero_eigenvalues': numerical_zero_count,
        'lid_degenerate': int(float(np.mean(logs)) >= -eps),
        'pca_negative_eigenvalue': unstable,
        'neighbor_video_ids': unique.tolist()}
    return row, spectrum, unstable


def _search(queries: np.ndarray, query_videos: list[str], gallery: np.ndarray,
            source_ids: np.ndarray, *, ks: tuple[int, ...], eps: float,
            batch_size: int, leave_video_out: bool = False) -> tuple[list[dict[str, Any]], np.ndarray]:
    if len(queries) != len(query_videos) or len(gallery) != len(source_ids):
        raise ValueError('Query/gallery source-video ID mapping mismatch')
    maximum = max(ks)
    if len(gallery) < maximum + (784 if leave_video_out else 0):
        raise RuntimeError('BLOCKED: gallery has too few non-self neighbors')
    index = ExactL2Index.build(gallery)
    # With one frame of 784 tokens per gallery video, requesting k+784
    # guarantees at least k neighbors from other source videos, even if all
    # same-video tokens are ranked first. No self-neighbor leakage.
    result = index.search(queries, maximum + (784 if leave_video_out else 0), batch_size=batch_size)
    all_rows: list[dict[str, Any]] = []
    spectra: list[np.ndarray] = []
    for n, video in enumerate(query_videos):
        nn = result.indices[n]
        distances = result.distances[n]
        if leave_video_out:
            keep = source_ids[nn] != video
            nn, distances = nn[keep][:maximum], distances[keep][:maximum]
        if len(nn) < maximum or (leave_video_out and np.any(source_ids[nn] == video)):
            raise RuntimeError('BLOCKED: gallery same-video exclusion failed')
        for k in ks:
            values, spectrum, _ = _geometry(distances, source_ids[nn], gallery[nn], video, k, eps)
            values['k'] = k
            values['feature_norm'] = float(np.linalg.norm(queries[n]))
            values['neighbor_token_indices'] = nn[:k].tolist() if k == 20 else None
            all_rows.append(values)
            if k == 20:
                spectra.append(spectrum)
        if (n + 1) % 1000 == 0:
            print(f'[m0 geometry] {n+1}/{len(queries)} ({"leave-video-out" if leave_video_out else "validation"})', flush=True)
    return all_rows, np.asarray(spectra, dtype=np.float32)


def _video_aggregation(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row['condition'], row['video_id'], int(row['k']))].append(row)
    results = []
    for (condition, video, k), tokens in sorted(groups.items()):
        summary: dict[str, Any] = {'condition': condition, 'video_id': video, 'k': k,
                                   'token_count': len(tokens), 'statistics': {}}
        for metric in METRICS:
            values = np.asarray([t[metric] for t in tokens], dtype=np.float64)
            q10, median, q90 = np.quantile(values, [.1, .5, .9])
            summary['statistics'][metric] = {
                'mean': float(np.mean(values)), 'median': float(median),
                'p10': float(q10), 'p50': float(median), 'p90': float(q90),
                'iqr': float(np.subtract(*np.quantile(values, [.75, .25]))),
            }
        summary['zero_distance_queries'] = sum(t['zero_distance_count'] > 0 for t in tokens)
        summary['duplicate_neighbor_queries'] = sum(t['neighbor_duplicate_count'] > 0 for t in tokens)
        summary['degenerate_pca_queries'] = sum(t['pca_degenerate'] for t in tokens)
        summary['pca_zero_eigenvalues_total'] = sum(t['pca_zero_eigenvalues'] for t in tokens)
        summary['degenerate_lid_queries'] = sum(t['lid_degenerate'] for t in tokens)
        summary['negative_eigenvalue_queries'] = sum(t['pca_negative_eigenvalue'] for t in tokens)
        results.append(summary)
    return results


def _bootstrap(values: np.ndarray, rng: np.random.Generator, replicates: int) -> dict[str, Any]:
    if not np.isfinite(values).all() or len(values) < 2:
        return {'estimate': None, 'ci95': None, 'videos': len(values)}
    estimate = float(np.mean(values))
    # Batch bootstrap indices to bound memory for any future larger cohort.
    samples = []
    for _ in range(0, replicates, 256):
        n = min(256, replicates - len(samples))
        idx = rng.integers(0, len(values), size=(n, len(values)))
        samples.extend(np.mean(values[idx], axis=1).tolist())
    return {'estimate': estimate, 'ci95': np.quantile(samples, [.025, .975]).tolist(),
            'videos': len(values), 'unit': 'video', 'estimator': 'mean of video medians'}


def _spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) < 3:
        return None
    from scipy.stats import spearmanr
    value = float(spearmanr(left, right).statistic)
    return value if np.isfinite(value) else None


def _source_arrays(m0: Config, base: Config, root: Path) -> tuple[GalleryArtifact, Path,
        list[dict[str, Any]], np.ndarray, dict[tuple[str, str], np.ndarray], list[dict[str, Any]]]:
    gallery, directory, grows = _gallery(base)
    inputs = read_json(root / 'input_manifest.json')
    if (inputs.get('gallery_manifest_sha256') != sha256_file(Path(str(gallery.metadata['manifest'])))
            or inputs.get('gallery_array_sha256') != sha256_file(directory / 'gallery.npy')):
        raise RuntimeError('BLOCKED: frozen gallery provenance differs from M0 input manifest')
    _gallery_features(base, gallery, grows)
    cfg, manifest, rows = _manifest(base, root)
    query, sources = _query_features(base, cfg, manifest, rows)
    expected = int(base.require('features.expected_dimension'))
    if gallery.vectors.shape != (64 * 784, expected) or set(query) != {
        (row['compression'], row['video_name']) for row in rows}:
        raise RuntimeError('BLOCKED: expected exact real-only gallery/query coverage or dimension')
    ids = np.repeat([g['video_name'] for g in grows], 784)
    if set(ids) & {r['video_name'] for r in rows}:
        raise RuntimeError('BLOCKED: gallery and query share source video')
    snapshot = {
        'status': 'verified', 'base_config_digest': config_digest(base),
        'backbone': base.require('backbone.architecture'),
        'weights': base.require('backbone.weights'), 'block_one_based': base.require('backbone.block_one_based'),
        'api_n': base.require('backbone.api_n'), 'norm': base.require('backbone.norm'),
        'reshape': base.require('backbone.reshape'), 'extra_l2_normalization': False,
        'grid': base.require('features.expected_patch_grid'), 'actual_dimension': int(gallery.vectors.shape[1]),
        'input_preprocessing': base.require('input'), 'face_crop': base.require('face_crop'),
        'distance': 'Euclidean L2, square-root of exact IndexFlatL2 squared distance',
        'gallery': str(directory), 'gallery_sha256': sha256_file(directory / 'gallery.npy'),
        'checkpoint_sha256': sha256_file(resolve_config_path(base, base.require('paths.dinov3_weights'))),
        'query_feature_sources': sources, 'mismatch': []}
    feature_audit = root / 'feature_config_audit.json'
    if feature_audit.exists() and read_json(feature_audit) != snapshot:
        raise RuntimeError('BLOCKED: feature/gallery provenance changed after the first M0 audit')
    write_json(feature_audit, snapshot)
    return gallery, directory, grows, ids, query, rows


def _baseline(m0: Config, base: Config, root: Path) -> dict[str, Any]:
    gallery, directory, grows, ids, query, records = _source_arrays(m0, base, root)
    tokens = _indices(m0)
    ks = tuple(int(v) for v in m0.require('k_sensitivity'))
    eps = float(m0.require('eps'))
    batch = int(m0.require('query_batch_size'))
    output: list[dict[str, Any]] = []
    spectra: dict[str, np.ndarray] = {}
    query_by_condition: dict[str, list[tuple[str, np.ndarray]]] = {
        'gallery_c23_leave_video_out': [
            (r['video_name'], gallery.vectors[n * 784 + tokens]) for n, r in enumerate(grows)],
        'validation_c23': [(r['video_name'], query[('c23', r['video_name'])])
            for r in records if r['compression'] == 'c23'],
        'validation_c40': [(r['video_name'], query[('c40', r['video_name'])])
            for r in records if r['compression'] == 'c40']}
    for condition, samples in query_by_condition.items():
        sample_records = grows if condition.startswith('gallery') else [
            r for r in records if r['compression'] == condition.rsplit('_', 1)[-1]]
        qvectors = np.concatenate([vec if condition.startswith('gallery') else vec[tokens]
                                    for _, vec in samples])
        qvideos = list(itertools.chain.from_iterable([[name] * len(tokens) for name, _ in samples]))
        rows, spectrum = _search(qvectors, qvideos, gallery.vectors, ids,
            ks=ks, eps=eps, batch_size=batch, leave_video_out=condition.startswith('gallery'))
        for j, row in enumerate(rows):
            q = j // len(ks)
            row.update(condition=condition, video_id=qvideos[q], token_index=int(tokens[q % len(tokens)]),
                       sample_id=sample_records[q // len(tokens)]['sample_id'])
        output.extend(rows)
        spectra[condition] = spectrum
    write_jsonl(root / 'per_query_geometry.jsonl', output)
    np.savez_compressed(root / 'local_pca_spectra.npz', **spectra, k_primary=20,
                        token_indices=tokens)
    video = _video_aggregation(output)
    write_jsonl(root / 'per_video_geometry.jsonl', video)
    write_json(root / 'baseline.provenance.json', {
        'per_query_sha256': sha256_file(root / 'per_query_geometry.jsonl'),
        'per_video_sha256': sha256_file(root / 'per_video_geometry.jsonl'),
        'feature_config_audit_sha256': sha256_file(root / 'feature_config_audit.json')})
    print(f'[m0] baseline completed: {len(output)} token/k rows, {len(video)} video/k rows', flush=True)
    return {'status': 'completed', 'per_query': len(output), 'per_video': len(video),
            'gallery_source': str(directory)}


def _load_baseline(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from .io import read_jsonl
    provenance = read_json(root / 'baseline.provenance.json')
    for name, key in (('per_query_geometry.jsonl', 'per_query_sha256'),
                      ('per_video_geometry.jsonl', 'per_video_sha256'),
                      ('feature_config_audit.json', 'feature_config_audit_sha256')):
        if provenance.get(key) != sha256_file(root / name):
            raise RuntimeError('BLOCKED: baseline input or output changed since generation: ' + name)
    queries = read_jsonl(root / 'per_query_geometry.jsonl')
    videos = read_jsonl(root / 'per_video_geometry.jsonl')
    if not queries or not videos:
        raise RuntimeError('BLOCKED: complete baseline is required first')
    return queries, videos


def _stability(m0: Config, base: Config, root: Path) -> dict[str, Any]:
    baseline, _ = _load_baseline(root)
    gallery, _, grows, _, queries, records = _source_arrays(m0, base, root)
    tokens = _indices(m0)
    qrecords = [r for r in records if r['compression'] == 'c23']
    qvectors = np.concatenate([queries[('c23', r['video_name'])][tokens] for r in qrecords])
    videos = list(itertools.chain.from_iterable([[r['video_name']] * len(tokens) for r in qrecords]))
    keys = [(name, int(t)) for name, t in zip(videos, np.tile(tokens, len(qrecords)))]
    reference = {(r['video_id'], r['token_index']): r for r in baseline
                 if r['condition'] == 'validation_c23' and r['k'] == 20}
    if set(keys) != set(reference):
        raise RuntimeError('BLOCKED: baseline and resampling cohort disagree')
    runs: dict[int, dict[tuple[str, int], dict[str, Any]]] = {}
    subset_size = int(round(0.8 * len(grows)))
    source = np.repeat([g['video_name'] for g in grows], 784)
    for seed in m0.require('gallery_resampling.seeds'):
        chosen = np.sort(np.random.default_rng(seed).choice(len(grows), subset_size, replace=False))
        indices = np.concatenate([np.arange(i * 784, (i + 1) * 784) for i in chosen])
        rows, _ = _search(qvectors, videos, gallery.vectors[indices], source[indices],
            ks=(20,), eps=float(m0.require('eps')), batch_size=int(m0.require('query_batch_size')))
        runs[int(seed)] = {key: {**row, 'neighbor_token_indices': indices[row['neighbor_token_indices']].tolist()}
                           for key, row in zip(keys, rows)}
        print(f'[m0 stability] seed={seed} selected_gallery_videos={subset_size}', flush=True)
    result = []
    for key in keys:
        source_row = reference[key]
        values: dict[str, Any] = {'video_id': key[0], 'token_index': key[1],
            'resample_seeds': m0.require('gallery_resampling.seeds'), 'metrics': {}}
        for metric in STABILITY:
            per_seed = [runs[seed][key][metric] for seed in runs]
            base_value = source_row[metric]
            values['metrics'][metric] = {'baseline': base_value, 'per_seed': per_seed,
                'relative_change': [(v - base_value) / max(abs(base_value), float(m0.require('eps')))
                                    for v in per_seed],
                'resample_mad': float(np.median(np.abs(np.asarray(per_seed) - np.median(per_seed))))}
        # Neighbors use original full-gallery row IDs for both baseline/resamples.
        original = set(source_row['neighbor_token_indices'])
        values['neighbor_token_jaccard'] = [len(original & set(runs[seed][key]['neighbor_token_indices'])) /
             len(original | set(runs[seed][key]['neighbor_token_indices'])) for seed in runs]
        original_videos = set(source_row['neighbor_video_ids'])
        values['neighbor_source_video_jaccard'] = [
            len(original_videos & set(runs[seed][key]['neighbor_video_ids'])) /
            len(original_videos | set(runs[seed][key]['neighbor_video_ids'])) for seed in runs]
        result.append(values)
    write_jsonl(root / 'neighbor_stability.jsonl', result)
    by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in result:
        by_video[item['video_id']].append(item)
    correlations: dict[str, Any] = {}
    video_mad: dict[str, Any] = {}
    for metric in STABILITY:
        correlations[metric] = {}
        video_mad[metric] = {}
        for left, right in itertools.combinations(runs, 2):
            correlations[metric][f'{left}-{right}'] = {
                'token_spearman': _spearman(np.asarray([runs[left][key][metric] for key in keys]),
                                           np.asarray([runs[right][key][metric] for key in keys])),
                'video_median_spearman': _spearman(np.asarray([np.median([runs[left][key][metric]
                                                for key in vkeys]) for vkeys in (
                                                [(name, int(i)) for i in tokens] for name in by_video)]),
                    np.asarray([np.median([runs[right][key][metric] for key in vkeys]) for vkeys in (
                                                [(name, int(i)) for i in tokens] for name in by_video)]))}
        for video, items in by_video.items():
            values = [np.median([next(item for item in items if item['token_index'] == int(idx))
                      ['metrics'][metric]['per_seed'][n] for idx in tokens]) for n in range(len(runs))]
            video_mad[metric][video] = float(np.median(np.abs(np.asarray(values) - np.median(values))))
    rng_ci = np.random.default_rng(int(m0.require('seed')) + 55)
    changes = {}
    for metric in STABILITY:
        changes[metric] = {}
        for seed in runs:
            video_deltas = np.asarray([np.median([runs[seed][(video, int(t))][metric] for t in tokens]) -
                np.median([reference[(video, int(t))][metric] for t in tokens]) for video in by_video])
            changes[metric][str(seed)] = _bootstrap(video_deltas, rng_ci,
                int(m0.require('bootstrap.replicates')))
    payload = {'seeds': list(runs), 'gallery_video_fraction': .8,
               'gallery_video_count': subset_size, 'pairwise_spearman': correlations,
               'video_median_resample_mad': video_mad,
               'per_seed_validation_video_bootstrap_delta': changes,
               'neighbor_token_jaccard_mean': float(np.mean([np.mean(r['neighbor_token_jaccard']) for r in result])),
               'neighbor_source_video_jaccard_mean': float(np.mean([
                   np.mean(r['neighbor_source_video_jaccard']) for r in result])),
               'status': 'completed'}
    write_json(root / 'resampling_summary.json', payload)
    return {'status': 'completed', 'query_tokens': len(result), 'seeds': list(runs)}


def _compression(m0: Config, root: Path) -> dict[str, Any]:
    _, videos = _load_baseline(root)
    groups = {condition: {row['video_id']: row for row in videos
                          if row['condition'] == condition and row['k'] == 20}
              for condition in ('validation_c23', 'validation_c40')}
    _, _, records = _manifest(load_config(resolve_config_path(m0, m0.require('base_config'))), root)
    by_compression = {c: {r['video_name']: r for r in records if r['compression'] == c}
                      for c in ('c23', 'c40')}
    common = sorted(set(groups['validation_c23']) & set(groups['validation_c40']))
    paired = (len(common) == len(groups['validation_c23']) == len(groups['validation_c40']) and
              all(by_compression['c23'][v]['frame_index'] == by_compression['c40'][v]['frame_index'] and
                  by_compression['c23'][v]['frame_count'] == by_compression['c40'][v]['frame_count'] and
                  by_compression['c23'][v]['crop_source_frame'] ==
                  by_compression['c40'][v]['crop_source_frame'] for v in common))
    result: dict[str, Any] = {'pairing': 'source_video_and_exact_center_frame' if paired else 'unavailable',
        'reason_if_unpaired': None if paired else 'source videos or exact frame alignment differ; no paired inference',
        'matched_video_count': len(common), 'metrics': {}}
    rng = np.random.default_rng(int(m0.require('seed')) + 41)
    for metric in METRICS:
        a = np.asarray([groups['validation_c23'][v]['statistics'][metric]['median'] for v in common])
        b = np.asarray([groups['validation_c40'][v]['statistics'][metric]['median'] for v in common])
        result['metrics'][metric] = {
            'c23_video_medians': {v: float(r['statistics'][metric]['median'])
                for v, r in groups['validation_c23'].items()},
            'c40_video_medians': {v: float(r['statistics'][metric]['median'])
                for v, r in groups['validation_c40'].items()},
            'spearman_over_matched_sources': _spearman(a, b) if paired else None,
        }
        if paired:
            delta = b - a
            result['metrics'][metric].update({
                'per_video_absolute_delta_c40_minus_c23': dict(zip(common, map(float, delta))),
                'per_video_relative_change': dict(zip(common, map(float, delta /
                    np.maximum(np.abs(a), float(m0.require('eps')))))),
                'paired_video_bootstrap': _bootstrap(delta, rng, int(m0.require('bootstrap.replicates'))),
            })
        else:
            for condition in groups:
                values = np.asarray([r['statistics'][metric]['median'] for r in groups[condition].values()])
                result['metrics'][metric][condition + '_unpaired_bootstrap'] = _bootstrap(
                    values, rng, int(m0.require('bootstrap.replicates')))
    write_json(root / 'compression_stability.json', result)
    return {'status': 'completed', 'pairing': result['pairing'], 'matched_videos': len(common)}


def _null(m0: Config, base: Config, root: Path) -> dict[str, Any]:
    _, real_videos = _load_baseline(root)
    gallery, _, grows, source, _, _ = _source_arrays(m0, base, root)
    tokens = _indices(m0)
    norms = np.linalg.norm(gallery.vectors.astype(np.float64), axis=1)
    real = {row['video_id']: row for row in real_videos
            if row['condition'] == 'gallery_c23_leave_video_out' and row['k'] == 20}
    if set(real) != {r['video_name'] for r in grows}:
        raise RuntimeError('BLOCKED: null and real gallery video cohorts disagree')
    results: dict[str, Any] = {'construction': 'isotropic Gaussian unit directions multiplied by each original token norm; '
        'same token count and 784-token source-video grouping; independent null query from same null gallery, '
        'leave-video-out', 'feature_dimension': gallery.vectors.shape[1], 'seeds': {},
        'real_minus_null_pooled_video': {}}
    per_query: list[dict[str, Any]] = []
    per_video: list[dict[str, Any]] = []
    spectra: dict[str, np.ndarray] = {}
    rng_ci = np.random.default_rng(int(m0.require('seed')) + 72)
    for seed in m0.require('null_control.seeds'):
        rng = np.random.default_rng(int(seed))
        directions = rng.standard_normal(gallery.vectors.shape, dtype=np.float32)
        direction_norm = np.linalg.norm(directions.astype(np.float64), axis=1)
        if np.any(direction_norm == 0):
            raise RuntimeError('BLOCKED: zero Gaussian direction in radial null')
        null_gallery = np.ascontiguousarray(directions * (norms / direction_norm)[:, None], dtype=np.float32)
        null_queries = np.concatenate([null_gallery[n * 784 + tokens] for n in range(len(grows))])
        qids = list(itertools.chain.from_iterable([[r['video_name']] * len(tokens) for r in grows]))
        rows, spectrum = _search(null_queries, qids, null_gallery, source,
            ks=(20,), eps=float(m0.require('eps')), batch_size=int(m0.require('query_batch_size')),
            leave_video_out=True)
        for index, row in enumerate(rows):
            row.update(condition=f'matched_radial_null_{seed}', video_id=qids[index],
                       token_index=int(tokens[index % len(tokens)]),
                       sample_id=grows[index // len(tokens)]['sample_id'], null_seed=seed)
        per_query.extend(rows)
        spectra[str(seed)] = spectrum
        summaries = _video_aggregation(rows)
        per_video.extend(summaries)
        replicate = {}
        null_by_video = {row['video_id']: row for row in summaries}
        for metric in NULL_METRICS:
            delta = np.asarray([real[v]['statistics'][metric]['median'] -
                null_by_video[v]['statistics'][metric]['median'] for v in sorted(real)])
            replicate[metric] = {'per_video_delta': dict(zip(sorted(real), map(float, delta))),
                'video_bootstrap': _bootstrap(delta, rng_ci, int(m0.require('bootstrap.replicates')))}
        results['seeds'][str(seed)] = replicate
        print(f'[m0 null] seed={seed} complete', flush=True)
        del null_gallery, null_queries, directions
    # Pooled null per-video mean across replicates, then paired video-cluster
    # bootstrap. Null replicate variation is separately retained above.
    for metric in NULL_METRICS:
        per_video_delta = np.asarray([np.mean([results['seeds'][str(seed)][metric]
            ['per_video_delta'][v] for seed in m0.require('null_control.seeds')]) for v in sorted(real)])
        replicate_estimates = [results['seeds'][str(seed)][metric]['video_bootstrap']['estimate']
                               for seed in m0.require('null_control.seeds')]
        results['real_minus_null_pooled_video'][metric] = {
            'per_video_delta': dict(zip(sorted(real), map(float, per_video_delta))),
            'paired_video_bootstrap': _bootstrap(per_video_delta, rng_ci,
                                                int(m0.require('bootstrap.replicates'))),
            'replicate_estimates': replicate_estimates,
            'replicate_std': float(np.std(replicate_estimates, ddof=1))}
    write_jsonl(root / 'null_per_query_geometry.jsonl', per_query)
    write_jsonl(root / 'null_per_video_geometry.jsonl', per_video)
    np.savez_compressed(root / 'null_local_pca_spectra.npz', **spectra,
                        k_primary=20, token_indices=tokens)
    write_json(root / 'null_control_summary.json', results)
    return {'status': 'completed', 'replicates': len(results['seeds']), 'null_control_skipped': False}


def _diagnostics(root: Path, queries: list[dict[str, Any]], videos: list[dict[str, Any]],
                 compression: dict[str, Any], resampling: dict[str, Any],
                 null: dict[str, Any]) -> dict[str, Any]:
    main = [r for r in queries if r['k'] == 20]
    conditions = sorted({r['condition'] for r in main})
    out: dict[str, Any] = {
        'status': 'completed', 'feature_config_mismatch': [], 'null_control_skipped': False,
        'official_test_locked': True, 'test_split_or_videos_read': False,
        'fake_manifest_images_masks_scores_read': False, 'per_condition': {},
        'compression_pairing': compression['pairing'],
        'resampling_seeds': resampling['seeds'], 'null_seeds': sorted(map(int, null['seeds']))}
    for condition in conditions:
        q = [r for r in main if r['condition'] == condition]
        v = [r for r in videos if r['condition'] == condition and r['k'] == 20]
        norm = np.asarray([r['feature_norm'] for r in q])
        source_hits: dict[str, int] = defaultdict(int)
        for row in q:
            for source_video, count in row['neighbor_video_counts'].items():
                source_hits[source_video] += count
        out['per_condition'][condition] = {
            'source_video_neighbor_hits': dict(sorted(source_hits.items())),
            'top_source_video_fraction_of_all_neighbor_slots': (
                float(max(source_hits.values()) / sum(source_hits.values())) if source_hits else None),
            'video_count': len(v), 'token_count': len(q),
            'zero_distance_queries': sum(r['zero_distance_count'] > 0 for r in q),
            'zero_distances_in_k20': int(sum(r['zero_distance_count'] for r in q)),
            'duplicate_neighbor_queries': sum(r['neighbor_duplicate_count'] > 0 for r in q),
            'duplicate_neighbor_tokens_counted_across_queries': int(sum(
                r['neighbor_duplicate_count'] for r in q)),
            'degenerate_pca_queries': sum(r['pca_degenerate'] for r in q),
            'pca_zero_eigenvalues_total': int(sum(r['pca_zero_eigenvalues'] for r in q)),
            'degenerate_lid_queries': sum(r['lid_degenerate'] for r in q),
            'negative_eigenvalue_queries': sum(r['pca_negative_eigenvalue'] for r in q),
            'same_video_neighbor_fraction_mean': float(np.mean([r['same_video_neighbor_fraction'] for r in q])),
            'feature_norm_median': float(np.median(norm)),
            'feature_norm_vs_metric_token_spearman_descriptive_only': {
                metric: _spearman(norm, np.asarray([r[metric] for r in q])) for metric in STABILITY},
            'feature_norm_vs_metric_video_median_spearman_descriptive_only': {
                metric: _spearman(
                    np.asarray([row['statistics']['feature_norm']['median'] for row in v]),
                    np.asarray([row['statistics'][metric]['median'] for row in v]))
                for metric in STABILITY},
            'video_median_metric_min_max': {metric: [float(min(row['statistics'][metric]['median'] for row in v)),
                    float(max(row['statistics'][metric]['median'] for row in v))] for metric in STABILITY},
            'neighbor_concentration_p90': float(np.quantile(
                    [r['neighbor_concentration'] for r in q], .9)),
            'note': 'Correlations cannot establish absence of norm or crop/blur confounds.'}
    write_json(root / 'diagnostics.json', out)
    return out


def _finalize(m0: Config, base: Config, root: Path) -> dict[str, Any]:
    queries, videos = _load_baseline(root)
    compression = read_json(root / 'compression_stability.json')
    resampling = read_json(root / 'resampling_summary.json')
    null = read_json(root / 'null_control_summary.json')
    _source_arrays(m0, base, root)  # Revalidate features before final inference.
    diagnostics = _diagnostics(root, queries, videos, compression, resampling, null)
    rng = np.random.default_rng(int(m0.require('seed')) + 90)
    summaries: dict[str, Any] = {'bootstrap_replicates': 2000, 'confidence_level': .95,
        'units': 'video; tokens are descriptive only', 'real_by_condition_and_k': {},
        'null_real_minus_null': null['real_minus_null_pooled_video'],
        'resampling_query_and_video_ranks': resampling['pairwise_spearman'],
        'resampling_video_mad': resampling['video_median_resample_mad'],
        'compression_video_bootstrap': {metric: v.get('paired_video_bootstrap') for metric, v
                                         in compression['metrics'].items()},
        'gallery_resampling_validation_video_bootstrap': resampling['per_seed_validation_video_bootstrap_delta'],
        'separate_bootstrap_streams': 'real validation, gallery-null, compression paired and '
                                      'gallery-resample are never pooled as independent tokens'}
    for condition in sorted({r['condition'] for r in videos}):
        summaries['real_by_condition_and_k'][condition] = {}
        for k in (10, 20, 40):
            rows = [r for r in videos if r['condition'] == condition and r['k'] == k]
            summaries['real_by_condition_and_k'][condition][str(k)] = {
                metric: _bootstrap(np.asarray([r['statistics'][metric]['median'] for r in rows]),
                    rng, 2000) for metric in METRICS}
    write_json(root / 'bootstrap_summary.json', summaries)
    # A is intentionally never automatically granted: norm effects, blur and
    # source-video dominance require qualitative confound review.
    null_ci = {metric: null['real_minus_null_pooled_video'][metric]
               ['paired_video_bootstrap']['ci95'] for metric in
               ('participation_ratio', 'effective_dim_90', 'local_anisotropy')}
    has_null_direction = (null_ci['local_anisotropy'] is not None and
        null_ci['local_anisotropy'][0] > 0 and
        ((null_ci['participation_ratio'] is not None and
          null_ci['participation_ratio'][1] < 0) or
         (null_ci['effective_dim_90'] is not None and
          null_ci['effective_dim_90'][1] < 0)))
    category = 'B (provisional; stability/confounds require review)' if has_null_direction else \
               'C (provisional; insufficient jointly directional structure evidence)'
    g = summaries['real_by_condition_and_k']
    lines = ['# M0-v1 static real-only manifold audit', '',
        '**This is a structural audit, not a localization-performance experiment.**',
        f'Provisional category: **{category}**. No automatic A classification or M1 unlock.',
        'No fake data, fake masks, fake manifest, fake scores, or official test were read. '
        'Official test remains locked. This does not change the failed Synthetic Gate B.', '',
        '## Frozen features and cohorts',
        f"- Backbone: {base.require('backbone.architecture')}, {base.require('backbone.weights')}; "
        f"block {base.require('backbone.block_one_based')} (api_n={base.require('backbone.api_n')}); "
        f"28×28 grid; actual dimension {read_json(root / 'feature_config_audit.json')['actual_dimension']}; "
        'float32 Euclidean L2 (square root of FAISS squared L2), no extra L2 normalization.',
        f"- Preprocessing: {json.dumps(base.require('input'), ensure_ascii=False)}; "
        f"face crop: {json.dumps(base.require('face_crop'), ensure_ascii=False)}.",
        f"- Frozen middle-frame train-real c23 gallery: {len([r for r in videos if r['condition'] == 'gallery_c23_leave_video_out' and r['k'] == 20])} videos. "
        f"Video-disjoint validation real: {len([r for r in videos if r['condition'] == 'validation_c23' and r['k'] == 20])} c23 "
        'and the same c40 sources; 128 fixed spatial token indices/video.',
        '- Gallery-internal neighbors exclude all 784 tokens of the source video. '
        'Validation videos do not enter gallery.', '',
        '## k=20 video-level estimates (mean of per-video token medians; 95% video bootstrap CI)',
        '| Condition | Metric | Estimate | 95% CI |', '|---|---|---:|---|']
    for condition in g:
        for metric in ('local_anisotropy', 'participation_ratio', 'effective_dim_90',
                       'effective_dim_95', 'lid_hat', 'r_k', 'median_r',
                       'unique_neighbor_videos_fraction', 'neighbor_concentration'):
            entry = g[condition]['20'][metric]
            lines.append(f"| {condition} | {metric} | {entry['estimate']:.5f} | {entry['ci95']} |")
    lines += ['', '## Matched-radial null (k=20; real minus mean null, paired gallery-video bootstrap)',
        '| Metric | Video-level delta | 95% CI | Null replicate SD |', '|---|---:|---|---:|']
    for metric in ('local_anisotropy', 'participation_ratio', 'effective_dim_90',
                   'effective_dim_95', 'r_k', 'median_r', 'neighbor_concentration'):
        entry = null['real_minus_null_pooled_video'][metric]
        lines.append(f"| {metric} | {entry['paired_video_bootstrap']['estimate']:.5f} | "
                     f"{entry['paired_video_bootstrap']['ci95']} | {entry['replicate_std']:.5f} |")
    lines += ['', '## Compression video-level median rank correlation (only if exact frame paired)',
        '| Metric | Spearman (c23/c40) | 95% paired video CI of c40-c23 |', '|---|---:|---|']
    for metric in ('local_anisotropy', 'participation_ratio', 'effective_dim_90',
                   'effective_dim_95', 'lid_hat', 'r_k', 'median_r'):
        entry = compression['metrics'][metric]
        ci = entry.get('paired_video_bootstrap', {}).get('ci95')
        lines.append(f"| {metric} | {entry['spearman_over_matched_sources']} | {ci} |")
    lines += ['', '## Null, stability and interpretation',
        '- Matched-radial null: five independent seeds; each original gallery token norm, '
        'source-video grouping and token count retained; random isotropic direction. '
        'Full per-video paired real-minus-null estimates/95% CIs and replicate variation: null_control_summary.json.',
        f"- Gallery resampling: five fixed seeds, 80% source videos without replacement; "
        f"mean neighbor token-set Jaccard={resampling['neighbor_token_jaccard_mean']:.4f}, "
        f"source-video-set Jaccard={resampling['neighbor_source_video_jaccard_mean']:.4f}. "
        'Metric ranks and per-video MAD: resampling_summary.json; query-level changes: neighbor_stability.jsonl.',
        f"- Compression: {compression['pairing']}. "
        'Per-video video-level median deltas, Spearman ranks and bootstrap: compression_stability.json.',
        '- Duplicate token/zero distances, feature norms, source-video concentration and negative PCA '
        'eigenvalue flags are reported in diagnostics.json. Norm-rank checks are descriptive, not a '
        'causal ruling-out of norm, crop jitter, blur or anatomical-region confounds.',
        '- Observed: the tables and stored per-query/per-video distributions. '
        'Speculative: treating observed PCA effective dimensions as mathematical manifold dimension, '
        'or inferring fake deviation/localization from any of these real-only results.',
        '- M1 real-vs-fake is **not automatically advised**: first review all null CIs, '
        'resampling ranks, c23/c40 ordering, dominance, duplicates, norm and provenance together. '
        'The provisional category is not a pass/fail gate.',
        '- Dynamic/temporal experiments are **not advised by M0**; T0-B lacked token correspondence. '
        'No change to Gate B or T0 and no official-test unlock.',
        '- Sensitivity k=10/40 and all listed video and token distributions are saved; '
        'no after-the-fact metric or k selection. Null is an isotropic radial baseline, not proof of a manifold.',
        '']
    atomic_write_text(root / 'summary.md', '\n'.join(lines))
    atomic_write_text(root / 'README.md', '# M0-v1 outputs\n\nSeparate real-only, validation-only static audit. '
        'Stages: prepare, audit, crops, features (only if audit reports missing exact cache), '
        'baseline, resampling, compression, null, finalize. Use summary.md and diagnostics.json. '
        'No official test, fake data, Gate B decision or localization metrics. '
        'Keep protocol.json, config_m0_v1.yaml and all per-query/per-video artifacts together.\n')
    return {'status': 'completed', 'provisional_category': category, 'official_test_locked': True,
            'm1_unlocked': False}


def run_stage(path: str | Path, stage: str) -> dict[str, Any]:
    m0, base, root = _context(path)
    print(f'[m0] stage={stage} output={root}; real-only, official test locked', flush=True)
    try:
        if stage == 'prepare':
            return prepare(m0, base, root)
        if stage == 'audit':
            cfg, manifest, rows = _manifest(base, root)
            gallery, directory, grows = _gallery(base)
            _gallery_features(base, gallery, grows)
            query, sources = _query_features(base, cfg, manifest, rows)
            return {'status': 'reused', 'real_query_frames': len(query),
                    'sources': {kind: sum(r['origin'] == kind for r in sources)
                                for kind in sorted({r['origin'] for r in sources})},
                    'gallery_videos': len(grows), 'gallery_source': str(directory)}
        if stage == 'crops':
            cfg, manifest, _ = _manifest(base, root)
            from .crops import cache_manifest_crops
            from dataclasses import asdict
            summary = asdict(cache_manifest_crops(manifest,
                crops_dir=resolve_config_path(cfg, cfg.require('paths.crops')),
                masks_dir=resolve_config_path(cfg, cfg.require('paths.masks')),
                input_size=int(cfg.require('input.size')),
                detector_upsample_primary=int(cfg.require('face_crop.upsample_primary')),
                detector_upsample_retry=int(cfg.require('face_crop.upsample_retry')),
                square_scale=float(cfg.require('face_crop.square_scale')),
                padding=str(cfg.require('face_crop.padding')),
                exclude_empty_fake_masks=False))
            audit = read_json(root / 'split_audit.json')
            audit['manifest_sha256'] = sha256_file(manifest)
            write_json(root / 'split_audit.json', audit)
            inputs = read_json(root / 'input_manifest.json')
            inputs['query_manifest_sha256'] = audit['manifest_sha256']
            write_json(root / 'input_manifest.json', inputs)
            if summary['failed_rows'] or summary['excluded_rows']:
                raise RuntimeError('BLOCKED: real-only crop stage has technical exclusions; do not filter them')
            return summary
        if stage == 'features':
            cfg, manifest, rows = _manifest(base, root)
            try:
                _, sources = _query_features(base, cfg, manifest, rows)
            except RuntimeError as exc:
                if not str(exc).startswith('BLOCKED: no exact real-only center-frame feature'):
                    raise
            else:
                return {'status': 'reused', 'query_frames': len(sources)}
            from .commands import extract_features
            return extract_features(cfg, subset='validation')
        if stage == 'baseline':
            return _baseline(m0, base, root)
        if stage == 'resampling':
            return _stability(m0, base, root)
        if stage == 'compression':
            return _compression(m0, root)
        if stage == 'null':
            return _null(m0, base, root)
        if stage == 'finalize':
            return _finalize(m0, base, root)
        raise ValueError(f'Unknown M0 stage: {stage}')
    except Exception as exc:
        # Do not mislabel incomplete or mismatched provenance as a completed M0.
        diagnostics = root / 'diagnostics.json'
        if stage != 'prepare' or not diagnostics.exists() or read_json(diagnostics).get('status') != 'completed':
            write_json(diagnostics, {'status': 'BLOCKED', 'reason': str(exc),
                'feature_config_mismatch': [str(exc)] if 'feature' in str(exc).lower() or
                    'provenance' in str(exc).lower() else [],
                'null_control_skipped': not (root / 'null_control_summary.json').is_file(),
                'official_test_locked': True, 'fake_data_read': False, 'test_data_read': False})
            atomic_write_text(root / 'summary.md', '# M0 BLOCKED\n\nM0 is a real-only structural '
                'audit, not a localization experiment. ' + str(exc) + '\n\n'
                'No M1 recommendation; original Gate B remains failed; official test locked.\n')
        raise
