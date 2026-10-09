"""Real-only gallery, clean-real calibration, and mask-free query scoring."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterator

import numpy as np

from src.cirlid.io import read_json, save_array, write_json
from .metrics import pixel_map as flatten_map
from . import METHODS
from .dataset import assert_disjoint
from .preprocessing import failure, read_array, read_regions
from .protocol import ProtocolError, Run, key
from .scoring import Descriptor, context_derangement, describe, score_gallery


def save_descriptor(path: Path, descriptor: Descriptor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('wb') as handle:
        np.savez(handle, unit_tokens=descriptor.unit_tokens,
                 relation=descriptor.relation, broken=descriptor.broken)
    temporary.replace(path)


def read_descriptor(run: Run, relative: str) -> Descriptor:
    with np.load(run.path(relative), allow_pickle=False) as archive:
        value = Descriptor(archive['unit_tokens'], archive['relation'], archive['broken'])
    if (value.unit_tokens.shape != (784, 768) or value.relation.shape != (784, 2)
            or value.broken.shape != (784, 2)
            or not all(np.isfinite(v).all() for v in (value.unit_tokens, value.relation, value.broken))):
        raise ProtocolError('invalid serialized gallery descriptor')
    return value


def gallery(run: Run) -> None:
    source = run.read_stage('features')
    roles = run.frozen()['roles']
    assert_disjoint(roles)
    with run.stage('gallery', ('features',)) as stage:
        if stage.complete:
            return
        permutation = context_derangement(run.config.require('protocol.seed'))
        rows = []
        for original in source:
            if original['role'] != 'gallery':
                continue
            if original['label'] != 0 or original['video_id'] not in roles['gallery_pool_ids']:
                raise ProtocolError('non-real or unauthorized gallery input')
            if roles['components'][original['video_id']] != original['component_id']:
                raise ProtocolError('gallery component mismatch')
            cached = stage.cached(original['sample_id'])
            if cached is not None:
                rows.append(cached)
                continue
            if original['status'] != 'ok':
                rows.append(stage.record(original))
                continue
            try:
                descriptor = describe(read_array(run, original['features_path']),
                                      read_regions(run, original['regions_path']), permutation)
                path = run.path(f'gallery/descriptors/{key(original["sample_id"])}.npz')
                save_descriptor(path, descriptor)
                rows.append(stage.record({**original, 'descriptor_path': run.relative(path)}, [path]))
            except (OSError, ValueError) as exc:
                if isinstance(exc, ProtocolError):
                    raise
                rows.append(stage.record(failure(original, 'gallery', f'{type(exc).__name__}: {exc}')))
        grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for row in rows:
            grouped[(row['video_id'], row['compression'])].append(row)
        indices = {}
        for seed, selected in roles['galleries'].items():
            valid_videos = []
            excluded = {}
            for video in selected:
                valid = all(len(grouped[(video, compression)]) == 3
                            and all(r['status'] == 'ok' for r in grouped[(video, compression)])
                            and len({r['frame_index'] for r in grouped[(video, compression)]}) == 3
                            for compression in ('c23', 'c40'))
                if valid:
                    valid_videos.append(video)
                else:
                    excluded[video] = {compression: [r.get('reason', r['status'])
                                                   for r in grouped[(video, compression)]]
                                       for compression in ('c23', 'c40')}
            for compression in ('c23', 'c40'):
                indices[f'{seed}/{compression}'] = {
                    'seed': int(seed), 'compression': compression,
                    'status': 'ok' if len(valid_videos) >= run.config.require('gallery.min_valid_videos') else 'failed',
                    'selected_video_ids': selected, 'valid_video_ids': valid_videos,
                    'excluded_videos': excluded, 'no_replacement': True,
                    'paired_compression_common_gallery': True,
                    'videos': {video: [r['descriptor_path'] for r in sorted(grouped[(video, compression)],
                                                                         key=lambda r: r['fraction'])]
                               for video in valid_videos},
                    'gallery_real_only': True, 'component_disjoint': True,
                    'duplicate_neighbor_audit': 'not_applicable_no_knn',
                    'duplicate_crop_content': duplicate_crops([r for video in valid_videos
                        for r in grouped[(video, compression)]]),
                }
        write_json(run.path('gallery/index.json'), indices)
        stage.finish(rows)


def references(run: Run, index: dict, query: dict) -> Iterator[tuple[str, Iterator[Descriptor]]]:
    if index['status'] != 'ok':
        raise ValueError('too_few_valid_gallery_videos')
    roles = run.frozen()['roles']
    assert_disjoint(roles)
    if query['component_id'] in {roles['components'][v] for v in index['videos']}:
        raise ProtocolError('query/reference component overlap')
    if index['compression'] != query['compression']:
        raise ProtocolError('reference compression differs from query')
    for video, paths in sorted(index['videos'].items()):
        # Generator keeps only one reference frame in memory at a time.
        yield video, (read_descriptor(run, path) for path in paths)


def query_score(run: Run, row: dict, index: dict,
                descriptor: Descriptor | None = None) -> tuple[dict[str, np.ndarray], dict]:
    if row['role'] not in ('calibration', 'evaluation_fake', 'evaluation_real', 'paired_real'):
        raise ProtocolError('only frozen non-gallery queries can be scored')
    regions = read_regions(run, row['regions_path'])
    if descriptor is None:
        descriptor = describe(read_array(run, row['features_path']), regions,
                              context_derangement(run.config.require('protocol.seed')))
    return score_gallery(descriptor, regions, references(run, index, row))


def calibrate(run: Run) -> None:
    source = [r for r in run.read_stage('features') if r['role'] == 'calibration']
    with run.stage('calibrate', ('features', 'gallery')) as stage:
        if stage.complete:
            return
        indices = read_json(run.path('gallery/index.json'))
        roles = run.frozen()['roles']
        if any(r['label'] != 0 or r['component_id'] not in roles['calibration_components'] for r in source):
            raise ProtocolError('non-real or unauthorized calibration input')
        outputs = []
        thresholds = {}
        for seed in run.config.require('gallery.seeds'):
            sample_id = f'thresholds/{seed}'
            cached = stage.cached(sample_id)
            if cached is not None:
                outputs.append(cached)
                thresholds[str(seed)] = cached
                continue
            index = indices[f'{seed}/c23']
            attempts = []
            good: dict[str, list[tuple[dict, dict[str, np.ndarray]]]] = defaultdict(list)
            for original in source:
                row = {**original, 'gallery_seed': seed}
                if original['status'] != 'ok':
                    attempts.append(row)
                    continue
                try:
                    maps, audit = query_score(run, original, index)
                    path = run.path(f'calibrate/scores/{seed}/{key(original["sample_id"])}.npz')
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with path.open('wb') as handle:
                        np.savez(handle, **maps)
                    # Retain only lightweight paths, not all maps/images in RAM.
                    good[original['video_id']].append((original, {'path': run.relative(path)}))
                    attempts.append({**row, 'score_path': run.relative(path), 'score_audit': audit})
                except (OSError, ValueError) as exc:
                    if isinstance(exc, ProtocolError):
                        raise
                    attempts.append(failure(row, 'calibrate', f'{type(exc).__name__}: {exc}'))
            eligible = {video: frames for video, frames in good.items()
                        if len(frames) == 3 and len({r['frame_index'] for r, _ in frames}) == 3}
            output = {'sample_id': sample_id, 'gallery_seed': seed,
                      'status': 'ok' if len(eligible) >= run.config.require('calibration.min_valid_videos') else 'failed',
                      'thresholds': {}, 'valid_video_ids': sorted(eligible),
                      'generated_frames': len(source), 'scored_frames': sum(len(v) for v in good.values()),
                      'eligible_frames': sum(len(v) for v in eligible.values()),
                      'source': 'component_disjoint_train_real_c23',
                      'quantile': 0.99, 'numpy_method': 'linear',
                      'prediction_rule': 'score_strictly_greater_than_threshold',
                      'transfer_to_c40': True, 'fake_data_used': False,
                      'gallery_status': index['status']}
            artifacts = [run.path(a["score_path"]) for a in attempts if a.get("score_path")]
            audit_path = run.path(f'calibrate/{seed}_attempts.json')
            write_json(audit_path, attempts)
            artifacts.append(audit_path)
            if output['status'] == 'ok':
                total = sum(int(read_regions(run, row['regions_path']).valid.sum())
                            for frames in eligible.values() for row, _ in frames)
                for method in METHODS:
                    pool_path = run.path(f'calibrate/pools/{seed}_{method}.npy')
                    pool_path.parent.mkdir(parents=True, exist_ok=True)
                    pool = np.lib.format.open_memmap(pool_path, mode='w+', dtype=np.float32, shape=(total,))
                    cursor = 0
                    for video in sorted(eligible):
                        for row, cached_maps in eligible[video]:
                            regions = read_regions(run, row['regions_path'])
                            with np.load(run.path(cached_maps['path']), allow_pickle=False) as archive:
                                values = flatten_map(archive[method])[regions.valid]
                            pool[cursor:cursor + len(values)] = values
                            cursor += len(values)
                    if cursor != total or total == 0:
                        raise ProtocolError('invalid clean-real calibration pixel count')
                    # This is a derived pool, not source data. In-place partition
                    # avoids allocating the complete pooled score array again.
                    threshold = float(np.quantile(pool, 0.99, method='linear', overwrite_input=True))
                    pool.flush()
                    del pool
                    output['thresholds'][method] = threshold
                    artifacts.append(pool_path)
                output['calibration_pixels'] = total
            else:
                output['reason'] = 'too_few_complete_clean_real_calibration_videos'
            outputs.append(stage.record(output, artifacts))
            thresholds[str(seed)] = output
        write_json(run.path('calibrate/thresholds.json'), thresholds)
        stage.finish(outputs)


def score(run: Run) -> None:
    source = [r for r in run.read_stage('features') if r['role'] in ('evaluation_fake', 'evaluation_real', 'paired_real')]
    indices = read_json(run.path('gallery/index.json'))
    thresholds = read_json(run.path('calibrate/thresholds.json'))
    with run.stage('score', ('features', 'gallery', 'calibrate')) as stage:
        if stage.complete:
            return
        rows = []
        for original in source:
            descriptor = None
            descriptor_error = None
            if original['status'] == 'ok':
                try:
                    descriptor = describe(read_array(run, original['features_path']),
                                          read_regions(run, original['regions_path']),
                                          context_derangement(run.config.require('protocol.seed')))
                except (OSError, ValueError) as exc:
                    if isinstance(exc, ProtocolError):
                        raise
                    descriptor_error = f'{type(exc).__name__}: {exc}'
            for seed in run.config.require('gallery.seeds'):
                row = {**original, 'input_sample_id': original['sample_id'],
                       'sample_id': f'seed{seed}/{original["sample_id"]}', 'gallery_seed': seed}
                cached = stage.cached(row['sample_id'])
                if cached is not None:
                    rows.append(cached)
                    continue
                if original['status'] != 'ok':
                    rows.append(stage.record(row))
                    continue
                if descriptor_error is not None:
                    rows.append(stage.record(failure(row, 'score', descriptor_error)))
                    continue
                index = indices[f'{seed}/{original["compression"]}']
                try:
                    maps, audit = query_score(run, original, index, descriptor)
                    path = run.path(f'score/maps/{key(row["sample_id"])}.npz')
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with path.open('wb') as handle:
                        np.savez_compressed(handle, **maps)
                    # Invalid calibration does not remove threshold-free AP.
                    row.update(score_path=run.relative(path), score_audit=audit,
                               calibration_status=thresholds[str(seed)]['status'],
                               paired_real_used_for_scoring=False, fake_mask_used_for_scoring=False)
                    rows.append(stage.record(row, [path]))
                except (OSError, ValueError) as exc:
                    if isinstance(exc, ProtocolError):
                        raise
                    rows.append(stage.record(failure(row, 'score', f'{type(exc).__name__}: {exc}')))
        stage.finish(rows)


def duplicate_crops(rows: list[dict]) -> dict[str, list[str]]:
    """Record equal saved RGB crops; do not silently prune selected references."""
    by_digest: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        if row.get('crop_sha256'):
            by_digest[row['crop_sha256']].append(row['sample_id'])
    return {digest: ids for digest, ids in sorted(by_digest.items()) if len(ids) > 1}
