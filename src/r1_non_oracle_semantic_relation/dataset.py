"""Train-only component roles. Masks are not located/read until evaluation."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from src.cirlid.config import Config, resolve_config_path
from src.cirlid.ffpp import FFPPIntegrationUnavailable, directed_fake_names, probe_video, read_split_pairs
from src.cirlid.geometry import fractional_frame_index
from src.cirlid.io import sha256_file, write_json
from .protocol import ProtocolError, Run


def component_map(pairs: list[tuple[str, str]]) -> dict[str, str]:
    """Same sorted union-find labels used by R0; both fake directions share a unit."""
    parent: dict[str, str] = {}

    def find(value: str) -> str:
        parent.setdefault(value, value)
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    for first, second in pairs:
        a, b = find(first), find(second)
        if a != b:
            parent[max(a, b)] = min(a, b)
    labels = {root: f'component-{i:04d}' for i, root in enumerate(sorted({find(v) for v in parent}), 1)}
    return {value: labels[find(value)] for value in parent}


def select_roles(pairs: list[tuple[str, str]], config: Config) -> dict:
    components = component_map(pairs)
    groups: dict[str, list[str]] = defaultdict(list)
    for video, component in sorted(components.items()):
        groups[component].append(video)
    names = sorted(groups)
    if len(names) < 3:
        raise ProtocolError('at least three components needed for evaluation/calibration/gallery')
    shuffled = list(np.random.default_rng(config.require('holdout.seed')).permutation(names))
    count = max(1, min(len(names) - 2, round(len(names) * config.require('holdout.fraction'))))
    holdout = set(shuffled[:count])
    exploration = sorted(set(names) - holdout)
    calibration_components: set[str] = set()
    calibration_ids: list[str] = []
    for component in np.random.default_rng(config.require('calibration.seed')).permutation(exploration):
        calibration_components.add(str(component))
        calibration_ids.extend(groups[str(component)])
        if len(calibration_ids) >= config.require('calibration.videos'):
            break
    # Entire selected components are excluded from every gallery even when a
    # component has more videos than the requested calibration count.
    calibration_ids = sorted(calibration_ids)[:config.require('calibration.videos')]
    pool = sorted(video for video, component in components.items()
                  if component not in holdout | calibration_components)
    if len(calibration_ids) != config.require('calibration.videos') or len(pool) < config.require('gallery.videos'):
        raise ProtocolError('train component-disjoint pool too small for fixed gallery/calibration sizes')
    galleries = {str(seed): sorted(str(v) for v in np.random.default_rng(seed).choice(
                    pool, size=config.require('gallery.videos'), replace=False))
                 for seed in config.require('gallery.seeds')}
    roles = {'components': components, 'evaluation_components': sorted(holdout),
             'evaluation_ids': sorted(v for v, g in components.items() if g in holdout),
             'calibration_components': sorted(calibration_components),
             'calibration_ids': calibration_ids, 'gallery_pool_ids': pool,
             'galleries': galleries, 'seed_selection_uses_scores_or_masks': False}
    assert_disjoint(roles)
    return roles


def assert_disjoint(roles: dict) -> None:
    components = roles['components']
    evaluation = set(roles['evaluation_components'])
    calibration = set(roles['calibration_components'])
    known_components = set(components.values())
    if not evaluation <= known_components or not calibration <= known_components:
        raise ProtocolError('unknown component in frozen roles')
    expected_evaluation = {v for v, c in components.items() if c in evaluation}
    if set(roles['evaluation_ids']) != expected_evaluation:
        raise ProtocolError('evaluation video role mismatch')
    if any(v not in components for v in roles['calibration_ids']):
        raise ProtocolError('unknown calibration video')
    if len(roles['calibration_ids']) != len(set(roles['calibration_ids'])):
        raise ProtocolError('duplicate calibration videos')
    expected_pool = {v for v, c in components.items() if c not in evaluation | calibration}
    if set(roles['gallery_pool_ids']) != expected_pool:
        raise ProtocolError('gallery pool role mismatch')
    if evaluation & calibration:
        raise ProtocolError('evaluation/calibration component leakage')
    for ids in roles['galleries'].values():
        if any(v not in components for v in ids):
            raise ProtocolError('unknown gallery video')
        if len(ids) != len(set(ids)):
            raise ProtocolError('duplicate gallery videos')
        if any(v not in expected_pool for v in ids):
            raise ProtocolError('gallery overlaps evaluation/calibration components')
    if any(components[v] not in calibration for v in roles['calibration_ids']):
        raise ProtocolError('calibration video role mismatch')


def video_path(config: Config, video: str, compression: str, *, fake: bool) -> Path:
    root = resolve_config_path(config, config.require('paths.ffpp_root')).resolve()
    if compression not in ('c23', 'c40'):
        raise ProtocolError('only frozen train c23/c40 are accessible')
    directory = (root / 'manipulated_sequences/Face2Face' if fake
                 else root / 'original_sequences/youtube')
    path = directory / compression / 'videos' / f'{video}.mp4'
    resolved = path.resolve()
    if root not in resolved.parents:
        raise ProtocolError('raw video symlink escapes authorized FF++ root')
    return resolved


def mask_path(run: Run, row: dict) -> Path:
    """Called exclusively by evaluate after score artifacts have been committed."""
    if row['role'] != 'evaluation_fake' or row['label'] != 1:
        raise ProtocolError('masks only accessible for frozen evaluation fake rows')
    root = resolve_config_path(run.config, run.config.require('paths.ffpp_root')).resolve()
    path = (root / 'manipulated_sequences/Face2Face/masks/videos' / f'{row["video_id"]}.mp4').resolve()
    if root not in path.parents:
        raise ProtocolError('mask path escaped authorized data root')
    return path


def check_raw(row: dict) -> None:
    path = Path(row['video_path'])
    current = path.stat()
    if current.st_size != row['video_size'] or current.st_mtime_ns != row['video_mtime_ns']:
        raise ProtocolError(f'raw video changed since prepare: {path}')


def prepare(run: Run) -> None:
    import cv2  # Fail early on missing dependency, not as thousands of missing videos.
    config = run.config
    if run.root.exists() and any(run.root.iterdir()):
        raise ProtocolError('prepare needs a new or empty R1 output directory')
    split = resolve_config_path(config, config.require('paths.split_files.train'))
    pairs = read_split_pairs(split)
    roles = select_roles(pairs, config)
    raw_cache: dict[str, dict] = {}

    def info(video: str, compression: str, fake: bool) -> dict:
        path = video_path(config, video, compression, fake=fake)
        name = str(path)
        if name not in raw_cache:
            try:
                count, width, height, fps = probe_video(path)
                stat = path.stat()
                raw_cache[name] = {'status': 'ok', 'video_path': name, 'frame_count': count,
                                   'width': width, 'height': height, 'fps': fps,
                                   'video_size': stat.st_size, 'video_mtime_ns': stat.st_mtime_ns}
            except (OSError, ValueError, FFPPIntegrationUnavailable) as exc:
                raw_cache[name] = {'status': 'failed', 'video_path': name,
                                   'reason': f'{type(exc).__name__}: {exc}'}
        return raw_cache[name]

    rows: list[dict] = []
    by_sample: dict[str, dict] = {}
    fractions = config.require('sampling.frame_fractions')

    def add(video: str, compression: str, role: str, fraction: float, meta: dict,
            index: int | None, component: str, **extra) -> dict:
        sample_id = f'{role}/{compression}/{video}/p{fraction:.2f}'
        if role == 'paired_real':
            sample_id += f'/f{index}'
        row = {'sample_id': sample_id, 'role': role, 'compression': compression,
               'video_id': video, 'component_id': component, 'fraction': fraction,
               'label': int(role == 'evaluation_fake'), **meta,
               'frame_index': index, **extra}
        if sample_id in by_sample:
            previous = by_sample[sample_id]
            if previous['frame_index'] != index:
                raise ProtocolError('inconsistent shared real frame request')
            return previous
        rows.append(row)
        by_sample[sample_id] = row
        return row

    union_gallery = sorted({video for ids in roles['galleries'].values() for video in ids})
    for role, ids, compressions in (
        ('gallery', union_gallery, ('c23', 'c40')),
        ('calibration', roles['calibration_ids'], ('c23',)),
        ('evaluation_real', roles['evaluation_ids'], ('c23', 'c40')),
    ):
        for video in ids:
            metadata = {c: info(video, c, False) for c in compressions}
            counts = [m['frame_count'] for m in metadata.values() if m['status'] == 'ok']
            shared_count = min(counts) if len(counts) == len(compressions) else None
            for compression in compressions:
                for fraction in fractions:
                    index = fractional_frame_index(shared_count, fraction) if shared_count else None
                    add(video, compression, role, fraction, metadata[compression], index,
                        roles['components'][video])
    for name, target, source in directed_fake_names(split):
        component = roles['components'][target]
        if component not in roles['evaluation_components']:
            continue
        metadata = {c: info(name, c, True) for c in ('c23', 'c40')}
        counts = [m['frame_count'] for m in metadata.values() if m['status'] == 'ok']
        shared_count = min(counts) if len(counts) == 2 else None
        for compression in ('c23', 'c40'):
            real_meta = info(target, compression, False)
            for fraction in fractions:
                index = fractional_frame_index(shared_count, fraction) if shared_count else None
                strict = bool(index is not None and real_meta['status'] == 'ok'
                              and index < real_meta['frame_count'])
                real_id = f'evaluation_real/{compression}/{target}/p{fraction:.2f}'
                real = by_sample[real_id]
                if strict and real['frame_index'] != index:
                    real = add(target, compression, 'paired_real', fraction, real_meta, index, component)
                add(name, compression, 'evaluation_fake', fraction, metadata[compression], index,
                    component, target_id=target, source_id=source,
                    source_group='pair__' + '__'.join(sorted((target, source))),
                    paired_real_id=real['sample_id'], strict_real_frame_match=strict)
    # Fingerprint both model assets once on the experiment machine; not locally.
    asset = resolve_config_path(config, config.require('mediapipe.asset_path'))
    asset_sha = sha256_file(asset)
    if asset_sha != config.require('mediapipe.asset_sha256'):
        raise ProtocolError('MediaPipe asset hash differs from configured official asset')
    checkpoint = resolve_config_path(config, config.require('paths.dinov3_weights'))
    repo = resolve_config_path(config, config.require('paths.dinov3_repo'))
    if not (repo / 'hubconf.py').is_file():
        raise ProtocolError('official DINOv3 repository with hubconf.py is required')
    repo_sources = {p.relative_to(repo).as_posix(): sha256_file(p) for p in sorted(repo.rglob('*.py'))}
    run.freeze({'train_split_sha256': sha256_file(split), 'roles': roles,
                'mediapipe_asset_sha256': asset_sha, 'dinov3_checkpoint_sha256': sha256_file(checkpoint),
                'dinov3_repo_source_hashes': repo_sources,
                'raw_video_integrity': 'size_and_mtime_not_full_content_hash',
                'frame_pairing': 'shared_numeric_index_across_compressions_real_secondary_only'})
    with run.stage('prepare', ()) as stage:
        write_json(run.path('prepare/roles.json'), roles)
        write_json(run.path('prepare/probe_audit.json'), raw_cache)
        stage.finish(rows)
