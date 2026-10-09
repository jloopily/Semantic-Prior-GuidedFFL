"""Independent per-input preprocessing. No target-real or mask fallback."""
from __future__ import annotations

from dataclasses import fields
from pathlib import Path
import os

import numpy as np

from src.cirlid.config import resolve_config_path
from src.cirlid.crops import _dlib_detector, detect_largest_face_box
from src.cirlid.dinov3 import extract_patch_tokens, load_official_model, preprocess_rgb, spec_from_config
from src.cirlid.ffpp import FFPPIntegrationUnavailable, decode_frame_rgb
from src.cirlid.geometry import crop_and_resize, square_bbox
from src.cirlid.io import save_array, sha256_file, write_json
from src.r0_semantic_relation.geometry import SemanticRegions, build_semantic_regions
from src.r0_semantic_relation.mediapipe_adapter import create_landmarker, detect_landmarks
from .dataset import check_raw
from .protocol import ProtocolError, Run, key


def resource_guard(run: Run, *, allow_heavy: bool) -> None:
    """Do not let a full stage silently run on the current 0.5-core/2-GiB host."""
    if not allow_heavy:
        raise ProtocolError('this stage requires explicit --allow-heavy on a suitably provisioned experiment host')
    cores = float(os.cpu_count() or 1)
    memory = None
    cpu_limit = Path('/sys/fs/cgroup/cpu.max')
    memory_limit = Path('/sys/fs/cgroup/memory.max')
    if cpu_limit.is_file():
        quota, period = cpu_limit.read_text().split()
        if quota != 'max':
            cores = min(cores, int(quota) / int(period))
    if memory_limit.is_file():
        value = memory_limit.read_text().strip()
        if value != 'max':
            memory = int(value)
    legacy_cpu = Path('/sys/fs/cgroup/cpu/cpu.cfs_quota_us')
    legacy_period = Path('/sys/fs/cgroup/cpu/cpu.cfs_period_us')
    if legacy_cpu.is_file() and legacy_period.is_file():
        quota = int(legacy_cpu.read_text())
        if quota > 0:
            cores = min(cores, quota / int(legacy_period.read_text()))
    legacy_memory = Path('/sys/fs/cgroup/memory/memory.limit_in_bytes')
    if legacy_memory.is_file():
        limit = int(legacy_memory.read_text())
        memory = min(memory, limit) if memory is not None else limit
    physical_memory = None
    meminfo = Path('/proc/meminfo')
    if meminfo.is_file():
        entries = {line.split(':', 1)[0]: line.split(':', 1)[1] for line in meminfo.read_text().splitlines()}
        physical_memory = int(entries['MemTotal'].split()[0]) * 1024
    if physical_memory is not None:
        memory = min(memory, physical_memory) if memory is not None else physical_memory
    if cores < run.config.require('runtime.minimum_cpu_cores'):
        raise ProtocolError(f'CPU quota {cores:g} cores is too low; no work started')
    if memory is not None and memory < run.config.require('runtime.minimum_memory_gib') * 1024**3:
        raise ProtocolError(f'memory limit {memory / 1024**3:g} GiB is too low; no work started')
    threads = str(run.config.require('runtime.threads'))
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = threads


def read_array(run: Run, relative: str) -> np.ndarray:
    return np.load(run.path(relative), allow_pickle=False)


def save_regions(path: Path, regions: SemanticRegions) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.npz.tmp')
    with temporary.open('wb') as handle:
        np.savez_compressed(handle, **{f.name: getattr(regions, f.name)
                                      for f in fields(SemanticRegions) if f.name != 'metadata'})
    temporary.replace(path)


def read_regions(run: Run, relative: str) -> SemanticRegions:
    with np.load(run.path(relative), allow_pickle=False) as archive:
        values = {f.name: archive[f.name] for f in fields(SemanticRegions) if f.name != 'metadata'}
    return SemanticRegions(**values, metadata={})


def failure(row: dict, stage: str, reason: str) -> dict:
    return {**row, 'status': 'failed', 'failed_stage': stage, 'reason': reason}


def crops(run: Run) -> None:
    source = run.read_stage('prepare')
    with run.stage('crops', ('prepare',)) as stage:
        if stage.complete:
            return
        detector = _dlib_detector()
        rows = []
        for original in source:
            sample_id = original['sample_id']
            cached = stage.cached(sample_id)
            if cached is not None:
                rows.append(cached)
                continue
            if original['status'] != 'ok' or original['frame_index'] is None:
                rows.append(stage.record(failure(original, 'crops', original.get('reason', 'no_cross_compression_frame'))))
                continue
            check_raw(original)
            try:
                image = decode_frame_rgb(original['video_path'], original['frame_index'])
                if image.shape[:2] != (original['height'], original['width']):
                    raise ValueError('decoded frame dimensions disagree with frozen metadata')
                bbox = square_bbox(detect_largest_face_box(image, detector=detector), scale=1.3)
                crop = crop_and_resize(image, bbox, 448)
                path = run.path(f'crops/arrays/{key(sample_id)}.npy')
                save_array(path, crop.astype(np.uint8))
                row = {**original, 'status': 'ok', 'crop_path': run.relative(path),
                       'crop_bbox': list(bbox.as_tuple()), 'crop_geometry_source': sample_id,
                       'detector_input_video': original['video_path'],
                       'crop_sha256': sha256_file(path)}
                rows.append(stage.record(row, [path]))
            except (OSError, ValueError, FFPPIntegrationUnavailable) as exc:
                if isinstance(exc, ProtocolError):
                    raise
                rows.append(stage.record(failure(original, 'crops', f'{type(exc).__name__}: {exc}')))
        stage.finish(rows)


def geometry(run: Run) -> None:
    source = run.read_stage('crops')
    with run.stage('geometry', ('crops',)) as stage:
        if stage.complete:
            return
        config = run.config
        asset = resolve_config_path(config, config.require('mediapipe.asset_path'))
        if sha256_file(asset) != run.frozen()['mediapipe_asset_sha256']:
            raise ProtocolError('MediaPipe asset changed after prepare')
        mp, landmarker = create_landmarker(asset,
            num_faces=config.require('mediapipe.num_faces'),
            min_face_detection_confidence=config.require('mediapipe.min_face_detection_confidence'),
            min_face_presence_confidence=config.require('mediapipe.min_face_presence_confidence'),
            min_tracking_confidence=config.require('mediapipe.min_tracking_confidence'))
        rows = []
        try:
            for original in source:
                sample_id = original['sample_id']
                cached = stage.cached(sample_id)
                if cached is not None:
                    rows.append(cached)
                    continue
                if original['status'] != 'ok':
                    rows.append(stage.record(original))
                    continue
                audit = {}
                try:
                    image = read_array(run, original['crop_path'])
                    landmarks, audit = detect_landmarks(mp, landmarker, image)
                    if audit['status'] != 'ok':
                        raise ValueError(f'landmark_failure:{audit["status"]}')
                    regions = build_semantic_regions(landmarks, size=448, token_grid=28, ring_radius=12)
                    for name in ('valid', 'mouth_core', 'perioral_ring', 'cheek_left', 'cheek_right'):
                        if not np.any(getattr(regions, name)):
                            raise ValueError(f'empty_semantic_region:{name}')
                    path = run.path(f'geometry/regions/{key(sample_id)}.npz')
                    points_path = run.path(f'geometry/landmarks/{key(sample_id)}.json')
                    save_regions(path, regions)
                    write_json(points_path, {'landmarks': [[float(p.x), float(p.y), float(p.z)] for p in landmarks],
                                              'status': audit, 'geometry_source': sample_id})
                    row = {**original, 'status': 'ok', 'landmark_status': audit,
                           'regions_path': run.relative(path), 'landmarks_path': run.relative(points_path),
                           'region_areas': {name: int(np.count_nonzero(getattr(regions, name)))
                                            for name in ('mouth_core', 'perioral_ring', 'cheek_left', 'cheek_right', 'valid')},
                           'region_token_weights': {name: float(getattr(regions, 'token_' + name).sum())
                                                   for name in ('mouth_core', 'perioral_ring', 'cheek_left', 'cheek_right', 'valid')}}
                    rows.append(stage.record(row, [path, points_path]))
                except (OSError, ValueError) as exc:
                    if isinstance(exc, ProtocolError):
                        raise
                    row = failure(original, 'geometry', f'{type(exc).__name__}: {exc}')
                    row['landmark_status'] = audit
                    rows.append(stage.record(row))
        finally:
            landmarker.close()
        stage.finish(rows)


def features(run: Run) -> None:
    source = run.read_stage('geometry')
    with run.stage('features', ('geometry',)) as stage:
        if stage.complete:
            return
        import torch
        config = run.config
        device = config.require('runtime.device')
        if device == 'cuda' and not torch.cuda.is_available():
            raise ProtocolError('CUDA features requested but no GPU available; nothing was extracted')
        torch.set_num_threads(config.require('runtime.threads'))
        checkpoint = resolve_config_path(config, config.require('paths.dinov3_weights'))
        if sha256_file(checkpoint) != run.frozen()['dinov3_checkpoint_sha256']:
            raise ProtocolError('DINOv3 checkpoint changed after prepare')
        repo = resolve_config_path(config, config.require('paths.dinov3_repo'))
        repo_sources = {p.relative_to(repo).as_posix(): sha256_file(p) for p in sorted(repo.rglob('*.py'))}
        if repo_sources != run.frozen()['dinov3_repo_source_hashes']:
            raise ProtocolError('DINOv3 repository source changed after prepare')
        spec = spec_from_config(config)
        model = load_official_model(repo,
                                    checkpoint, spec, device)
        rows: list[dict] = []
        pending: list[dict] = []

        def flush() -> None:
            if not pending:
                return
            images = [read_array(run, row['crop_path']) for row in pending]
            batch = preprocess_rgb(images, config)
            values = extract_patch_tokens(model, batch, spec)
            if len(values) != len(pending):
                raise ProtocolError('DINOv3 batch output count differs from requested sample count')
            for original, value in zip(pending, values):
                path = run.path(f'features/arrays/{key(original["sample_id"])}.npy')
                save_array(path, value.astype(np.float32))
                rows.append(stage.record({**original, 'features_path': run.relative(path),
                                          'feature_shape': list(value.shape), 'feature_dtype': 'float32',
                                          'extra_l2_normalization': False}, [path]))
            pending.clear()

        for original in source:
            cached = stage.cached(original['sample_id'])
            if cached is not None:
                rows.append(cached)
            elif original['status'] != 'ok':
                rows.append(stage.record(original))
            else:
                pending.append(original)
                if len(pending) >= config.require('runtime.batch_size'):
                    flush()
        flush()
        stage.finish(sorted(rows, key=lambda row: row['sample_id']))
