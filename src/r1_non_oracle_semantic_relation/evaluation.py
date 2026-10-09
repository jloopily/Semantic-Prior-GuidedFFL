"""Mask access is isolated here and happens only after every scoring stage."""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from src.cirlid.ffpp import FFPPIntegrationUnavailable, decode_mask, probe_video
from src.cirlid.geometry import BBox, crop_and_resize, map_bbox
from src.cirlid.io import read_json, save_array, write_jsonl
from . import METHODS
from .dataset import check_raw, mask_path
from .metrics import pixel_map, pixel_metrics, warp_real_to_fake
from .preprocessing import failure, read_array, read_regions
from .protocol import ProtocolError, Run, key


def load_maps(run: Run, row: dict) -> dict[str, np.ndarray]:
    with np.load(run.path(row['score_path']), allow_pickle=False) as archive:
        return {method: pixel_map(archive[method]) for method in METHODS}


def load_fake_truth(run: Run, row: dict) -> tuple[np.ndarray, dict]:
    path = mask_path(run, row)
    count, width, height, _ = probe_video(path)
    # No temporal interpolation of masks: strict numeric frame correspondence.
    if count != row['frame_count']:
        raise ValueError('mask_video_frame_count_mismatch_no_remapping')
    check_raw(row)
    mask = decode_mask(path, row['frame_index'])
    if mask.shape != (height, width):
        raise ValueError('decoded mask dimensions disagree with mask metadata')
    bbox = map_bbox(BBox(*row['crop_bbox']), source_size=(row['width'], row['height']),
                    target_size=(width, height))
    truth = crop_and_resize(mask, bbox, 448, is_mask=True, mask_policy='nonzero')
    stat = path.stat()
    return truth, {'raw_mask_path': str(path), 'mask_frame_count': count,
                   'mask_frame_index': row['frame_index'], 'mask_frame_remapped': False,
                   'raw_mask_size': stat.st_size, 'raw_mask_mtime_ns': stat.st_mtime_ns,
                   'mask_used_only_for_evaluation': True}


def paired_maps(run: Run, fake: dict, real: dict, fake_maps: dict[str, np.ndarray],
                truth: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]:
    fake_regions = read_regions(run, fake['regions_path'])
    real_regions = read_regions(run, real['regions_path'])
    real_maps = load_maps(run, real)
    warped = {}
    common = fake_regions.valid.copy()
    for method in METHODS:
        warped[method], overlap = warp_real_to_fake(real_maps[method], real_regions.valid, fake, real)
        common &= overlap
    if not common.any():
        raise ValueError('independent_real_fake_crops_have_no_common_valid_domain')
    return {name: value[common] for name, value in fake_maps.items()}, {
        name: value[common] for name, value in warped.items()}, truth[common]


def evaluate(run: Run) -> None:
    source = run.read_stage('score')
    by_id = {row['sample_id']: row for row in source}
    thresholds = read_json(run.path('calibrate/thresholds.json'))
    with run.stage('evaluate', ('score', 'geometry', 'calibrate')) as stage:
        if stage.complete:
            return
        frames = []
        for original in source:
            if original['role'] == 'paired_real':
                continue
            cached = stage.cached(original['sample_id'])
            if cached is not None:
                frames.append(cached)
                continue
            if original['status'] != 'ok':
                frames.append(stage.record({**original, 'evaluable': False}))
                continue
            artifacts = []
            try:
                maps = load_maps(run, original)
                regions = read_regions(run, original['regions_path'])
                if original['role'] == 'evaluation_fake':
                    truth, mask_audit = load_fake_truth(run, original)
                    mask_artifact = run.path(f'evaluate/masks/{key(original["input_sample_id"])}.npy')
                    if mask_artifact.exists():
                        if not np.array_equal(read_array(run, run.relative(mask_artifact)), truth):
                            raise ProtocolError('mask content changed between gallery seeds/resume')
                    else:
                        save_array(mask_artifact, truth)
                    artifacts.append(mask_artifact)
                    row = {**original, **mask_audit, 'mask_artifact': run.relative(mask_artifact)}
                else:
                    truth = np.zeros((448, 448), dtype=np.uint8)
                    row = {**original, 'mask_used_only_for_evaluation': False}
                threshold_record = thresholds[str(original['gallery_seed'])]
                method_thresholds = threshold_record.get('thresholds', {})
                row['metrics'] = {method: pixel_metrics(value[regions.valid], truth[regions.valid],
                                                      method_thresholds.get(method))
                                  for method, value in maps.items()}
                row['region_metrics'] = {}
                for region in run.config.require('statistics.secondary_domains'):
                    domain = getattr(regions, region) & regions.valid
                    row['region_metrics'][region] = {
                        method: pixel_metrics(value[domain], truth[domain], method_thresholds.get(method))
                        for method, value in maps.items()}
                row['evaluable'] = bool(original['role'] == 'evaluation_fake'
                                        and all(row['metrics'][m]['ap'] is not None for m in METHODS))
                row['scored_and_mask_evaluated'] = True
                row['empty_valid_mask'] = not bool(truth[regions.valid].any())
                row['full_valid_mask'] = bool(truth[regions.valid].all())
                row['paired_comparison_status'] = 'not_applicable'
                row['mask_positive_pixels_in_crop'] = int(truth.sum())
                row['mask_positive_pixels_in_valid_domain'] = int(truth[regions.valid].sum())
                semantic = regions.mouth_core | regions.perioral_ring
                row['mask_fraction_in_mouth_perioral'] = float(truth[semantic].sum() / truth.sum()) if truth.sum() else None
                if original['role'] == 'evaluation_fake':
                    real = by_id.get(f'seed{original["gallery_seed"]}/{original["paired_real_id"]}')
                    if not original['strict_real_frame_match'] or real is None or real['status'] != 'ok':
                        row['paired_comparison_status'] = 'unavailable_real_frame_or_geometry'
                    else:
                        try:
                            fmaps, rmaps, common_truth = paired_maps(run, original, real, maps, truth)
                            row['paired_metrics'] = {}
                            for method in METHODS:
                                fake_metric = pixel_metrics(fmaps[method], common_truth, method_thresholds.get(method))
                                real_metric = pixel_metrics(rmaps[method], common_truth, method_thresholds.get(method))
                                row['paired_metrics'][method] = {'fake': fake_metric, 'real_on_fake_mask': real_metric,
                                    'fake_minus_real_ap': fake_metric['ap'] - real_metric['ap']
                                    if fake_metric['ap'] is not None and real_metric['ap'] is not None else None}
                            row['paired_comparison_status'] = 'ok'
                            row['paired_common_pixels'] = len(common_truth)
                        except (OSError, ValueError) as exc:
                            if isinstance(exc, ProtocolError):
                                raise
                            row['paired_comparison_status'] = f'unavailable:{type(exc).__name__}:{exc}'
                frames.append(stage.record(row, artifacts))
            except (OSError, ValueError, FFPPIntegrationUnavailable) as exc:
                if isinstance(exc, ProtocolError):
                    raise
                frames.append(stage.record({**failure(original, 'evaluate', f'{type(exc).__name__}: {exc}'),
                                             'evaluable': False}))
        grouped: dict[tuple[int, str, str, str], list[dict]] = defaultdict(list)
        for row in frames:
            grouped[(row['gallery_seed'], row['compression'], row['role'], row['video_id'])].append(row)
        videos = [pool_video(run, frame_rows, by_id, thresholds[str(group[0])])
                  for group, frame_rows in sorted(grouped.items())]
        write_jsonl(run.path('evaluate/per_video_metrics.jsonl'), videos)
        write_jsonl(run.path('evaluate/per_query_metrics.jsonl'), frames)
        stage.finish(frames)


def pool_video(run: Run, rows: list[dict], score_rows: dict[str, dict], threshold_record: dict) -> dict:
    first = rows[0]
    good = [row for row in rows if row['status'] == 'ok' and row.get('scored_and_mask_evaluated')]
    coverage = (len(rows) == 3 and len(good) == 3
                and sorted(row['fraction'] for row in rows) == [0.25, 0.50, 0.75]
                and len({row['frame_index'] for row in good}) == 3)
    output = {name: first.get(name) for name in ('gallery_seed', 'compression', 'role', 'video_id',
              'component_id', 'source_group', 'source_id', 'target_id')}
    output.update(video_key=f'{first["role"]}/{first["compression"]}/{first["video_id"]}',
                  generated_frames=len(rows), scored_frames=len(good),
                  evaluable_frames=sum(bool(r.get('evaluable')) for r in rows),
                  undefined_ap_frames=sum(r['status'] == 'ok' and not r.get('evaluable') for r in rows),
                  empty_valid_mask_frames=sum(bool(r.get('empty_valid_mask')) for r in rows),
                  full_valid_mask_frames=sum(bool(r.get('full_valid_mask')) for r in rows),
                  frame_coverage_ok=coverage, status='ok' if coverage else 'insufficient_frame_coverage',
                  calibration_status=threshold_record['status'], metrics={}, region_metrics={}, paired_metrics={})
    for method in METHODS:
        output[f'{method}_ap'] = None
        output[f'{method}_f1'] = None
        output[f'{method}_fpr'] = None
    output['relation_ap_minus_prevalence'] = None
    output['relation_fake_minus_real_ap'] = None
    output['relation_minus_real_template_ap'] = None
    if not coverage:
        return output
    # At most one video's three frames are pooled. Include empty/full-mask
    # frames; undefined frame AP does not authorize dropping their pixels.
    truths = [read_array(run, row['mask_artifact']) if first['role'] == 'evaluation_fake'
              else np.zeros((448, 448), np.uint8) for row in good]
    regions = [read_regions(run, row['regions_path']) for row in good]
    frame_maps = [load_maps(run, row) for row in good]
    method_thresholds = threshold_record.get('thresholds', {})
    for method in METHODS:
        values = np.concatenate([maps[method][r.valid] for maps, r in zip(frame_maps, regions)])
        labels = np.concatenate([truth[r.valid] for truth, r in zip(truths, regions)])
        metric = pixel_metrics(values, labels, method_thresholds.get(method))
        output['metrics'][method] = metric
        for field in ('ap', 'f1', 'fpr'):
            output[f'{method}_{field}'] = metric[field]
        if method == 'relation':
            output['relation_ap_minus_prevalence'] = metric['ap_minus_prevalence']
    for region in run.config.require('statistics.secondary_domains'):
        domains = [getattr(r, region) & r.valid for r in regions]
        labels = np.concatenate([truth[domain] for truth, domain in zip(truths, domains)])
        output['region_metrics'][region] = {
            method: pixel_metrics(np.concatenate([maps[method][domain] for maps, domain in zip(frame_maps, domains)]),
                                  labels, method_thresholds.get(method)) for method in METHODS}
    if first['role'] == 'evaluation_fake' and all(r.get('paired_comparison_status') == 'ok' for r in good):
        fparts = {m: [] for m in METHODS}
        rparts = {m: [] for m in METHODS}
        labels = []
        for row, maps, truth in zip(good, frame_maps, truths):
            real = score_rows[f'seed{row["gallery_seed"]}/{row["paired_real_id"]}']
            fmaps, rmaps, common_truth = paired_maps(run, row, real, maps, truth)
            labels.append(common_truth)
            for method in METHODS:
                fparts[method].append(fmaps[method])
                rparts[method].append(rmaps[method])
        pooled_truth = np.concatenate(labels)
        for method in METHODS:
            fake_metric = pixel_metrics(np.concatenate(fparts[method]), pooled_truth, method_thresholds.get(method))
            real_metric = pixel_metrics(np.concatenate(rparts[method]), pooled_truth, method_thresholds.get(method))
            output['paired_metrics'][method] = {'fake': fake_metric, 'real_on_fake_mask': real_metric,
                'fake_minus_real_ap': fake_metric['ap'] - real_metric['ap']
                if fake_metric['ap'] is not None and real_metric['ap'] is not None else None}
        output['relation_fake_minus_real_ap'] = output['paired_metrics']['relation']['fake_minus_real_ap']
        fake_ap = output['paired_metrics']['relation']['fake']['ap']
        real_template_ap = output['paired_metrics']['position_template']['real_on_fake_mask']['ap']
        output['relation_minus_real_template_ap'] = fake_ap - real_template_ap if fake_ap is not None and real_template_ap is not None else None
        output['paired_common_pixels'] = len(pooled_truth)
    output['evaluable'] = all(output[f'{m}_ap'] is not None for m in METHODS)
    return output
