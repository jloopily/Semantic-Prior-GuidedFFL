"""All-seed/component-clustered reports, without selecting the best outcome."""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone

import numpy as np

from src.cirlid.io import atomic_write_text, read_json, read_jsonl, write_csv, write_json
from src.r0_semantic_relation.bootstrap import paired_group_bootstrap
from . import METHODS
from .dataset import assert_disjoint
from .protocol import Run


COMPARISONS = {
    'relation_minus_local': ('relation_ap', 'local_difference_ap'),
    'relation_minus_template': ('relation_ap', 'position_template_ap'),
    'relation_minus_broken': ('relation_ap', 'broken_context_ap'),
    'relation_minus_prevalence': ('relation_ap_minus_prevalence', '_zero'),
    'relation_fake_minus_real': ('relation_fake_minus_real_ap', '_zero'),
    'relation_minus_real_template': ('relation_minus_real_template_ap', '_zero'),
}


def bootstrap(rows: list[dict], first: str, second: str, run: Run) -> dict:
    selected = [{**row, '_zero': 0.0} for row in rows]
    try:
        return {'status': 'ok', **paired_group_bootstrap(selected, first, second,
            group_key='component_id', replicates=run.config.require('statistics.bootstrap_replicates'),
            seed=run.config.require('protocol.bootstrap_seed'))}
    except ValueError as exc:
        return {'status': 'unavailable', 'reason': str(exc)}


def statistics(rows: list[dict], run: Run) -> dict:
    summaries = {}
    for method in METHODS:
        summaries[method] = {}
        for metric in ('ap', 'f1', 'fpr'):
            values = [r[f'{method}_{metric}'] for r in rows if r.get(f'{method}_{metric}') is not None]
            summaries[method][metric] = {'ordinary_video_mean': float(np.mean(values)) if values else None,
                                          'videos': len(values)}
        summaries[method]['ap_ci_component_weighted'] = bootstrap(rows, f'{method}_ap', '_zero', run)
    differences = {name: bootstrap(rows, first, second, run) for name, (first, second) in COMPARISONS.items()}
    return {'methods': summaries, 'differences': differences,
            'generated_videos': len(rows),
            'evaluable_videos': sum(all(r.get(f'{m}_ap') is not None for m in METHODS) for r in rows),
            'evaluable_components': len({r['component_id'] for r in rows if r.get('relation_ap') is not None}),
            'ordinary_means_and_component_weighted_differences_are_different_estimands': True}


def average_seeds(rows: list[dict], seeds: list[int]) -> list[dict]:
    """A seed is a repeated reference choice, NEVER an independent subject."""
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row['compression'], row['role'], row['video_id'])].append(row)
    outputs = []
    fields = [f'{m}_{metric}' for m in METHODS for metric in ('ap', 'f1', 'fpr')]
    fields += ['relation_ap_minus_prevalence', 'relation_fake_minus_real_ap', 'relation_minus_real_template_ap']
    for _, group in sorted(grouped.items()):
        if len(group) != len(seeds) or {r['gallery_seed'] for r in group} != set(seeds):
            continue
        row = {key: group[0].get(key) for key in ('compression', 'role', 'video_id', 'component_id',
                                                 'source_group', 'target_id', 'source_id')}
        row['seeds'] = seeds
        for field in fields:
            values = [r.get(field) for r in group]
            row[field] = float(np.mean(values)) if all(v is not None for v in values) else None
        row['evaluable'] = all(row[f'{m}_ap'] is not None for m in METHODS)
        outputs.append(row)
    return outputs


def positive(result: dict) -> bool:
    return result.get('status') == 'ok' and result['ci95'][0] > 0


def finalize(run: Run) -> None:
    frames = run.read_stage('evaluate')
    videos = read_jsonl(run.path('evaluate/per_video_metrics.jsonl'))
    geometry = run.read_stage('geometry')
    roles = run.frozen()['roles']
    assert_disjoint(roles)
    indices = read_json(run.path('gallery/index.json'))
    calibration = read_json(run.path('calibrate/thresholds.json'))
    with run.stage('finalize', ('evaluate', 'gallery', 'calibrate', 'geometry')) as stage:
        if stage.complete:
            return
        seeds = run.config.require('gallery.seeds')
        seed_tables = {}
        quality = {}
        for seed in seeds:
            for compression in ('c23', 'c40'):
                name = f'{seed}/{compression}'
                subset = [r for r in videos if r['gallery_seed'] == seed
                          and r['compression'] == compression and r['role'] == 'evaluation_fake']
                seed_tables[name] = statistics(subset, run)
                attempts = [r for r in frames if r['gallery_seed'] == seed
                            and r['compression'] == compression and r['role'] == 'evaluation_fake']
                generated = len(attempts)
                evaluable = sum(bool(r.get('evaluable')) for r in attempts)
                fraction = evaluable / generated if generated else 0.0
                components = seed_tables[name]['evaluable_components']
                quality[name] = {
                    'generated_frames': generated, 'scored_frames': sum(r['status'] == 'ok' for r in attempts),
                    'evaluable_frames': evaluable, 'evaluable_frame_fraction': fraction,
                    'coverage_domain': 'non_oracle_fake_full_face_ap_all_methods',
                    'paired_secondary_available_frames': sum(r.get('paired_comparison_status') == 'ok' for r in attempts),
                    'paired_secondary_defined_ap_frames': sum(r.get('paired_metrics', {}).get('relation', {}).get('fake_minus_real_ap') is not None for r in attempts),
                    'undefined_ap_frames': sum(r['status'] == 'ok' and not r.get('evaluable') for r in attempts),
                    'failure_reasons': dict(Counter(r.get('reason', 'undefined_mask_ap')
                                                   for r in attempts if not r.get('evaluable'))),
                    'evaluable_components': components,
                    'coverage_gate_pass': fraction >= run.config.require('quality.min_evaluable_frame_fraction'),
                    'independence_gate_pass': components >= run.config.require('quality.min_independent_components'),
                    'gallery_gate_pass': indices[name]['status'] == 'ok',
                    'calibration_gate_pass': calibration[str(seed)]['status'] == 'ok',
                    'coverage_gate_sensitivity': {str(t): fraction >= t for t in (0.5, 0.6, 0.7, 0.8)},
                }
        averaged = average_seeds(videos, seeds)
        primary_tables = {compression: statistics([r for r in averaged if r['compression'] == compression
                                      and r['role'] == 'evaluation_fake'], run)
                          for compression in ('c23', 'c40')}
        transfer_rows = []
        compressed = {(r['video_id'], r['compression']): r for r in averaged if r['role'] == 'evaluation_fake'}
        for video in sorted({r['video_id'] for r in averaged if r['role'] == 'evaluation_fake'}):
            c23, c40 = compressed.get((video, 'c23')), compressed.get((video, 'c40'))
            if not c23 or not c40:
                continue
            transfer = {'video_id': video, 'component_id': c23['component_id']}
            for method in METHODS:
                transfer[f'{method}_c23'] = c23[f'{method}_ap']
                transfer[f'{method}_c40'] = c40[f'{method}_ap']
            transfer_rows.append(transfer)
        transfer = {method: bootstrap(transfer_rows, f'{method}_c40', f'{method}_c23', run) for method in METHODS}
        clean_real = {}
        for compression in ('c23', 'c40'):
            subset = [r for r in averaged if r['compression'] == compression and r['role'] == 'evaluation_real']
            clean_real[compression] = {m: {'ordinary_video_mean_fpr': float(np.mean(values)) if values else None,
                                           'videos': len(values)}
                                       for m in METHODS
                                       for values in [[r[f'{m}_fpr'] for r in subset if r.get(f'{m}_fpr') is not None]]}
        geometry_audit = {}
        for role in ('gallery', 'calibration', 'evaluation_fake', 'evaluation_real', 'paired_real'):
            for compression in ('c23', 'c40'):
                subset = [r for r in geometry if r['role'] == role and r['compression'] == compression]
                geometry_audit[f'{role}/{compression}'] = {
                    'generated': len(subset), 'geometry_ok': sum(r['status'] == 'ok' for r in subset),
                    'landmark_statuses': dict(Counter(r.get('landmark_status', {}).get('status', 'not_attempted') for r in subset)),
                    'failure_reasons': dict(Counter(r.get('reason') for r in subset if r['status'] != 'ok')),
                    'geometry_success_fraction': sum(r['status'] == 'ok' for r in subset) / len(subset) if subset else None,
                    'region_token_weight_mean_on_success': {name: float(np.mean(values)) if values else None
                        for name in ('mouth_core', 'perioral_ring', 'cheek_left', 'cheek_right', 'valid')
                        for values in [[r['region_token_weights'][name] for r in subset if r['status'] == 'ok']]},
                    'region_area_mean_on_success': {name: float(np.mean(values)) if values else None
                        for name in ('mouth_core', 'perioral_ring', 'cheek_left', 'cheek_right', 'valid')
                        for values in [[r['region_areas'][name] for r in subset if r['status'] == 'ok']]},
                }
        # Region summaries stay secondary and include every seed/compression.
        region_tables = {}
        for seed in seeds:
            for compression in ('c23', 'c40'):
                for region in run.config.require('statistics.secondary_domains'):
                    subset = []
                    for video in videos:
                        if video['gallery_seed'] != seed or video['compression'] != compression or video['role'] != 'evaluation_fake':
                            continue
                        metrics = video.get('region_metrics', {}).get(region, {})
                        subset.append({'component_id': video['component_id'],
                                       **{f'{m}_ap': metrics.get(m, {}).get('ap') for m in METHODS}})
                    region_tables[f'{seed}/{compression}/{region}'] = {
                        'ordinary_video_mean_ap': {m: float(np.mean(values)) if values else None
                            for m in METHODS for values in [[r[f'{m}_ap'] for r in subset if r[f'{m}_ap'] is not None]]},
                        'differences': {
                        name: bootstrap(subset, first, second, run)
                        for name, (first, second) in list(COMPARISONS.items())[:3]}}
        primary = primary_tables['c23']['differences']
        secondary = primary_tables['c40']['differences']
        primary_component_gate = all(t['evaluable_components'] >= run.config.require('quality.min_independent_components')
                                     for t in primary_tables.values())
        quality_ok = primary_component_gate and all(q['coverage_gate_pass'] and q['independence_gate_pass'] and q['gallery_gate_pass']
                         for q in quality.values())
        calibration_ok = all(q['calibration_gate_pass'] for q in quality.values())
        local_support = positive(primary['relation_minus_local']) and positive(primary['relation_minus_prevalence'])
        controls_support = positive(primary['relation_minus_template']) and positive(primary['relation_minus_broken'])
        paired_support = positive(primary['relation_fake_minus_real']) and positive(primary['relation_minus_real_template'])
        robust = positive(secondary['relation_minus_local']) and all(
            positive(seed_tables[f'{s}/c23']['differences']['relation_minus_local']) for s in seeds)
        if quality_ok and calibration_ok and local_support and controls_support and paired_support and robust:
            decision = 'PASS_CONTINUE'
        elif local_support:
            decision = 'PARTIAL_PASS_DIAGNOSTIC'
        elif not quality_ok:
            decision = 'PARTIAL_PASS_DIAGNOSTIC'
        else:
            decision = 'FAIL_ORACLE_DEPENDENT'
        margin = run.config.require('statistics.practical_margin_ap')
        conclusion = {
            'decision': decision, 'engineering_stages_completed': True, 'protocol_integrity_checked': True,
            'quality_ok': quality_ok, 'calibration_ok': calibration_ok,
            'all_seed_complete_case_component_gate': primary_component_gate,
            'local_baseline_support': local_support, 'controls_support': controls_support,
            'paired_secondary_support': paired_support, 'compression_and_seed_support': robust,
            'practical_margin_ap': margin,
            'practical_margin_interpretation': 'descriptive_post_R0_margin_not_equivalence_or_preregistered_noninferiority',
            'control_differences_within_margin': {name: abs(primary[name]['point']) <= margin
                if primary[name].get('status') == 'ok' else None
                for name in ('relation_minus_template', 'relation_minus_broken')},
            'continue_R2': decision == 'PASS_CONTINUE', 'official_test_locked': True,
            'failure_label_caveat': 'FAIL_ORACLE_DEPENDENT is a pilot label, not proof of causal oracle dependence',
            'no_publication_acceptance_claim': True,
        }
        audit = {'gallery_real_only': True, 'component_disjoint': True,
                 'identity_disjoint_status': 'source_video_components_only_biometric_identity_not_verified',
                 'gallery_selection_used_fake_scores_or_masks': False,
                 'paired_real_only_for_secondary_evaluation': True,
                 'independent_fake_crop_and_geometry': True,
                 'fake_mask_access_stage': 'evaluate_only_after_score_commit',
                 'official_test_locked': True, 'official_test_accessed_this_run': False,
                 'correspondence': run.config.require('gallery.correspondence'),
                 'R0_math_reused_but_reference_and_geometry_changed': True,
                 'coverage_threshold': run.config.require('quality.min_evaluable_frame_fraction'),
                 'coverage_threshold_origin': run.config.require('quality.coverage_origin'),
                 'sensitivity_interpretation': 'gate_only_same_samples_same_metrics_no_reselection',
                 'paired_compression_common_gallery': True}
        summary = {'created_utc': datetime.now(timezone.utc).isoformat(), 'decision': conclusion,
                   'primary_all_seed_complete_case': primary_tables, 'per_seed': seed_tables,
                   'c40_minus_c23_paired': transfer, 'clean_real_fpr_threshold_transfer': clean_real,
                   'quality': quality, 'geometry_audit': geometry_audit,
                   'region_tables': region_tables, 'protocol_audit': audit,
                   'all_seeds_reported': seeds, 'bootstrap_unit': 'source_video_connected_component',
                   'bootstrap_replicates': 2000, 'seed_not_an_independent_unit': True,
                   'limitations': ['independent crop grids are not landmark-registered',
                                   'AP on evaluable subset; missingness may be nonrandom',
                                   'one manipulation only: Face2Face; no cross-manipulation result claimed',
                                   'coverage and practical-margin choices were specified after R0',
                                   'all comparison CIs are descriptive, without multiple-testing correction']}
        write_json(run.path('finalize/summary.json'), summary)
        write_json(run.path('finalize/decision.json'), conclusion)
        write_json(run.path('finalize/quality_audit.json'), quality)
        write_json(run.path('finalize/protocol_audit.json'), audit)
        write_json(run.path('finalize/bootstrap_differences.json'), {
            'all_seed_primary': {c: t['differences'] for c, t in primary_tables.items()},
            'per_seed': {k: t['differences'] for k, t in seed_tables.items()},
            'c40_minus_c23': transfer, 'resampling_unit': 'component_id'})
        write_json(run.path('finalize/condition_diagnostics.json'), {
            'quality': quality, 'per_seed': seed_tables, 'regions': region_tables,
            'clean_real_fpr_transfer': clean_real,
            'zero_distance_queries': {m: sum(bool(r.get('score_audit', {}).get('zero_pairwise_distances', {}).get(m)) for r in frames)
                                       for m in ('relation', 'local_difference', 'broken_context')}})
        write_json(run.path('finalize/gallery_audit.json'), indices)
        write_json(run.path('finalize/semantic_region_summary.json'), geometry_audit)
        write_json(run.path('finalize/threshold_summary.json'), calibration)
        write_csv(run.path('finalize/all_seed_video_metrics.csv'), averaged,
                  fieldnames=['compression', 'role', 'video_id', 'component_id', 'source_group',
                              'target_id', 'source_id', 'seeds', 'evaluable',
                              *[f'{m}_{metric}' for m in METHODS for metric in ('ap', 'f1', 'fpr')],
                              'relation_ap_minus_prevalence', 'relation_fake_minus_real_ap',
                              'relation_minus_real_template_ap'])
        write_csv(run.path('finalize/source_group_metrics.csv'), component_rows(videos),
                  fieldnames=['gallery_seed', 'compression', 'component_id', 'videos',
                              *[f'{m}_mean_ap' for m in METHODS]])
        atomic_write_text(run.path('finalize/summary.md'), markdown(summary))
        atomic_write_text(run.path('finalize/analysis_report.md'), markdown(summary))
        stage.finish([])


def component_rows(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[int, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        if row['role'] == 'evaluation_fake' and row.get('evaluable'):
            groups[(row['gallery_seed'], row['compression'], row['component_id'])].append(row)
    return [{'gallery_seed': seed, 'compression': compression, 'component_id': component,
             'videos': len(group), **{f'{m}_mean_ap': float(np.mean([r[f'{m}_ap'] for r in group])) for m in METHODS}}
            for (seed, compression, component), group in sorted(groups.items())]


def markdown(summary: dict) -> str:
    lines = ['# R1 Non-Oracle Semantic Relation Localization Pilot', '',
             f"Decision: **{summary['decision']['decision']}**", '',
             'All registered seeds are reported. Primary table averages seeds per video only when all seeds are available.',
             'Ordinary video means below are not the component-weighted bootstrap estimand.', '',
             '| Condition | relation | local_difference | position_template | broken_context |',
             '|---|---:|---:|---:|---:|']
    for compression, table in summary['primary_all_seed_complete_case'].items():
        values = [table['methods'][m]['ap']['ordinary_video_mean'] for m in METHODS]
        lines.append(f'| {compression} | ' + ' | '.join(f'{v:.6f}' if v is not None else 'undefined' for v in values) + ' |')
    lines += ['', '## Component-weighted paired differences (AP percentage points)', '',
              '| Condition | Comparison | Difference | 95% CI | Components |', '|---|---|---:|---|---:|']
    for compression, table in summary['primary_all_seed_complete_case'].items():
        for name, value in table['differences'].items():
            if value['status'] == 'ok':
                lines.append(f'| {compression} | {name} | {100 * value["point"]:+.4f} | '
                             f'[{100 * value["ci95"][0]:+.4f}, {100 * value["ci95"][1]:+.4f}] | {value["groups"]} |')
            else:
                lines.append(f'| {compression} | {name} | undefined | {value["reason"]} | — |')
    lines += ['', '## Coverage and failures', '',
              '| Seed/condition | Evaluable/generated frames | Fraction | Quality gate |', '|---|---:|---:|---|']
    for name, q in summary['quality'].items():
        lines.append(f'| {name} | {q["evaluable_frames"]}/{q["generated_frames"]} | '
                     f'{100 * q["evaluable_frame_fraction"]:.2f}% | {q["coverage_gate_pass"]} |')
    lines += ['', 'Coverage threshold: ' + str(summary['protocol_audit']['coverage_threshold']),
              '50/60/70/80% sensitivity is a gate comparison on unchanged data, not new AP experiments.', '',
              '## Scope', '', 'Real-only, source-video-component-disjoint gallery. No test unlock, training, or target adaptation.',
              'Paired-real AP is a secondary diagnostic using independently generated real maps warped into fake source coordinates.',
              'Fake masks are accessed only after score commit. No best-seed selection. Undefined results remain in the audit.', '',
              '## Limitations', '', *['- ' + text for text in summary['limitations']], '',
              'Full per-seed, region, clean-real FPR, c23/c40 paired CIs and audit details: summary.json.',
              'A small control difference is descriptive, not a statistical equivalence claim.',
              'No conclusions about a face manifold, universal localization, or publication acceptance are made.', '']
    lines.extend(question_answers(summary))
    return '\n'.join(lines)


def comparison_description(value: dict) -> str:
    if value.get('status') != 'ok':
        return '未定义：' + value.get('reason', 'insufficient observations')
    lo, hi = value['ci95']
    support = 'CI 下界 > 0，有正向支持' if lo > 0 else (
        'CI 上界 < 0，方向为负' if hi < 0 else 'CI 跨越/触及 0，尚未区分方向')
    return (f"{100 * value['point']:+.4f} pp；95% CI [{100 * lo:+.4f}, {100 * hi:+.4f}]；"
            f"{value['rows']} 个视频、{value['groups']} 个独立来源组件；{support}。")


def question_answers(summary: dict) -> list[str]:
    """Explicit report answers; never equate a small difference with random error."""
    primary = summary['primary_all_seed_complete_case']['c23']['differences']
    decision = summary['decision']
    lines = ['', '## R1 关键问题逐项回答', '',
             '1. **是否去掉 oracle reference？** 是：评分只用组件隔离的训练真实图库。fake、reference、'
             'calibration 的裁剪、landmark 和 DINOv3 特征分别生成；配对真实帧只用于后置诊断。',
             '2. **图库是否 source-disjoint？** train 配对图的连通组件与留出集、校准集隔离。'
             '这不等同于已验证的人脸身份隔离；生物身份未核实。',
             '3. **relation 是否优于 local_difference？** ' + comparison_description(primary['relation_minus_local']),
             '4. **relation 是否优于 position_template？** ' + comparison_description(primary['relation_minus_template']),
             '5. **relation 是否优于 broken_context？** ' + comparison_description(primary['relation_minus_broken']),
             '6. **c23/c40 是否稳定？** 两个条件分开报告，c23 的 clean-real 阈值原样迁移到 c40。'
             '以下是相同视频上 c40−c23 的组件加权差异（不是两个非配对平均值相减）：']
    for method in METHODS:
        lines.append('   - ' + method + ': ' + comparison_description(summary['c40_minus_c23_paired'][method]))
    lines.append('7. **是否跨 gallery seed 稳定？** 每个 seed 均报告，不选择最优 seed。'
                 '以下为 c23 的 relation−local_difference；完整控制比较见 per_seed：')
    for seed in summary['all_seeds_reported']:
        lines.append(f'   - seed {seed}: ' + comparison_description(
            summary['per_seed'][f'{seed}/c23']['differences']['relation_minus_local']))
    lines.extend([
        '8. **是否排除了空间位置混杂？** ' + (
            '两个控制比较均有正向支持，但这仍不是对所有空间混杂的完全排除。'
            if decision['controls_support'] else
            '尚未排除。模板/破坏上下文对照仍须与 relation 并列解释，不能把小差异自动称为随机误差。'),
        '9. **是否排除了区域面积混杂？** 未完全排除。主指标使用各方法相同的全脸有效域；'
        '逐区域结果、区域像素面积和 token 权重均报告，但没有新增面积匹配或语义打乱实验。',
        '10. **真实/伪造 landmark 是否存在不对称失败？** 以下使用独立 evaluation_real 与 evaluation_fake '
        '的 geometry 成功率；不是只计算评分成功帧：'])
    for compression in ('c23', 'c40'):
        real = summary['geometry_audit'][f'evaluation_real/{compression}']
        fake = summary['geometry_audit'][f'evaluation_fake/{compression}']
        rrate, frate = real['geometry_success_fraction'], fake['geometry_success_fraction']
        difference = f'{100 * (frate - rrate):+.2f} pp' if rrate is not None and frate is not None else 'undefined'
        lines.append(f'   - {compression}: real {real["geometry_ok"]}/{real["generated"]}；'
                     f'fake {fake["geometry_ok"]}/{fake["generated"]}；fake−real 成功率 {difference}。')
    lines.extend([
        '11. **是否有 mask/test 泄漏？** 本流程只读取授权 train.json 及其视频。官方 test 保持锁定；'
        'mask 在 score 已提交后才由 evaluate 读取，不参与裁剪、区域、图库、评分或阈值拟合。',
        f'12. **最终判定？** {decision["decision"]}。工程完成、协议状态、主比较、控制比较和'
        '稳健性证据分开记录；FAIL_ORACLE_DEPENDENT 是试验标签，不是 oracle 因果依赖的证明。',
        '', f'实际贡献幅度以点估计/CI 原值报告；{100 * decision["practical_margin_ap"]:.2f} pp '
        '仅是用户在 R0 之后选择的描述性参考线，不是预注册等效界限，也不是统一最低提升要求。', ''])
    return lines
