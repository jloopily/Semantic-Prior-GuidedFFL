"""T0 regression tests for execution on the server; not run locally."""
from __future__ import annotations

import numpy as np

from src.cirlid.geometry import fractional_frame_index
from src.cirlid.t0 import BUDGETS, MANIPULATIONS, METHODS, _statistics, _pixel, _linear_quantile_inplace


def _rows():
    rows = []
    for method in METHODS:
        for compression in ('c23', 'c40'):
            for group_index, manipulation in enumerate(MANIPULATIONS):
                # Deliberately unequal manipulation sizes: test macro, not global pooling.
                for video in range(group_index + 1):
                    for budget_index, budget in enumerate(BUDGETS):
                        rows.append({'method': method, 'compression': compression,
                            'manipulation': manipulation, 'video_name': str(video),
                            'budget': budget, 'is_real': False,
                            'pixel_ap': 0.1 * group_index + 0.01 * budget_index,
                            'pixel_auroc': 0.6, 'f1': 0.3, 'iou': 0.2})
    return rows


def test_t0_macro_not_global_video_pool():
    aggregate, _ = _statistics(_rows(), 123)
    row = next(r for r in aggregate if r['metric'] == 'pixel_ap' and r['method'] == METHODS[0]
               and r['compression'] == 'c23' and r['budget'] == 'T1_center')
    assert np.isclose(row['mean'], 0.15)
    assert row['n_videos'] == 10


def test_t0_paired_draws_constant_effect():
    _, differences = _statistics(_rows(), 123)
    row = next(r for r in differences if r['metric'] == 'pixel_ap' and r['method'] == METHODS[0]
               and r['compression'] == 'c23' and r['comparison'] == 'T9_uniform - T3_frozen')
    assert np.isclose(row['absolute_delta'], 0.02)
    assert np.allclose(row['absolute_ci95'], [0.02, 0.02])
    assert row['shared_video_draws_all_budgets'] is True


def test_t0_undefined_metric_intersection():
    rows = _rows()
    for row in rows:
        if row['manipulation'] == 'NeuralTextures' and row['video_name'] == '0' and row['budget'] == 'T1_center':
            row['pixel_ap'] = None
    aggregate, _ = _statistics(rows, 123)
    rows = [r for r in aggregate if r['metric'] == 'pixel_ap' and r['method'] == METHODS[0]
            and r['compression'] == 'c23']
    assert len(rows) == 4
    assert all(r['n_videos'] == 9 for r in rows)
    assert all(r['undefined_videos']['NeuralTextures'] == ['0'] for r in rows)


def test_t0_short_video_budget_dedup():
    indices = [fractional_frame_index(2, p) for p in BUDGETS['T9_uniform']]
    assert len(set(indices)) == 2
    assert len(indices) - len(set(indices)) == 7


def test_t0_empty_mask_not_zero_ap():
    row = _pixel(np.array([0.1, 0.5]), np.zeros(2, dtype=np.uint8), 0.4)
    assert row['pixel_ap'] is None
    assert row['pixel_auroc'] is None
    assert row['undefined_reason'] == 'single_class_mask'
    assert row['f1'] == 0.0


def test_t0_all_positive_ap_defined_auroc_undefined():
    row = _pixel(np.array([0.1, 0.5]), np.ones(2, dtype=np.uint8), 0.4)
    assert row['pixel_ap'] == 1.0
    assert row['pixel_auroc'] is None


def test_t0_exact_linear_quantile():
    for values in ([0.1], [0.1, 0.2, 0.2, 0.4], [2.0, 0.0, 9.0, 4.0, 1.0]):
        pool = np.asarray(values, dtype=np.float32)
        expected = np.quantile(pool.astype(np.float64), 0.99, method='linear')
        assert np.isclose(_linear_quantile_inplace(pool.copy(), 0.99), expected, rtol=0, atol=1e-12)


def test_t0_rounding_preserves_frozen_half_up():
    # 0.5 maps to 1 for a two-frame video, unlike Python bankers-rounding.
    assert fractional_frame_index(2, 0.5) == 1
