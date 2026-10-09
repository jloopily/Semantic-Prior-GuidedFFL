"""Machine-readable frozen Gate A/B/D decisions."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Mapping


@dataclass(frozen=True)
class GateDecision:
    name: str
    passed: bool
    checks: Mapping[str, bool]
    observed: Mapping[str, float]
    thresholds: Mapping[str, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def gate_alpha(median_sp_ratio: float, paired_ci_lower: float, *, ratio_min: float = 0.50, ci_lower_min: float = 0.0) -> GateDecision:
    checks = {
        "median_sp_ratio": median_sp_ratio >= ratio_min,
        "paired_ci_lower": paired_ci_lower > ci_lower_min,
    }
    return GateDecision(
        "A-alpha",
        all(checks.values()),
        checks,
        {"median_sp_ratio": median_sp_ratio, "paired_ci_lower": paired_ci_lower},
        {"ratio_min": ratio_min, "ci_lower_min_exclusive": ci_lower_min},
    )


def gate_beta(slope_ci_upper: float, auroc_drop_at_0_5: float, *, slope_ci_upper_max: float = 0.0, drop_min: float = 0.05) -> GateDecision:
    checks = {
        "negative_slope_ci": slope_ci_upper < slope_ci_upper_max,
        "auroc_drop": auroc_drop_at_0_5 >= drop_min,
    }
    return GateDecision(
        "A-beta",
        all(checks.values()),
        checks,
        {"slope_ci_upper": slope_ci_upper, "auroc_drop_at_0_5": auroc_drop_at_0_5},
        {"slope_ci_upper_max_exclusive": slope_ci_upper_max, "drop_min": drop_min},
    )


def gate_synthetic(relative_q_reduction: float, *, reduction_min: float = 0.10) -> GateDecision:
    checks = {"relative_q_reduction": relative_q_reduction >= reduction_min}
    return GateDecision(
        "B-synthetic",
        all(checks.values()),
        checks,
        {"relative_q_reduction": relative_q_reduction},
        {"reduction_min": reduction_min},
    )


def gate_b(delta_pixel_ap: float, paired_ci_lower: float, *, delta_min: float = 0.02, ci_lower_min: float = 0.0) -> GateDecision:
    checks = {"delta_pixel_ap": delta_pixel_ap >= delta_min, "paired_ci_lower": paired_ci_lower > ci_lower_min}
    return GateDecision(
        "B-real",
        all(checks.values()),
        checks,
        {"delta_pixel_ap": delta_pixel_ap, "paired_ci_lower": paired_ci_lower},
        {"delta_min": delta_min, "ci_lower_min_exclusive": ci_lower_min},
    )


def gate_d(
    delta_vs_sp: float,
    ci_lower_vs_sp: float,
    delta_vs_c2: float,
    ci_lower_vs_c2: float,
    *,
    delta_min: float = 0.02,
    ci_lower_min: float = 0.0,
) -> GateDecision:
    checks = {
        "delta_vs_sp": delta_vs_sp >= delta_min,
        "ci_lower_vs_sp": ci_lower_vs_sp > ci_lower_min,
        "delta_vs_c2": delta_vs_c2 >= delta_min,
        "ci_lower_vs_c2": ci_lower_vs_c2 > ci_lower_min,
    }
    return GateDecision(
        "D",
        all(checks.values()),
        checks,
        {
            "delta_vs_sp": delta_vs_sp,
            "ci_lower_vs_sp": ci_lower_vs_sp,
            "delta_vs_c2": delta_vs_c2,
            "ci_lower_vs_c2": ci_lower_vs_c2,
        },
        {"delta_min": delta_min, "ci_lower_min_exclusive": ci_lower_min},
    )
