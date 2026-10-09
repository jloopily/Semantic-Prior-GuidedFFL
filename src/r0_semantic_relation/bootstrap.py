"""Source-group clustered paired bootstrap for R0.

The unit resampled with replacement is the source group, not an individual
frame or an individual video.  Each draw contains the complete set of rows in
the selected source groups, and the estimand is the equally weighted mean of
source-group means.
"""
from __future__ import annotations

import numpy as np


def paired_group_bootstrap(
    rows,
    first: str,
    second: str,
    *,
    group_key: str = "source_group",
    replicates: int = 2000,
    seed: int = 20260930,
):
    groups: dict[str, list[float]] = {}
    for row in rows:
        if row.get(first) is None or row.get(second) is None:
            continue
        groups.setdefault(str(row[group_key]), []).append(float(row[first]) - float(row[second]))
    if len(groups) < 2:
        raise ValueError("at least two independent source groups are required for bootstrap CI")
    if int(replicates) < 1:
        raise ValueError("replicates must be positive")

    keys = sorted(groups)
    group_means = np.asarray([np.mean(groups[key]) for key in keys], dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = np.empty(int(replicates), dtype=np.float64)
    for i in range(int(replicates)):
        selected = rng.integers(0, len(keys), size=len(keys))
        # Resample source groups with replacement.  A group contributes its
        # complete within-group mean; frames/videos are never treated as the
        # independent bootstrap units.
        draws[i] = float(np.mean(group_means[selected]))
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return {
        "point": float(np.mean(group_means)),
        "ci95": [float(lo), float(hi)],
        "replicates": int(replicates),
        "seed": int(seed),
        "groups": len(keys),
        "rows": int(sum(len(values) for values in groups.values())),
        "group_sizes": {key: len(groups[key]) for key in keys},
        "resampling_unit": f"{group_key}_with_replacement",
        "group_key": group_key,
        "estimand": f"equal_weight_mean_of_{group_key}_means",
    }
