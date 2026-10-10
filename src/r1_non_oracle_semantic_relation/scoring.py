"""R0 pairwise mathematics with independent geometry and equal-video references.

Reference correspondence is the SAME normalized crop-grid index, not a claim
of landmark registration. Changing reference geometry/aggregation is explicit.
No masks, paired real frames, labels, or target-conditioned reference selection
are accepted by the scoring API.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from src.r0_semantic_relation.geometry import SemanticRegions
from src.r0_semantic_relation.scoring import _relation_vectors, _unit, position_template_score
from . import METHODS, SEMANTIC_CONTROL, SEMANTIC_CONTROL_METHODS


@dataclass(frozen=True)
class Descriptor:
    unit_tokens: np.ndarray
    relation: np.ndarray
    broken: np.ndarray


def context_derangement(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    while True:
        permutation = rng.permutation(784)
        if not np.any(permutation == np.arange(784)):
            return permutation


def describe(tokens: np.ndarray, regions: SemanticRegions, permutation: np.ndarray) -> Descriptor:
    value = np.asarray(tokens, dtype=np.float32)
    if value.shape != (784, 768) or not np.isfinite(value).all():
        raise ValueError('R1 requires finite DINOv3 [784,768] features')
    perm = np.asarray(permutation, dtype=np.int64)
    if perm.shape != (784,) or not np.array_equal(np.sort(perm), np.arange(784)):
        raise ValueError('invalid context permutation')
    if np.any(perm == np.arange(784)):
        raise ValueError('broken context requires a derangement')
    relation, valid, _ = _relation_vectors(value, value, regions)
    broken, broken_valid, _ = _relation_vectors(value, value[perm], regions)
    if not (valid and broken_valid):
        raise ValueError('one or both cheek context descriptors are invalid')
    if not np.any((regions.token_mouth_core + regions.token_perioral_ring) > 0):
        raise ValueError('empty mouth/perioral score support')
    return Descriptor(_unit(value), relation, broken)


def pair_distances(query: Descriptor, reference: Descriptor) -> dict[str, np.ndarray]:
    return {
        'relation': np.linalg.norm(query.relation - reference.relation, axis=1).astype(np.float32),
        'local_difference': np.linalg.norm(query.unit_tokens - reference.unit_tokens, axis=1).astype(np.float32),
        'broken_context': np.linalg.norm(query.broken - reference.broken, axis=1).astype(np.float32),
    }


def score_gallery(query: Descriptor, regions: SemanticRegions,
                  references: Iterable[tuple[str, Iterable[Descriptor]]], *,
                  methods: tuple[str, ...] = METHODS) -> tuple[dict[str, np.ndarray], dict]:
    """Streaming distances: mean over each video's frames, then mean over videos.

    Never distance to an averaged descriptor, nearest-reference selection, or
    per-query selection. References must have been role-checked by the caller.
    """
    if methods not in (METHODS, SEMANTIC_CONTROL_METHODS):
        raise ValueError('unsupported R1 method registry')
    # The supplement is derived from the final local map; no second reference
    # traversal, distance calculation, or change in reference/video weights.
    totals = {name: np.zeros(784, dtype=np.float64)
              for name in ('relation', 'local_difference', 'broken_context')}
    counts: dict[str, int] = {}
    zero_distances = {name: 0 for name in totals}
    for video_id, frames in references:
        if video_id in counts:
            raise ValueError('duplicate reference video would alter equal-video weights')
        within = {name: np.zeros(784, dtype=np.float64) for name in totals}
        count = 0
        for reference in frames:
            distances = pair_distances(query, reference)
            for name in totals:
                within[name] += distances[name]
                zero_distances[name] += int(np.count_nonzero(distances[name] <= 1e-12))
            count += 1
        if not count:
            raise ValueError('gallery video has zero eligible reference frames')
        counts[video_id] = count
        for name in totals:
            totals[name] += within[name] / count
    if not counts:
        raise ValueError('empty reference gallery')
    maps = {name: (value / len(counts)).astype(np.float32) for name, value in totals.items()}
    support = ((regions.token_mouth_core + regions.token_perioral_ring) > 0) & (regions.token_valid > 0)
    maps['relation'][~support] = 0
    maps['broken_context'][~support] = 0
    maps['position_template'], _ = position_template_score(regions)
    if SEMANTIC_CONTROL in methods:
        maps[SEMANTIC_CONTROL] = maps['local_difference'].copy()
        maps[SEMANTIC_CONTROL][~support] = 0
    maps = {name: maps[name] for name in methods}
    return maps, {'gallery_videos': len(counts), 'frames_per_video': counts,
                  'methods': list(methods),
                  'semantic_local_control': {
                      'enabled': SEMANTIC_CONTROL in methods,
                      'base_method': 'local_difference',
                      'derivation': 'copy_aggregated_local_then_zero_outside_query_relation_token_support',
                      'same_reference_frames_and_weights': True,
                      'same_token_support_as_relation': True,
                      'mask_used': False,
                      'evaluation_domain': 'full_face_valid_domain',
                  },
                  'aggregation': 'mean_frame_distances_then_equal_video_mean',
                  'correspondence': 'same_normalized_independent_crop_token_index',
                  'score_support_tokens': int(support.sum()),
                  'full_face_valid_tokens': int(np.count_nonzero(regions.token_valid)),
                  'zero_pairwise_distances': zero_distances,
                  'query_zero_feature_tokens': int(np.count_nonzero(np.linalg.norm(query.unit_tokens, axis=1) <= 1e-12)),
                  'score_ranges': {name: [float(value.min()), float(value.max())] for name, value in maps.items()}}
