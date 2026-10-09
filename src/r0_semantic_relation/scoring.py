"""Frozen lightweight relation and control scorers for R0.

The primary relation score is deliberately an oracle paired comparison. Fake
and target-real images each form their own mouth-to-cheek relation vector;
the score is the per-token change between those two vectors. The semantic
regions are defined once from target-real landmarks and reused for both images.
"""
from __future__ import annotations

import numpy as np

from .geometry import SemanticRegions


def _unit(x: np.ndarray) -> np.ndarray:
    value = np.asarray(x, dtype=np.float32)
    norm = np.linalg.norm(value, axis=-1, keepdims=True)
    return value / np.maximum(norm, 1e-12)


def _context(tokens: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, bool]:
    token_value = np.asarray(tokens, dtype=np.float32).reshape(784, -1)
    weight_value = np.asarray(weights, dtype=np.float32).reshape(-1)
    if weight_value.size != token_value.shape[0]:
        raise ValueError("context weights must have one value per token")
    total = float(weight_value.sum())
    if total <= 0.0:
        return np.zeros(token_value.shape[-1], dtype=np.float32), False
    context = np.sum(token_value * weight_value[:, None], axis=0) / total
    valid = bool(np.isfinite(context).all() and np.linalg.norm(context) > 1e-12)
    return context.astype(np.float32), valid


def _relation_vectors(query_tokens: np.ndarray, context_tokens: np.ndarray, regions: SemanticRegions) -> tuple[np.ndarray, bool, dict]:
    """Build two cosine-to-left/right-cheek relation channels."""
    query = np.asarray(query_tokens, dtype=np.float32).reshape(784, -1)
    context = np.asarray(context_tokens, dtype=np.float32).reshape(784, -1)
    left, left_valid = _context(context, regions.token_cheek_left)
    right, right_valid = _context(context, regions.token_cheek_right)
    audit = {
        "left_context_valid": left_valid,
        "right_context_valid": right_valid,
        "query_zero_feature_count": int(np.count_nonzero(np.linalg.norm(query, axis=1) <= 1e-12)),
        "context_zero_feature_count": int(np.count_nonzero(np.linalg.norm(context, axis=1) <= 1e-12)),
    }
    if not (left_valid and right_valid):
        return np.zeros((784, 2), dtype=np.float32), False, audit
    query_unit = _unit(query)
    relation = np.stack((query_unit @ _unit(left), query_unit @ _unit(right)), axis=1).astype(np.float32)
    return relation, True, audit


def relation_score(fake_tokens: np.ndarray, real_tokens: np.ndarray, regions: SemanticRegions) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return fake-vs-real relation change, valid domain, and audit.

    Fake uses fake query tokens and fake cheek context. Target-real uses real
    query tokens and real cheek context. The score is their two-channel L2
    relation change, restricted to mouth core plus perioral ring tokens.
    """
    fake = np.asarray(fake_tokens, dtype=np.float32).reshape(784, -1)
    real = np.asarray(real_tokens, dtype=np.float32).reshape(784, -1)
    fake_rel, fake_valid, fake_audit = _relation_vectors(fake, fake, regions)
    real_rel, real_valid, real_audit = _relation_vectors(real, real, regions)
    domain = ((regions.token_mouth_core + regions.token_perioral_ring) > 0) & (regions.token_valid > 0)
    if not (fake_valid and real_valid):
        return np.zeros(784, dtype=np.float32), np.zeros(784, dtype=bool), {
            "fake_context_valid": fake_valid,
            "real_context_valid": real_valid,
            "fake_relation": fake_audit,
            "real_relation": real_audit,
        }
    score = np.linalg.norm(fake_rel - real_rel, axis=1).astype(np.float32)
    score[~domain] = 0.0
    return score, domain, {
        "fake_context_valid": fake_valid,
        "real_context_valid": real_valid,
        "fake_relation": fake_audit,
        "real_relation": real_audit,
        "zero_feature_count": fake_audit["query_zero_feature_count"] + real_audit["query_zero_feature_count"],
    }


def local_difference_score(fake_tokens: np.ndarray, real_tokens: np.ndarray, regions: SemanticRegions) -> tuple[np.ndarray, np.ndarray]:
    """Content-only paired local feature difference baseline."""
    fake = np.asarray(fake_tokens, np.float32).reshape(784, -1)
    real = np.asarray(real_tokens, np.float32).reshape(784, -1)
    scores = np.linalg.norm(_unit(fake) - _unit(real), axis=1).astype(np.float32)
    return scores, regions.token_valid > 0


def position_template_score(regions: SemanticRegions) -> tuple[np.ndarray, np.ndarray]:
    """Position-only semantic template baseline."""
    domain = regions.token_valid > 0
    score = ((regions.token_mouth_core + regions.token_perioral_ring) > 0).astype(np.float32)
    return score, domain


def _validate_permutation(permutation: np.ndarray) -> np.ndarray:
    perm = np.asarray(permutation, dtype=np.int64).reshape(-1)
    if perm.size != 784 or not np.array_equal(np.sort(perm), np.arange(784)):
        raise ValueError("context permutation must be a permutation of 0..783")
    return perm


def broken_context_relation_score(fake_tokens: np.ndarray, real_tokens: np.ndarray, regions: SemanticRegions, *, permutation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Break only cheek context while preserving mouth/query token positions."""
    perm = _validate_permutation(permutation)
    fake = np.asarray(fake_tokens, dtype=np.float32).reshape(784, -1)
    real = np.asarray(real_tokens, dtype=np.float32).reshape(784, -1)
    fake_rel, fake_valid, _ = _relation_vectors(fake, fake[perm], regions)
    real_rel, real_valid, _ = _relation_vectors(real, real[perm], regions)
    domain = ((regions.token_mouth_core + regions.token_perioral_ring) > 0) & (regions.token_valid > 0)
    if not (fake_valid and real_valid):
        return np.zeros(784, dtype=np.float32), np.zeros(784, dtype=bool)
    score = np.linalg.norm(fake_rel - real_rel, axis=1).astype(np.float32)
    score[~domain] = 0.0
    return score, domain
