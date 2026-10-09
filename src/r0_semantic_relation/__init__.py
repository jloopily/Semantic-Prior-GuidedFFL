"""Isolated R0 semantic-relation oracle diagnostic.

This package is intentionally separate from the frozen CIR-LID/T0/M0/M1 code.
It uses paired target-real frames and must never be presented as a deployable
single-image detector.
"""

from .geometry import SemanticRegions, build_semantic_regions, polygon_is_simple
from .scoring import relation_score, local_difference_score, position_template_score

__all__ = [
    "SemanticRegions",
    "build_semantic_regions",
    "polygon_is_simple",
    "relation_score",
    "local_difference_score",
    "position_template_score",
]
