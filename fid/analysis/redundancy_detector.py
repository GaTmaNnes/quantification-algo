"""Redundancy detection and comparison logic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np

from .functional_similarity import FunctionalSimilarity
from .weight_fingerprint import WeightFingerprint


@dataclass
class RedundancyCandidate:
    """Candidate pair flagged as potentially redundant."""

    left: str
    right: str
    relative_error: float
    cosine_similarity: float
    score: float
    notes: str = ""


class RedundancyDetector:
    """Minimal redundancy detector for preliminary model analysis."""

    def __init__(self) -> None:
        self.similarity = FunctionalSimilarity()

    def compare_tensors(self, left: Any, right: Any, *, left_name: str = "left", right_name: str = "right") -> RedundancyCandidate:
        relative_error = self.similarity.relative_error(left, right)
        cosine_similarity = self.similarity.cosine_similarity(left, right)
        score = 1.0 - min(1.0, relative_error)
        if cosine_similarity > 0.99:
            notes = "high representational similarity"
        elif relative_error < 0.05:
            notes = "low functional discrepancy"
        else:
            notes = "requires further validation"

        return RedundancyCandidate(
            left=left_name,
            right=right_name,
            relative_error=relative_error,
            cosine_similarity=cosine_similarity,
            score=score,
            notes=notes,
        )

    def summarize_fingerprint(self, tensor: Any, *, name: str = "tensor", module: str = "") -> WeightFingerprint:
        return WeightFingerprint.from_array(name, tensor, module=module)


__all__ = ["RedundancyCandidate", "RedundancyDetector"]
