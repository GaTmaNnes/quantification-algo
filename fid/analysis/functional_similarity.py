"""Functional similarity and representation comparison utilities."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


class FunctionalSimilarity:
    """Compute simplified functional similarity between two vectors or arrays."""

    def __init__(self, eps: float = 1e-8) -> None:
        self.eps = eps

    def relative_error(self, a: Any, b: Any) -> float:
        left = np.asarray(a, dtype=np.float64)
        right = np.asarray(b, dtype=np.float64)
        diff = np.linalg.norm(left - right)
        denom = np.linalg.norm(left) + self.eps
        return float(diff / denom)

    def cosine_similarity(self, a: Any, b: Any) -> float:
        left = np.asarray(a, dtype=np.float64).reshape(-1)
        right = np.asarray(b, dtype=np.float64).reshape(-1)
        numerator = float(np.dot(left, right))
        denom = np.linalg.norm(left) * np.linalg.norm(right)
        if np.isclose(denom, 0.0):
            return 0.0
        return float(numerator / denom)


__all__ = ["FunctionalSimilarity"]
