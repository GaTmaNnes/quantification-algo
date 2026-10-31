"""Tests for redundancy comparisons."""

from __future__ import annotations

import numpy as np

from fid.analysis.redundancy_detector import RedundancyDetector


def test_compare_tensors_similarity() -> None:
    detector = RedundancyDetector()
    left = np.ones((4, 4), dtype=np.float64)
    right = 1.05 * np.ones((4, 4), dtype=np.float64)
    candidate = detector.compare_tensors(left, right, left_name="a", right_name="b")

    assert candidate.score >= 0.0
    assert candidate.left == "a"
    assert candidate.right == "b"


def test_module_signature_distance() -> None:
    detector = RedundancyDetector()
    left = np.ones((4, 4), dtype=np.float64)
    right = 1.0 + 0.01 * np.ones((4, 4), dtype=np.float64)
    result = detector.compare_tensors(left, right)

    assert result.relative_error >= 0.0
    assert result.cosine_similarity <= 1.0
    assert result.cosine_similarity >= -1.0
