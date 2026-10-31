"""Quick example for FID analysis workflow."""

from __future__ import annotations

import numpy as np

from fid.analysis.functional_similarity import FunctionalSimilarity
from fid.analysis.redundancy_detector import RedundancyDetector


def main() -> None:
    left = np.random.randn(16, 16).astype(np.float64)
    right = left + 0.01 * np.random.randn(16, 16)

    detector = RedundancyDetector()
    candidate = detector.compare_tensors(left, right, left_name="A", right_name="B")
    print("Redundancy candidate:")
    print(candidate)

    sim = FunctionalSimilarity()
    print(f"Relative error: {sim.relative_error(left, right):.6f}")
    print(f"Cosine similarity: {sim.cosine_similarity(left, right):.6f}")


if __name__ == "__main__":
    main()
