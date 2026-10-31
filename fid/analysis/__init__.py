"""Redundancy detection and comparison logic."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np

from fid.analysis.weight_fingerprint import WeightFingerprint
from fid.core.tensor_inventory import TensorInventory


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
    """Lightweight redundancy detector for early model analysis."""

    def __init__(self, *, eps: float = 1e-8) -> None:
        self.eps = eps

    def relative_error(self, left: Any, right: Any) -> float:
        left_array = np.asarray(left, dtype=np.float64).reshape(-1)
        right_array = np.asarray(right, dtype=np.float64).reshape(-1)
        diff = np.linalg.norm(left_array - right_array)
        denom = np.linalg.norm(left_array) + self.eps
        return float(diff / denom)

    def cosine_similarity(self, left: Any, right: Any) -> float:
        left_array = np.asarray(left, dtype=np.float64).reshape(-1)
        right_array = np.asarray(right, dtype=np.float64).reshape(-1)
        numerator = float(np.dot(left_array, right_array))
        denom = np.linalg.norm(left_array) * np.linalg.norm(right_array)
        if np.isclose(denom, 0.0):
            return 0.0
        return float(numerator / denom)

    def compare_tensors(self, left: Any, right: Any, *, left_name: str = "left", right_name: str = "right") -> RedundancyCandidate:
        relative_error = self.relative_error(left, right)
        cosine_similarity = self.cosine_similarity(left, right)
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

    def fingerprints_from_model(self, model: Any) -> List[WeightFingerprint]:
        inventory = TensorInventory.from_model(model)
        fingerprints: List[WeightFingerprint] = []
        for record in inventory.as_list():
            parameter = None
            try:
                parameter = model.get_parameter(record.name)
            except Exception:
                parameter = None
            if parameter is not None:
                fingerprints.append(WeightFingerprint.from_parameter(record.name, parameter, module=record.module))
        return fingerprints

    def detect_similar_modules(self, model: Any, *, limit: int = 10) -> List[Dict[str, Any]]:
        """Compare module-level signatures heuristically to spot similar blocks."""
        inventory = TensorInventory.from_model(model)
        by_module: Dict[str, List[WeightFingerprint]] = defaultdict(list)

        for record in inventory.as_list():
            parameter = None
            try:
                parameter = model.get_parameter(record.name)
            except Exception:
                parameter = None
            if parameter is not None:
                by_module[record.module].append(
                    WeightFingerprint.from_parameter(record.name, parameter, module=record.module)
                )

        results: List[Dict[str, Any]] = []
        module_names = sorted(by_module)
        for i, left_name in enumerate(module_names):
            left = by_module[left_name]
            if not left:
                continue
            left_mean = np.mean([fp.abs_mean for fp in left])
            left_std = np.mean([fp.std for fp in left])
            for right_name in module_names[i + 1 :]:
                right = by_module[right_name]
                if not right:
                    continue
                right_mean = np.mean([fp.abs_mean for fp in right])
                right_std = np.mean([fp.std for fp in right])

                relative_gap = abs(left_mean - right_mean) / (abs(left_mean) + self.eps)
                variance_gap = abs(left_std - right_std) / (left_std + self.eps)
                if relative_gap < 0.5 and variance_gap < 0.5:
                    results.append(
                        {
                            "left_module": left_name,
                            "right_module": right_name,
                            "mean_abs_gap": float(relative_gap),
                            "std_gap": float(variance_gap),
                            "score": float(1.0 - min(1.0, relative_gap + variance_gap)),
                        }
                    )

        return sorted(results, key=lambda entry: entry["score"], reverse=True)[:limit]


__all__ = ["RedundancyCandidate", "RedundancyDetector"]
