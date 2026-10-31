"""Redundancy detection and comparison logic."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np

from fid.core.tensor_inventory import TensorInventory

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

    def __init__(self, eps: float = 1e-8) -> None:
        self.similarity = FunctionalSimilarity(eps=eps)

    def compare_tensors(
        self,
        left: Any,
        right: Any,
        *,
        left_name: str = "left",
        right_name: str = "right",
    ) -> RedundancyCandidate:
        left_array = np.asarray(left, dtype=np.float64)
        right_array = np.asarray(right, dtype=np.float64)

        left_flat = left_array.reshape(-1)
        right_flat = right_array.reshape(-1)
        size = min(left_flat.size, right_flat.size)
        if size == 0:
            raise ValueError("Cannot compare empty tensors.")

        left_flat = left_flat[:size]
        right_flat = right_flat[:size]

        relative_error = self.similarity.relative_error(left_flat, right_flat)
        cosine_similarity = self.similarity.cosine_similarity(left_flat, right_flat)
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
        """Build fingerprints for all parameters and buffers of a PyTorch model."""
        inventory = TensorInventory.from_model(model)
        parameter_map = {name: value for name, value in model.named_parameters()}
        parameter_map.update({name: value for name, value in model.named_buffers()})

        fingerprints: List[WeightFingerprint] = []
        for record in inventory.as_list():
            tensor = parameter_map.get(record.name)
            if tensor is None:
                continue
            array = tensor.detach().cpu().numpy()
            if array.size == 0:
                continue
            fingerprints.append(self.summarize_fingerprint(array, name=record.name, module=record.module))
        return fingerprints

    def detect_similar_modules(self, model: Any, *, limit: int = 10) -> List[Dict[str, Any]]:
        """Compare parameter tensors across modules and return the strongest candidates."""
        grouped: Dict[str, List[tuple[str, Any]]] = defaultdict(list)
        for name, parameter in model.named_parameters():
            module_name = name.rsplit(".", 1)[0] if "." in name else ""
            grouped[module_name].append((name, parameter))

        results: List[Dict[str, Any]] = []
        module_names = sorted(grouped)
        for index, left_module in enumerate(module_names):
            for right_module in module_names[index + 1 :]:
                best: Dict[str, Any] | None = None
                for left_name, left_param in grouped[left_module]:
                    for right_name, right_param in grouped[right_module]:
                        if left_param.shape != right_param.shape:
                            continue
                        candidate = self.compare_tensors(
                            left_param.detach().cpu().numpy(),
                            right_param.detach().cpu().numpy(),
                            left_name=f"{left_module}.{left_name}",
                            right_name=f"{right_module}.{right_name}",
                        )
                        item = {
                            "left_module": left_module,
                            "right_module": right_module,
                            "left_name": left_name,
                            "right_name": right_name,
                            "relative_error": candidate.relative_error,
                            "cosine_similarity": candidate.cosine_similarity,
                            "score": candidate.score,
                            "notes": candidate.notes,
                        }
                        if best is None or item["score"] > best["score"]:
                            best = item
                if best is not None:
                    results.append(best)

        results.sort(key=lambda item: item["score"], reverse=True)
        return results[:limit]

    def detect_moe_experts(self, model: Any, *, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for candidate expert pairs within MoE-style modules."""
        expert_names: List[str] = []
        expert_tensors: Dict[str, Any] = {}
        for name, parameter in model.named_parameters():
            if "expert" in name.lower() or "moe" in name.lower():
                expert_names.append(name)
                expert_tensors[name] = parameter

        if len(expert_names) < 2:
            return []

        results: List[Dict[str, Any]] = []
        for index, left_name in enumerate(expert_names):
            for right_name in expert_names[index + 1 :]:
                left = expert_tensors[left_name]
                right = expert_tensors[right_name]
                if left.shape != right.shape:
                    continue
                candidate = self.compare_tensors(
                    left.detach().cpu().numpy(),
                    right.detach().cpu().numpy(),
                    left_name=left_name,
                    right_name=right_name,
                )
                results.append(
                    {
                        "left_expert": left_name,
                        "right_expert": right_name,
                        "relative_error": candidate.relative_error,
                        "cosine_similarity": candidate.cosine_similarity,
                        "score": candidate.score,
                        "notes": candidate.notes,
                    }
                )

        results.sort(key=lambda item: item["score"], reverse=True)
        return results[:limit]


__all__ = ["RedundancyCandidate", "RedundancyDetector"]
