"""Redundancy detection and comparison logic."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

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


@dataclass
class ModuleSignature:
    """Aggregated statistics for a module-level comparison."""

    module: str
    mean_abs: float
    std: float
    l1_norm: float
    parameter_count: int
    tensor_count: int

    def distance_to(self, other: "ModuleSignature") -> float:
        """A lightweight scale-free distance between module signatures."""
        mean_gap = abs(self.mean_abs - other.mean_abs) / (abs(self.mean_abs) + 1e-8)
        std_gap = abs(self.std - other.std) / (self.std + 1e-8)
        l1_gap = abs(self.l1_norm - other.l1_norm) / (self.l1_norm + 1e-8)
        return float(mean_gap + std_gap + l1_gap) / 3.0


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

    def _build_module_summaries(self, model: Any) -> Dict[str, ModuleSignature]:
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

        summaries: Dict[str, ModuleSignature] = {}
        for module_name, fingerprints in by_module.items():
            total_params = sum(fp.numel for fp in fingerprints)
            mean_abs = float(np.mean([fp.abs_mean for fp in fingerprints]))
            std = float(np.mean([fp.std for fp in fingerprints]))
            l1_norm = float(np.sum([fp.l1_norm for fp in fingerprints]))
            summaries[module_name] = ModuleSignature(
                module=module_name,
                mean_abs=mean_abs,
                std=std,
                l1_norm=l1_norm,
                parameter_count=total_params,
                tensor_count=len(fingerprints),
            )

        return summaries

    def detect_similar_modules(self, model: Any, *, limit: int = 10) -> List[Dict[str, Any]]:
        """Compare module-level signatures heuristically to spot similar blocks."""
        summaries = self._build_module_summaries(model)
        results: List[Dict[str, Any]] = []
        module_names = sorted(summaries)

        for i, left_name in enumerate(module_names):
            left = summaries[left_name]
            for right_name in module_names[i + 1 :]:
                right = summaries[right_name]
                distance = left.distance_to(right)
                if distance < 0.5:
                    score = 1.0 - min(1.0, distance)
                    results.append(
                        {
                            "left_module": left_name,
                            "right_module": right_name,
                            "distance": float(distance),
                            "score": float(score),
                        }
                    )

        return sorted(results, key=lambda entry: entry["score"], reverse=True)[:limit]

    def detect_moe_experts(self, model: Any, *, limit: int = 10) -> List[Dict[str, Any]]:
        """Heuristically compare experts when module names suggest a MoE layout.

        This is intentionally shallow and should be treated as a first-pass signal,
        not as a proof of equivalence.
        """
        summaries = self._build_module_summaries(model)
        expert_modules = [name for name in summaries if "expert" in name.lower()]

        results: List[Dict[str, Any]] = []
        for i, left_name in enumerate(expert_modules):
            left = summaries[left_name]
            for right_name in expert_modules[i + 1 :]:
                right = summaries[right_name]
                distance = left.distance_to(right)
                score = 1.0 - min(1.0, distance)
                if distance < 0.75:
                    results.append(
                        {
                            "left_expert": left_name,
                            "right_expert": right_name,
                            "distance": float(distance),
                            "score": float(score),
                        }
                    )

        return sorted(results, key=lambda entry: entry["score"], reverse=True)[:limit]


__all__ = ["ModuleSignature", "RedundancyCandidate", "RedundancyDetector"]


"""Command line entry points for FID."""

from __future__ import annotations

import argparse

from fid.analysis.redundancy_detector import RedundancyDetector
from fid.core.model_loader import ModelLoader
from fid.reports.analysis_report import AnalysisReport


def main() -> None:
    parser = argparse.ArgumentParser(description="FID - Functional Information Deduplication")
    parser.add_argument("--model", default="sshleifer/tiny-gpt2", help="Hugging Face model identifier")
    parser.add_argument("--revision", default="main", help="Model revision or branch")
    parser.add_argument("--device", default="cpu", help="Torch device: cpu or cuda")
    parser.add_argument("--limit", type=int, default=10, help="Number of tensors to print in the summary")
    parser.add_argument("--top-similar", type=int, default=5, help="Number of similar module pairs to print")
    args = parser.parse_args()

    print("FID - Functional Information Deduplication")
    print(f"Loading model: {args.model} (rev={args.revision}, device={args.device})")

    loader = ModelLoader(args.model, revision=args.revision, device=args.device)
    spec = loader.load()

    inventory = spec.inventory
    if inventory is None:
        raise RuntimeError("Tensor inventory was not created. The model could not be loaded properly.")

    summary = {
        "model": spec.name,
        "device": spec.device,
        "parameters": spec.metadata.get("num_parameters", 0),
        "inventory_count": inventory.count(),
    }

    print("Model loaded successfully.")
    print(f"Kind: {spec.kind}")
    print(f"Parameters: {summary['parameters']}")
    print(f"Tensor inventory entries: {inventory.count()}")
    print(f"Modules discovered: {len(loader.list_modules())}")
    print("Sample tensors:")
    for record in inventory.as_list()[: args.limit]:
        print(f"- {record.name} | module={record.module} | shape={record.shape} | dtype={record.dtype}")

    detector = RedundancyDetector()
    fingerprints = detector.fingerprints_from_model(spec.model)
    if fingerprints:
        print("\nFingerprint summary:")
        for fp in fingerprints[: args.limit]:
            print(
                f"- {fp.name} | module={fp.module} | mean={fp.mean:.5f} | "
                f"std={fp.std:.5f} | l1={fp.l1_norm:.5f} | sparsity={fp.sparsity:.3f}"
            )

        similar = detector.detect_similar_modules(spec.model, limit=args.top_similar)
        moe_candidates = detector.detect_moe_experts(spec.model, limit=args.top_similar)
        report = AnalysisReport()
        rendered = report.render(summary, similar)
        print("\n" + rendered)
        if moe_candidates:
            print("\nPossible MoE expert candidates:")
            for candidate in moe_candidates:
                print(
                    f"- {candidate['left_expert']} vs {candidate['right_expert']} "
                    f"| score={candidate['score']:.4f}"
                )
    else:
        print("No parameter fingerprints could be computed for this model.")


if __name__ == "__main__":
    main()
