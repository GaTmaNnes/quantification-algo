#!/usr/bin/env python3
"""CPU-only smoke test for inventory generation and fingerprinting.

This script validates the code paths intended for low-resource local analysis
without requiring GPU or internet access to a remote model.
"""

from __future__ import annotations

import torch
from torch import nn

from fid.analysis.redundancy_detector import RedundancyDetector
from fid.core.tensor_inventory import TensorInventory


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 8)


def main() -> None:
    model = TinyModel().to("cpu")
    inventory = TensorInventory.from_model(model)
    detector = RedundancyDetector()

    print("CPU smoke test: FID inventory + fingerprint pipeline")
    print(f"Inventory count: {inventory.count()}")
    print(f"Modules: {sorted(set(t.module for t in inventory.as_list()))}")

    parameter = model.fc1.weight
    fingerprint = detector.summarize_fingerprint(parameter.detach().cpu().numpy(), name="fc1.weight", module="fc1")
    print(
        "Fingerprint sample: "
        f"mean={fingerprint.mean:.4f}, std={fingerprint.std:.4f}, "
        f"l1={fingerprint.l1_norm:.4f}, sparsity={fingerprint.sparsity:.4f}"
    )

    candidates = detector.detect_similar_modules(model, limit=5)
    print(f"Similar-module candidates: {len(candidates)}")
    for candidate in candidates:
        print(
            f"- {candidate['left_module']} vs {candidate['right_module']} "
            f"score={candidate['score']:.4f}"
        )


if __name__ == "__main__":
    main()
