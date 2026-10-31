#!/usr/bin/env python3
"""CPU-only smoke test for FID inventory generation and fingerprinting."""

from __future__ import annotations

from torch import nn

from fid.analysis.redundancy_detector import RedundancyDetector
from fid.core.model_loader import ModelLoader
from fid.core.tensor_inventory import TensorInventory


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 8)


def test_tensor_inventory_from_model() -> None:
    model = TinyModel()
    inventory = TensorInventory.from_model(model)
    assert inventory.count() == 4
    names = {record.name for record in inventory.as_list()}
    assert {"fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"} <= names


def test_redundancy_detector_compare_tensors() -> None:
    detector = RedundancyDetector()
    left = [1.0, 2.0, 3.0]
    right = [1.02, 2.04, 3.06]
    candidate = detector.compare_tensors(left, right, left_name="a", right_name="b")
    assert candidate.left == "a"
    assert candidate.right == "b"
    assert candidate.score >= 0.0
    assert -1.0 <= candidate.cosine_similarity <= 1.0


def test_model_loader_from_torch_model() -> None:
    model = TinyModel()
    spec = ModelLoader.from_torch_model(model, name="tiny-model", device="cpu")
    assert spec.model is model
    assert spec.inventory is not None
    assert spec.inventory.count() == 4
    assert spec.num_parameters > 0
