"""Tests for the FID tensor inventory."""

from __future__ import annotations

from torch import nn

from fid.core.tensor_inventory import TensorInventory


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear1 = nn.Linear(4, 8)
        self.linear2 = nn.Linear(8, 2)


def test_tensor_inventory_from_model() -> None:
    model = TinyModel()
    inventory = TensorInventory.from_model(model)

    assert inventory.count() == 4
    names = {tensor.name for tensor in inventory.as_list()}
    assert "linear1.weight" in names
    assert "linear1.bias" in names
    assert "linear2.weight" in names
    assert "linear2.bias" in names


def test_group_by_module() -> None:
    model = TinyModel()
    inventory = TensorInventory.from_model(model)
    grouped = inventory.group_by_module()

    assert "linear1" in grouped
    assert "linear2" in grouped
    assert len(grouped["linear1"]) == 2
    assert len(grouped["linear2"]) == 2
