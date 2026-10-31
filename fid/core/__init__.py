"""Minimal package exports for fid.core."""

from __future__ import annotations

from fid.core.model_loader import ModelLoader, ModelSpec
from fid.core.tensor_inventory import TensorInventory, TensorRecord

__all__ = ["ModelLoader", "ModelSpec", "TensorInventory", "TensorRecord"]
