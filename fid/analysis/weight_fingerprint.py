"""Tensor inventory utilities for a model graph."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class TensorRecord:
    """Metadata for a single tensor in a model."""

    name: str
    shape: tuple[int, ...]
    dtype: str
    module: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class TensorInventory:
    """A lightweight inventory for tensors discovered in a model."""

    def __init__(self) -> None:
        self._tensors: List[TensorRecord] = []

    def add_tensor(self, name: str, shape: tuple[int, ...], dtype: str, module: str, **metadata: Any) -> None:
        self._tensors.append(
            TensorRecord(
                name=name,
                shape=shape,
                dtype=dtype,
                module=module,
                metadata=metadata,
            )
        )

    @classmethod
    def from_model(cls, model: Any) -> "TensorInventory":
        """Populate an inventory from a real PyTorch or HF model."""
        inventory = cls()

        for name, param in model.named_parameters():
            module_name = name.rsplit(".", 1)[0] if "." in name else ""
            inventory.add_tensor(
                name=name,
                shape=tuple(param.shape),
                dtype=str(param.dtype),
                module=module_name,
                kind="parameter",
            )

        if hasattr(model, "named_buffers"):
            for name, buffer in model.named_buffers():
                if buffer is None:
                    continue
                module_name = name.rsplit(".", 1)[0] if "." in name else ""
                inventory.add_tensor(
                    name=name,
                    shape=tuple(buffer.shape),
                    dtype=str(buffer.dtype),
                    module=module_name,
                    kind="buffer",
                )

        return inventory

    def as_list(self) -> List[TensorRecord]:
        return list(self._tensors)

    def count(self) -> int:
        return len(self._tensors)

    def group_by_module(self) -> Dict[str, List[TensorRecord]]:
        grouped: Dict[str, List[TensorRecord]] = {}
        for tensor in self._tensors:
            grouped.setdefault(tensor.module, []).append(tensor)
        return grouped


__all__ = ["TensorInventory", "TensorRecord"]
