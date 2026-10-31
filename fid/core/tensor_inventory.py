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

    def as_list(self) -> List[TensorRecord]:
        return list(self._tensors)

    def count(self) -> int:
        return len(self._tensors)


__all__ = ["TensorInventory", "TensorRecord"]
