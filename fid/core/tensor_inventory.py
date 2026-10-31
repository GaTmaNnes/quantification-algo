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
    kind: str = "parameter"
    metadata: Dict[str, Any] = field(default_factory=dict)


class TensorInventory:
    """A lightweight inventory for tensors discovered in a model."""

    def __init__(self) -> None:
        self._tensors: List[TensorRecord] = []

    @classmethod
    def from_model(
        cls,
        model: Any,
        *,
        include_parameters: bool = True,
        include_buffers: bool = True,
    ) -> "TensorInventory":
        """Build an inventory from a PyTorch module.

        Parameters are prioritized because they are the main object of interest for
        weight fingerprinting and redundancy detection. Buffers are also kept when
        present (e.g. running statistics in normalization layers).
        """
        inventory = cls()

        if include_parameters:
            for name, parameter in model.named_parameters():
                module_name = name.rsplit(".", 1)[0] if "." in name else ""
                inventory.add_tensor(
                    name,
                    tuple(parameter.shape),
                    str(parameter.dtype),
                    module_name,
                    kind="parameter",
                    requires_grad=parameter.requires_grad,
                )

        if include_buffers:
            for name, buffer in model.named_buffers():
                module_name = name.rsplit(".", 1)[0] if "." in name else ""
                inventory.add_tensor(
                    name,
                    tuple(buffer.shape),
                    str(buffer.dtype),
                    module_name,
                    kind="buffer",
                )

        return inventory

    def add_tensor(
        self,
        name: str,
        shape: tuple[int, ...],
        dtype: str,
        module: str,
        *,
        kind: str = "parameter",
        **metadata: Any,
    ) -> None:
        self._tensors.append(
            TensorRecord(
                name=name,
                shape=shape,
                dtype=dtype,
                module=module,
                kind=kind,
                metadata=metadata,
            )
        )

    def as_list(self) -> List[TensorRecord]:
        return list(self._tensors)

    def count(self) -> int:
        return len(self._tensors)

    def by_module(self) -> Dict[str, List[TensorRecord]]:
        grouped: Dict[str, List[TensorRecord]] = {}
        for tensor in self._tensors:
            grouped.setdefault(tensor.module, []).append(tensor)
        return grouped


__all__ = ["TensorInventory", "TensorRecord"]
