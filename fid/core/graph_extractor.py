"""Graph extraction utilities for FID."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass
class NodeInfo:
    """Descriptor of a parameter or module node within a model graph."""

    name: str
    module: str
    kind: str
    shape: Optional[Tuple[int, ...]] = None
    dtype: str = "float32"


class GraphExtractor:
    """Extract lightweight graph metadata from a PyTorch model."""

    def __init__(self, model: Any) -> None:
        self.model = model

    def extract_parameters(self) -> List[NodeInfo]:
        """Return a flattened view of all parameters in a model."""
        nodes: List[NodeInfo] = []
        for name, param in self.model.named_parameters():
            module_name = name.rsplit(".", 1)[0] if "." in name else ""
            nodes.append(
                NodeInfo(
                    name=name,
                    module=module_name,
                    kind="parameter",
                    shape=tuple(param.shape),
                    dtype=str(param.dtype),
                )
            )
        return nodes

    def module_summary(self) -> Dict[str, int]:
        """Summarize counts of parameters per module."""
        summary: Dict[str, int] = defaultdict(int)
        for node in self.extract_parameters():
            summary[node.module] += 1
        return dict(summary)

    def extract_modules(self) -> List[str]:
        """Return module names from the underlying model."""
        return list(self.model.named_modules())


__all__ = ["GraphExtractor", "NodeInfo"]
