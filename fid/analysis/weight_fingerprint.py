"""Weight fingerprinting utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np


@dataclass
class WeightFingerprint:
    """A simplified fingerprint summarizing tensor-level statistics."""

    name: str
    shape: tuple[int, ...]
    mean: float
    std: float
    l1_norm: float
    l2_norm: float
    sparsity: float
    metadata: Dict[str, Any]

    @classmethod
    def from_array(cls, name: str, array: Any, *, module: str = "") -> "WeightFingerprint":
        values = np.asarray(array, dtype=np.float64)
        flat = values.reshape(-1)
        mean = float(np.mean(flat))
        std = float(np.std(flat))
        l1_norm = float(np.sum(np.abs(flat)))
        l2_norm = float(np.linalg.norm(flat))
        sparsity = float(np.mean(np.abs(flat) < 1e-8))
        metadata = {"module": module}
        return cls(
            name=name,
            shape=tuple(values.shape),
            mean=mean,
            std=std,
            l1_norm=l1_norm,
            l2_norm=l2_norm,
            sparsity=sparsity,
            metadata=metadata,
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "shape": self.shape,
            "mean": self.mean,
            "std": self.std,
            "l1_norm": self.l1_norm,
            "l2_norm": self.l2_norm,
            "sparsity": self.sparsity,
            "metadata": self.metadata,
        }


__all__ = ["WeightFingerprint"]
