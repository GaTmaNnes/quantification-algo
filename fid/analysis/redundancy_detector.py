"""Weight fingerprinting utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np


@dataclass
class WeightFingerprint:
    """A small but useful summary of a weight tensor."""

    name: str
    module: str
    shape: tuple[int, ...]
    dtype: str
    numel: int
    mean: float
    std: float
    variance: float
    abs_mean: float
    l1_norm: float
    l2_norm: float
    max_abs: float
    sparsity: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_array(cls, name: str, array: Any, *, module: str = "", dtype: str = "float32") -> "WeightFingerprint":
        values = np.asarray(array, dtype=np.float64)
        flat = values.reshape(-1)
        mean = float(np.mean(flat))
        std = float(np.std(flat))
        variance = float(np.var(flat))
        abs_mean = float(np.mean(np.abs(flat)))
        l1_norm = float(np.sum(np.abs(flat)))
        l2_norm = float(np.linalg.norm(flat))
        max_abs = float(np.max(np.abs(flat))) if flat.size else 0.0
        sparsity = float(np.mean(np.abs(flat) < 1e-8)) if flat.size else 0.0
        return cls(
            name=name,
            module=module,
            shape=tuple(values.shape),
            dtype=dtype,
            numel=int(flat.size),
            mean=mean,
            std=std,
            variance=variance,
            abs_mean=abs_mean,
            l1_norm=l1_norm,
            l2_norm=l2_norm,
            max_abs=max_abs,
            sparsity=sparsity,
            metadata={"module": module},
        )

    @classmethod
    def from_parameter(cls, name: str, parameter: Any, *, module: str = "") -> "WeightFingerprint":
        values = parameter.detach().cpu().numpy()
        return cls.from_array(name, values, module=module, dtype=str(parameter.dtype))

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "module": self.module,
            "shape": self.shape,
            "dtype": self.dtype,
            "numel": self.numel,
            "mean": self.mean,
            "std": self.std,
            "variance": self.variance,
            "abs_mean": self.abs_mean,
            "l1_norm": self.l1_norm,
            "l2_norm": self.l2_norm,
            "max_abs": self.max_abs,
            "sparsity": self.sparsity,
            "metadata": self.metadata,
        }


__all__ = ["WeightFingerprint"]
