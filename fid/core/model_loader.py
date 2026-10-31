"""Utilities to load models from Hugging Face and test checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class ModelSpec:
    """Basic metadata container for a model instance."""

    name: str
    kind: str = "transformer"
    revision: str = "main"
    device: str = "cpu"
    dtype: str = "float32"
    metadata: Optional[Dict[str, Any]] = None


class ModelLoader:
    """Minimal model loader interface for research workflows."""

    def __init__(self, model_name: str, *, revision: str = "main", device: str = "cpu") -> None:
        self.model_name = model_name
        self.revision = revision
        self.device = device

    def load(self) -> ModelSpec:
        """Return a lightweight representation of a model.

        This is a prototype implementation intentionally designed to be
        extended with Hugging Face or local checkpoint loading.
        """
        return ModelSpec(
            name=self.model_name,
            kind="transformer",
            revision=self.revision,
            device=self.device,
            dtype="float32",
            metadata={"loader": "prototype", "source": "not_loaded"},
        )

    def list_modules(self) -> List[str]:
        """Return a placeholder list of module names."""
        return ["embed_tokens", "self_attn", "mlp", "lm_head"]


__all__ = ["ModelLoader", "ModelSpec"]
