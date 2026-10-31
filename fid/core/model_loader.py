"""Utilities to load models from Hugging Face and inspect them."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

from fid.core.tensor_inventory import TensorInventory

try:  # pragma: no cover - dependency is optional at import time
    from transformers import AutoModel, AutoModelForCausalLM
except ImportError:  # pragma: no cover
    AutoModel = None
    AutoModelForCausalLM = None


@dataclass
class ModelSpec:
    """Resolved model instance with metadata and inventory."""

    name: str
    model: Any
    kind: str = "transformer"
    revision: str = "main"
    device: str = "cpu"
    dtype: str = "float32"
    inventory: Optional[TensorInventory] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.model.parameters())


class ModelLoader:
    """Load a Hugging Face model and expose a lightweight model specification."""

    def __init__(self, model_name: str, *, revision: str = "main", device: str = "cpu") -> None:
        self.model_name = model_name
        self.revision = revision
        self.device = device
        self.model: Any | None = None

    def _resolve_device(self) -> torch.device:
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device(self.device)

    def _load_pretrained_model(self) -> Any:
        if AutoModel is None or AutoModelForCausalLM is None:
            raise RuntimeError(
                "transformers is not installed. Install the project dependencies first: "
                "python -m pip install -r requirements.txt"
            )

        last_error: Exception | None = None
        loaders = [
            ("AutoModelForCausalLM", AutoModelForCausalLM),
            ("AutoModel", AutoModel),
        ]

        for loader_name, loader in loaders:
            try:
                model = loader.from_pretrained(
                    self.model_name,
                    revision=self.revision,
                    local_files_only=False,
                    trust_remote_code=True,
                )
                return model
            except Exception as exc:  # pragma: no cover - network or config can vary
                last_error = exc

        if last_error is not None:
            raise RuntimeError(f"Unable to load model '{self.model_name}' with transformers: {last_error}") from last_error
        raise RuntimeError(f"Unable to load model '{self.model_name}' with transformers.")

    def load(self) -> ModelSpec:
        """Load the model, move it to the target device and build an inventory."""
        device = self._resolve_device()
        model = self._load_pretrained_model()
        model.to(device)
        model.eval()

        self.model = model
        inventory = TensorInventory.from_model(model)
        metadata = {
            "loader": "transformers",
            "source": "huggingface",
            "num_parameters": sum(parameter.numel() for parameter in model.parameters()),
            "num_modules": len(list(model.named_modules())),
            "device": str(device),
        }
        return ModelSpec(
            name=self.model_name,
            model=model,
            kind="transformer",
            revision=self.revision,
            device=str(device),
            dtype=str(next(model.parameters()).dtype),
            inventory=inventory,
            metadata=metadata,
        )

    @classmethod
    def from_torch_model(cls, model: Any, *, name: str = "custom-model", device: str = "cpu") -> ModelSpec:
        loader = cls(name, device=device)
        loader.model = model
        inventory = TensorInventory.from_model(model)
        metadata = {
            "loader": "torch-module",
            "source": "local",
            "num_parameters": sum(parameter.numel() for parameter in model.parameters()),
            "num_modules": len(list(model.named_modules())),
            "device": str(torch.device(device)),
        }
        return ModelSpec(
            name=name,
            model=model,
            kind="transformer",
            device=str(torch.device(device)),
            dtype=str(next(model.parameters()).dtype),
            inventory=inventory,
            metadata=metadata,
        )

    def list_modules(self) -> List[str]:
        if self.model is None:
            return []
        return [name for name, _ in self.model.named_modules()]


__all__ = ["ModelLoader", "ModelSpec"]
