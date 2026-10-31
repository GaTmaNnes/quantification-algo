"""Utilities to load models from Hugging Face and build a tensor inventory."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from fid.core.tensor_inventory import TensorInventory


@dataclass
class ModelSpec:
    """Metadata container for a loaded model instance."""

    name: str
    kind: str = "transformer"
    revision: str = "main"
    device: str = "cpu"
    dtype: str = "float32"
    metadata: Dict[str, Any] = field(default_factory=dict)
    model: Optional[Any] = None
    tokenizer: Optional[Any] = None
    inventory: Optional[TensorInventory] = None


class ModelLoader:
    """Load a Hugging Face model and expose its tensor inventory."""

    def __init__(self, model_name: str, *, revision: str = "main", device: str = "cpu") -> None:
        self.model_name = model_name
        self.revision = revision
        self.device = device
        self.model: Optional[Any] = None
        self.tokenizer: Optional[Any] = None

    def load(self, *, trust_remote_code: bool = False) -> ModelSpec:
        """Load a real model and build a tensor inventory.

        Strategy:
        1. prefer a causal LM for LLM workloads,
        2. fallback to a generic AutoModel,
        3. move to the target device,
        4. enumerate all parameters/buffers into a TensorInventory.
        """
        resolved_device = self.device
        if resolved_device == "cuda" and not torch.cuda.is_available():
            resolved_device = "cpu"

        model = None
        try:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                revision=self.revision,
                trust_remote_code=trust_remote_code,
            )
        except Exception:
            try:
                model = AutoModel.from_pretrained(
                    self.model_name,
                    revision=self.revision,
                    trust_remote_code=trust_remote_code,
                )
            except Exception as exc:  # pragma: no cover - network- or env-dependent
                raise RuntimeError(
                    f"Unable to load model '{self.model_name}' from Hugging Face. "
                    "Check the model name, access, and installed dependencies."
                ) from exc

        model.to(torch.device(resolved_device))
        model.eval()

        tokenizer = None
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name, revision=self.revision)
        except Exception:  # pragma: no cover - tokenizer is optional here
            tokenizer = None

        inventory = TensorInventory.from_model(model)
        self.model = model
        self.tokenizer = tokenizer

        metadata = {
            "loader": "transformers",
            "source": "huggingface",
            "num_parameters": sum(p.numel() for p in model.parameters()),
            "num_trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "inventory_count": inventory.count(),
        }

        return ModelSpec(
            name=self.model_name,
            kind="transformer",
            revision=self.revision,
            device=resolved_device,
            dtype=str(next(model.parameters()).dtype),
            metadata=metadata,
            model=model,
            tokenizer=tokenizer,
            inventory=inventory,
        )

    def list_modules(self) -> List[str]:
        """Return the list of module names when a real model is loaded."""
        if self.model is None:
            return ["embed_tokens", "self_attn", "mlp", "lm_head"]
        return [name for name, _ in self.model.named_modules()]


__all__ = ["ModelLoader", "ModelSpec"]
