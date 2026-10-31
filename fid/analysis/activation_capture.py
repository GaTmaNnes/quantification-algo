"""Activation capture utilities for functional analysis."""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional


class ActivationCapture:
    """Capture intermediate activations by registering forward hooks."""

    def __init__(self, model: Any) -> None:
        self.model = model
        self.activations: Dict[str, Any] = {}
        self._handles = []

    def _hook(self, name: str) -> Callable[[Any, Any, Any], None]:
        def _inner(module: Any, inputs: Any, outputs: Any) -> None:
            self.activations[name] = outputs

        return _inner

    def capture(self, *args: Any, target_modules: Optional[Iterable[str]] = None, **kwargs: Any) -> Dict[str, Any]:
        """Run a forward pass and capture activations for selected modules."""
        self.activations.clear()

        for name, module in self.model.named_modules():
            if target_modules is not None and not any(name == target or name.startswith(f"{target}.") for target in target_modules):
                continue
            self._handles.append(module.register_forward_hook(self._hook(name)))

        try:
            self.model(*args, **kwargs)
        finally:
            for handle in self._handles:
                handle.remove()
            self._handles.clear()

        return dict(self.activations)


__all__ = ["ActivationCapture"]
