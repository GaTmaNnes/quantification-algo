"""Validation helpers for candidate redundancy checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class ValidationResult:
    """Report whether a candidate passes a lightweight validation rule."""

    name: str
    passed: bool
    relative_error: float
    cosine_similarity: float
    message: str = ""


class Validator:
    """Check preliminary candidate quality before deeper scientific analysis."""

    def __init__(self, relative_error_threshold: float = 0.05, cosine_threshold: float = 0.95) -> None:
        self.relative_error_threshold = relative_error_threshold
        self.cosine_threshold = cosine_threshold

    def validate(self, name: str, relative_error: float, cosine_similarity: float) -> ValidationResult:
        passed = relative_error <= self.relative_error_threshold and cosine_similarity >= self.cosine_threshold
        if passed:
            message = "Candidate looks promising; proceed to deeper validation."
        else:
            message = "Candidate is not yet validated; inspect structure and calibration."

        return ValidationResult(
            name=name,
            passed=passed,
            relative_error=relative_error,
            cosine_similarity=cosine_similarity,
            message=message,
        )


__all__ = ["ValidationResult", "Validator"]
