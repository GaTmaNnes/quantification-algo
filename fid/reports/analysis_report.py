"""Pretty-print report generation for FID analyses."""

from __future__ import annotations

from typing import Any, Dict, Iterable


class AnalysisReport:
    """Simple report builder for terminal output."""

    def __init__(self, title: str = "FID analysis report") -> None:
        self.title = title

    def render(self, summary: Dict[str, Any], candidates: Iterable[Dict[str, Any]] | None = None) -> str:
        lines = [self.title, "=" * len(self.title)]
        lines.append(f"model: {summary.get('model', 'unknown')}")
        lines.append(f"device: {summary.get('device', 'cpu')}")
        lines.append(f"parameters: {summary.get('parameters', 0)}")
        lines.append(f"inventory_count: {summary.get('inventory_count', 0)}")

        if candidates is not None:
            lines.append("\nSimilar module candidates:")
            for index, candidate in enumerate(candidates, start=1):
                left_name = candidate.get("left_module", candidate.get("left_expert", "?"))
                right_name = candidate.get("right_module", candidate.get("right_expert", "?"))
                lines.append(
                    f"{index}. {left_name} vs {right_name} | score={candidate.get('score', 0.0):.4f}"
                )

        return "\n".join(lines)


__all__ = ["AnalysisReport"]
