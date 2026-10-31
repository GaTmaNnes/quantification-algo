"""Simple similarity report helpers."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List


class SimilarityReport:
    """Render a compact human-readable summary of model comparison results."""

    def __init__(self, title: str = "FID similarity report") -> None:
        self.title = title

    def render(self, entries: Iterable[Dict[str, Any]]) -> str:
        lines = [self.title, "-" * len(self.title)]
        for idx, entry in enumerate(entries, start=1):
            left = entry.get("left", "left")
            right = entry.get("right", "right")
            score = entry.get("score", 0.0)
            relative_error = entry.get("relative_error", 0.0)
            cosine = entry.get("cosine_similarity", 0.0)
            lines.append(
                f"{idx}. {left} vs {right} | score={score:.4f} | rel_err={relative_error:.4f} | cosine={cosine:.4f}"
            )
        return "\n".join(lines)


__all__ = ["SimilarityReport"]
