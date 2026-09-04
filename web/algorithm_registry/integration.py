from __future__ import annotations

from typing import Any

from .models import InstalledAlgorithm
from .registry import list_installed


def imported_algorithms() -> tuple[InstalledAlgorithm, ...]:
    """Return currently installed packages for Streamlit page integration."""
    return list_installed()


def imported_animation_data() -> dict[str, dict[str, Any]]:
    entries: dict[str, dict[str, Any]] = {}
    for algorithm in imported_algorithms():
        animation = algorithm.manifest.animation
        if animation is None:
            continue

        label = algorithm.manifest.name
        if label in entries:
            label = f"{label} ({algorithm.manifest.algorithm_id})"
        entries[label] = {
            "title": algorithm.manifest.name,
            "video_path": str(algorithm.path / animation["file"]),
            "description": (
                animation.get("concept_markdown")
                or algorithm.manifest.summary
            ),
            "latex": animation.get("formula", ""),
            "symbols": animation.get("symbols", []),
            "highlights": animation.get("highlights", []),
            "viewing_flow": animation.get("viewing_flow", []),
            "derivation_steps": animation.get("derivation_steps", []),
            "imported": True,
        }
    return entries
