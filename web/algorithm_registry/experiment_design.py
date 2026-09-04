from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class ExperimentDesignPreset:
    preset_id: str
    label: str
    description: str
    design: Mapping[str, Any]


_FROZEN_LAKE_4X4 = ExperimentDesignPreset(
    preset_id="frozen-lake-4x4-v1",
    label="Standard 4×4 FrozenLake",
    description=(
        "A deterministic 4×4 teaching grid with Start, safe ice, holes, and Goal."
    ),
    design={
        "provenance": {
            "type": "platform_preset",
            "preset_id": "frozen-lake-4x4-v1",
            "label": "Standard 4×4 FrozenLake",
            "note": (
                "Platform-supplied teaching scenario; it is not claimed to be "
                "an exact passage from the uploaded source."
            ),
        },
        "task": {
            "mission": (
                "Cross the frozen lake from Start (S) to Goal (G) without "
                "falling into a Hole (H)."
            ),
            "dynamics": [
                "Actions are deterministic; the agent moves left, down, right, or up.",
                "Moving beyond the grid keeps the agent in the current cell.",
                "Goal and Hole cells end the episode.",
            ],
            "rewards": [
                "Goal: +1 and the episode ends.",
                "Safe cell or boundary: 0.",
                "Hole: 0 and the episode ends.",
            ],
        },
        "environment_map": {
            "kind": "grid",
            "layout": ["SFFF", "FHFH", "FFFH", "HFFG"],
            "legend": {
                "S": {
                    "label": "START", "role": "start", "terminal": False,
                    "color": "#8ecae6",
                },
                "F": {
                    "label": "ICE", "role": "normal", "terminal": False,
                    "icon": "❄️",
                },
                "H": {
                    "label": "HOLE", "role": "hazard", "terminal": True,
                    "color": "#444444", "text_color": "#ffffff",
                },
                "G": {
                    "label": "GOAL", "role": "goal", "terminal": True,
                    "color": "#66cc66",
                },
            },
            "actions": {
                "0": {"label": "Left", "arrow": "←"},
                "1": {"label": "Down", "arrow": "↓"},
                "2": {"label": "Right", "arrow": "→"},
                "3": {"label": "Up", "arrow": "↑"},
            },
        },
        "transition_model": {
            "kind": "deterministic_grid",
            "out_of_bounds": "stay",
            "start_symbol": "S",
            "goal_symbols": ["G"],
            "hazard_symbols": ["H"],
            "terminal_symbols": ["G", "H"],
            "step_reward": 0.0,
            "goal_reward": 1.0,
            "hazard_reward": 0.0,
        },
    },
)


_CLIFF_WALKING_4X12 = ExperimentDesignPreset(
    preset_id="cliff-walking-4x12-v1",
    label="Standard 4×12 CliffWalking",
    description=(
        "The classic 4×12 cliff task with a start, a goal, and reset-on-cliff cells."
    ),
    design={
        "provenance": {
            "type": "platform_preset",
            "preset_id": "cliff-walking-4x12-v1",
            "label": "Standard 4×12 CliffWalking",
            "note": (
                "Platform-supplied teaching scenario; it is not claimed to be "
                "an exact passage from the uploaded source."
            ),
        },
        "task": {
            "mission": "Reach Goal (G) from Start (S) while avoiding the cliff.",
            "dynamics": [
                "Actions are deterministic; the agent moves up, right, down, or left.",
                "Moving beyond the grid keeps the agent in the current cell.",
                "Entering a Cliff (C) cell returns the agent to Start.",
            ],
            "rewards": [
                "Every normal move: -1.",
                "Cliff: -100 and return to Start.",
                "Goal ends the episode.",
            ],
        },
        "environment_map": {
            "kind": "grid",
            "layout": [
                "FFFFFFFFFFFF",
                "FFFFFFFFFFFF",
                "FFFFFFFFFFFF",
                "SCCCCCCCCCCG",
            ],
            "legend": {
                "S": {"label": "START", "role": "start", "terminal": False},
                "F": {"label": "PATH", "role": "normal", "terminal": False},
                "C": {
                    "label": "CLIFF", "role": "hazard", "terminal": False,
                    "color": "#ff4444", "text_color": "#ffffff",
                },
                "G": {"label": "GOAL", "role": "goal", "terminal": True},
            },
            "actions": {
                "0": {"label": "Up", "arrow": "↑"},
                "1": {"label": "Right", "arrow": "→"},
                "2": {"label": "Down", "arrow": "↓"},
                "3": {"label": "Left", "arrow": "←"},
            },
        },
        "transition_model": {
            "kind": "deterministic_grid",
            "out_of_bounds": "stay",
            "start_symbol": "S",
            "goal_symbols": ["G"],
            "hazard_symbols": ["C"],
            "terminal_symbols": ["G"],
            "hazard_behavior": "reset_to_start",
            "step_reward": -1.0,
            "goal_reward": -1.0,
            "hazard_reward": -100.0,
        },
    },
)


EXPERIMENT_DESIGN_PRESETS: Mapping[str, ExperimentDesignPreset] = {
    preset.preset_id: preset
    for preset in (_FROZEN_LAKE_4X4, _CLIFF_WALKING_4X12)
}


def get_experiment_design_preset(preset_id: str) -> dict[str, Any]:
    try:
        preset = EXPERIMENT_DESIGN_PRESETS[preset_id]
    except KeyError as exc:
        raise ValueError(f"unknown experiment design preset: {preset_id}") from exc
    return deepcopy(dict(preset.design))


def recommend_experiment_design_preset(
    *,
    algorithm_id: str = "",
    name: str = "",
    summary: str = "",
    supported_environments: Iterable[str] = (),
) -> str | None:
    haystack = " ".join(
        [algorithm_id, name, summary, *[str(item) for item in supported_environments]]
    ).lower()
    if "cliffwalking" in haystack or "cliff walking" in haystack:
        return _CLIFF_WALKING_4X12.preset_id
    if any(
        token in haystack
        for token in ("frozenlake", "frozen lake", "gridworld", "grid world")
    ):
        return _FROZEN_LAKE_4X4.preset_id
    return None


def validate_experiment_design(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("experiment_design must be an object")
    design = deepcopy(dict(value))
    provenance = design.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("experiment_design.provenance must be an object")
    if provenance.get("type") not in {
        "source_derived",
        "platform_preset",
        "agent_proposed",
    }:
        raise ValueError("experiment_design.provenance.type is invalid")
    for field in ("label", "note"):
        if not isinstance(provenance.get(field), str) or not provenance[field].strip():
            raise ValueError(f"experiment_design.provenance.{field} is required")
    if provenance.get("type") == "platform_preset":
        preset_id = provenance.get("preset_id")
        if preset_id not in EXPERIMENT_DESIGN_PRESETS:
            raise ValueError("experiment_design platform preset is unknown")

    task = design.get("task")
    if not isinstance(task, Mapping) or not isinstance(task.get("mission"), str):
        raise ValueError("experiment_design.task.mission is required")
    for field in ("dynamics", "rewards"):
        entries = task.get(field, [])
        if not isinstance(entries, list) or not all(
            isinstance(item, str) and item.strip() for item in entries
        ):
            raise ValueError(f"experiment_design.task.{field} is invalid")

    environment = design.get("environment_map")
    if not isinstance(environment, Mapping) or environment.get("kind") != "grid":
        raise ValueError("experiment_design.environment_map must be a grid")
    layout = environment.get("layout")
    if (
        not isinstance(layout, list)
        or not layout
        or len(layout) > 20
        or not all(isinstance(row, str) and row for row in layout)
        or len({len(row) for row in layout}) != 1
        or len(layout[0]) > 20
    ):
        raise ValueError("experiment_design.environment_map layout is invalid")
    legend = environment.get("legend")
    symbols = set("".join(layout))
    if not isinstance(legend, Mapping) or not symbols.issubset(legend):
        raise ValueError("experiment_design.environment_map legend is incomplete")
    actions = environment.get("actions")
    if not isinstance(actions, Mapping) or not actions:
        raise ValueError("experiment_design.environment_map actions are required")
    transition_model = design.get("transition_model")
    if not isinstance(transition_model, Mapping):
        raise ValueError("experiment_design.transition_model is required")
    from .contracts import validate_experiment_spec

    validate_experiment_spec(
        {
            "parameters": {},
            "presentation": {
                "task": task,
                "environment_map": environment,
            },
        }
    )
    return design
