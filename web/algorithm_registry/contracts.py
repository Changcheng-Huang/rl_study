from __future__ import annotations

import math
from pathlib import PurePosixPath
from typing import Any, Mapping


class ContractValidationError(ValueError):
    pass


PARAMETER_TYPES = {"int", "float", "bool", "string", "choice"}
ARTIFACT_TYPES = {"image", "table", "text", "video"}
MAX_GRID_ROWS = 20
MAX_GRID_COLUMNS = 20


def _is_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float))


def _non_empty_string_list(value: Any, field: str) -> None:
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise ContractValidationError(f"{field} must be a list of non-empty strings")


def _validate_presentation(presentation: Any) -> None:
    if not isinstance(presentation, Mapping):
        raise ContractValidationError("experiment presentation must be an object")

    task = presentation.get("task")
    if task is not None:
        if not isinstance(task, Mapping):
            raise ContractValidationError("presentation task must be an object")
        mission = task.get("mission")
        if not isinstance(mission, str) or not mission.strip():
            raise ContractValidationError(
                "presentation task must define a non-empty mission"
            )
        for field in ("dynamics", "rewards"):
            if field in task:
                _non_empty_string_list(task[field], f"presentation task {field}")

    environment = presentation.get("environment_map")
    if environment is None:
        return
    if not isinstance(environment, Mapping):
        raise ContractValidationError("presentation environment_map must be an object")
    if environment.get("kind") != "grid":
        raise ContractValidationError("environment_map kind must be 'grid'")

    layout = environment.get("layout")
    if not isinstance(layout, list) or not layout:
        raise ContractValidationError("environment_map layout must be a non-empty list")
    if len(layout) > MAX_GRID_ROWS:
        raise ContractValidationError(
            f"environment_map cannot exceed {MAX_GRID_ROWS} rows"
        )
    if not all(isinstance(row, str) and row for row in layout):
        raise ContractValidationError("environment_map rows must be non-empty strings")
    columns = len(layout[0])
    if columns > MAX_GRID_COLUMNS:
        raise ContractValidationError(
            f"environment_map cannot exceed {MAX_GRID_COLUMNS} columns"
        )
    if any(len(row) != columns for row in layout):
        raise ContractValidationError("environment_map layout must be rectangular")

    legend = environment.get("legend")
    if not isinstance(legend, Mapping):
        raise ContractValidationError("environment_map legend must be an object")
    symbols = set("".join(layout))
    missing = symbols - {str(key) for key in legend}
    if missing:
        raise ContractValidationError(
            "environment_map legend is missing symbols: "
            + ", ".join(sorted(missing))
        )
    allowed_roles = {"normal", "start", "goal", "hazard", "obstacle"}
    for symbol, definition in legend.items():
        if not isinstance(symbol, str) or len(symbol) != 1:
            raise ContractValidationError(
                "environment_map legend keys must be single characters"
            )
        if not isinstance(definition, Mapping):
            raise ContractValidationError(
                f"environment_map legend '{symbol}' must be an object"
            )
        label = definition.get("label")
        role = definition.get("role", "normal")
        if not isinstance(label, str) or not label.strip():
            raise ContractValidationError(
                f"environment_map legend '{symbol}' must define a label"
            )
        if role not in allowed_roles:
            raise ContractValidationError(
                f"environment_map legend '{symbol}' has an invalid role"
            )
        if "terminal" in definition and not isinstance(
            definition["terminal"], bool
        ):
            raise ContractValidationError(
                f"environment_map legend '{symbol}' terminal must be boolean"
            )
        for optional_field in ("color", "text_color", "icon"):
            optional_value = definition.get(optional_field)
            if optional_value is not None and (
                not isinstance(optional_value, str) or not optional_value.strip()
            ):
                raise ContractValidationError(
                    f"environment_map legend '{symbol}' {optional_field} "
                    "must be a non-empty string"
                )

    actions = environment.get("actions")
    if not isinstance(actions, Mapping) or not actions:
        raise ContractValidationError("environment_map actions must be an object")
    for action, definition in actions.items():
        if not isinstance(action, (str, int)) or isinstance(action, bool):
            raise ContractValidationError(
                "environment_map action keys must be strings or integers"
            )
        if not isinstance(definition, Mapping):
            raise ContractValidationError(
                f"environment_map action '{action}' must be an object"
            )
        label = definition.get("label")
        arrow = definition.get("arrow")
        if not isinstance(label, str) or not label.strip():
            raise ContractValidationError(
                f"environment_map action '{action}' must define a label"
            )
        if not isinstance(arrow, str) or not arrow.strip():
            raise ContractValidationError(
                f"environment_map action '{action}' must define an arrow"
            )


def _validate_parameter_value(name: str, definition: Mapping[str, Any], value: Any) -> None:
    parameter_type = definition["type"]

    if parameter_type == "int":
        valid_type = isinstance(value, int) and not isinstance(value, bool)
    elif parameter_type == "float":
        valid_type = _is_number(value)
    elif parameter_type == "bool":
        valid_type = isinstance(value, bool)
    elif parameter_type in {"string", "choice"}:
        valid_type = isinstance(value, (str, int, float, bool))
    else:
        valid_type = False

    if not valid_type:
        raise ContractValidationError(
            f"parameter '{name}' default/value does not match type '{parameter_type}'"
        )

    if parameter_type in {"int", "float"}:
        minimum = definition.get("min")
        maximum = definition.get("max")
        if minimum is not None and (not _is_number(minimum) or value < minimum):
            raise ContractValidationError(f"parameter '{name}' is below its minimum")
        if maximum is not None and (not _is_number(maximum) or value > maximum):
            raise ContractValidationError(f"parameter '{name}' is above its maximum")

    if parameter_type == "choice" and value not in definition["options"]:
        raise ContractValidationError(f"parameter '{name}' is not one of its options")


def validate_experiment_spec(spec: Any) -> None:
    if not isinstance(spec, Mapping):
        raise ContractValidationError("experiment spec must be an object")

    parameters = spec.get("parameters")
    if not isinstance(parameters, Mapping):
        raise ContractValidationError("experiment spec must contain a parameters object")

    for name, definition in parameters.items():
        if not isinstance(name, str) or not name.strip():
            raise ContractValidationError("parameter names must be non-empty strings")
        if not isinstance(definition, Mapping):
            raise ContractValidationError(f"parameter '{name}' definition must be an object")

        parameter_type = definition.get("type")
        if parameter_type not in PARAMETER_TYPES:
            raise ContractValidationError(
                f"parameter '{name}' type must be one of {sorted(PARAMETER_TYPES)}"
            )
        if "default" not in definition:
            raise ContractValidationError(f"parameter '{name}' must define a default")

        if parameter_type == "choice":
            options = definition.get("options")
            if not isinstance(options, list) or not options:
                raise ContractValidationError(
                    f"choice parameter '{name}' must define non-empty options"
                )

        if "step" in definition:
            step = definition["step"]
            if not _is_number(step) or step <= 0:
                raise ContractValidationError(f"parameter '{name}' step must be positive")

        if "min" in definition and "max" in definition:
            minimum = definition["min"]
            maximum = definition["max"]
            if not _is_number(minimum) or not _is_number(maximum) or minimum > maximum:
                raise ContractValidationError(f"parameter '{name}' has an invalid range")

        _validate_parameter_value(name, definition, definition["default"])

    if "presentation" in spec:
        _validate_presentation(spec["presentation"])


def normalize_parameters(
    spec: Mapping[str, Any], parameters: Mapping[str, Any]
) -> dict[str, Any]:
    validate_experiment_spec(spec)
    if not isinstance(parameters, Mapping):
        raise ContractValidationError("experiment parameters must be an object")

    definitions = spec["parameters"]
    unknown = set(parameters) - set(definitions)
    if unknown:
        raise ContractValidationError(
            f"unknown experiment parameters: {', '.join(sorted(unknown))}"
        )

    normalized: dict[str, Any] = {}
    for name, definition in definitions.items():
        value = parameters.get(name, definition["default"])
        _validate_parameter_value(name, definition, value)
        normalized[name] = value
    return normalized


def _validate_artifact_path(path_value: Any) -> None:
    if not isinstance(path_value, str) or not path_value.strip():
        raise ContractValidationError("artifact path must be a non-empty string")
    normalized = path_value.replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute() or ".." in path.parts:
        raise ContractValidationError("artifact path must remain inside the algorithm package")


def validate_experiment_result(
    result: Any,
    spec: Mapping[str, Any] | None = None,
) -> None:
    if not isinstance(result, Mapping):
        raise ContractValidationError("experiment result must be an object")

    metrics = result.get("metrics", {})
    summary = result.get("summary", {})
    artifacts = result.get("artifacts", [])

    if not isinstance(metrics, Mapping):
        raise ContractValidationError("result metrics must be an object")
    for name, values in metrics.items():
        if not isinstance(name, str) or not name.strip():
            raise ContractValidationError("metric names must be non-empty strings")
        if not isinstance(values, list):
            raise ContractValidationError(f"metric '{name}' values must be a list")
        for value in values:
            if not _is_number(value) or not math.isfinite(float(value)):
                raise ContractValidationError(f"metric '{name}' contains a non-finite number")

    if not isinstance(summary, Mapping):
        raise ContractValidationError("result summary must be an object")
    for name, value in summary.items():
        if not isinstance(name, str) or not name.strip():
            raise ContractValidationError("summary names must be non-empty strings")
        if value is not None and not isinstance(value, (str, int, float, bool)):
            raise ContractValidationError(
                f"summary '{name}' must be a scalar JSON-compatible value"
            )
        if isinstance(value, float) and not math.isfinite(value):
            raise ContractValidationError(f"summary '{name}' must be finite")

    if not isinstance(artifacts, list):
        raise ContractValidationError("result artifacts must be a list")
    for artifact in artifacts:
        if not isinstance(artifact, Mapping):
            raise ContractValidationError("each result artifact must be an object")
        if artifact.get("type") not in ARTIFACT_TYPES:
            raise ContractValidationError(
                f"artifact type must be one of {sorted(ARTIFACT_TYPES)}"
            )
        _validate_artifact_path(artifact.get("path"))
        if "title" in artifact and not isinstance(artifact["title"], str):
            raise ContractValidationError("artifact title must be a string")

    views = result.get("views", {})
    if not isinstance(views, Mapping):
        raise ContractValidationError("result views must be an object")
    unknown_views = set(views) - {"policy_grid"}
    if unknown_views:
        raise ContractValidationError(
            "unknown result views: " + ", ".join(sorted(unknown_views))
        )
    policy_grid = views.get("policy_grid")
    if policy_grid is None:
        return
    if not isinstance(policy_grid, Mapping):
        raise ContractValidationError("policy_grid view must be an object")
    if spec is None:
        raise ContractValidationError(
            "policy_grid requires the validated experiment spec"
        )
    environment = (
        spec.get("presentation", {}).get("environment_map")
        if isinstance(spec.get("presentation"), Mapping)
        else None
    )
    if not isinstance(environment, Mapping) or environment.get("kind") != "grid":
        raise ContractValidationError(
            "policy_grid requires a grid environment_map in get_spec()"
        )
    layout = environment["layout"]
    cell_count = len(layout) * len(layout[0])
    state_values = policy_grid.get("state_values")
    best_actions = policy_grid.get("best_actions")
    if not isinstance(state_values, list) or len(state_values) != cell_count:
        raise ContractValidationError(
            "policy_grid state_values length must match the environment grid"
        )
    if not isinstance(best_actions, list) or len(best_actions) != cell_count:
        raise ContractValidationError(
            "policy_grid best_actions length must match the environment grid"
        )
    for value in state_values:
        if value is not None and (
            not _is_number(value) or not math.isfinite(float(value))
        ):
            raise ContractValidationError(
                "policy_grid state_values must contain finite numbers or null"
            )
    allowed_actions = {str(action) for action in environment["actions"]}
    for action in best_actions:
        if action is not None and str(action) not in allowed_actions:
            raise ContractValidationError(
                "policy_grid best_actions contains an undeclared action"
            )
