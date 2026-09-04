from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.patches import Rectangle

from algorithm_registry import (
    ExperimentReporter,
    ExperimentUnavailableError,
    load_experiment,
)


def _parameter_widget(
    algorithm_id: str,
    name: str,
    definition: Mapping[str, Any],
) -> Any:
    label = definition.get("label", name.replace("_", " ").title())
    help_text = definition.get("help")
    key = f"imported_{algorithm_id}_{name}"
    parameter_type = definition["type"]
    default = definition["default"]

    if parameter_type == "bool":
        return st.checkbox(label, value=default, key=key, help=help_text)
    if parameter_type == "choice":
        options = definition["options"]
        return st.selectbox(
            label,
            options,
            index=options.index(default),
            key=key,
            help=help_text,
        )
    if parameter_type == "string":
        return st.text_input(label, value=str(default), key=key, help=help_text)
    if parameter_type == "int":
        kwargs = {
            "label": label,
            "value": int(default),
            "step": int(definition.get("step", 1)),
            "key": key,
            "help": help_text,
        }
        if "min" in definition:
            kwargs["min_value"] = int(definition["min"])
        if "max" in definition:
            kwargs["max_value"] = int(definition["max"])
        return st.number_input(**kwargs)
    if parameter_type == "float":
        kwargs = {
            "label": label,
            "value": float(default),
            "step": float(definition.get("step", 0.01)),
            "key": key,
            "help": help_text,
            "format": "%.4f",
        }
        if "min" in definition:
            kwargs["min_value"] = float(definition["min"])
        if "max" in definition:
            kwargs["max_value"] = float(definition["max"])
        return st.number_input(**kwargs)
    raise ValueError(f"unsupported parameter type: {parameter_type}")


def _safe_artifact_path(package_root: Path, relative: str) -> Path | None:
    candidate = (package_root / relative).resolve()
    try:
        candidate.relative_to(package_root.resolve())
    except ValueError:
        return None
    return candidate if candidate.is_file() else None


def _render_grid_map(
    environment: Mapping[str, Any],
    *,
    title: str,
    policy: Mapping[str, Any] | None = None,
) -> None:
    layout = environment["layout"]
    rows, columns = len(layout), len(layout[0])
    values = policy.get("state_values") if policy is not None else None
    actions = policy.get("best_actions") if policy is not None else None
    matrix = np.zeros((rows, columns), dtype=float)
    if values is not None:
        matrix = np.array(
            [float(value) if value is not None else np.nan for value in values]
        ).reshape((rows, columns))

    figure, axis = plt.subplots(
        figsize=(max(6.0, columns * 0.85), max(3.5, rows * 0.85))
    )
    finite = matrix[np.isfinite(matrix)]
    minimum = float(finite.min()) if finite.size else 0.0
    maximum = float(finite.max()) if finite.size else 1.0
    if minimum == maximum:
        maximum = minimum + 1.0
    image = axis.imshow(matrix, cmap="YlGnBu_r", vmin=minimum, vmax=maximum)
    if policy is not None:
        figure.colorbar(image, ax=axis, label="Estimated State Value")

    role_colors = {
        "start": "#8ecae6",
        "goal": "#66cc66",
        "hazard": "#e76f51",
        "obstacle": "#555555",
    }
    legend = environment["legend"]
    action_definitions = {
        str(key): value for key, value in environment["actions"].items()
    }
    for row in range(rows):
        for column in range(columns):
            state = row * columns + column
            symbol = layout[row][column]
            definition = legend[symbol]
            role = definition.get("role", "normal")
            cell_color = definition.get("color") or role_colors.get(role)
            if cell_color:
                axis.add_patch(
                    Rectangle(
                        (column - 0.5, row - 0.5),
                        1,
                        1,
                        color=cell_color,
                        alpha=0.85,
                    )
                )
            label = definition["label"]
            terminal = bool(definition.get("terminal", False))
            if role != "normal" or terminal:
                text_color = definition.get("text_color") or (
                    "white" if role in {"hazard", "obstacle"} else "black"
                )
                axis.text(
                    column,
                    row,
                    label,
                    ha="center",
                    va="center",
                    color=text_color,
                    weight="bold",
                    fontsize=9,
                )
            elif actions is None:
                axis.text(
                    column,
                    row,
                    definition.get("icon", label),
                    ha="center",
                    va="center",
                    fontsize=14 if definition.get("icon") else 9,
                )

            if policy is not None and values is not None and values[state] is not None:
                axis.text(
                    column - 0.43,
                    row - 0.34,
                    f"{float(values[state]):.2f}",
                    ha="left",
                    va="top",
                    fontsize=7,
                )
            if (
                actions is not None
                and actions[state] is not None
                and not terminal
                and role not in {"hazard", "obstacle", "goal"}
            ):
                action = action_definitions[str(actions[state])]
                axis.text(
                    column,
                    row,
                    action["arrow"],
                    ha="center",
                    va="center",
                    fontsize=20,
                    weight="bold",
                )

    axis.set_xticks(np.arange(-0.5, columns, 1), minor=True)
    axis.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    axis.grid(which="minor", color="#eeeeee", linewidth=1.5)
    axis.tick_params(which="minor", bottom=False, left=False)
    axis.set_xticks(range(columns))
    axis.set_yticks(range(rows))
    axis.set_xlabel("Grid Column")
    axis.set_ylabel("Grid Row")
    axis.set_title(title)
    st.pyplot(figure, clear_figure=True)


def _render_presentation(spec: Mapping[str, Any]) -> None:
    presentation = spec.get("presentation")
    if not isinstance(presentation, Mapping):
        return
    task = presentation.get("task")
    environment = presentation.get("environment_map")
    if task is None and environment is None:
        return
    task_column, map_column = st.columns([1, 1.5], vertical_alignment="top")
    with task_column:
        if isinstance(task, Mapping):
            st.subheader("Task")
            st.markdown(f"**Mission:** {task['mission']}")
            for heading, key in (("Dynamics", "dynamics"), ("Rewards", "rewards")):
                entries = task.get(key, [])
                if entries:
                    st.markdown(f"**{heading}**")
                    st.markdown("\n".join(f"- {entry}" for entry in entries))
    with map_column:
        if isinstance(environment, Mapping):
            st.subheader("Environment Map")
            _render_grid_map(environment, title="Initial Environment Map")
    st.divider()


def _render_result(
    result: Mapping[str, Any],
    package_root: Path,
    spec: Mapping[str, Any],
) -> None:
    summary = result.get("summary", {})
    if summary:
        st.subheader("Summary")
        columns = st.columns(min(len(summary), 4))
        for index, (name, value) in enumerate(summary.items()):
            columns[index % len(columns)].metric(name.replace("_", " ").title(), value)

    metrics = result.get("metrics", {})
    policy_grid = result.get("views", {}).get("policy_grid")
    environment = spec.get("presentation", {}).get("environment_map")
    if policy_grid is not None and environment is not None and metrics:
        policy_tab, metrics_tab = st.tabs(["🗺️ Learned Policy", "📈 Training Curves"])
        with policy_tab:
            _render_grid_map(
                environment,
                title="Learned Policy",
                policy=policy_grid,
            )
        with metrics_tab:
            frame = pd.DataFrame(
                {name: pd.Series(values) for name, values in metrics.items()}
            )
            st.line_chart(frame)
    else:
        if policy_grid is not None and environment is not None:
            st.subheader("Learned Policy")
            _render_grid_map(
                environment,
                title="Learned Policy",
                policy=policy_grid,
            )
        if metrics:
            st.subheader("Metrics")
            frame = pd.DataFrame(
                {name: pd.Series(values) for name, values in metrics.items()}
            )
            st.line_chart(frame)

    for artifact in result.get("artifacts", []):
        path = _safe_artifact_path(package_root, artifact["path"])
        if path is None:
            st.warning(f"Artifact is unavailable: {artifact['path']}")
            continue
        title = artifact.get("title")
        if title:
            st.subheader(title)
        artifact_type = artifact["type"]
        if artifact_type == "image":
            st.image(str(path))
        elif artifact_type == "video":
            st.video(str(path))
        elif artifact_type == "text":
            st.text(path.read_text(encoding="utf-8"))
        elif artifact_type == "table":
            if path.suffix.lower() == ".csv":
                st.dataframe(pd.read_csv(path), use_container_width=True)
            else:
                st.json(json.loads(path.read_text(encoding="utf-8")))


def render_imported_experiment(algorithm_id: str) -> None:
    try:
        experiment = load_experiment(algorithm_id)
    except ExperimentUnavailableError as exc:
        st.error(str(exc))
        return

    manifest = experiment.algorithm.manifest
    st.header(manifest.name)
    st.caption(manifest.summary)
    _render_presentation(experiment.spec)

    parameters: dict[str, Any] = {}
    with st.expander("Experiment Settings", expanded=True):
        definitions = experiment.spec["parameters"]
        if definitions:
            columns = st.columns(2)
            for index, (name, definition) in enumerate(definitions.items()):
                with columns[index % 2]:
                    parameters[name] = _parameter_widget(
                        algorithm_id, name, definition
                    )
        start = st.button(
            "Run Experiment",
            type="primary",
            use_container_width=True,
            key=f"run_imported_{algorithm_id}",
        )

    if not start:
        return

    progress_bar = st.progress(0)
    status = st.empty()

    def on_progress(event) -> None:
        progress_bar.progress(event.current / event.total)
        if event.message:
            status.caption(event.message)

    reporter = ExperimentReporter(on_progress=on_progress)
    try:
        with st.spinner("Running experiment..."):
            result = experiment.run(parameters, reporter)
    except Exception as exc:
        st.error(f"Experiment failed: {exc}")
        return

    progress_bar.progress(1.0)
    status.success("Experiment complete.")
    _render_result(result, experiment.algorithm.path, experiment.spec)
