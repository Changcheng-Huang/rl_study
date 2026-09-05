from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

import nbformat
import streamlit as st
from algorithm_registry.notebook_links import manual_publication_for

from algorithm_registry import (
    EXPERIMENT_DESIGN_PRESETS,
    AgentSpecError,
    AlgorithmPackageError,
    DraftInput,
    approve_module,
    cancel_change_request,
    build_animation_creator_kit,
    create_draft,
    create_revision_draft,
    describe_agent_error,
    extract_source_text,
    generate_animation_guidance_with_agent,
    generate_animation_options_with_agent,
    generate_default_animation_guidance,
    generate_default_animation_options,
    generate_module_with_agent,
    get_agent_configuration,
    get_animation_planner_configuration,
    get_experiment_design_preset,
    get_module_agent_configuration,
    install_approved_draft,
    install_package,
    list_drafts,
    list_installed,
    list_rejected_drafts,
    load_animation_guidance,
    load_animation_options,
    module_content_ready,
    recommend_experiment_design_preset,
    regenerate_module,
    reject_draft,
    remove_animation_module,
    replace_module_file,
    resolve_placeholder_blocker,
    request_changes,
    restore_rejected_draft,
    save_algorithm_spec,
    save_animation_guidance,
    save_animation_module,
    save_experiment_design,
    save_theory,
    select_animation_option,
    suggest_algorithm_spec,
    trash_rejected_draft,
    uninstall_package,
    validate_package,
)


CORE_MODULES = ("theory", "notebook", "experiment")
ALGORITHM_SPEC_USER_GUIDE = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "algorithm_spec_user_guide.md"
)
ANIMATION_WORKFLOW_USER_GUIDE = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "animation_workflow_user_guide.md"
)


def _next_patch_version(version: str) -> str:
    major, minor, patch = version.split("-", 1)[0].split("+", 1)[0].split(".")
    return f"{int(major)}.{int(minor)}.{int(patch) + 1}"


def _write_upload(uploaded_file) -> Path:
    temporary = tempfile.NamedTemporaryFile(
        prefix="algorithm-upload-", suffix=".zip", delete=False
    )
    try:
        temporary.write(uploaded_file.getvalue())
        temporary.flush()
    finally:
        temporary.close()
    return Path(temporary.name)


def _render_report(report) -> None:
    if report.manifest is not None:
        manifest = report.manifest
        st.markdown(f"#### {manifest.name}")
        st.caption(
            f"Schema v{manifest.schema_version} · ID: `{manifest.algorithm_id}` · "
            f"Version: `{manifest.version}` · Category: `{manifest.category}`"
        )
        st.write(manifest.summary)
        modules = ["Theory"]
        if manifest.animation is not None:
            modules.append("Animation")
        if manifest.notebook is not None:
            modules.append("Jupyter")
        if manifest.experiment is not None:
            modules.append("RL Laboratory")
        st.write("Will integrate into: " + ", ".join(modules))

    for issue in report.errors:
        st.error(f"[{issue.code}] {issue.message}")
    for issue in report.warnings:
        st.warning(f"[{issue.code}] {issue.message}")
    if report.valid:
        st.success("Package validation passed.")


def _lines(value: str) -> tuple[str, ...]:
    return tuple(line.strip() for line in value.splitlines() if line.strip())


def _editor_records(value) -> list[dict]:
    if value is None:
        return []
    if hasattr(value, "to_dict"):
        return [dict(item) for item in value.to_dict(orient="records")]
    return [dict(item) for item in value]


def _animation_symbols(value) -> tuple[dict[str, str], ...]:
    symbols = []
    for item in _editor_records(value):
        symbol = str(item.get("symbol", "")).strip()
        meaning = str(item.get("meaning", "")).strip()
        if not symbol and not meaning:
            continue
        if not symbol or not meaning:
            raise ValueError("Each symbol row needs both a symbol and a meaning.")
        symbols.append({"symbol": symbol, "meaning": meaning})
    return tuple(symbols)


def _derivation_steps(value) -> tuple[dict, ...]:
    steps = []
    for item in _editor_records(value):
        title = str(item.get("title", "")).strip()
        text = str(item.get("text", "")).strip()
        latex = _lines(str(item.get("latex", "")))
        if not title and not text and not latex:
            continue
        if not title or not text:
            raise ValueError(
                "Each derivation row needs a step title and explanation."
            )
        steps.append({"title": title, "text": text, "latex": list(latex)})
    return tuple(steps)


def _symbol_rows(value) -> list[dict[str, str]]:
    return [
        {"symbol": str(item.get("symbol", "")), "meaning": str(item.get("meaning", ""))}
        for item in value
    ]


def _derivation_rows(value) -> list[dict[str, str]]:
    rows = []
    for item in value:
        latex = item.get("latex", [])
        if isinstance(latex, str):
            latex_text = latex
        else:
            latex_text = "\n".join(str(block) for block in latex)
        rows.append(
            {
                "title": str(item.get("title", item.get("name", ""))),
                "text": str(item.get("text", item.get("content", ""))),
                "latex": latex_text,
            }
        )
    return rows


def _render_symbol_editor(label: str, values, *, key: str, disabled: bool = False):
    st.caption(
        "Optional video legend: each row explains one formula symbol shown beside "
        "the animation. It does not create a program variable. Empty rows are ignored."
    )
    rows = _symbol_rows(values)
    edited = st.data_editor(
        rows,
        key=key,
        num_rows="dynamic",
        disabled=disabled,
        width="stretch",
        column_config={
            "symbol": st.column_config.TextColumn("Symbol", required=True),
            "meaning": st.column_config.TextColumn("Meaning", required=True),
        },
        column_order=("symbol", "meaning"),
    )
    return rows if edited is None else edited


def _render_derivation_editor(
    label: str, values, *, key: str, disabled: bool = False
):
    st.caption(
        f"{label}: each row becomes one learner-facing explanation step. Put "
        "multiple formulas on separate lines."
    )
    rows = _derivation_rows(values)
    edited = st.data_editor(
        rows,
        key=key,
        num_rows="dynamic",
        disabled=disabled,
        width="stretch",
        column_config={
            "title": st.column_config.TextColumn("Step title", required=True),
            "text": st.column_config.TextColumn("Explanation", required=True),
            "latex": st.column_config.TextColumn("Formula(s), one per line"),
        },
        column_order=("title", "text", "latex"),
    )
    return rows if edited is None else edited


def _form_scalar(value):
    if not isinstance(value, str):
        return value
    cleaned = value.strip()
    lowered = cleaned.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return int(cleaned)
    except ValueError:
        try:
            return float(cleaned)
        except ValueError:
            return cleaned


def _hyperparameter_rows(value) -> list[dict]:
    rows = []
    for name, configuration in value.items():
        details = configuration if isinstance(configuration, dict) else {}
        default = details.get("default", configuration)
        choices = details.get("choices", details.get("allowed_values", []))
        if not isinstance(choices, (list, tuple)):
            choices = [choices] if choices not in (None, "") else []
        rows.append(
            {
                "name": str(name),
                "default": str(default),
                "description": str(details.get("description", "")),
                "minimum": str(details.get("minimum", details.get("min", ""))),
                "maximum": str(details.get("maximum", details.get("max", ""))),
                "step": str(details.get("step", "")),
                "choices": ", ".join(str(item) for item in choices),
            }
        )
    return rows


def _hyperparameters_from_editor(value) -> dict:
    result = {}
    for row in _editor_records(value):
        name = str(row.get("name", "")).strip()
        if not name and not any(str(item).strip() for item in row.values()):
            continue
        if not name:
            raise ValueError("Each hyperparameter row needs a name.")
        if name in result:
            raise ValueError(f"Hyperparameter '{name}' appears more than once.")
        configuration = {"default": _form_scalar(row.get("default", ""))}
        description = str(row.get("description", "")).strip()
        minimum = str(row.get("minimum", "")).strip()
        maximum = str(row.get("maximum", "")).strip()
        step = str(row.get("step", "")).strip()
        choices = [
            item.strip()
            for item in str(row.get("choices", "")).split(",")
            if item.strip()
        ]
        if description:
            configuration["description"] = description
        if minimum:
            configuration["minimum"] = _form_scalar(minimum)
        if maximum:
            configuration["maximum"] = _form_scalar(maximum)
        if step:
            configuration["step"] = _form_scalar(step)
        if choices:
            configuration["choices"] = [_form_scalar(item) for item in choices]
        result[name] = configuration
    return result


def _render_hyperparameter_editor(values, *, key: str, disabled: bool = False):
    st.caption(
        "Each row defines one adjustable experiment control. For example, an "
        "epsilon row can become an Exploration rate control whose initial value "
        "is taken from Starting value."
    )
    rows = _hyperparameter_rows(values)
    edited = st.data_editor(
        rows,
        key=key,
        num_rows="dynamic",
        disabled=disabled,
        width="stretch",
        column_config={
            "name": st.column_config.TextColumn("Setting", required=True),
            "default": st.column_config.TextColumn("Starting value", required=True),
            "description": st.column_config.TextColumn("What it changes"),
            "minimum": st.column_config.TextColumn("Lowest value"),
            "maximum": st.column_config.TextColumn("Highest value"),
            "step": st.column_config.TextColumn("Change per click"),
            "choices": st.column_config.TextColumn("Allowed values, comma-separated"),
        },
        column_order=(
            "name",
            "default",
            "description",
            "minimum",
            "maximum",
            "step",
            "choices",
        ),
    )
    return rows if edited is None else edited


def _review_modules(raw: dict) -> tuple[str, ...]:
    modules = list(CORE_MODULES)
    if "animation" in raw.get("modules", {}):
        modules.append("animation")
    return tuple(modules)


def _render_create_draft(provider: str) -> None:
    create_agent_request_in_progress = any(
        str(key).startswith("algorithm_spec_agent_request_in_progress_")
        and bool(value)
        for key, value in st.session_state.items()
    )
    st.subheader("Create Schema v2 Draft")
    st.caption(
        "Upload evidence, use the Agent to propose an AlgorithmSpec, correct and "
        "confirm it, then create a review draft and generate each module separately."
    )
    st.info(
        "Output language: English. Uploaded evidence remains in its original "
        "language so reviewers can verify exact source excerpts."
    )
    st.markdown("### Step 1 · Upload source material")
    mode_label = st.selectbox(
        "Generation profile",
        ["Monte Carlo Control preset", "Generic scaffold"],
        disabled=create_agent_request_in_progress,
    )
    preset = mode_label.startswith("Monte Carlo")
    profile_key = "monte_carlo" if preset else "generic"
    suggestion_key = f"algorithm_spec_agent_suggestion_{profile_key}"
    suggestion_history_key = (
        f"algorithm_spec_agent_suggestion_history_{profile_key}"
    )
    suggestion_request_key = (
        f"algorithm_spec_agent_request_in_progress_{profile_key}"
    )
    suggestion_error_key = f"algorithm_spec_agent_error_{profile_key}"
    revision_key = f"algorithm_spec_form_revision_{profile_key}"
    request_in_progress = bool(st.session_state.get(suggestion_request_key))
    if request_in_progress:
        st.info(
            "AlgorithmSpec Agent request is running. Source, profile, and "
            "AlgorithmSpec form controls are locked until it finishes."
        )
    if preset:
        st.info(
            "**Bundled preset:** copies the repository's reviewed Monte Carlo "
            "Theory, Notebook, and Experiment. Uploaded material is stored only "
            "as provenance and does not change those bundled files."
        )
    else:
        st.info(
            "**Generic workflow starts empty:** upload source material, then ask "
            "the Agent for an editable AlgorithmSpec or enter one manually. No "
            "example algorithm values are inserted into the form."
        )
    st.markdown(
        "**What the form does:** its values become `manifest.json / AlgorithmSpec`. "
        "You can fill them manually or ask the Agent for an editable suggestion. "
        "Nothing is saved as a draft until you confirm the specification and "
        "create the review scaffold."
    )
    defaults = {
        "id": "monte-carlo-control" if preset else "",
        "name": "Monte Carlo Control" if preset else "",
        "category": "model-free-control" if preset else "",
        "summary": (
            "Learn an action-value function from complete FrozenLake episodes "
            "without bootstrapping."
            if preset
            else ""
        ),
        "objective": (
            "Estimate an action-value function from complete sampled episodes "
            "and improve an epsilon-greedy policy."
            if preset
            else ""
        ),
    }

    uploaded = st.file_uploader(
        "Source material",
        type=["md", "txt", "pdf"],
        help=(
            "PDF text is extracted without OCR. For the Monte Carlo preset, the "
            "bundled theory is used when no file is selected."
        ),
        key="v2_source",
        disabled=create_agent_request_in_progress,
    )
    source_name: str | None = None
    source_bytes: bytes | None = None
    extracted: str | None = None
    if uploaded is not None:
        try:
            extracted = extract_source_text(uploaded.name, uploaded.getvalue())
        except AlgorithmPackageError as exc:
            st.error(f"Source extraction failed: {exc}")
        else:
            source_name = uploaded.name
            source_bytes = uploaded.getvalue()
            st.success(
                f"Source ready: `{uploaded.name}` · {len(uploaded.getvalue()):,} "
                f"bytes · {len(extracted):,} extracted characters"
            )
            with st.expander("Extracted source preview", expanded=False):
                st.text(extracted[:4000])
                if len(extracted) > 4000:
                    st.caption("Preview truncated to the first 4,000 characters.")
    elif preset:
        bundled = (
            Path(__file__).resolve().parents[1]
            / "algorithm_packages"
            / "examples"
            / "monte_carlo_control"
            / "theory.md"
        )
        source_name = "monte-carlo-control-source.md"
        source_bytes = bundled.read_bytes()
        extracted = extract_source_text(source_name, source_bytes)
        st.caption(
            "No upload selected. The bundled Monte Carlo theory file will be "
            "stored as the source record."
        )

    st.markdown("### Step 2 · Extract and correct AlgorithmSpec")
    current_source_digest = (
        hashlib.sha256(extracted.encode("utf-8")).hexdigest()
        if extracted is not None
        else None
    )
    stored_suggestion = st.session_state.get(suggestion_key)
    suggestion_is_current = bool(
        stored_suggestion
        and current_source_digest
        and stored_suggestion.get("source_sha256") == current_source_digest
    )
    if stored_suggestion and not suggestion_is_current:
        st.warning(
            "The source has changed since the current Agent suggestion was "
            "generated. Generate a new suggestion before using it."
        )

    configuration = get_agent_configuration()
    with st.container(border=True):
        st.markdown("#### Agent-assisted AlgorithmSpec")
        st.caption(
            "The Agent reads extracted source text and proposes editable form "
            "values with supporting excerpts. It does not create or approve a "
            "draft. Clicking the button sends extracted text to the configured "
            "OpenAI-compatible endpoint."
        )
        st.write(f"Model: `{configuration.model}`")
        st.caption(
            "Adapter: LangChain ChatOpenAI · "
            f"Structured output: `{configuration.structured_output_method}` · "
            f"Endpoint: `{'custom base URL' if configuration.base_url else 'OpenAI default'}`"
        )
        st.caption(
            "Requests are never retried automatically. A failed request can be "
            "started again manually without changing saved draft files."
        )
        if configuration.configured:
            st.success("LangChain Agent is configured.")
        else:
            st.info(configuration.message)
            st.code(
                'export ALGORITHM_AGENT_API_KEY="your_api_key_here"\n'
                'export ALGORITHM_AGENT_BASE_URL="https://provider.example/v1"\n'
                'export ALGORITHM_AGENT_MODEL="provider-model-name"\n'
                'export ALGORITHM_AGENT_STRUCTURED_METHOD="function_calling"\n'
                "uv run streamlit run web/app.py",
                language="bash",
            )
        request_error = st.session_state.get(suggestion_error_key)
        if request_error:
            st.error(request_error["summary"])
            if request_error.get("request_id"):
                st.caption(f"Provider request ID: `{request_error['request_id']}`")
            with st.expander("Technical details", expanded=False):
                st.code(request_error["technical"], language="text")
        overwrite_confirmed = True
        if stored_suggestion is not None:
            st.warning(
                "Regenerating makes another model request, may incur additional "
                "token cost, and replaces the current AlgorithmSpec form values. "
                "Any unsubmitted manual edits in the form may be lost."
            )
            overwrite_confirmed = st.checkbox(
                "I understand and want to overwrite the current AlgorithmSpec suggestion",
                key=f"confirm_regenerate_algorithm_spec_{profile_key}",
                disabled=request_in_progress,
            )

        generate_column, clear_column = st.columns(2)
        if generate_column.button(
            (
                "Regenerate AlgorithmSpec Suggestion"
                if stored_suggestion is not None
                else (
                    "Retry AlgorithmSpec generation"
                    if request_error
                    else "Suggest AlgorithmSpec with Agent"
                )
            ),
            key=f"suggest_algorithm_spec_{profile_key}",
            disabled=(
                not configuration.configured
                or extracted is None
                or not overwrite_confirmed
                or request_in_progress
            ),
            type="primary",
            width="stretch",
        ):
            st.session_state[suggestion_request_key] = {
                "source_sha256": current_source_digest,
            }
            st.rerun()

        if clear_column.button(
            "Clear Agent Suggestions",
            key=f"clear_algorithm_spec_agent_{profile_key}",
            disabled=stored_suggestion is None or request_in_progress,
            width="stretch",
        ):
            st.session_state.pop(suggestion_key, None)
            st.session_state.pop(suggestion_history_key, None)
            st.session_state[revision_key] = (
                int(st.session_state.get(revision_key, 0)) + 1
            )
            _queue_action_response(
                "success",
                "Suggestions cleared",
                "The current AlgorithmSpec suggestion and its local history were cleared.",
            )
            st.rerun()

        if suggestion_is_current:
            st.success(
                "Agent suggestions are applied to the editable form below. "
                "They are not automatically trusted or saved."
            )
            metadata = (
                f"Generated {stored_suggestion['generated_at']} · "
                f"`{stored_suggestion['model']}`"
            )
            usage = stored_suggestion.get("usage", {})
            if usage.get("total_tokens"):
                metadata += f" · {usage['total_tokens']:,} tokens"
            st.caption(metadata)
            for warning in stored_suggestion.get("platform_warnings", []):
                st.warning(warning)
            agent_warnings = stored_suggestion.get("agent_warnings", [])
            if agent_warnings:
                with st.expander(
                    f"Agent cautions ({len(agent_warnings)})",
                    expanded=False,
                ):
                    st.caption(
                        "These are model-supplied cautions. Platform evidence "
                        "verification above remains authoritative."
                    )
                    for warning in agent_warnings:
                        st.warning(warning)
            evidence = stored_suggestion.get("evidence", [])
            if not evidence:
                st.error(
                    "No source excerpts could be verified. The suggestion is not "
                    "traceable to a matching passage in the uploaded material. "
                    "Review every field manually before confirmation."
                )
            with st.expander(
                f"Verified source evidence ({len(evidence)})",
                expanded=False,
            ):
                if evidence:
                    st.dataframe(
                        [
                            {
                                "Fields": ", ".join(
                                    item.get("supports_fields", [])
                                ),
                                "Source excerpt": item.get(
                                    "source_excerpt", ""
                                ),
                                "Verification": item.get(
                                    "verification", "exact"
                                ),
                                "Why it supports the suggestion": item.get(
                                    "explanation", ""
                                ),
                            }
                            for item in evidence
                        ],
                        width="stretch",
                    )
                else:
                    st.info(
                        "The model returned no evidence, or its excerpts could "
                        "not be matched against the extracted source. Line-wrap "
                        "whitespace and letter case are ignored; paraphrases, "
                        "translations, and changed formulas are rejected."
                    )

        suggestion_history = st.session_state.get(
            suggestion_history_key, []
        )
        if suggestion_history:
            with st.expander(
                f"Previous AlgorithmSpec suggestions ({len(suggestion_history)})",
                expanded=False,
            ):
                st.dataframe(
                    [
                        {
                            "Generated at": item.get("generated_at", ""),
                            "Model": item.get("model", ""),
                            "Method": item.get(
                                "structured_output_method", ""
                            ),
                            "Verified evidence": len(
                                item.get("evidence", [])
                            ),
                            "Warnings": len(item.get("warnings", [])),
                            "Tokens": item.get("usage", {}).get(
                                "total_tokens"
                            ),
                        }
                        for item in reversed(suggestion_history)
                    ],
                    width="stretch",
                )

    agent_values = (
        dict(stored_suggestion.get("values", {}))
        if suggestion_is_current
        else {}
    )
    form_revision = int(st.session_state.get(revision_key, 0))
    form_key_prefix = f"v2_{profile_key}_{form_revision}"

    st.markdown("### Step 3 · Confirm AlgorithmSpec and create review draft")
    if not provider.strip():
        st.warning(
            "Enter the Provider name in the sidebar and select Apply role names. "
            "It will be recorded with the confirmed AlgorithmSpec snapshot."
        )
    if not preset:
        st.info(
            "Creating the draft does not call Theory, Notebook, or Experiment "
            "Agents. It saves safe scaffolds with `not_generated` status. Generate "
            "and review each module separately in Review Drafts."
        )

    animation_upload = st.file_uploader(
        "Optional finished Animation (MP4)",
        type=["mp4"],
        key=f"v2_animation_{profile_key}",
        disabled=request_in_progress,
        help=(
            "The platform does not generate or execute animation code in this "
            "version. Upload a finished MP4 made with Manim or another tool."
        ),
    )
    if animation_upload is not None:
        st.video(animation_upload.getvalue())
        st.caption(
            f"Animation ready: `{animation_upload.name}` · "
            f"{len(animation_upload.getvalue()):,} bytes"
        )

    st.markdown("**Generated output**")
    output_files = [
        "manifest.json",
        "theory.md",
        "notebook.ipynb",
        "experiment.py",
        "sources/<uploaded-file>",
    ]
    if animation_upload is not None:
        output_files.append("animation.mp4")
    st.code("\n".join(output_files), language="text")

    default_assumptions = (
        ["Episodic environment", "Finite state and action spaces"]
        if preset
        else []
    )
    default_inputs = (
        ["Environment", "Number of episodes", "Discount factor"]
        if preset
        else []
    )
    default_outputs = ["Action-value table", "Improved policy"] if preset else []
    default_equations = (
        [
            r"G_t = R_{t+1} + \gamma G_{t+1}",
            r"Q(s,a) = \frac{1}{N(s,a)}\sum_i G_i",
        ]
        if preset
        else []
    )
    default_pseudocode = (
        [
            "Generate a complete episode",
            "Compute returns backwards",
            "Update first-visit state-action values",
            "Improve the epsilon-greedy policy",
        ]
        if preset
        else []
    )
    default_hyperparameters = (
        {
            "episodes": {"default": 2000},
            "gamma": {"default": 0.99},
            "epsilon": {"default": 0.2},
        }
        if preset
        else {}
    )

    experiment_design = None
    if not preset:
        supported_for_recommendation = agent_values.get(
            "supported_environments", []
        )
        recommended_design_id = (
            recommend_experiment_design_preset(
                algorithm_id=agent_values.get("algorithm_id", defaults["id"]),
                name=agent_values.get("name", defaults["name"]),
                summary=agent_values.get("summary", defaults["summary"]),
                supported_environments=supported_for_recommendation,
            )
            if suggestion_is_current
            else None
        )
        design_ids = []
        if recommended_design_id is not None:
            design_ids.append(recommended_design_id)
        else:
            design_ids.append("none")
        design_ids.extend(
            preset_id
            for preset_id in EXPERIMENT_DESIGN_PRESETS
            if preset_id not in design_ids
        )
        if "none" not in design_ids:
            design_ids.append("none")

        def _design_label(preset_id: str) -> str:
            if preset_id == "none":
                return "No declarative map · expert decides later"
            label = EXPERIMENT_DESIGN_PRESETS[preset_id].label
            return (
                f"Recommended · {label}"
                if preset_id == recommended_design_id
                else label
            )

        st.markdown("#### Teaching experiment scenario")
        st.caption(
            "No scenario is selected before an Agent suggestion exists. After "
            "generation, the platform may recommend a reviewed teaching preset; "
            "the Provider still decides whether to use it."
        )
        selected_design_id = st.selectbox(
            "Experiment scenario",
            design_ids,
            format_func=_design_label,
            key=f"{form_key_prefix}_experiment_design",
            disabled=request_in_progress,
            help=(
                "Choose a platform teaching preset, or defer the map to a domain "
                "expert. The choice is stored in AlgorithmSpec with provenance."
            ),
        )
        if selected_design_id == "none":
            st.info(
                "No map will be requested from the Experiment Agent. The "
                "algorithm can still generate a non-grid experiment."
            )
        else:
            selected_preset = EXPERIMENT_DESIGN_PRESETS[selected_design_id]
            experiment_design = get_experiment_design_preset(selected_design_id)
            provenance = experiment_design["provenance"]
            st.success(
                f"Platform preset selected: {selected_preset.label}. "
                "A professional can still review or replace it before publishing."
            )
            st.caption(provenance["note"])
            task = experiment_design["task"]
            st.markdown(f"**Mission:** {task['mission']}")
            st.code(
                "\n".join(experiment_design["environment_map"]["layout"]),
                language="text",
            )

    with st.form(f"create_v2_draft_{profile_key}_{form_revision}"):
        c1, c2 = st.columns(2)
        algorithm_id = c1.text_input(
            "Algorithm ID",
            value=agent_values.get("algorithm_id", defaults["id"]),
            key=f"{form_key_prefix}_algorithm_id",
            disabled=request_in_progress,
        )
        version = c2.text_input(
            "Version",
            value="1.0.0",
            key=f"{form_key_prefix}_version",
            disabled=request_in_progress,
        )
        name = c1.text_input(
            "Name",
            value=agent_values.get("name", defaults["name"]),
            key=f"{form_key_prefix}_name",
            disabled=request_in_progress,
        )
        category = c2.text_input(
            "Category",
            value=agent_values.get("category", defaults["category"]),
            key=f"{form_key_prefix}_category",
            disabled=request_in_progress,
        )
        summary = st.text_area(
            "Summary",
            value=agent_values.get("summary", defaults["summary"]),
            key=f"{form_key_prefix}_summary",
            disabled=request_in_progress,
        )
        objective = st.text_area(
            "Objective",
            value=agent_values.get("objective", defaults["objective"]),
            key=f"{form_key_prefix}_objective",
            disabled=request_in_progress,
        )
        c1, c2 = st.columns(2)
        assumptions = c1.text_area(
            "Assumptions (one per line)",
            value="\n".join(
                agent_values.get("assumptions", default_assumptions)
            ),
            key=f"{form_key_prefix}_assumptions",
            disabled=request_in_progress,
        )
        inputs = c2.text_area(
            "Inputs (one per line)",
            value="\n".join(agent_values.get("inputs", default_inputs)),
            key=f"{form_key_prefix}_inputs",
            disabled=request_in_progress,
        )
        states = c1.text_area(
            "States (one per line)",
            value="\n".join(
                agent_values.get(
                    "states", ["Environment state"] if preset else []
                )
            ),
            key=f"{form_key_prefix}_states",
            disabled=request_in_progress,
        )
        actions = c2.text_area(
            "Actions (one per line)",
            value="\n".join(
                agent_values.get(
                    "actions", ["Available action"] if preset else []
                )
            ),
            key=f"{form_key_prefix}_actions",
            disabled=request_in_progress,
        )
        outputs = st.text_area(
            "Outputs (one per line)",
            value="\n".join(
                agent_values.get("outputs", default_outputs)
            ),
            key=f"{form_key_prefix}_outputs",
            disabled=request_in_progress,
        )
        equations = st.text_area(
            "Core equations (one per line)",
            value="\n".join(
                agent_values.get("core_equations", default_equations)
            ),
            key=f"{form_key_prefix}_equations",
            disabled=request_in_progress,
        )
        pseudocode = st.text_area(
            "Pseudocode (one step per line)",
            value="\n".join(
                agent_values.get("pseudocode", default_pseudocode)
            ),
            key=f"{form_key_prefix}_pseudocode",
            disabled=request_in_progress,
        )
        environments = st.text_area(
            "Supported environments (one per line)",
            value="\n".join(
                agent_values.get(
                    "supported_environments",
                    ["FrozenLake-v1"] if preset else [],
                )
            ),
            key=f"{form_key_prefix}_environments",
            disabled=request_in_progress,
        )
        st.markdown("**Hyperparameters**")
        st.info(
            "These are the settings learners will be able to adjust in the "
            "experiment page. The table is empty until the Agent or Provider "
            "defines a setting."
        )
        hyperparameters_editor = _render_hyperparameter_editor(
            agent_values.get("hyperparameters", default_hyperparameters),
            key=f"{form_key_prefix}_hyperparameters",
            disabled=request_in_progress,
        )
        references = st.text_area(
            "Reference URLs (one per line)",
            value="",
            key=f"{form_key_prefix}_references",
            disabled=request_in_progress,
        )
        if animation_upload is not None:
            st.markdown("**Optional Animation metadata**")
            animation_concept_markdown = st.text_area(
                "Animation concept (Markdown)",
                value=summary,
                key=f"{form_key_prefix}_animation_concept",
                disabled=request_in_progress,
                help="Concept text displayed below the installed video.",
            )
            animation_formula = st.text_input(
                "Animation formula",
                key=f"{form_key_prefix}_animation_formula",
                disabled=request_in_progress,
                help="Main formula displayed or explained in the video.",
            )
            animation_symbols_editor = _render_symbol_editor(
                "Animation symbols",
                [],
                key=f"{form_key_prefix}_animation_symbols",
                disabled=request_in_progress,
            )
            animation_highlights = st.text_area(
                "Animation highlights (one per line)",
                key=f"{form_key_prefix}_animation_highlights",
                disabled=request_in_progress,
                help="Short teaching points shown beside the installed video.",
            )
            animation_viewing_flow = st.text_area(
                "Animation viewing flow (one step per line)",
                key=f"{form_key_prefix}_animation_viewing_flow",
                disabled=request_in_progress,
                help="Suggested sequence for watching and interpreting the video.",
            )
            animation_steps_editor = _render_derivation_editor(
                "Animation derivation steps",
                [],
                key=f"{form_key_prefix}_animation_steps",
                disabled=request_in_progress,
            )
        else:
            animation_concept_markdown = ""
            animation_formula = ""
            animation_symbols_editor = []
            animation_highlights = ""
            animation_viewing_flow = ""
            animation_steps_editor = []
        if preset:
            st.caption(
                "The Monte Carlo preset uses the repository's trusted module files."
            )
        else:
            st.caption(
                "Only safe scaffold files are created now. No module model calls "
                "are made until you choose a module in Review Drafts."
            )
        suggestion_evidence = (
            stored_suggestion.get("evidence", [])
            if suggestion_is_current
            else []
        )
        basis_options = (
            ["Bundled reviewed preset"] if preset else ["Manual specification"]
        )
        if suggestion_is_current and suggestion_evidence:
            basis_options.insert(0, "Verified Agent suggestion")
        specification_basis = st.selectbox(
            "Specification basis",
            basis_options,
            help=(
                "An Agent suggestion is selectable only when at least one source "
                "excerpt passed exact platform verification."
            ),
        )
        manual_confirmation_reason = ""
        if specification_basis == "Manual specification":
            manual_confirmation_reason = st.text_area(
                "Manual specification reason",
                help=(
                    "Explain why the Provider is confirming the fields without "
                    "traceable Agent evidence."
                ),
            )
        confirm_spec = st.checkbox(
            "I reviewed and confirm this AlgorithmSpec as the module source of truth",
            disabled=request_in_progress,
            help=(
                "The confirmed snapshot hash and Provider name are "
                "recorded in manifest.json."
            ),
        )
        required_spec_values = (
            algorithm_id,
            name,
            category,
            summary,
            objective,
            assumptions,
            inputs,
            states,
            actions,
            outputs,
            equations,
            pseudocode,
            environments,
        )
        specification_incomplete = (
            source_bytes is None
            or any(not str(value).strip() for value in required_spec_values)
        )
        if specification_incomplete:
            st.caption(
                "Create remains disabled until source material and all required "
                "AlgorithmSpec fields are present. Generate a suggestion or fill "
                "the empty fields manually."
            )
        submitted = st.form_submit_button(
            "Confirm AlgorithmSpec and Create Review Draft",
            type="primary",
            width="stretch",
            disabled=(
                request_in_progress
                or not provider.strip()
                or specification_incomplete
            ),
        )

    if request_in_progress:
        pending_request = st.session_state.get(suggestion_request_key, {})
        suggestion = None
        try:
            if (
                current_source_digest is None
                or pending_request.get("source_sha256")
                != current_source_digest
            ):
                raise AgentSpecError(
                    "The source changed before the Agent request started. "
                    "Please generate the suggestion again."
                )
            with st.spinner(
                "Agent is reading the source and building suggestions..."
            ):
                suggestion = suggest_algorithm_spec(
                    source_name or "source.txt",
                    extracted or "",
                    model=configuration.model,
                )
        except AgentSpecError as exc:
            details = describe_agent_error(exc)
            st.session_state[suggestion_error_key] = details
            _queue_action_response(
                "error",
                "AlgorithmSpec generation failed",
                details["summary"],
                request_id=details["request_id"],
                technical=details["technical"],
            )
        else:
            st.session_state.pop(suggestion_error_key, None)
            if stored_suggestion is not None:
                suggestion_history = list(
                    st.session_state.get(suggestion_history_key, [])
                )
                suggestion_history.append(dict(stored_suggestion))
                st.session_state[suggestion_history_key] = suggestion_history[-5:]
            st.session_state[suggestion_key] = suggestion.as_session_value()
            st.session_state[revision_key] = (
                int(st.session_state.get(revision_key, 0)) + 1
            )
            st.session_state["algorithm_admin_message"] = (
                "Agent suggestions generated. Review every field before "
                "creating the draft."
            )
        finally:
            st.session_state.pop(suggestion_request_key, None)
        st.rerun()

    if not submitted:
        return
    try:
        if not confirm_spec:
            raise ValueError(
                "Confirm the reviewed AlgorithmSpec before creating the draft."
            )
        if (
            specification_basis == "Verified Agent suggestion"
            and not suggestion_evidence
        ):
            raise ValueError(
                "A verified Agent suggestion requires at least one verified source excerpt."
            )
        if (
            specification_basis == "Manual specification"
            and not manual_confirmation_reason.strip()
        ):
            raise ValueError(
                "Enter a reason for confirming a manual specification."
            )
        hyperparameters = _hyperparameters_from_editor(hyperparameters_editor)
        animation_steps = _derivation_steps(animation_steps_editor)
        animation_symbols = _animation_symbols(animation_symbols_editor)
        if source_name is None or source_bytes is None:
            raise ValueError("A source file is required for a generic scaffold.")
        algorithm_spec_agent = None
        if specification_basis == "Verified Agent suggestion":
            algorithm_spec_agent = {
                "provider": stored_suggestion.get(
                    "provider", configuration.provider
                ),
                "framework": "langchain",
                "model": stored_suggestion.get("model"),
                "structured_output_method": stored_suggestion.get(
                    "structured_output_method"
                ),
                "response_id": stored_suggestion.get("response_id"),
                "generated_at": stored_suggestion.get("generated_at"),
                "source_text_sha256": stored_suggestion.get("source_sha256"),
                "source_characters": stored_suggestion.get(
                    "source_characters"
                ),
                "submitted_characters": stored_suggestion.get(
                    "submitted_characters"
                ),
                "source_truncated": stored_suggestion.get("source_truncated"),
                "evidence": stored_suggestion.get("evidence", []),
                "warnings": stored_suggestion.get("warnings", []),
                "agent_warnings": stored_suggestion.get("agent_warnings", []),
                "platform_warnings": stored_suggestion.get(
                    "platform_warnings", []
                ),
                "usage": stored_suggestion.get("usage", {}),
                "accepted_after_manual_review": True,
            }
        draft = create_draft(
            DraftInput(
                algorithm_id=algorithm_id.strip(),
                name=name.strip(),
                version=version.strip(),
                category=category.strip(),
                summary=summary.strip(),
                objective=objective.strip(),
                assumptions=_lines(assumptions),
                inputs=_lines(inputs),
                outputs=_lines(outputs),
                states=_lines(states),
                actions=_lines(actions),
                hyperparameters=hyperparameters,
                core_equations=_lines(equations),
                pseudocode=_lines(pseudocode),
                supported_environments=_lines(environments),
                source_name=source_name,
                source_bytes=source_bytes,
                reference_urls=_lines(references),
                generation_mode=(
                    "monte-carlo-preset" if preset else "template"
                ),
                animation_name=(
                    animation_upload.name
                    if animation_upload is not None
                    else None
                ),
                animation_bytes=(
                    animation_upload.getvalue()
                    if animation_upload is not None
                    else None
                ),
                animation_concept_markdown=animation_concept_markdown,
                animation_formula=animation_formula,
                animation_symbols=animation_symbols,
                animation_highlights=_lines(animation_highlights),
                animation_viewing_flow=_lines(animation_viewing_flow),
                animation_derivation_steps=animation_steps,
                algorithm_spec_agent=algorithm_spec_agent,
                algorithm_spec_confirmation={
                    "confirmed_by": provider.strip(),
                    "note": (
                        "Provider confirmed a verified Agent suggestion."
                        if specification_basis == "Verified Agent suggestion"
                        else (
                            "Provider confirmed the bundled reviewed preset."
                            if specification_basis == "Bundled reviewed preset"
                            else "Provider confirmed a manual specification: "
                            + manual_confirmation_reason.strip()
                        )
                    ),
                },
                experiment_design=experiment_design,
            )
        )
    except (AlgorithmPackageError, OSError, ValueError) as exc:
        _queue_action_response(
            "error",
            "Draft creation failed",
            str(exc),
            technical=str(exc),
        )
        st.rerun()
    else:
        st.session_state["algorithm_admin_message"] = (
            f"Draft {draft.key} created from a confirmed AlgorithmSpec. "
            "Generate and review Theory, Notebook, and Experiment separately."
        )
        st.session_state["algorithm_admin_next_section"] = "Review Drafts"
        st.rerun()


def _render_notebook_preview(path: Path) -> None:
    try:
        notebook = nbformat.read(path, as_version=4)
    except Exception as exc:
        st.error(f"Notebook preview failed: {exc}")
        return
    st.caption(f"{len(notebook.cells)} cells")
    for index, cell in enumerate(notebook.cells[:12], start=1):
        if cell.cell_type == "markdown":
            st.markdown(cell.source)
        elif cell.cell_type == "code":
            st.code(cell.source, language="python")
        else:
            st.text(cell.source)
        if index != min(len(notebook.cells), 12):
            st.divider()
    if len(notebook.cells) > 12:
        st.info("Preview is limited to the first 12 cells.")


def _queue_action_response(
    level: str,
    title: str,
    message: str,
    *,
    request_id: str | None = None,
    technical: str | None = None,
) -> None:
    st.session_state["algorithm_admin_action_response"] = {
        "level": level,
        "title": title,
        "message": message,
        "request_id": request_id,
        "technical": technical,
    }


@st.dialog("Action result", width="medium")
def _render_action_response_dialog(response: dict) -> None:
    st.markdown(f"### {response.get('title', 'Action result')}")
    level = response.get("level", "info")
    renderer = {
        "success": st.success,
        "error": st.error,
        "warning": st.warning,
    }.get(level, st.info)
    renderer(str(response.get("message", "")))
    if response.get("request_id"):
        st.caption(f"Provider request ID: `{response['request_id']}`")
    if response.get("technical"):
        with st.expander("Technical details", expanded=False):
            st.code(str(response["technical"]), language="text")
    if st.button("Close", key="close_algorithm_action_response", width="stretch"):
        st.rerun()


def _render_pending_action_response() -> None:
    response = st.session_state.pop("algorithm_admin_action_response", None)
    if response:
        _render_action_response_dialog(response)


def _run_and_rerun(
    action,
    *args,
    success: str,
    return_section: str | None = None,
    **kwargs,
) -> None:
    try:
        action(*args, **kwargs)
    except (AlgorithmPackageError, AgentSpecError, OSError, ValueError) as exc:
        _queue_action_response(
            "error",
            "Action failed",
            str(exc),
            technical=str(exc),
        )
        st.rerun()
    else:
        _queue_action_response("success", "Action completed", success)
        if return_section is not None:
            st.session_state["algorithm_admin_next_section"] = return_section
        st.rerun()


def _render_status(status: str, note: str = "") -> None:
    messages = {
        "awaiting_review": (
            st.info,
            "Awaiting review. Approve it or request changes.",
        ),
        "changes_requested": (
            st.warning,
            "Changes requested. Edit, upload a replacement, or regenerate before approval.",
        ),
        "approved": (
            st.success,
            "Approved and locked. Use Reopen for Changes to edit it again.",
        ),
        "validation_failed": (
            st.error,
            "Validation failed. Correct or replace the module before review.",
        ),
        "not_generated": (
            st.warning,
            "Safe scaffold only. Generate, edit, or replace this module before review.",
        ),
        "draft": (st.info, "Draft generated but not yet validated."),
        "installed": (st.success, "Installed."),
    }
    renderer, message = messages.get(
        status, (st.caption, f"Current status: {status}")
    )
    if note:
        message += f"\n\nLatest note: {note}"
    renderer(message)


def _render_spec_editor(draft, provider: str) -> None:
    manifest = draft.manifest
    if manifest is None:
        return
    raw = manifest.raw
    st.caption(
        "ID and version are immutable because they define the draft directory. "
        "Saving AlgorithmSpec sends every generated module back to Needs Changes."
    )
    with st.form(f"edit_spec_{draft.key}"):
        current_algorithm = raw["algorithm"]
        left, right = st.columns(2)
        left.text_input("Algorithm ID", value=manifest.algorithm_id, disabled=True)
        right.text_input("Version", value=manifest.version, disabled=True)
        name = left.text_input("Name", value=manifest.name)
        category = right.text_input("Category", value=manifest.category)
        summary = st.text_area("Summary", value=manifest.summary)
        objective = st.text_area(
            "Learning objective", value=current_algorithm.get("objective", "")
        )
        assumptions = st.text_area(
            "Assumptions (one per line)",
            value="\n".join(current_algorithm.get("assumptions", [])),
        )
        inputs = st.text_area(
            "Inputs (one per line)",
            value="\n".join(current_algorithm.get("inputs", [])),
        )
        outputs = st.text_area(
            "Outputs (one per line)",
            value="\n".join(current_algorithm.get("outputs", [])),
        )
        states = st.text_area(
            "State description (one per line)",
            value="\n".join(current_algorithm.get("states", [])),
        )
        actions = st.text_area(
            "Action description (one per line)",
            value="\n".join(current_algorithm.get("actions", [])),
        )
        st.markdown("**Hyperparameters**")
        hyperparameters_editor = _render_hyperparameter_editor(
            current_algorithm.get("hyperparameters", {}),
            key=f"review_hyperparameters_{draft.key}",
        )
        equations = st.text_area(
            "Core equations (one per line)",
            value="\n".join(current_algorithm.get("core_equations", [])),
        )
        pseudocode = st.text_area(
            "Pseudocode (one step per line)",
            value="\n".join(current_algorithm.get("pseudocode", [])),
        )
        environments = st.text_area(
            "Supported environments (one per line)",
            value="\n".join(current_algorithm.get("supported_environments", [])),
        )
        submitted = st.form_submit_button(
            "Save AlgorithmSpec",
            disabled=not provider.strip(),
            width="stretch",
        )
    if submitted:
        try:
            standard_fields = {
                "objective",
                "assumptions",
                "inputs",
                "outputs",
                "states",
                "actions",
                "hyperparameters",
                "core_equations",
                "pseudocode",
                "supported_environments",
            }
            algorithm = {
                key: value
                for key, value in current_algorithm.items()
                if key not in standard_fields
            }
            algorithm.update(
                {
                    "objective": objective.strip(),
                    "assumptions": list(_lines(assumptions)),
                    "inputs": list(_lines(inputs)),
                    "outputs": list(_lines(outputs)),
                    "states": list(_lines(states)),
                    "actions": list(_lines(actions)),
                    "hyperparameters": _hyperparameters_from_editor(
                        hyperparameters_editor
                    ),
                    "core_equations": list(_lines(equations)),
                    "pseudocode": list(_lines(pseudocode)),
                    "supported_environments": list(_lines(environments)),
                }
            )
            save_algorithm_spec(
                draft.key,
                name=name,
                category=category,
                summary=summary,
                algorithm=algorithm,
                reviewer=provider,
            )
        except (AlgorithmPackageError, OSError, ValueError) as exc:
            _queue_action_response(
                "error",
                "AlgorithmSpec update failed",
                str(exc),
                technical=str(exc),
            )
            st.rerun()
        else:
            st.session_state["algorithm_admin_message"] = (
                "AlgorithmSpec saved. All generated modules now require changes."
            )
            st.rerun()


def _render_experiment_design_editor(draft, provider: str) -> None:
    manifest = draft.manifest
    if manifest is None:
        return
    current = manifest.algorithm.get("experiment_design")
    current_preset_id = None
    if isinstance(current, dict):
        provenance = current.get("provenance", {})
        if isinstance(provenance, dict):
            candidate = provenance.get("preset_id")
            if candidate in EXPERIMENT_DESIGN_PRESETS:
                current_preset_id = candidate

    option_ids = list(EXPERIMENT_DESIGN_PRESETS)
    if current is not None and current_preset_id is None:
        option_ids.insert(0, "keep-current")
    option_ids.append("none")
    default_id = current_preset_id or (
        "keep-current" if current is not None else "none"
    )

    def _label(option_id: str) -> str:
        if option_id == "none":
            return "No declarative map · expert decides later"
        if option_id == "keep-current":
            return "Keep current custom/source-derived scenario"
        preset = EXPERIMENT_DESIGN_PRESETS[option_id]
        suffix = " · current" if option_id == current_preset_id else ""
        return preset.label + suffix

    with st.container(border=True):
        st.markdown("#### Experiment teaching scenario")
        st.caption(
            "This is a scoped AlgorithmSpec edit: changing it resets only "
            "Experiment, while Theory, Notebook, and Animation keep their "
            "current review states."
        )
        selected_id = st.selectbox(
            "Experiment scenario",
            option_ids,
            index=option_ids.index(default_id),
            format_func=_label,
            key=f"review_experiment_design_{draft.key}",
        )
        if selected_id == "none":
            selected_design = None
            st.info("Experiment can run without a Task map or policy grid.")
        elif selected_id == "keep-current":
            selected_design = current
        else:
            selected_design = get_experiment_design_preset(selected_id)
            st.caption(selected_design["provenance"]["note"])
            st.markdown(f"**Mission:** {selected_design['task']['mission']}")
            st.code(
                "\n".join(selected_design["environment_map"]["layout"]),
                language="text",
            )
        if st.button(
            "Save Experiment Scenario",
            key=f"save_experiment_design_{draft.key}",
            disabled=(selected_design == current or not provider.strip()),
            width="stretch",
        ):
            _run_and_rerun(
                save_experiment_design,
                draft.key,
                selected_design,
                provider,
                success=(
                    "Experiment scenario saved. Only Experiment now requires "
                    "regeneration or replacement."
                ),
            )


def _render_publication_blockers(draft, reviewer: str) -> None:
    manifest = draft.manifest
    if manifest is None:
        return
    raw = manifest.raw
    blockers = raw["generation"].get("blocking_flags", [])
    if not blockers:
        return
    st.error("Installation blocked: " + ", ".join(blockers))
    if "placeholder_content" not in blockers:
        return
    completion = {
        "Theory replaced by an edit or Agent generation": module_content_ready(
            raw, "theory"
        ),
        "Notebook replaced by upload or Agent generation": module_content_ready(
            raw, "notebook"
        ),
        "Experiment replaced by upload or Agent generation": module_content_ready(
            raw, "experiment"
        ),
    }
    st.markdown("**Scaffold replacement checklist**")
    for label, done in completion.items():
        st.markdown(f"{'✅' if done else '⬜'} {label}")
    st.caption(
        "When all three files come from successful Agent generation, the blocker "
        "is cleared automatically. Mixed Agent/manual completion still requires "
        "this confirmation. Completed modules can be approved individually while "
        "the installation blocker remains."
    )
    completion_note = st.text_input(
        "Implementation completion note",
        key=f"completion_note_{draft.key}",
    )
    confirm = st.checkbox(
        "I confirm the placeholder algorithm content was replaced and validated",
        key=f"completion_confirm_{draft.key}",
    )
    if st.button(
        "Clear Placeholder Installation Blocker",
        disabled=(
            not all(completion.values())
            or not reviewer.strip()
            or not completion_note.strip()
            or not confirm
        ),
        width="stretch",
    ):
        _run_and_rerun(
            resolve_placeholder_blocker,
            draft.key,
            reviewer,
            completion_note,
        success="Content completion recorded; installation blocker cleared.",
    )


def _render_module_review(
    draft, module: str, provider: str, reviewer: str
) -> None:
    manifest = draft.manifest
    if manifest is None:
        return
    raw = manifest.raw
    state = raw["review"]["modules"][module]
    status = state["status"]
    open_module = st.session_state.get("algorithm_admin_open_module", "")
    current_module = f"{draft.key}:{module}"
    agent_request_key = f"module_agent_request_{draft.key}_{module}"
    agent_feedback_key = f"module_agent_feedback_{draft.key}_{module}"
    pending_agent_request = st.session_state.get(agent_request_key)
    agent_request_in_progress = bool(pending_agent_request)
    with st.container(border=True):
        st.subheader(f"{module.title()} · {status}")
        _render_status(status, state.get("note", ""))
        agent_feedback = st.session_state.get(agent_feedback_key)
        if agent_feedback:
            feedback_message = str(agent_feedback.get("message", ""))
            if agent_feedback.get("level") == "error":
                st.error(feedback_message)
                if agent_feedback.get("request_id"):
                    st.caption(
                        f"Provider request ID: `{agent_feedback['request_id']}`"
                    )
                if agent_feedback.get("technical"):
                    with st.expander("Technical details", expanded=False):
                        st.code(agent_feedback["technical"], language="text")
            elif agent_feedback.get("level") == "success":
                st.success(feedback_message)
            else:
                st.info(feedback_message)
        content_ready = module_content_ready(raw, module)
        if (
            "placeholder_content"
            in raw["generation"].get("blocking_flags", [])
            and not content_ready
        ):
            st.info(
                "No learner-facing content has been generated for this module. "
                "The platform keeps an internal safety file for validation, but "
                "it is hidden here and cannot be approved."
            )
        if not provider.strip():
            st.caption(
                "🔒 Enter Provider name in the sidebar and select Apply role "
                "names to enable generation and editing. Reviewer name is "
                "required only for review actions."
            )
        if agent_request_in_progress:
            st.info(
                f"{module.title()} Agent request is running. Module controls "
                "are temporarily disabled."
            )
        locked = status == "approved" or agent_request_in_progress
        if module == "theory":
            st.caption(
                "Revision paths: edit the Markdown and save, or ask the Theory "
                "Agent to regenerate from the review note."
            )
        elif module in {"notebook", "experiment"}:
            st.caption(
                f"Revision paths: ask the {module.title()} Agent to regenerate "
                "from the review note, or download the current file, edit it "
                "locally, and upload a validated replacement. Needs Changes "
                "records the request; it is not an online code editor."
            )
        show_module_content = status != "not_generated" or content_ready
        if not show_module_content and module in CORE_MODULES:
            st.markdown(f"#### No {module.title()} content yet")
            if module == "theory":
                st.caption(
                    "Generate Theory with the Agent. After generation succeeds, "
                    "the editable Theory Markdown field will appear here."
                )
            else:
                st.caption(
                    f"Generate {module.title()} with its Agent, or upload a "
                    "replacement file. Nothing is prefilled from the internal scaffold."
                )

        if module == "theory" and show_module_content:
            theory_path = draft.path / manifest.theory_file
            content = (
                theory_path.read_text(encoding="utf-8")
                if show_module_content
                else ""
            )
            edited = st.text_area(
                "Theory Markdown",
                value=content,
                height=420,
                key=f"theory_{draft.key}",
                disabled=locked,
            )
            if st.button(
                "Save Theory",
                key=f"save_theory_{draft.key}",
                disabled=locked or not provider.strip(),
            ):
                _run_and_rerun(
                    save_theory,
                    draft.key,
                    edited,
                    provider,
                    success="Theory saved and approval reset.",
                )
        elif module == "notebook" and show_module_content:
            notebook_path = draft.path / manifest.notebook["file"]
            st.download_button(
                "Download current Notebook",
                data=notebook_path.read_bytes(),
                file_name=notebook_path.name,
                mime="application/x-ipynb+json",
                key=f"download_{draft.key}_notebook",
                width="stretch",
            )
            _render_notebook_preview(notebook_path)
        elif module == "experiment" and show_module_content:
            experiment_path = draft.path / manifest.experiment["module"]
            st.download_button(
                "Download current Experiment",
                data=experiment_path.read_bytes(),
                file_name=experiment_path.name,
                mime="text/x-python",
                key=f"download_{draft.key}_experiment",
                width="stretch",
            )
            source = experiment_path.read_text(encoding="utf-8")
            st.code(source, language="python", line_numbers=True)
        elif module == "animation":
            animation = raw["modules"]["animation"]
            st.video(str(draft.path / animation["file"]))
            replacement = st.file_uploader(
                "Replace Animation MP4",
                type=["mp4"],
                key=f"replacement_{draft.key}_animation",
                disabled=locked,
                help=(
                    "Upload a finished MP4. This version does not generate "
                    "animations or execute Manim source code."
                ),
            )
            animation_concept_markdown = st.text_area(
                "Animation concept (Markdown)",
                value=animation.get("concept_markdown", raw.get("summary", "")),
                key=f"animation_concept_{draft.key}",
                disabled=locked,
            )
            animation_formula = st.text_input(
                "Animation formula",
                value=animation.get("formula", ""),
                key=f"animation_formula_{draft.key}",
                disabled=locked,
            )
            animation_symbols_editor = _render_symbol_editor(
                "Animation symbols",
                animation.get("symbols", []),
                key=f"animation_symbols_{draft.key}",
                disabled=locked,
            )
            animation_highlights = st.text_area(
                "Animation highlights (one per line)",
                value="\n".join(animation.get("highlights", [])),
                key=f"animation_highlights_{draft.key}",
                disabled=locked,
            )
            animation_viewing_flow = st.text_area(
                "Animation viewing flow (one step per line)",
                value="\n".join(animation.get("viewing_flow", [])),
                key=f"animation_viewing_flow_{draft.key}",
                disabled=locked,
            )
            animation_steps_editor = _render_derivation_editor(
                "Animation derivation steps",
                animation.get("derivation_steps", []),
                key=f"animation_steps_{draft.key}",
                disabled=locked,
            )
            with st.expander("Animation page preview", expanded=False):
                st.markdown(animation_concept_markdown or "_No concept text._")
                if animation_highlights.strip():
                    st.markdown("**What to look for**")
                    st.markdown(
                        "\n".join(
                            f"- {item}" for item in _lines(animation_highlights)
                        )
                    )
                if animation_formula.strip():
                    st.markdown("**Mathematical core**")
                    st.latex(animation_formula)
                try:
                    preview_symbols = _animation_symbols(animation_symbols_editor)
                    preview_steps = _derivation_steps(animation_steps_editor)
                except ValueError as exc:
                    st.warning(f"Preview metadata is invalid: {exc}")
                else:
                    if preview_symbols:
                        st.markdown(
                            "**Symbols:** "
                            + ", ".join(
                                f"{item['symbol']}={item['meaning']}"
                                for item in preview_symbols
                            )
                        )
                    if animation_viewing_flow.strip():
                        st.markdown("**Suggested viewing flow**")
                        st.markdown(
                            "\n".join(
                                f"- {item}"
                                for item in _lines(animation_viewing_flow)
                            )
                        )
                    if preview_steps:
                        st.markdown("**Derivation Notes**")
                        for index, step in enumerate(preview_steps, start=1):
                            st.markdown(
                                f"**Step {index}: {step.get('title', '').strip()}**"
                            )
                            if step.get("text"):
                                st.markdown(step["text"])
                            latex = step.get("latex")
                            if isinstance(latex, list):
                                for block in latex:
                                    st.latex(block)
                            elif latex:
                                st.latex(latex)
            if st.button(
                "Save Animation / Upload Replacement",
                key=f"save_animation_{draft.key}",
                disabled=locked or not provider.strip(),
                width="stretch",
            ):
                try:
                    steps = _derivation_steps(animation_steps_editor)
                    symbols = _animation_symbols(animation_symbols_editor)
                except ValueError as exc:
                    _queue_action_response(
                        "error",
                        "Animation metadata is incomplete",
                        str(exc),
                    )
                    st.rerun()
                else:
                    _run_and_rerun(
                        save_animation_module,
                        draft.key,
                        file_name=(
                            replacement.name if replacement is not None else None
                        ),
                        payload=(
                            replacement.getvalue()
                            if replacement is not None
                            else None
                        ),
                        concept_markdown=animation_concept_markdown,
                        formula=animation_formula,
                        symbols=symbols,
                        highlights=_lines(animation_highlights),
                        viewing_flow=_lines(animation_viewing_flow),
                        derivation_steps=steps,
                        reviewer=provider,
                        success="Animation validated and saved.",
                    )
            confirm_remove = st.checkbox(
                "Move this Animation to recoverable trash",
                key=f"remove_animation_confirm_{draft.key}",
                disabled=locked,
            )
            if st.button(
                "Remove Optional Animation",
                key=f"remove_animation_{draft.key}",
                disabled=(
                    locked
                    or not confirm_remove
                    or not provider.strip()
                ),
            ):
                _run_and_rerun(
                    remove_animation_module,
                    draft.key,
                    provider,
                    success="Animation removed and its MP4 moved to recoverable trash.",
                )

        if module in {"notebook", "experiment"}:
            replacement = st.file_uploader(
                f"Replace {module.title()} file",
                type=["ipynb"] if module == "notebook" else ["py"],
                key=f"replacement_{draft.key}_{module}",
                disabled=locked,
                help=(
                    "The uploaded file is validated in a temporary copy before "
                    "it replaces the current module."
                ),
            )
            if st.button(
                f"Upload {module.title()} Replacement",
                key=f"replace_{draft.key}_{module}",
                disabled=(
                    locked
                    or replacement is None
                    or not provider.strip()
                ),
                width="stretch",
            ):
                _run_and_rerun(
                    replace_module_file,
                    draft.key,
                    module,
                    replacement.name,
                    replacement.getvalue(),
                    provider,
                    success=(
                        f"{module.title()} replacement validated and saved."
                    ),
                )

        generation_mode = raw["generation"]["mode"]
        is_revision = bool(raw["generation"].get("revision_of"))
        scaffold_revision_protected = is_revision and generation_mode == "template"
        module_agent_configuration = (
            get_module_agent_configuration(module)
            if module in CORE_MODULES
            else None
        )
        use_agent_regeneration = bool(
            module_agent_configuration
            and module_agent_configuration.configured
            and generation_mode == "template"
        )
        module_generation_history = raw["generation"].get(
            "module_generations", {}
        ).get(module, [])
        if module == "animation":
            regeneration_help = (
                "Animation has no generator in this version. Use Needs Changes, "
                "then upload a revised finished MP4 or edit its metadata."
            )
        elif generation_mode == "monte-carlo-preset":
            regeneration_help = (
                "Regeneration copies the same bundled Monte Carlo file. It does "
                "not interpret review notes."
            )
        elif use_agent_regeneration:
            regeneration_help = (
                f"The {module.title()} Agent uses the confirmed AlgorithmSpec, "
                "original source, current file, and the latest change request. "
                f"Model: `{module_agent_configuration.model}`."
            )
        elif scaffold_revision_protected:
            regeneration_help = (
                "This is a revision of an installed implementation. Scaffold "
                "regeneration is disabled so the carried-forward working file "
                "cannot be replaced by placeholder code. Configure the module "
                "Agent, or download, edit, and upload a validated replacement."
            )
        else:
            regeneration_help = (
                "Regeneration rebuilds the deterministic template from the "
                "current AlgorithmSpec and original source. It does not interpret "
                "review notes. Configure the module Agent to use model generation."
            )
        st.caption(regeneration_help)
        note = st.text_area(
            "Review note / required change reason",
            key=f"review_note_{draft.key}_{module}",
        )
        if not note.strip():
            if status == "changes_requested":
                st.caption(
                    "Cancel Change Request stays disabled until a cancellation "
                    "reason is entered above."
                )
            elif status == "awaiting_review":
                st.caption(
                    "Enter a reason above, then use Needs Changes."
                )
            elif status == "approved":
                st.caption(
                    "To edit this approved module, enter a reason above and use "
                    "Reopen for Changes. The Provider can edit it after it reopens."
                )
        generation_action = ""
        confirm_regenerate = True
        if module != "animation":
            generation_action = (
                "Retry generation"
                if agent_feedback and agent_feedback.get("level") == "error"
                else
                "Regenerate with Agent"
                if use_agent_regeneration and module_generation_history
                else (
                    "Generate with Agent"
                    if use_agent_regeneration
                    else (
                        "Module Agent unavailable"
                        if scaffold_revision_protected
                        else "Regenerate Scaffold"
                    )
                )
            )
            requires_overwrite_confirmation = bool(
                module_generation_history or content_ready
            )
            if requires_overwrite_confirmation:
                confirm_regenerate = st.checkbox(
                    f"I understand {generation_action} overwrites the current "
                    f"{module.title()} file",
                    key=f"confirm_regenerate_{draft.key}_{module}",
                    disabled=locked,
                )
                if not confirm_regenerate:
                    st.caption(
                        "Confirm the overwrite above to enable regeneration."
                    )
            else:
                st.caption(
                    "This is the first generation for this scaffold. It can be "
                    "started with one click."
                )
        action_columns = st.columns(2 if module == "animation" else 3)
        left, middle = action_columns[:2]
        blocking_flags = raw["generation"].get("blocking_flags", [])
        approval_blocked = any(
            flag
            for flag in blocking_flags
            if flag != "placeholder_content"
        ) or (
            "placeholder_content" in blocking_flags
            and not content_ready
        )
        approve_disabled = (
            status != "awaiting_review"
            or not draft.report.valid
            or approval_blocked
            or not reviewer.strip()
            or agent_request_in_progress
        )
        if left.button(
            "Approve",
            key=f"approve_{draft.key}_{module}",
            disabled=approve_disabled,
            width="stretch",
        ):
            _run_and_rerun(
                approve_module,
                draft.key,
                module,
                reviewer,
                note,
                success=f"{module.title()} approved.",
            )
        if status == "changes_requested":
            if middle.button(
                "Cancel Change Request",
                key=f"cancel_changes_{draft.key}_{module}",
                disabled=(
                    not reviewer.strip()
                    or agent_request_in_progress
                ),
                width="stretch",
            ):
                if not note.strip():
                    _queue_action_response(
                        "warning",
                        "Cancellation reason required",
                        "Enter a cancellation reason in the review note field, "
                        "then select Cancel Change Request again.",
                    )
                    st.rerun()
                else:
                    st.session_state["algorithm_admin_open_module"] = current_module
                    _run_and_rerun(
                        cancel_change_request,
                        draft.key,
                        module,
                        reviewer,
                        note,
                        success=(
                            f"{module.title()} change request cancelled; "
                            "the current revision is awaiting review."
                        ),
                    )
        elif middle.button(
            "Reopen for Changes" if status == "approved" else "Needs Changes",
            key=f"changes_{draft.key}_{module}",
            disabled=(
                status not in {"awaiting_review", "approved"}
                or not reviewer.strip()
                or agent_request_in_progress
            ),
            width="stretch",
        ):
            if not note.strip():
                _queue_action_response(
                    "warning",
                    "Change reason required",
                    "Enter the required change reason in the review note field, "
                    "then select this action again.",
                )
                st.rerun()
            else:
                st.session_state["algorithm_admin_open_module"] = current_module
                _run_and_rerun(
                    request_changes,
                    draft.key,
                    module,
                    reviewer,
                    note,
                    success=f"{module.title()} reopened for Provider changes.",
                )
        if module != "animation":
            right = action_columns[2]
            if right.button(
                generation_action,
                key=f"regenerate_{draft.key}_{module}",
                disabled=(
                    locked
                    or not provider.strip()
                    or not confirm_regenerate
                    or (scaffold_revision_protected and not use_agent_regeneration)
                ),
                width="stretch",
            ):
                st.session_state["algorithm_admin_open_module"] = current_module
                if use_agent_regeneration:
                    st.session_state[agent_feedback_key] = {
                        "level": "info",
                        "message": (
                            f"{module.title()} Agent request started. Keep this "
                            "page open; the result or error will remain here."
                        ),
                    }
                    st.session_state[agent_request_key] = {
                        "provider": provider,
                        "note": note,
                    }
                    st.rerun()
                else:
                    _run_and_rerun(
                        regenerate_module,
                        draft.key,
                        module,
                        provider,
                        success=f"{module.title()} scaffold regenerated.",
                    )

        if agent_request_in_progress:
            try:
                with st.spinner(
                    f"{module.title()} Agent is generating and validating..."
                ):
                    generate_module_with_agent(
                        draft.key,
                        module,
                        str(pending_agent_request.get("provider", "")),
                        str(pending_agent_request.get("note", "")),
                    )
            except (
                AlgorithmPackageError,
                AgentSpecError,
                OSError,
                ValueError,
            ) as exc:
                details = describe_agent_error(exc)
                st.session_state[agent_feedback_key] = {
                    "level": "error",
                    "message": details["summary"],
                    "request_id": details["request_id"],
                    "technical": details["technical"],
                }
                _queue_action_response(
                    "error",
                    f"{module.title()} generation failed",
                    details["summary"],
                    request_id=details["request_id"],
                    technical=details["technical"],
                )
            else:
                st.session_state[agent_feedback_key] = {
                    "level": "success",
                    "message": (
                        f"{module.title()} generated and validated successfully. "
                        "The new revision is awaiting review."
                    ),
                }
                _queue_action_response(
                    "success",
                    f"{module.title()} generated",
                    f"{module.title()} generated by Agent and is awaiting review.",
                )
            finally:
                st.session_state.pop(agent_request_key, None)
            st.session_state["algorithm_admin_open_module"] = current_module
            st.session_state["algorithm_admin_next_section"] = "Review Drafts"
            st.rerun()


def _render_add_animation(draft, provider: str) -> None:
    with st.container(border=True):
        st.markdown("#### Step 4 · Upload MP4")
        st.info(
            "Animation is optional and does not block installation while absent. "
            "Use the creator guidance above to make it outside the website, then "
            "upload the finished MP4 here."
        )
        try:
            guidance = load_animation_guidance(draft.key)
        except AlgorithmPackageError as exc:
            st.error(f"Saved animation guidance is invalid: {exc}")
            guidance = None
        metadata = guidance.get("metadata", {}) if guidance else {}
        uploaded = st.file_uploader(
            "Finished Animation MP4",
            type=["mp4"],
            key=f"add_animation_file_{draft.key}",
        )
        if uploaded is not None:
            st.video(uploaded.getvalue())
        elif not provider.strip():
            st.caption("Disabled: Provider name required and an MP4 must be selected.")
        else:
            st.caption("Disabled until a finished MP4 is selected.")
        concept_markdown = st.text_area(
            "Animation concept (Markdown)",
            value=(
                metadata.get("concept_markdown", "")
                or (draft.manifest.summary if draft.manifest is not None else "")
            ),
            key=f"add_animation_concept_{draft.key}",
        )
        formula = st.text_input(
            "Animation formula",
            value=metadata.get("formula", ""),
            key=f"add_animation_formula_{draft.key}",
        )
        symbols_editor = _render_symbol_editor(
            "Animation symbols",
            metadata.get("symbols", []),
            key=f"add_animation_symbols_{draft.key}",
        )
        highlights = st.text_area(
            "Animation highlights (one per line)",
            value="\n".join(metadata.get("highlights", [])),
            key=f"add_animation_highlights_{draft.key}",
        )
        viewing_flow = st.text_area(
            "Animation viewing flow (one step per line)",
            value="\n".join(metadata.get("viewing_flow", [])),
            key=f"add_animation_viewing_flow_{draft.key}",
        )
        steps_editor = _render_derivation_editor(
            "Animation derivation steps",
            metadata.get("derivation_steps", []),
            key=f"add_animation_steps_{draft.key}",
        )
        if st.button(
            "Add Animation to Draft",
            key=f"add_animation_{draft.key}",
            disabled=uploaded is None or not provider.strip(),
            width="stretch",
        ):
            try:
                steps = _derivation_steps(steps_editor)
                symbols = _animation_symbols(symbols_editor)
            except ValueError as exc:
                _queue_action_response(
                    "error",
                    "Animation metadata is incomplete",
                    str(exc),
                )
                st.rerun()
            else:
                _run_and_rerun(
                    save_animation_module,
                    draft.key,
                    file_name=uploaded.name,
                    payload=uploaded.getvalue(),
                    concept_markdown=concept_markdown,
                    formula=formula,
                    symbols=symbols,
                    highlights=_lines(highlights),
                    viewing_flow=_lines(viewing_flow),
                    derivation_steps=steps,
                    reviewer=provider,
                    success="Animation added and is awaiting review.",
                )


def _render_animation_guidance(draft, provider: str) -> None:
    """Render creator planning independently from the optional MP4 module."""

    with st.container(border=True):
        st.markdown("### Animation production workflow")
        st.info(
            "This produces a storyboard and production checklist only. The "
            "website does not execute Manim/FFmpeg or render video. A creator "
            "makes the MP4 externally and uploads it for review. Generated "
            "guidance and Creator Kit documents use English."
        )
        with st.expander("How the Animation workflow works", expanded=False):
            if ANIMATION_WORKFLOW_USER_GUIDE.is_file():
                animation_guide = ANIMATION_WORKFLOW_USER_GUIDE.read_text(
                    encoding="utf-8"
                )
                st.markdown(animation_guide)
                st.download_button(
                    "Download Animation Workflow User Guide",
                    data=animation_guide.encode("utf-8"),
                    file_name="animation-workflow-user-guide.md",
                    mime="text/markdown",
                    key=f"animation_workflow_guide_{draft.key}",
                    width="stretch",
                )
            else:
                st.info("The Animation workflow guide is unavailable.")
        try:
            options_state = load_animation_options(draft.key)
        except AlgorithmPackageError as exc:
            st.error(f"Saved animation concepts are invalid: {exc}")
            return
        configuration = get_animation_planner_configuration()
        option_error_key = f"animation_option_error_{draft.key}"
        option_error = st.session_state.get(option_error_key)
        if option_error:
            st.error(option_error["summary"])
            if option_error.get("request_id"):
                st.caption(f"Provider request ID: `{option_error['request_id']}`")
            with st.expander("Technical details", expanded=False):
                st.code(option_error["technical"], language="text")

        st.markdown("#### Step 1 · Generate three concepts")
        provider_note = st.text_area(
            "Concept note for the Agent (optional)",
            key=f"animation_option_note_{draft.key}",
            placeholder="Example: compare two value tables and stay under 90 seconds.",
        )
        option_left, option_right = st.columns(2)
        if option_left.button(
            "Create three starter concepts",
            key=f"animation_options_default_{draft.key}",
            disabled=not provider.strip(),
            width="stretch",
        ):
            _run_and_rerun(
                generate_default_animation_options,
                draft.key,
                provider,
                success="Three animation concepts created from AlgorithmSpec.",
                return_section="Review Drafts",
            )
        if option_right.button(
            "Generate three concepts with Agent",
            key=f"animation_options_agent_{draft.key}",
            disabled=not provider.strip() or not configuration.configured,
            width="stretch",
        ):
            try:
                with st.spinner("Animation Concept Agent is preparing three options..."):
                    generate_animation_options_with_agent(
                        draft.key, provider, provider_note
                    )
            except (AlgorithmPackageError, AgentSpecError, OSError, ValueError) as exc:
                details = describe_agent_error(exc)
                st.session_state[option_error_key] = details
                _queue_action_response(
                    "error",
                    "Animation concept generation failed",
                    details["summary"],
                    request_id=details["request_id"],
                    technical=details["technical"],
                )
            else:
                st.session_state.pop(option_error_key, None)
                st.session_state["algorithm_admin_message"] = (
                    "Three animation concepts were generated. Select one in Animation."
                )
            st.session_state["algorithm_admin_next_section"] = "Review Drafts"
            st.rerun()
        if not provider.strip():
            st.caption("Disabled: Provider name required.")
        elif not configuration.configured:
            st.caption(
                "Agent concept generation disabled: Planning Agent unavailable. "
                "The three starter concepts remain available."
            )

        selected_option = None
        st.markdown("#### Step 2 · Select one concept")
        if options_state is None:
            st.caption("Generate three concepts before selecting one.")
        else:
            options = options_state["options"]
            labels = {
                (
                    f"{item['title']} · {item['complexity']} complexity · "
                    f"{item['estimated_duration_seconds']}s"
                ): item
                for item in options
            }
            current_id = options_state.get("selected_option_id")
            default_index = next(
                (
                    index
                    for index, item in enumerate(labels.values())
                    if item["option_id"] == current_id
                ),
                0,
            )
            selected_label = st.radio(
                "Animation concepts",
                list(labels),
                index=default_index,
                key=f"animation_option_choice_{draft.key}",
            )
            selected_option = labels[selected_label]
            with st.container(border=True):
                st.markdown(f"**Teaching focus:** {selected_option['teaching_focus']}")
                st.markdown(f"**Visual approach:** {selected_option['visual_approach']}")
                st.markdown(f"**Production cost:** {selected_option['production_cost']}")
                st.markdown(f"**Best use case:** {selected_option['best_use_case']}")
                st.markdown("**Trade-offs**")
                for item in selected_option["trade_offs"]:
                    st.markdown(f"- {item}")
            if st.button(
                "Select this concept",
                key=f"select_animation_option_{draft.key}",
                disabled=(
                    not provider.strip()
                    or selected_option["option_id"] == current_id
                ),
                width="stretch",
            ):
                _run_and_rerun(
                    select_animation_option,
                    draft.key,
                    selected_option["option_id"],
                    provider,
                    success=f"Selected animation concept: {selected_option['title']}.",
                    return_section="Review Drafts",
                )
            if selected_option["option_id"] == current_id:
                st.success("This concept is selected for storyboard generation.")

        st.markdown("#### Step 3 · Build and review the storyboard")
        try:
            guidance = load_animation_guidance(draft.key)
        except AlgorithmPackageError as exc:
            st.error(f"Saved animation guidance is invalid: {exc}")
            return
        guidance_record = draft.manifest.raw.get("generation", {}).get(
            "animation_guidance", {}
        )
        if guidance is not None and guidance_record.get("stale_since"):
            st.warning(
                "AlgorithmSpec changed after this guidance was saved. Review "
                "it carefully or replace it before giving it to the creator."
            )
        if configuration.configured:
            st.caption(
                "Planning Agent: available · "
                f"model `{configuration.model}` · "
                f"structured output `{configuration.structured_output_method}`"
            )
        else:
            st.caption(
                "Planning Agent: unavailable · " + configuration.message
                + " The deterministic AlgorithmSpec starter still works."
            )

        overwrite = True
        if guidance is not None:
            overwrite = st.checkbox(
                "Replace the current creator guidance",
                key=f"animation_guidance_overwrite_{draft.key}",
                help=(
                    "This replaces only the storyboard/production advice. It "
                    "does not change the uploaded MP4 or Animation review state."
                ),
            )
        left, right = st.columns(2)
        if left.button(
            "Create AlgorithmSpec Starter" if guidance is None else "Recreate Starter",
            key=f"animation_guidance_default_{draft.key}",
            disabled=(
                not provider.strip()
                or not overwrite
                or options_state is None
                or not options_state.get("selected_option_id")
            ),
            width="stretch",
        ):
            _run_and_rerun(
                generate_default_animation_guidance,
                draft.key,
                provider,
                success="Animation creator guidance created from AlgorithmSpec.",
                return_section="Review Drafts",
            )
        agent_note = st.text_area(
            "Planning note for the Agent (optional)",
            key=f"animation_guidance_agent_note_{draft.key}",
            placeholder=(
                "Example: focus on the 4×4 GridWorld transition and keep the "
                "video under 90 seconds."
            ),
        )
        guidance_error_key = f"animation_guidance_error_{draft.key}"
        guidance_error = st.session_state.get(guidance_error_key)
        if guidance_error:
            st.error(guidance_error["summary"])
            if guidance_error.get("request_id"):
                st.caption(f"Provider request ID: `{guidance_error['request_id']}`")
            with st.expander("Technical details", expanded=False):
                st.code(guidance_error["technical"], language="text")
        if right.button(
            "Retry storyboard generation"
            if guidance_error
            else "Generate with Planning Agent",
            key=f"animation_guidance_agent_{draft.key}",
            disabled=(
                not provider.strip()
                or not configuration.configured
                or not overwrite
                or options_state is None
                or not options_state.get("selected_option_id")
            ),
            width="stretch",
        ):
            try:
                with st.spinner(
                    "Planning Agent is building the storyboard. It will not render video..."
                ):
                    generate_animation_guidance_with_agent(
                        draft.key, provider, agent_note
                    )
            except (AlgorithmPackageError, AgentSpecError, OSError, ValueError) as exc:
                details = describe_agent_error(exc)
                st.session_state[guidance_error_key] = details
                _queue_action_response(
                    "error",
                    "Storyboard generation failed",
                    details["summary"],
                    request_id=details["request_id"],
                    technical=details["technical"],
                )
            else:
                st.session_state.pop(guidance_error_key, None)
                st.session_state["algorithm_admin_message"] = (
                    "Animation creator guidance generated by Agent."
                )
            st.session_state["algorithm_admin_next_section"] = "Review Drafts"
            st.rerun()

        if not provider.strip():
            st.caption("Storyboard generation disabled: Provider name required.")
        elif options_state is None or not options_state.get("selected_option_id"):
            st.caption("Storyboard generation disabled: select one concept first.")
        elif not configuration.configured:
            st.caption(
                "Agent storyboard generation disabled: Planning Agent unavailable. "
                "The AlgorithmSpec starter remains available."
            )

        if guidance is None:
            st.caption(
                "No creator guidance has been saved yet. Creating it is optional "
                "and does not affect installation."
            )
            return

        st.markdown(f"#### {guidance['title']}")
        st.caption(
            f"{guidance['target_duration_seconds']} seconds · "
            f"{guidance['aspect_ratio']} · {guidance['resolution']} · "
            f"{guidance['fps']} FPS"
        )
        st.write(guidance["audience"])
        for index, scene in enumerate(guidance["scenes"], start=1):
            with st.expander(
                f"Scene {index} · {scene['title']} · {scene['duration_seconds']}s"
            ):
                st.markdown(f"**Teaching purpose:** {scene['purpose']}")
                st.markdown(f"**Narration:** {scene['narration']}")
                st.markdown("**Visual direction**")
                st.markdown("\n".join(f"- {item}" for item in scene["visuals"]))
                if scene["formulas"]:
                    st.markdown("**Formula checks**")
                    for formula in scene["formulas"]:
                        st.latex(formula)

        st.download_button(
            "Download Creator Kit (.zip)",
            data=build_animation_creator_kit(guidance),
            file_name=f"{draft.manifest.algorithm_id}-animation-creator-kit.zip",
            mime="application/zip",
            key=f"animation_guidance_kit_{draft.key}",
            width="stretch",
        )
        with st.expander("Edit creator guidance with a form", expanded=False):
            st.caption(
                "Edit the document fields below. The platform maps the form to "
                "its internal data; you do not need to read or write JSON. "
                "Saving guidance never changes an uploaded MP4."
            )
            with st.form(f"animation_guidance_form_{draft.key}"):
                title = st.text_input("Document title", value=guidance["title"])
                audience = st.text_area(
                    "Target audience", value=guidance["audience"]
                )
                spec_left, spec_middle, spec_right = st.columns(3)
                aspect_ratio = spec_left.text_input(
                    "Aspect ratio", value=guidance["aspect_ratio"]
                )
                resolution = spec_middle.text_input(
                    "Resolution", value=guidance["resolution"]
                )
                fps = spec_right.number_input(
                    "FPS", min_value=12, max_value=120, value=guidance["fps"]
                )
                learning_objectives = st.text_area(
                    "Learning objectives (one per line)",
                    value="\n".join(guidance["learning_objectives"]),
                )
                recommended_tools = st.text_area(
                    "Recommended tools (one per line)",
                    value="\n".join(guidance["recommended_tools"]),
                )
                visual_style = st.text_area(
                    "Visual style rules (one per line)",
                    value="\n".join(guidance["visual_style"]),
                )

                scene_values = []
                st.markdown("#### Storyboard scenes")
                for index, scene in enumerate(guidance["scenes"], start=1):
                    with st.expander(
                        f"Scene {index} · {scene['title']}", expanded=index == 1
                    ):
                        scene_title = st.text_input(
                            "Scene title",
                            value=scene["title"],
                            key=f"guidance_scene_title_{draft.key}_{index}",
                        )
                        duration = st.number_input(
                            "Estimated seconds",
                            min_value=3,
                            max_value=120,
                            value=scene["duration_seconds"],
                            key=f"guidance_scene_duration_{draft.key}_{index}",
                        )
                        purpose = st.text_area(
                            "Teaching purpose",
                            value=scene["purpose"],
                            key=f"guidance_scene_purpose_{draft.key}_{index}",
                        )
                        narration = st.text_area(
                            "Narration",
                            value=scene["narration"],
                            key=f"guidance_scene_narration_{draft.key}_{index}",
                        )
                        on_screen_text = st.text_area(
                            "On-screen text (one per line)",
                            value="\n".join(scene["on_screen_text"]),
                            key=f"guidance_scene_text_{draft.key}_{index}",
                        )
                        visuals = st.text_area(
                            "Visual direction (one per line)",
                            value="\n".join(scene["visuals"]),
                            key=f"guidance_scene_visuals_{draft.key}_{index}",
                        )
                        formulas = st.text_area(
                            "Formula checks (one per line)",
                            value="\n".join(scene["formulas"]),
                            key=f"guidance_scene_formulas_{draft.key}_{index}",
                        )
                        transition = st.text_area(
                            "Transition to the next scene",
                            value=scene["transition_to_next"],
                            key=f"guidance_scene_transition_{draft.key}_{index}",
                        )
                        scene_values.append(
                            {
                                "scene_id": scene["scene_id"],
                                "title": scene_title,
                                "purpose": purpose,
                                "duration_seconds": int(duration),
                                "narration": narration,
                                "on_screen_text": list(_lines(on_screen_text)),
                                "visuals": list(_lines(visuals)),
                                "formulas": list(_lines(formulas)),
                                "transition_to_next": transition,
                            }
                        )

                required_assets = st.text_area(
                    "Required assets (one per line)",
                    value="\n".join(guidance["required_assets"]),
                )
                production_notes = st.text_area(
                    "Production notes (one per line)",
                    value="\n".join(guidance["production_notes"]),
                )
                accessibility_notes = st.text_area(
                    "Accessibility checks (one per line)",
                    value="\n".join(guidance["accessibility_notes"]),
                )

                metadata = guidance["metadata"]
                st.markdown("#### Text shown beside the installed video")
                concept_markdown = st.text_area(
                    "Concept explanation", value=metadata["concept_markdown"]
                )
                formula = st.text_input("Main formula", value=metadata["formula"])
                symbols_editor = _render_symbol_editor(
                    "Formula symbols",
                    metadata["symbols"],
                    key=f"guidance_symbols_{draft.key}",
                )
                highlights = st.text_area(
                    "Highlights (one per line)",
                    value="\n".join(metadata["highlights"]),
                )
                viewing_flow = st.text_area(
                    "Viewing flow (one per line)",
                    value="\n".join(metadata["viewing_flow"]),
                )
                derivation_editor = _render_derivation_editor(
                    "Formula derivation",
                    metadata["derivation_steps"],
                    key=f"guidance_derivation_{draft.key}",
                )
                submitted = st.form_submit_button(
                    "Save Creator Guidance",
                    disabled=not provider.strip(),
                    width="stretch",
                )
            if submitted:
                try:
                    updated_guidance = {
                        "title": title.strip(),
                        "audience": audience.strip(),
                        "target_duration_seconds": sum(
                            scene["duration_seconds"] for scene in scene_values
                        ),
                        "aspect_ratio": aspect_ratio.strip(),
                        "resolution": resolution.strip(),
                        "fps": int(fps),
                        "learning_objectives": list(_lines(learning_objectives)),
                        "recommended_tools": list(_lines(recommended_tools)),
                        "visual_style": list(_lines(visual_style)),
                        "scenes": scene_values,
                        "required_assets": list(_lines(required_assets)),
                        "production_notes": list(_lines(production_notes)),
                        "accessibility_notes": list(_lines(accessibility_notes)),
                        "metadata": {
                            "concept_markdown": concept_markdown.strip(),
                            "formula": formula.strip(),
                            "symbols": list(_animation_symbols(symbols_editor)),
                            "highlights": list(_lines(highlights)),
                            "viewing_flow": list(_lines(viewing_flow)),
                            "derivation_steps": list(
                                _derivation_steps(derivation_editor)
                            ),
                        },
                        "warnings": guidance["warnings"],
                    }
                    save_animation_guidance(
                        draft.key, updated_guidance, provider
                    )
                except (AlgorithmPackageError, OSError, ValueError) as exc:
                    _queue_action_response(
                        "error",
                        "Creator guidance was not saved",
                        str(exc),
                        technical=str(exc),
                    )
                    st.rerun()
                else:
                    st.session_state["algorithm_admin_message"] = (
                        "Animation creator guidance saved."
                    )
                    st.session_state["algorithm_admin_next_section"] = (
                        "Review Drafts"
                    )
                    st.rerun()


def _render_algorithm_overview(draft, provider: str) -> None:
    manifest = draft.manifest
    if manifest is None:
        return
    raw = manifest.raw
    algorithm = raw["algorithm"]
    left, right = st.columns(2)
    with left.container(border=True):
        st.markdown("#### What it is")
        st.caption("The problem this algorithm is intended to solve.")
        st.write(manifest.summary)
        st.write(algorithm["objective"])
    with right.container(border=True):
        st.markdown("#### How it learns")
        st.caption("The ordered decisions and updates used by the algorithm.")
        for index, step in enumerate(algorithm["pseudocode"], start=1):
            st.markdown(f"{index}. {step}")
    with left.container(border=True):
        st.markdown("#### Inputs")
        st.caption("Information and settings required to run the algorithm.")
        for item in algorithm["inputs"]:
            st.markdown(f"- {item}")
    with right.container(border=True):
        st.markdown("#### Outputs")
        st.caption("Values, policies, metrics, or artifacts produced by a run.")
        for item in algorithm["outputs"]:
            st.markdown(f"- {item}")
    with left.container(border=True):
        st.markdown("#### Settings")
        st.caption(
            "Controls learners can adjust on the experiment page. Starting value "
            "is the value used when the experiment first opens."
        )
        rows = _hyperparameter_rows(algorithm["hyperparameters"])
        display_rows = [
            {
                "Setting": row["name"],
                "Starting value": row["default"],
                "What it changes": row["description"],
                "Lowest value": row["minimum"],
                "Highest value": row["maximum"],
                "Change per click": row["step"],
                "Allowed values": row["choices"],
            }
            for row in rows
        ]
        if display_rows:
            st.dataframe(display_rows, width="stretch", hide_index=True)
        else:
            st.info("No adjustable experiment settings have been defined.")
    with right.container(border=True):
        st.markdown("#### Risks & Evidence")
        st.caption("Assumptions and traceable source support requiring human review.")
        for item in algorithm["assumptions"]:
            st.markdown(f"- {item}")
        agent_record = raw["generation"].get("algorithm_spec_agent") or {}
        evidence = agent_record.get("evidence", [])
        if evidence:
            st.success(f"{len(evidence)} source excerpt(s) passed platform verification.")
        elif agent_record:
            st.error("No Agent evidence excerpt passed platform verification.")
        else:
            st.info("This specification was confirmed as a manual specification.")

    st.markdown("#### Core equations")
    st.caption("The mathematical updates that generated modules must preserve.")
    for equation in algorithm["core_equations"]:
        st.latex(equation)

    with st.expander("How to read this AlgorithmSpec", expanded=False):
        if ALGORITHM_SPEC_USER_GUIDE.is_file():
            guide_text = ALGORITHM_SPEC_USER_GUIDE.read_text(encoding="utf-8")
            st.markdown(guide_text)
            st.download_button(
                "Download AlgorithmSpec User Guide",
                data=guide_text.encode("utf-8"),
                file_name="algorithm-spec-user-guide.md",
                mime="text/markdown",
                key=f"algorithm_spec_guide_{draft.key}",
                width="stretch",
            )
        else:
            st.info("The AlgorithmSpec user guide is unavailable.")

    with st.expander("Advanced editing", expanded=False):
        st.caption(
            "Changing AlgorithmSpec sends generated modules back to Needs Changes."
        )
        _render_experiment_design_editor(draft, provider)
        _render_spec_editor(draft, provider)
    with st.expander("Sources and platform validation", expanded=False):
        st.dataframe(raw["sources"], width="stretch")
        _render_report(draft.report)
    with st.expander("Review history", expanded=False):
        history = raw["review"]["history"]
        if history:
            st.dataframe(history, width="stretch")
        else:
            st.info("No review actions have been recorded.")


def _render_review_drafts(provider: str, reviewer: str) -> None:
    st.subheader("Review Schema v2 Drafts")
    if not reviewer.strip():
        st.warning(
            "Review and publishing actions are disabled until Reviewer name is "
            "entered. Provider generation and editing remain separate."
        )
    drafts = list_drafts()
    if not drafts:
        st.info("No drafts are awaiting review.")
        return
    labels = {
        (
            f"{draft.manifest.name} · {draft.key}"
            if draft.manifest is not None
            else f"Invalid draft · {draft.key}"
        ): draft
        for draft in drafts
    }
    draft = labels[st.selectbox("Draft", list(labels))]
    if draft.manifest is None:
        _render_report(draft.report)
        return
    manifest = draft.manifest
    raw = manifest.raw
    st.markdown(f"### {manifest.name}")
    st.caption(
        f"`{manifest.algorithm_id}` · v{manifest.version} · {manifest.category}"
    )
    st.write(manifest.summary)
    generation_mode = raw["generation"]["mode"]
    generation_label = (
        "Bundled Monte Carlo file copy"
        if generation_mode == "monte-carlo-preset"
        else "Local deterministic scaffold"
    )
    agent_record = raw["generation"].get("algorithm_spec_agent")
    generation_caption = (
        f"Module generator: **{generation_label}** · "
        f"`{raw['generation']['generator_version']}`"
    )
    if agent_record:
        generation_caption += (
            " · AlgorithmSpec suggested through LangChain "
            f"`{agent_record.get('model', 'unknown-model')}` · "
            f"`{agent_record.get('structured_output_method', 'unknown-method')}`"
        )
    else:
        generation_caption += " · AlgorithmSpec entered manually"
    module_generations = raw["generation"].get("module_generations", {})
    if module_generations:
        generated_labels = []
        for module in CORE_MODULES:
            records = module_generations.get(module, [])
            if records:
                generated_labels.append(
                    f"{module.title()}={records[-1].get('model', 'unknown-model')}"
                )
        if generated_labels:
            generation_caption += " · Module Agents: " + ", ".join(
                generated_labels
            )
    st.caption(generation_caption)
    review_modules = _review_modules(raw)

    status_columns = st.columns(len(review_modules))
    for index, module in enumerate(review_modules):
        status_columns[index].metric(
            module.title(),
            raw["review"]["modules"][module]["status"],
        )

    module_statuses = {
        module.title(): raw["review"]["modules"][module]["status"].replace(
            "_", " "
        )
        for module in review_modules
    }
    animation_options = load_animation_options(draft.key)
    animation_guidance = load_animation_guidance(draft.key)
    if "animation" in review_modules:
        animation_status = module_statuses["Animation"]
    elif animation_guidance is not None:
        animation_status = (
            "storyboard stale"
            if animation_guidance.get("stale")
            else "storyboard ready"
        )
    elif animation_options and animation_options.get("selected_option_id"):
        animation_status = "concept selected"
    elif animation_options:
        animation_status = "concepts ready"
    else:
        animation_status = "not started"
    publish_ready = (
        draft.report.valid
        and not raw["generation"].get("blocking_flags", [])
        and all(
            raw["review"]["modules"][module]["status"] == "approved"
            for module in review_modules
        )
    )
    workspace_labels = {
        "Overview": "Overview",
        "Theory": f"Theory · {module_statuses.get('Theory', 'not included')}",
        "Notebook": (
            f"Notebook · {module_statuses.get('Notebook', 'not included')}"
        ),
        "Experiment": (
            f"Experiment · {module_statuses.get('Experiment', 'not included')}"
        ),
        "Animation": f"Animation · {animation_status}",
        "Publish": f"Publish · {'ready' if publish_ready else 'blocked'}",
    }
    workspace_selection_key = f"draft_workspace_selection_{draft.key}"
    selected_workspace = st.session_state.get(
        workspace_selection_key, "Overview"
    )
    if selected_workspace not in workspace_labels:
        selected_workspace = "Overview"
    label_signature = hashlib.sha256(
        "|".join(workspace_labels.values()).encode("utf-8")
    ).hexdigest()[:12]
    workspace = st.segmented_control(
        "Draft workspace",
        list(workspace_labels),
        default=selected_workspace,
        format_func=workspace_labels.__getitem__,
        key=f"draft_workspace_{draft.key}_{label_signature}",
        width="stretch",
    )
    if workspace is not None:
        st.session_state[workspace_selection_key] = workspace
    if workspace == "Overview":
        _render_algorithm_overview(draft, provider)
    elif workspace in {"Theory", "Notebook", "Experiment"}:
        _render_module_review(draft, workspace.lower(), provider, reviewer)
    elif workspace == "Animation":
        _render_animation_guidance(draft, provider)
        if "animation" in review_modules:
            _render_module_review(draft, "animation", provider, reviewer)
        else:
            _render_add_animation(draft, provider)
    elif workspace == "Publish":
        blockers = raw["generation"].get("blocking_flags", [])
        _render_publication_blockers(draft, reviewer)
        statuses = {
            module: raw["review"]["modules"][module]["status"]
            for module in review_modules
        }
        ready = (
            draft.report.valid
            and not blockers
            and all(status == "approved" for status in statuses.values())
        )
        installed_conflict = any(
            algorithm.manifest.algorithm_id == manifest.algorithm_id
            for algorithm in list_installed()
        )
        if installed_conflict:
            st.info(
                f"Installed version conflict: `{manifest.algorithm_id}` is still live. "
                "Remove it from Installed after completing this revision."
            )
        left, right = st.columns(2)
        if left.button(
            "Install Approved Draft",
            type="primary",
            disabled=not ready or installed_conflict or not reviewer.strip(),
            width="stretch",
        ):
            _run_and_rerun(
                install_approved_draft,
                draft.key,
                reviewer,
                success=f"{manifest.name} installed from approved v2 draft.",
                return_section="Installed",
            )
        reject_reason = st.text_input(
            "Draft rejection reason", key=f"reject_reason_{draft.key}"
        )
        confirm_reject = st.checkbox(
            "Confirm draft rejection", key=f"confirm_reject_{draft.key}"
        )
        if right.button(
            "Reject Draft",
            disabled=(
                not reviewer.strip()
                or not reject_reason.strip()
                or not confirm_reject
            ),
            width="stretch",
        ):
            _run_and_rerun(
                reject_draft,
                draft.key,
                reviewer,
                reject_reason,
                success=f"{manifest.name} moved to the rejected archive.",
                return_section="Rejected Drafts",
            )


def _render_rejected_drafts(reviewer: str) -> None:
    st.subheader("Rejected Drafts")
    st.caption(
        "Rejected drafts are archived, not deleted. Restore one for another "
        "review cycle or move it to recoverable trash."
    )
    rejected = list_rejected_drafts()
    if not rejected:
        st.info("No rejected drafts are archived.")
        return
    labels = {
        (
            f"{record.manifest.name} · {record.key}"
            if record.manifest is not None
            else f"Invalid rejected draft · {record.key}"
        ): record
        for record in rejected
    }
    record = labels[st.selectbox("Rejected draft", list(labels))]
    if record.manifest is None:
        _render_report(record.report)
        return
    manifest = record.manifest
    st.markdown(f"### {manifest.name}")
    st.caption(
        f"`{manifest.algorithm_id}` · v{manifest.version} · archived at `{record.path}`"
    )
    history = manifest.review.get("history", [])
    rejection_events = [
        event for event in history if event.get("action") == "draft_rejected"
    ]
    if rejection_events:
        latest = rejection_events[-1]
        st.warning(
            f"Rejected by **{latest.get('reviewer', 'unknown')}**: "
            f"{latest.get('note', '')}"
        )
    with st.expander("Review history", expanded=False):
        st.dataframe(history, width="stretch")

    left, right = st.columns(2)
    if left.button(
        "Restore to Active Drafts",
        type="primary",
        disabled=not reviewer.strip(),
        width="stretch",
    ):
        _run_and_rerun(
            restore_rejected_draft,
            record.key,
            reviewer,
            success=f"{manifest.name} restored and marked Needs Changes.",
            return_section="Review Drafts",
        )
    confirm_trash = st.checkbox(
        "Confirm move to recoverable trash",
        key=f"trash_rejected_confirm_{record.key}",
    )
    if right.button(
        "Move to Trash",
        disabled=not reviewer.strip() or not confirm_trash,
        width="stretch",
    ):
        _run_and_rerun(
            trash_rejected_draft,
            record.key,
            reviewer,
            success=f"{manifest.name} moved to recoverable trash.",
        )


def _render_legacy_install() -> None:
    st.subheader("Schema v1 Legacy ZIP")
    st.caption(
        "The existing trusted-package validation and direct-install flow remains "
        "available for backward compatibility."
    )
    uploaded_file = st.file_uploader(
        "Algorithm package (.zip)",
        type=["zip"],
        key="legacy_zip",
        help="The ZIP must contain a valid schema v1 manifest.",
    )
    if uploaded_file is None:
        return
    upload_path = _write_upload(uploaded_file)
    try:
        report = validate_package(upload_path)
        _render_report(report)
        is_v1 = report.manifest is not None and report.manifest.schema_version == 1
        if report.manifest is not None and not is_v1:
            st.error("Schema v2 packages must be created and approved as drafts.")
        if st.button(
            "Install Legacy Package",
            type="primary",
            disabled=not report.valid or not is_v1,
            width="stretch",
        ):
            try:
                installed = install_package(upload_path)
            except AlgorithmPackageError as exc:
                _queue_action_response(
                    "error",
                    "Installation failed",
                    str(exc),
                    technical=str(exc),
                )
                st.rerun()
            else:
                st.session_state["algorithm_admin_message"] = (
                    f"{installed.manifest.name} installed successfully."
                )
                st.session_state["algorithm_admin_next_section"] = "Installed"
                st.rerun()
    finally:
        upload_path.unlink(missing_ok=True)


def _render_installed(reviewer: str) -> None:
    st.subheader("Installed Algorithms")
    algorithms = list_installed()
    if not algorithms:
        st.info("No imported algorithms are installed.")
        return
    for algorithm in algorithms:
        manifest = algorithm.manifest
        with st.container(border=True):
            left, right = st.columns([3, 1])
            with left:
                st.markdown(f"#### {manifest.name}")
                st.caption(
                    f"Schema v{manifest.schema_version} · "
                    f"`{manifest.algorithm_id}` · v{manifest.version} · "
                    f"{manifest.category}"
                )
                st.write(manifest.summary)
                if manifest.experiment is not None and not algorithm.experiment_available:
                    st.warning("Experiment is disabled because dependencies are missing.")
                    for dependency in algorithm.dependencies:
                        if not dependency.available:
                            st.code(dependency.install_hint, language="bash")
                if manifest.notebook is not None:
                    publication = manual_publication_for(algorithm)
                    if publication.status == "ready":
                        st.success("A matching manual repository copy is available.")
                        st.link_button(
                            "Open in Colab",
                            publication.colab_url,
                            use_container_width=True,
                        )
                    else:
                        st.caption(publication.message)
                        st.code(publication.relative_path)
            with right:
                confirm = st.checkbox(
                    "Confirm removal",
                    key=f"remove_confirm_{manifest.algorithm_id}",
                )
                if st.button(
                    "Remove",
                    key=f"remove_{manifest.algorithm_id}",
                    type="secondary",
                    width="stretch",
                    disabled=not confirm,
                ):
                    trash_path = uninstall_package(manifest.algorithm_id)
                    st.session_state["algorithm_admin_message"] = (
                        f"{manifest.name} was removed. Recovery copy: {trash_path}"
                    )
                    st.rerun()
            if manifest.schema_version == 2:
                with st.expander("Create Revision Draft", expanded=False):
                    st.caption(
                        "Copies this installed package into review without "
                        "changing the live version. Carried-forward modules stay "
                        "approved and locked; use Needs Changes on the modules "
                        "you intend to revise. Remove the old installed ID only "
                        "after the revision is fully approved."
                    )
                    revision_version = st.text_input(
                        "Revision version",
                        value=_next_patch_version(manifest.version),
                        key=f"revision_version_{manifest.algorithm_id}",
                    )
                    if st.button(
                        "Create Revision Draft",
                        key=f"create_revision_{manifest.algorithm_id}",
                        type="primary",
                        disabled=not reviewer.strip(),
                        width="stretch",
                    ):
                        _run_and_rerun(
                            create_revision_draft,
                            manifest.algorithm_id,
                            revision_version.strip(),
                            reviewer,
                            success=(
                                f"Revision {revision_version.strip()} created for "
                                f"{manifest.name}."
                            ),
                            return_section="Review Drafts",
                        )


def show_algorithm_admin() -> None:
    st.header("Algorithm Package Manager")
    st.caption(
        "Generate and review Schema v2 drafts, retain Schema v1 compatibility, "
        "and manage installed teaching algorithms."
    )
    st.session_state.setdefault("applied_algorithm_provider", "")
    st.session_state.setdefault("applied_algorithm_reviewer", "")
    with st.sidebar.form(
        "algorithm_role_names",
        clear_on_submit=False,
        enter_to_submit=False,
    ):
        provider_input = st.text_input(
            "Provider name",
            value=st.session_state["applied_algorithm_provider"],
            key="algorithm_provider_input",
            help=(
                "Required for generating, editing, replacing files, selecting an "
                "animation concept, and uploading an MP4."
            ),
        )
        reviewer_input = st.text_input(
            "Reviewer name",
            value=st.session_state["applied_algorithm_reviewer"],
            key="algorithm_reviewer_input",
            help=(
                "Required only for approvals, change requests, rejection, and "
                "publishing."
            ),
        )
        roles_submitted = st.form_submit_button(
            "Apply role names",
            width="stretch",
        )
    if roles_submitted:
        st.session_state["applied_algorithm_provider"] = provider_input.strip()
        st.session_state["applied_algorithm_reviewer"] = reviewer_input.strip()
        _queue_action_response(
            "success",
            "Role names applied",
            "Provider and Reviewer names were applied. Your current workspace "
            "selection was preserved.",
        )
    provider = str(st.session_state["applied_algorithm_provider"])
    reviewer = str(st.session_state["applied_algorithm_reviewer"])
    st.sidebar.caption(
        "Provider prepares the material: generate, edit, replace files, choose "
        "an animation concept, and upload the MP4."
    )
    st.sidebar.caption(
        "Reviewer makes the release decision: approve, request changes, reject, "
        "and install. Reviewer controls never generate or rewrite content."
    )
    st.sidebar.caption(
        f"Active roles — Provider: **{provider or 'not set'}** · "
        f"Reviewer: **{reviewer or 'not set'}**"
    )
    with st.expander("Provider and Reviewer — who does what?", expanded=False):
        provider_column, reviewer_column = st.columns(2)
        with provider_column.container(border=True):
            st.markdown("#### Provider")
            st.write("Creates and corrects the learning package.")
            st.markdown(
                "- Generate modules\n"
                "- Edit or replace files\n"
                "- Select an animation concept\n"
                "- Upload the finished MP4"
            )
        with reviewer_column.container(border=True):
            st.markdown("#### Reviewer")
            st.write("Checks quality and controls publication.")
            st.markdown(
                "- Approve a module\n"
                "- Request changes\n"
                "- Reject a draft\n"
                "- Install the approved package"
            )
    message = st.session_state.pop("algorithm_admin_message", None)
    if message:
        _queue_action_response("success", "Action completed", str(message))
    error_message = st.session_state.pop("algorithm_admin_error", None)
    if error_message:
        _queue_action_response("error", "Action failed", str(error_message))
    _render_pending_action_response()
    sections = [
        "Create v2 Draft",
        "Review Drafts",
        "Rejected Drafts",
        "Legacy v1 ZIP",
        "Installed",
    ]
    next_section = st.session_state.pop("algorithm_admin_next_section", None)
    if next_section in sections:
        st.session_state["algorithm_admin_section"] = next_section
    selected_section = st.segmented_control(
        "Manager section",
        sections,
        default="Create v2 Draft",
        key="algorithm_admin_section",
        label_visibility="collapsed",
        width="stretch",
    )
    if selected_section == "Create v2 Draft":
        _render_create_draft(provider)
    elif selected_section == "Review Drafts":
        _render_review_drafts(provider, reviewer)
    elif selected_section == "Rejected Drafts":
        _render_rejected_drafts(reviewer)
    elif selected_section == "Legacy v1 ZIP":
        _render_legacy_install()
    elif selected_section == "Installed":
        _render_installed(reviewer)
