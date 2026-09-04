from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import nbformat

from .animation_planner import (
    default_animation_options,
    default_animation_guidance,
    plan_animation_options,
    plan_animation_guidance,
    validate_animation_options,
    validate_animation_guidance,
)
from .models import (
    AlgorithmManifest,
    DraftNotFoundError,
    DraftStateError,
    DraftValidationError,
    InstalledAlgorithm,
    ValidationReport,
)
from .experiment_design import validate_experiment_design
from .package import (
    ALGORITHM_ID_PATTERN,
    CORE_MODULES,
    VERSION_PATTERN,
    default_installed_root,
    install_package,
    iter_package_files,
    project_root,
    validate_source_directory,
)


GENERATOR_VERSION = "template-v2.0"
MAX_SOURCE_SIZE = 20 * 1024 * 1024
MAX_ANIMATION_SIZE = 200 * 1024 * 1024


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_packages_root() -> Path:
    configured_root = os.getenv("ALGORITHM_PACKAGES_ROOT", "").strip()
    if configured_root:
        return Path(configured_root).resolve()
    return project_root() / "algorithm_packages"


def default_drafts_root() -> Path:
    return default_packages_root() / "drafts"


def default_rejected_root() -> Path:
    return default_packages_root() / "rejected"


@dataclass(frozen=True)
class DraftInput:
    algorithm_id: str
    name: str
    version: str
    category: str
    summary: str
    objective: str
    assumptions: tuple[str, ...]
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    states: tuple[str, ...]
    actions: tuple[str, ...]
    hyperparameters: Mapping[str, Any]
    core_equations: tuple[str, ...]
    pseudocode: tuple[str, ...]
    supported_environments: tuple[str, ...]
    source_name: str
    source_bytes: bytes
    reference_urls: tuple[str, ...] = ()
    generation_mode: str = "template"
    animation_name: str | None = None
    animation_bytes: bytes | None = None
    animation_concept_markdown: str = ""
    animation_formula: str = ""
    animation_symbols: tuple[Mapping[str, str], ...] = ()
    animation_highlights: tuple[str, ...] = ()
    animation_viewing_flow: tuple[str, ...] = ()
    animation_derivation_steps: tuple[Mapping[str, Any], ...] = ()
    algorithm_spec_agent: Mapping[str, Any] | None = None
    algorithm_spec_confirmation: Mapping[str, Any] | None = None
    experiment_design: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class DraftRecord:
    path: Path
    manifest: AlgorithmManifest | None
    report: ValidationReport

    @property
    def key(self) -> str:
        return self.path.name


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "wb") as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    _atomic_write(
        path,
        (json.dumps(data, indent=2, ensure_ascii=False) + "\n").encode("utf-8"),
    )


def _write_text(path: Path, text: str) -> None:
    _atomic_write(path, text.encode("utf-8"))


def extract_source_text(file_name: str, payload: bytes) -> str:
    if not payload:
        raise DraftValidationError("source file cannot be empty")
    if len(payload) > MAX_SOURCE_SIZE:
        raise DraftValidationError("source file exceeds the 20 MiB limit")

    suffix = Path(file_name).suffix.lower()
    if suffix in {".md", ".txt"}:
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise DraftValidationError("Markdown/TXT sources must use UTF-8") from exc
    elif suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise DraftValidationError(
                "PDF extraction requires the 'pypdf' dependency"
            ) from exc
        try:
            reader = PdfReader(io.BytesIO(payload))
            text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception as exc:
            raise DraftValidationError(f"PDF text extraction failed: {exc}") from exc
    else:
        raise DraftValidationError("source must be a Markdown, TXT, or PDF file")

    if not text.strip():
        raise DraftValidationError("source contains no extractable text")
    return text.strip()


def _validate_animation_payload(file_name: str, payload: bytes) -> None:
    if Path(file_name).suffix.lower() != ".mp4":
        raise DraftValidationError("animation must use the .mp4 extension")
    if not payload:
        raise DraftValidationError("animation MP4 cannot be empty")
    if len(payload) > MAX_ANIMATION_SIZE:
        raise DraftValidationError("animation MP4 exceeds the 200 MiB limit")
    if len(payload) < 12 or payload[4:8] != b"ftyp":
        raise DraftValidationError(
            "animation file does not contain a recognizable MP4 header"
        )


def _validate_animation_metadata(
    concept_markdown: str,
    formula: str,
    symbols: Iterable[Mapping[str, str]],
    highlights: Iterable[str],
    viewing_flow: Iterable[str],
    derivation_steps: Iterable[Mapping[str, Any]],
) -> None:
    if not isinstance(concept_markdown, str):
        raise DraftValidationError("animation concept_markdown must be a string")
    if not isinstance(formula, str):
        raise DraftValidationError("animation formula must be a string")
    for index, item in enumerate(symbols):
        if not isinstance(item, Mapping) or set(item) != {"symbol", "meaning"}:
            raise DraftValidationError(
                f"animation symbol {index} must contain symbol and meaning"
            )
        if not all(
            isinstance(item[field], str) and item[field].strip()
            for field in ("symbol", "meaning")
        ):
            raise DraftValidationError(
                f"animation symbol {index} values must be non-empty strings"
            )
    for field_name, values in (
        ("highlights", highlights),
        ("viewing_flow", viewing_flow),
    ):
        if not all(isinstance(item, str) and item.strip() for item in values):
            raise DraftValidationError(
                f"animation {field_name} must contain non-empty strings"
            )
    for index, step in enumerate(derivation_steps):
        if not isinstance(step, Mapping):
            raise DraftValidationError(
                f"animation derivation step {index} must be an object"
            )
        unknown = set(step) - {"title", "text", "latex"}
        if unknown:
            raise DraftValidationError(
                f"animation derivation step {index} contains unsupported fields: "
                + ", ".join(sorted(unknown))
            )
        for field_name in ("title", "text", "latex"):
            value = step.get(field_name)
            if field_name == "latex" and isinstance(value, list):
                if not all(isinstance(item, str) and item.strip() for item in value):
                    raise DraftValidationError(
                        f"animation derivation step {index}.latex is invalid"
                    )
            elif value is not None and not isinstance(value, str):
                raise DraftValidationError(
                    f"animation derivation step {index}.{field_name} is invalid"
                )
        if not any(step.get(field) for field in ("title", "text", "latex")):
            raise DraftValidationError(
                f"animation derivation step {index} cannot be empty"
            )


def _clean_string_list(name: str, values: Iterable[str]) -> list[str]:
    cleaned = [str(value).strip() for value in values if str(value).strip()]
    if not cleaned:
        raise DraftValidationError(f"{name} must contain at least one item")
    return cleaned


def _reject_structured_string_fragments(name: str, values: Iterable[str]) -> None:
    """Reject model-produced JSON fragments where plain teaching text is required."""

    suspicious_keys = re.compile(
        r'^"?(?:id|latex|description|variables|step_number|instruction|notes)"?\s*:'
    )
    for index, value in enumerate(values, start=1):
        cleaned = str(value).strip()
        if cleaned in {"{", "}", "[", "]"} or suspicious_keys.match(cleaned):
            raise DraftValidationError(
                f"{name} item {index} looks like a JSON fragment; use one plain "
                "entry per item"
            )
        if cleaned.startswith(("{", "[")):
            try:
                parsed = json.loads(cleaned)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, (Mapping, list)):
                raise DraftValidationError(
                    f"{name} item {index} must be plain text, not JSON"
                )


def _validate_algorithm_semantics(
    hyperparameters: Mapping[str, Any],
    core_equations: Iterable[str],
    pseudocode: Iterable[str],
) -> None:
    if isinstance(core_equations, (str, bytes)) or isinstance(
        pseudocode, (str, bytes)
    ):
        raise DraftValidationError(
            "core_equations and pseudocode must be lists of plain text entries"
        )
    _reject_structured_string_fragments("core_equations", core_equations)
    _reject_structured_string_fragments("pseudocode", pseudocode)
    for name, definition in hyperparameters.items():
        if not isinstance(definition, Mapping):
            raise DraftValidationError(f"hyperparameter '{name}' must be an object")
        if isinstance(definition.get("default"), (Mapping, list)):
            raise DraftValidationError(
                f"hyperparameter '{name}' default must be a scalar value"
            )


def _validate_input(data: DraftInput) -> None:
    if not ALGORITHM_ID_PATTERN.fullmatch(data.algorithm_id):
        raise DraftValidationError(
            "algorithm id must use lowercase letters, numbers, and single hyphens"
        )
    if not VERSION_PATTERN.fullmatch(data.version):
        raise DraftValidationError("version must follow semantic version format")
    for field_name in ("name", "category", "summary", "objective", "source_name"):
        if not str(getattr(data, field_name)).strip():
            raise DraftValidationError(f"{field_name} is required")
    for field_name in (
        "assumptions",
        "inputs",
        "outputs",
        "states",
        "actions",
        "core_equations",
        "pseudocode",
        "supported_environments",
    ):
        _clean_string_list(field_name, getattr(data, field_name))
    if not isinstance(data.hyperparameters, Mapping):
        raise DraftValidationError("hyperparameters must be an object")
    _validate_algorithm_semantics(
        data.hyperparameters, data.core_equations, data.pseudocode
    )
    if data.algorithm_spec_agent is not None and not isinstance(
        data.algorithm_spec_agent, Mapping
    ):
        raise DraftValidationError("algorithm_spec_agent must be an object")
    if (
        isinstance(data.algorithm_spec_agent, Mapping)
        and "evidence" in data.algorithm_spec_agent
        and not data.algorithm_spec_agent.get("evidence")
    ):
        raise DraftValidationError(
            "an Agent-confirmed AlgorithmSpec requires verified source evidence; "
            "use manual specification mode with a reason instead"
        )
    if data.algorithm_spec_confirmation is not None and not isinstance(
        data.algorithm_spec_confirmation, Mapping
    ):
        raise DraftValidationError("algorithm_spec_confirmation must be an object")
    if data.experiment_design is not None:
        try:
            validate_experiment_design(data.experiment_design)
        except ValueError as exc:
            raise DraftValidationError(str(exc)) from exc
    if data.generation_mode not in {"template", "monte-carlo-preset"}:
        raise DraftValidationError("unsupported generation mode")
    if data.generation_mode == "monte-carlo-preset" and (
        data.algorithm_id != "monte-carlo-control"
    ):
        raise DraftValidationError(
            "the Monte Carlo preset requires algorithm id 'monte-carlo-control'"
        )
    for url in data.reference_urls:
        if not url.startswith(("http://", "https://")):
            raise DraftValidationError("reference URLs must use HTTP(S)")
    if data.animation_name is not None or data.animation_bytes is not None:
        _validate_animation_payload(
            data.animation_name or "",
            data.animation_bytes or b"",
        )
        _validate_animation_metadata(
            data.animation_concept_markdown,
            data.animation_formula,
            data.animation_symbols,
            data.animation_highlights,
            data.animation_viewing_flow,
            data.animation_derivation_steps,
        )


def _review_state(
    status: str = "draft",
    modules: Iterable[str] = CORE_MODULES,
) -> dict[str, Any]:
    timestamp = utc_now()
    return {
        "modules": {
            module: {
                "status": status,
                "reviewer": None,
                "note": "",
                "updated_at": timestamp,
            }
            for module in modules
        },
        "history": [],
    }


def _manifest_for(
    data: DraftInput,
    source_relative_path: str,
    source_digest: str,
) -> dict[str, Any]:
    sources: list[dict[str, Any]] = [
        {
            "type": "file",
            "path": source_relative_path,
            "sha256": source_digest,
            "name": Path(data.source_name).name,
        }
    ]
    sources.extend({"type": "url", "url": url} for url in data.reference_urls)
    blocking_flags = (
        ["placeholder_content"] if data.generation_mode == "template" else []
    )
    requirements = []
    if data.generation_mode == "monte-carlo-preset":
        requirements = [
            {"package": "gymnasium>=1.2.3", "import": "gymnasium"},
            {"package": "numpy>=2.2", "import": "numpy"},
        ]
    modules: dict[str, Any] = {
        "theory": {"file": "theory.md"},
        "notebook": {"file": "notebook.ipynb"},
        "experiment": {
            "module": "experiment.py",
            "requirements": requirements,
        },
    }
    review_modules = list(CORE_MODULES)
    if data.animation_bytes is not None:
        modules["animation"] = {
            "file": "animation.mp4",
            "concept_markdown": data.animation_concept_markdown.strip(),
            "formula": data.animation_formula.strip(),
            "symbols": [dict(item) for item in data.animation_symbols],
            "highlights": list(data.animation_highlights),
            "viewing_flow": list(data.animation_viewing_flow),
            "derivation_steps": [
                dict(step) for step in data.animation_derivation_steps
            ],
        }
        review_modules.append("animation")
    generation: dict[str, Any] = {
        "mode": data.generation_mode,
        "generator_version": GENERATOR_VERSION,
        "generated_at": utc_now(),
        "source_sha256": source_digest,
        "blocking_flags": blocking_flags,
    }
    if data.algorithm_spec_agent is not None:
        generation["algorithm_spec_agent"] = dict(data.algorithm_spec_agent)
    algorithm = {
        "objective": data.objective.strip(),
        "assumptions": _clean_string_list("assumptions", data.assumptions),
        "inputs": _clean_string_list("inputs", data.inputs),
        "outputs": _clean_string_list("outputs", data.outputs),
        "states": _clean_string_list("states", data.states),
        "actions": _clean_string_list("actions", data.actions),
        "hyperparameters": dict(data.hyperparameters),
        "core_equations": _clean_string_list(
            "core_equations", data.core_equations
        ),
        "pseudocode": _clean_string_list("pseudocode", data.pseudocode),
        "supported_environments": _clean_string_list(
            "supported_environments", data.supported_environments
        ),
    }
    if data.experiment_design is not None:
        algorithm["experiment_design"] = validate_experiment_design(
            data.experiment_design
        )
    manifest = {
        "schema_version": 2,
        "id": data.algorithm_id,
        "name": data.name.strip(),
        "version": data.version,
        "summary": data.summary.strip(),
        "category": data.category.strip(),
        "sources": sources,
        "algorithm": algorithm,
        "modules": modules,
        "generation": generation,
        "review": _review_state(modules=review_modules),
    }
    if data.algorithm_spec_confirmation is not None:
        confirmation = dict(data.algorithm_spec_confirmation)
        confirmation.setdefault("confirmed_at", utc_now())
        confirmation["spec_sha256"] = _algorithm_spec_sha256(manifest)
        generation["algorithm_spec_confirmation"] = confirmation
        _append_history(
            manifest,
            module=None,
            action="algorithm_spec_confirmed",
            old_status=None,
            new_status=None,
            reviewer=str(confirmation.get("confirmed_by", "")).strip(),
            note=str(confirmation.get("note", "")).strip(),
        )
    return manifest


def _algorithm_spec_value(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "id": manifest["id"],
        "name": manifest["name"],
        "version": manifest["version"],
        "category": manifest["category"],
        "summary": manifest["summary"],
        "algorithm": manifest["algorithm"],
    }


def _algorithm_spec_sha256(manifest: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        _algorithm_spec_value(manifest),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _generic_theory(manifest: Mapping[str, Any], source_text: str) -> str:
    algorithm = manifest["algorithm"]
    bullets = lambda values: "\n".join(f"- {value}" for value in values)
    equations = "\n\n".join(f"$$\n{value}\n$$" for value in algorithm["core_equations"])
    pseudocode = "\n".join(
        f"{index}. {step}" for index, step in enumerate(algorithm["pseudocode"], 1)
    )
    excerpt = source_text[:2000]
    return f"""# {manifest['name']}

> This file is a deterministic teaching scaffold. It requires expert review and
> a real experiment implementation before publication.

## Summary

{manifest['summary']}

## Objective

{algorithm['objective']}

## Assumptions

{bullets(algorithm['assumptions'])}

## Inputs and outputs

### Inputs

{bullets(algorithm['inputs'])}

### Outputs

{bullets(algorithm['outputs'])}

## States and actions

### States

{bullets(algorithm['states'])}

### Actions

{bullets(algorithm['actions'])}

## Core equations

{equations}

## Pseudocode

{pseudocode}

## Supported environments

{bullets(algorithm['supported_environments'])}

## Source excerpt

{excerpt}
"""


def _generic_notebook(manifest: Mapping[str, Any]) -> nbformat.NotebookNode:
    notebook = nbformat.v4.new_notebook()
    notebook.metadata["algorithm_id"] = manifest["id"]
    notebook.metadata["algorithm_version"] = manifest["version"]
    notebook.cells = [
        nbformat.v4.new_markdown_cell(
            f"# {manifest['name']}\n\n"
            "**Scaffold only:** replace placeholder implementation cells before publication."
        ),
        nbformat.v4.new_markdown_cell(
            "## Objective\n\n" + manifest["algorithm"]["objective"]
        ),
        nbformat.v4.new_markdown_cell(
            "## Pseudocode\n\n"
            + "\n".join(
                f"{index}. {step}"
                for index, step in enumerate(
                    manifest["algorithm"]["pseudocode"], start=1
                )
            )
        ),
        nbformat.v4.new_code_cell(
            "algorithm_spec = "
            + repr(
                {
                    "id": manifest["id"],
                    "hyperparameters": manifest["algorithm"]["hyperparameters"],
                    "supported_environments": manifest["algorithm"][
                        "supported_environments"
                    ],
                }
            )
            + "\nalgorithm_spec"
        ),
        nbformat.v4.new_code_cell(
            "raise NotImplementedError("
            "'Implement the reviewed algorithm before publishing this notebook.'"
            ")"
        ),
    ]
    return notebook


GENERIC_EXPERIMENT = '''from __future__ import annotations


def get_spec():
    return {
        "parameters": {
            "episodes": {
                "type": "int",
                "default": 100,
                "min": 1,
                "max": 10000,
                "step": 1,
                "label": "Episodes"
            }
        }
    }


def run(parameters, reporter):
    raise NotImplementedError(
        "This generated experiment is a scaffold and cannot be published."
    )
'''


def _write_generated_modules(
    draft_path: Path,
    manifest: Mapping[str, Any],
    source_text: str,
    module_name: str | None = None,
) -> None:
    mode = manifest["generation"]["mode"]
    selected = set(CORE_MODULES if module_name is None else (module_name,))
    if mode == "monte-carlo-preset":
        example = (
            project_root()
            / "algorithm_packages"
            / "examples"
            / "monte_carlo_control"
        )
        files = {
            "theory": "theory.md",
            "notebook": "notebook.ipynb",
            "experiment": "experiment.py",
        }
        for name in selected:
            _atomic_write(
                draft_path / files[name],
                (example / files[name]).read_bytes(),
            )
        return

    if "theory" in selected:
        _write_text(draft_path / "theory.md", _generic_theory(manifest, source_text))
    if "notebook" in selected:
        notebook = _generic_notebook(manifest)
        buffer = io.StringIO()
        nbformat.write(notebook, buffer)
        _write_text(draft_path / "notebook.ipynb", buffer.getvalue())
    if "experiment" in selected:
        _write_text(draft_path / "experiment.py", GENERIC_EXPERIMENT)


def _load_raw(draft_path: Path) -> dict[str, Any]:
    manifest_path = draft_path / "manifest.json"
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DraftValidationError(f"draft manifest cannot be read: {exc}") from exc
    if not isinstance(data, dict):
        raise DraftValidationError("draft manifest must be an object")
    return data


def _append_history(
    manifest: dict[str, Any],
    *,
    module: str | None,
    action: str,
    old_status: str | None,
    new_status: str | None,
    reviewer: str,
    note: str = "",
) -> None:
    manifest["review"]["history"].append(
        {
            "module": module,
            "action": action,
            "from": old_status,
            "to": new_status,
            "reviewer": reviewer,
            "note": note,
            "timestamp": utc_now(),
        }
    )


def _set_status(
    manifest: dict[str, Any],
    module: str,
    status: str,
    reviewer: str,
    note: str = "",
    action: str = "status_changed",
) -> None:
    state = manifest["review"]["modules"][module]
    old_status = state["status"]
    state.update(
        {
            "status": status,
            "reviewer": reviewer or None,
            "note": note,
            "updated_at": utc_now(),
        }
    )
    _append_history(
        manifest,
        module=module,
        action=action,
        old_status=old_status,
        new_status=status,
        reviewer=reviewer,
        note=note,
    )


def _declared_review_modules(manifest: Mapping[str, Any]) -> tuple[str, ...]:
    modules = list(CORE_MODULES)
    if "animation" in manifest.get("modules", {}):
        modules.append("animation")
    return tuple(modules)


def _history_after_latest_spec_update(
    manifest: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    history = list(manifest.get("review", {}).get("history", []))
    latest_spec_update = max(
        (
            index
            for index, event in enumerate(history)
            if event.get("action") == "spec_updated"
        ),
        default=-1,
    )
    return history[latest_spec_update + 1 :]


def module_content_ready(
    manifest: Mapping[str, Any],
    module: str,
) -> bool:
    if module == "animation":
        return "animation" in manifest.get("modules", {})
    if module not in CORE_MODULES:
        return False
    if manifest.get("generation", {}).get("mode") != "template":
        return True
    accepted_actions = {
        "theory": {"edited", "agent_generated"},
        "notebook": {"file_replaced", "agent_generated"},
        "experiment": {"file_replaced", "agent_generated"},
    }
    return any(
        event.get("module") == module
        and event.get("action") in accepted_actions[module]
        for event in _history_after_latest_spec_update(manifest)
    )


def create_draft(
    data: DraftInput,
    drafts_root: str | os.PathLike[str] | None = None,
    installed_root: str | os.PathLike[str] | None = None,
) -> DraftRecord:
    _validate_input(data)
    source_text = extract_source_text(data.source_name, data.source_bytes)
    root = Path(drafts_root).resolve() if drafts_root else default_drafts_root()
    installed = (
        Path(installed_root).resolve()
        if installed_root is not None
        else default_installed_root()
    )
    draft_path = root / f"{data.algorithm_id}-{data.version}"
    if draft_path.exists():
        raise DraftStateError(f"draft '{draft_path.name}' already exists")
    if (installed / data.algorithm_id).exists():
        raise DraftStateError(
            f"algorithm '{data.algorithm_id}' is already installed; remove it first"
        )

    root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".draft-create-", dir=root))
    try:
        safe_name = Path(data.source_name).name
        source_path = temporary / "sources" / safe_name
        source_path.parent.mkdir(parents=True)
        _atomic_write(source_path, data.source_bytes)
        source_digest = hashlib.sha256(data.source_bytes).hexdigest()
        manifest = _manifest_for(
            data,
            source_path.relative_to(temporary).as_posix(),
            source_digest,
        )
        _write_json(temporary / "manifest.json", manifest)
        _write_generated_modules(temporary, manifest, source_text)
        if data.animation_bytes is not None:
            _atomic_write(temporary / "animation.mp4", data.animation_bytes)
        report = validate_source_directory(temporary)
        manifest = _load_raw(temporary)
        for module in manifest["review"]["modules"]:
            is_scaffold = (
                data.generation_mode == "template"
                and module in CORE_MODULES
            )
            status = (
                "validation_failed"
                if not report.valid
                else ("not_generated" if is_scaffold else "awaiting_review")
            )
            _set_status(
                manifest,
                module,
                status,
                reviewer="generator",
                note=(
                    "Safe scaffold created. Generate or replace this module "
                    "before review."
                    if is_scaffold
                    else "Initial trusted generation completed."
                ),
                action="scaffold_created" if is_scaffold else "generated",
            )
        _write_json(temporary / "manifest.json", manifest)
        os.replace(temporary, draft_path)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return load_draft(draft_path.name, root)


def create_revision_draft(
    algorithm_id: str,
    version: str,
    reviewer: str,
    drafts_root: str | os.PathLike[str] | None = None,
    installed_root: str | os.PathLike[str] | None = None,
) -> DraftRecord:
    if not reviewer.strip():
        raise DraftStateError("reviewer name is required")
    if not ALGORITHM_ID_PATTERN.fullmatch(algorithm_id):
        raise DraftValidationError("invalid installed algorithm id")
    if not VERSION_PATTERN.fullmatch(version):
        raise DraftValidationError("revision version must use semantic versioning")

    installed = (
        Path(installed_root).resolve()
        if installed_root is not None
        else default_installed_root()
    )
    installed_path = installed / algorithm_id
    report = validate_source_directory(installed_path)
    if not report.valid or report.manifest is None:
        raise DraftValidationError("installed algorithm is not a valid package")
    if report.manifest.schema_version != 2:
        raise DraftStateError("only Schema v2 algorithms can create revision drafts")
    if version == report.manifest.version:
        raise DraftStateError("revision version must differ from the installed version")

    drafts = (
        Path(drafts_root).resolve()
        if drafts_root is not None
        else default_drafts_root()
    )
    destination = drafts / f"{algorithm_id}-{version}"
    if destination.exists():
        raise DraftStateError(f"draft '{destination.name}' already exists")
    drafts.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".revision-create-", dir=drafts))
    try:
        shutil.copytree(installed_path, temporary, dirs_exist_ok=True)
        manifest = _load_raw(temporary)
        previous_version = str(manifest["version"])
        manifest["version"] = version
        if report.manifest.animation is not None:
            manifest["modules"]["animation"] = dict(report.manifest.animation)
        generation = manifest.setdefault("generation", {})
        generation["revision_of"] = {
            "id": algorithm_id,
            "version": previous_version,
            "created_at": utc_now(),
            "created_by": reviewer.strip(),
        }
        for module in _declared_review_modules(manifest):
            _set_status(
                manifest,
                module,
                "approved",
                reviewer.strip(),
                note=(
                    f"Approved content carried forward from installed version "
                    f"{previous_version}. Use Needs Changes to revise it."
                ),
                action="revision_content_carried_forward",
            )
        _append_history(
            manifest,
            module=None,
            action="revision_draft_created",
            old_status=None,
            new_status=None,
            reviewer=reviewer.strip(),
            note=(
                f"Revision draft {version} created from installed version "
                f"{previous_version}."
            ),
        )
        _write_json(temporary / "manifest.json", manifest)
        validation = validate_source_directory(temporary)
        if not validation.valid:
            details = "; ".join(issue.message for issue in validation.errors)
            raise DraftValidationError(
                f"revision draft validation failed: {details}"
            )
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return load_draft(destination.name, drafts)


def load_draft(
    draft_key: str,
    drafts_root: str | os.PathLike[str] | None = None,
) -> DraftRecord:
    root = Path(drafts_root).resolve() if drafts_root else default_drafts_root()
    path = (root / draft_key).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise DraftNotFoundError("draft path escapes the drafts root") from exc
    if not path.is_dir() or path.name.startswith("."):
        raise DraftNotFoundError(f"draft '{draft_key}' does not exist")
    report = validate_source_directory(path)
    return DraftRecord(path=path, manifest=report.manifest, report=report)


def list_drafts(
    drafts_root: str | os.PathLike[str] | None = None,
) -> tuple[DraftRecord, ...]:
    root = Path(drafts_root).resolve() if drafts_root else default_drafts_root()
    if not root.is_dir():
        return ()
    records = []
    for path in sorted(root.iterdir()):
        if path.is_dir() and not path.name.startswith("."):
            records.append(load_draft(path.name, root))
    return tuple(records)


def load_animation_options(
    draft_key: str,
    drafts_root: str | os.PathLike[str] | None = None,
) -> dict[str, Any] | None:
    record = load_draft(draft_key, drafts_root)
    if record.manifest is None:
        raise DraftValidationError("draft manifest is invalid")
    options_record = record.manifest.raw.get("generation", {}).get(
        "animation_options"
    )
    if not isinstance(options_record, Mapping):
        return None
    relative_path = str(options_record.get("file", "")).strip()
    if not relative_path:
        raise DraftValidationError("animation options file is not declared")
    path = (record.path / relative_path).resolve()
    try:
        path.relative_to(record.path)
        payload = path.read_bytes()
        value = json.loads(payload.decode("utf-8"))
    except (ValueError, OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DraftValidationError(f"animation options cannot be read: {exc}") from exc
    expected_digest = str(options_record.get("sha256", "")).strip()
    if expected_digest and hashlib.sha256(payload).hexdigest() != expected_digest:
        raise DraftValidationError("animation options SHA-256 does not match")
    if not isinstance(value, Mapping):
        raise DraftValidationError("animation options must be a JSON object")
    try:
        options = validate_animation_options(value.get("options", []))
    except (TypeError, ValueError) as exc:
        raise DraftValidationError(f"animation options are invalid: {exc}") from exc
    selected = options_record.get("selected_option_id")
    if selected is not None and selected not in {
        item["option_id"] for item in options
    }:
        raise DraftValidationError("selected animation option does not exist")
    return {
        "options": [dict(item) for item in options],
        "selected_option_id": selected,
        "record": dict(options_record),
    }


def save_animation_options(
    draft_key: str,
    options: Iterable[Mapping[str, Any]],
    provider: str,
    *,
    source: str = "manual",
    agent_metadata: Mapping[str, Any] | None = None,
    drafts_root: str | os.PathLike[str] | None = None,
) -> DraftRecord:
    if not provider.strip():
        raise DraftStateError("provider name is required")
    try:
        normalized = validate_animation_options(tuple(options))
    except (TypeError, ValueError) as exc:
        raise DraftValidationError(str(exc)) from exc
    record = load_draft(draft_key, drafts_root)
    if record.manifest is None:
        raise DraftValidationError("draft manifest is invalid")
    manifest = _load_raw(record.path)
    relative_path = "animation_options.json"
    payload = (
        json.dumps({"options": normalized}, ensure_ascii=False, indent=2) + "\n"
    ).encode("utf-8")
    timestamp = utc_now()
    previous = manifest["generation"].get("animation_options", {})
    runs = list(previous.get("agent_runs", [])) if isinstance(previous, Mapping) else []
    if agent_metadata is not None:
        runs.append(dict(agent_metadata))
    manifest["generation"]["animation_options"] = {
        "file": relative_path,
        "source": source,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "updated_at": timestamp,
        "updated_by": provider.strip(),
        "selected_option_id": None,
        "selected_by": None,
        "selected_at": None,
        "agent_runs": runs,
    }
    guidance = manifest["generation"].get("animation_guidance")
    if isinstance(guidance, dict):
        guidance["stale_since"] = timestamp
        guidance["stale_reason"] = "Animation concept options were replaced"
    _append_history(
        manifest,
        module=None,
        action="animation_options_saved",
        old_status=None,
        new_status=None,
        reviewer=provider.strip(),
        note=f"Three animation concepts saved from {source} by the Provider.",
    )
    _atomic_write(record.path / relative_path, payload)
    _write_json(record.path / "manifest.json", manifest)
    return load_draft(draft_key, drafts_root)


def generate_default_animation_options(
    draft_key: str,
    provider: str,
    drafts_root: str | os.PathLike[str] | None = None,
) -> DraftRecord:
    record = load_draft(draft_key, drafts_root)
    if record.manifest is None:
        raise DraftValidationError("draft manifest is invalid")
    options = default_animation_options(_algorithm_spec_value(record.manifest.raw))
    return save_animation_options(
        draft_key,
        options,
        provider,
        source="deterministic AlgorithmSpec concepts",
        drafts_root=drafts_root,
    )


def generate_animation_options_with_agent(
    draft_key: str,
    provider: str,
    provider_note: str = "",
    *,
    chat_model: Any | None = None,
    environ: Mapping[str, str] | None = None,
    drafts_root: str | os.PathLike[str] | None = None,
) -> DraftRecord:
    if not provider.strip():
        raise DraftStateError("provider name is required")
    record = load_draft(draft_key, drafts_root)
    if record.manifest is None:
        raise DraftValidationError("draft manifest is invalid")
    manifest = record.manifest.raw
    source = next(
        (item for item in manifest["sources"] if item.get("type") == "file"),
        None,
    )
    if source is None:
        raise DraftValidationError("draft has no local source for animation planning")
    source_path = record.path / source["path"]
    source_text = extract_source_text(
        source.get("name", source_path.name), source_path.read_bytes()
    )
    result = plan_animation_options(
        _algorithm_spec_value(manifest),
        source_text,
        provider_note=provider_note,
        chat_model=chat_model,
        environ=environ,
    )
    return save_animation_options(
        draft_key,
        result.options,
        provider,
        source="Animation Concept Agent",
        agent_metadata=result.metadata,
        drafts_root=drafts_root,
    )


def select_animation_option(
    draft_key: str,
    option_id: str,
    provider: str,
    drafts_root: str | os.PathLike[str] | None = None,
) -> DraftRecord:
    if not provider.strip():
        raise DraftStateError("provider name is required")
    loaded = load_animation_options(draft_key, drafts_root)
    if loaded is None:
        raise DraftStateError("generate animation concepts before selecting one")
    identifiers = {item["option_id"] for item in loaded["options"]}
    if option_id not in identifiers:
        raise DraftValidationError("selected animation option does not exist")
    record = load_draft(draft_key, drafts_root)
    manifest = _load_raw(record.path)
    options_record = manifest["generation"]["animation_options"]
    previous = options_record.get("selected_option_id")
    if previous == option_id:
        raise DraftStateError("animation option is already selected")
    timestamp = utc_now()
    options_record["selected_option_id"] = option_id
    options_record["selected_by"] = provider.strip()
    options_record["selected_at"] = timestamp
    guidance = manifest["generation"].get("animation_guidance")
    if isinstance(guidance, dict) and guidance.get("selected_option_id") != option_id:
        guidance["stale_since"] = timestamp
        guidance["stale_reason"] = "Provider selected a different animation concept"
    _append_history(
        manifest,
        module=None,
        action="animation_option_selected",
        old_status=None,
        new_status=None,
        reviewer=provider.strip(),
        note=f"Provider selected animation concept '{option_id}'.",
    )
    _write_json(record.path / "manifest.json", manifest)
    return load_draft(draft_key, drafts_root)


def load_animation_guidance(
    draft_key: str,
    drafts_root: str | os.PathLike[str] | None = None,
) -> dict[str, Any] | None:
    """Load the creator brief without changing draft or module state."""

    record = load_draft(draft_key, drafts_root)
    if record.manifest is None:
        raise DraftValidationError("draft manifest is invalid")
    guidance_record = record.manifest.raw.get("generation", {}).get(
        "animation_guidance"
    )
    if not isinstance(guidance_record, Mapping):
        return None
    relative_path = str(guidance_record.get("file", "")).strip()
    if not relative_path:
        raise DraftValidationError("animation guidance file is not declared")
    path = (record.path / relative_path).resolve()
    try:
        path.relative_to(record.path)
    except ValueError as exc:
        raise DraftValidationError("animation guidance path escapes the draft") from exc
    try:
        payload = path.read_bytes()
        value = json.loads(payload.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DraftValidationError(f"animation guidance cannot be read: {exc}") from exc
    expected_digest = str(guidance_record.get("sha256", "")).strip()
    actual_digest = hashlib.sha256(payload).hexdigest()
    if expected_digest and expected_digest != actual_digest:
        raise DraftValidationError("animation guidance SHA-256 does not match")
    if not isinstance(value, Mapping):
        raise DraftValidationError("animation guidance must be a JSON object")
    return validate_animation_guidance(value)


def save_animation_guidance(
    draft_key: str,
    guidance: Mapping[str, Any],
    reviewer: str,
    *,
    source: str = "manual",
    agent_metadata: Mapping[str, Any] | None = None,
    drafts_root: str | os.PathLike[str] | None = None,
) -> DraftRecord:
    """Atomically save planning data; never mutate the MP4 or review status."""

    if not reviewer.strip():
        raise DraftStateError("provider name is required")
    normalized = validate_animation_guidance(guidance)
    record = load_draft(draft_key, drafts_root)
    if record.manifest is None:
        raise DraftValidationError("draft manifest is invalid")
    manifest = _load_raw(record.path)
    options_state = load_animation_options(draft_key, drafts_root)
    selected_option_id = (
        options_state.get("selected_option_id") if options_state else None
    )
    if not selected_option_id:
        raise DraftStateError(
            "select an animation concept before saving detailed guidance"
        )
    relative_path = "animation_guidance.json"
    payload = (json.dumps(normalized, ensure_ascii=False, indent=2) + "\n").encode(
        "utf-8"
    )
    timestamp = utc_now()
    previous = manifest["generation"].get("animation_guidance", {})
    history = list(previous.get("agent_runs", [])) if isinstance(previous, Mapping) else []
    if agent_metadata is not None:
        history.append(dict(agent_metadata))
    manifest["generation"]["animation_guidance"] = {
        "file": relative_path,
        "source": source,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "updated_at": timestamp,
        "updated_by": reviewer.strip(),
        "selected_option_id": selected_option_id,
        "agent_runs": history,
    }
    _append_history(
        manifest,
        module=None,
        action="animation_guidance_saved",
        old_status=None,
        new_status=None,
        reviewer=reviewer.strip(),
        note=(
            f"Animation creator guidance saved from {source}. "
            "This did not create or modify an MP4."
        ),
    )
    # Write the guidance first. If the manifest write fails, it remains an
    # undeclared recovery file and therefore cannot affect validation/install.
    _atomic_write(record.path / relative_path, payload)
    _write_json(record.path / "manifest.json", manifest)
    return load_draft(draft_key, drafts_root)


def generate_default_animation_guidance(
    draft_key: str,
    reviewer: str,
    drafts_root: str | os.PathLike[str] | None = None,
) -> DraftRecord:
    record = load_draft(draft_key, drafts_root)
    if record.manifest is None:
        raise DraftValidationError("draft manifest is invalid")
    options_state = load_animation_options(draft_key, drafts_root)
    if not options_state or not options_state.get("selected_option_id"):
        raise DraftStateError(
            "select an animation concept before building the storyboard"
        )
    selected = next(
        item
        for item in options_state["options"]
        if item["option_id"] == options_state["selected_option_id"]
    )
    guidance = default_animation_guidance(_algorithm_spec_value(record.manifest.raw))
    guidance["title"] = selected["title"]
    guidance["target_duration_seconds"] = selected["estimated_duration_seconds"]
    guidance["visual_style"] = [
        selected["visual_approach"],
        *guidance["visual_style"][:2],
    ]
    guidance["production_notes"] = [
        f"Selected concept: {selected['title']}.",
        *guidance["production_notes"],
    ]
    return save_animation_guidance(
        draft_key,
        guidance,
        reviewer,
        source="deterministic AlgorithmSpec starter",
        drafts_root=drafts_root,
    )


def generate_animation_guidance_with_agent(
    draft_key: str,
    reviewer: str,
    review_note: str = "",
    *,
    chat_model: Any | None = None,
    environ: Mapping[str, str] | None = None,
    drafts_root: str | os.PathLike[str] | None = None,
) -> DraftRecord:
    if not reviewer.strip():
        raise DraftStateError("provider name is required")
    record = load_draft(draft_key, drafts_root)
    if record.manifest is None:
        raise DraftValidationError("draft manifest is invalid")
    manifest = record.manifest.raw
    source = next(
        (item for item in manifest["sources"] if item.get("type") == "file"),
        None,
    )
    if source is None:
        raise DraftValidationError("draft has no local source for animation planning")
    source_path = record.path / source["path"]
    source_text = extract_source_text(
        source.get("name", source_path.name), source_path.read_bytes()
    )
    current = load_animation_guidance(draft_key, drafts_root)
    options_state = load_animation_options(draft_key, drafts_root)
    if not options_state or not options_state.get("selected_option_id"):
        raise DraftStateError(
            "select an animation concept before building the storyboard"
        )
    selected = next(
        item
        for item in options_state["options"]
        if item["option_id"] == options_state["selected_option_id"]
    )
    result = plan_animation_guidance(
        _algorithm_spec_value(manifest),
        source_text,
        review_note=review_note,
        current_guidance=current,
        selected_option=selected,
        chat_model=chat_model,
        environ=environ,
    )
    return save_animation_guidance(
        draft_key,
        result.guidance,
        reviewer,
        source="Animation Planning Agent",
        agent_metadata=result.metadata,
        drafts_root=drafts_root,
    )


def list_rejected_drafts(
    rejected_root: str | os.PathLike[str] | None = None,
) -> tuple[DraftRecord, ...]:
    root = (
        Path(rejected_root).resolve()
        if rejected_root is not None
        else default_rejected_root()
    )
    if not root.is_dir():
        return ()
    records = []
    for path in sorted(root.iterdir(), reverse=True):
        if path.is_dir() and not path.name.startswith("."):
            records.append(load_draft(path.name, root))
    return tuple(records)


def validate_draft(
    draft_key: str,
    drafts_root: str | os.PathLike[str] | None = None,
) -> ValidationReport:
    return load_draft(draft_key, drafts_root).report


def _validate_candidate(
    draft_path: Path,
    manifest: Mapping[str, Any],
    replacements: Mapping[str, bytes] | None = None,
) -> None:
    with tempfile.TemporaryDirectory(prefix="draft-candidate-") as temp_dir:
        candidate = Path(temp_dir) / "draft"
        shutil.copytree(draft_path, candidate)
        _write_json(candidate / "manifest.json", manifest)
        for relative_path, payload in (replacements or {}).items():
            _atomic_write(candidate / relative_path, payload)
        report = validate_source_directory(candidate)
        if not report.valid:
            raise DraftValidationError(
                "; ".join(issue.message for issue in report.errors)
            )


def save_algorithm_spec(
    draft_key: str,
    *,
    name: str,
    category: str,
    summary: str,
    algorithm: Mapping[str, Any],
    reviewer: str,
    drafts_root: str | os.PathLike[str] | None = None,
) -> DraftRecord:
    if not reviewer.strip():
        raise DraftStateError("reviewer name is required")
    if not all(value.strip() for value in (name, category, summary)):
        raise DraftValidationError("name, category, and summary are required")
    if not isinstance(algorithm, Mapping):
        raise DraftValidationError("algorithm must be an object")
    hyperparameters = algorithm.get("hyperparameters")
    if not isinstance(hyperparameters, Mapping):
        raise DraftValidationError("algorithm.hyperparameters must be an object")
    _validate_algorithm_semantics(
        hyperparameters,
        algorithm.get("core_equations", []),
        algorithm.get("pseudocode", []),
    )
    record = load_draft(draft_key, drafts_root)
    if record.manifest is None:
        raise DraftValidationError("draft manifest is invalid")
    manifest = _load_raw(record.path)
    previous_hash = _algorithm_spec_sha256(manifest)
    manifest["name"] = name.strip()
    manifest["category"] = category.strip()
    manifest["summary"] = summary.strip()
    manifest["algorithm"] = dict(algorithm)
    _validate_candidate(record.path, manifest)
    current_hash = _algorithm_spec_sha256(manifest)
    if current_hash == previous_hash:
        raise DraftStateError("AlgorithmSpec has no saved changes")
    timestamp = utc_now()
    manifest["generation"]["algorithm_spec_confirmation"] = {
        "confirmed_by": reviewer.strip(),
        "confirmed_at": timestamp,
        "spec_sha256": current_hash,
        "note": "AlgorithmSpec updated during draft review.",
    }
    manifest["generation"].setdefault("algorithm_spec_revisions", []).append(
        {
            "previous_sha256": previous_hash,
            "spec_sha256": current_hash,
            "reviewer": reviewer.strip(),
            "updated_at": timestamp,
        }
    )
    manifest["generation"]["module_generations"] = {}
    animation_guidance = manifest["generation"].get("animation_guidance")
    if isinstance(animation_guidance, dict):
        animation_guidance["stale_since"] = timestamp
        animation_guidance["stale_reason"] = "AlgorithmSpec was updated"
    animation_options = manifest["generation"].get("animation_options")
    if isinstance(animation_options, dict):
        animation_options["stale_since"] = timestamp
        animation_options["stale_reason"] = "AlgorithmSpec was updated"
    flags = manifest["generation"].setdefault("blocking_flags", [])
    if (
        manifest["generation"].get("mode") == "template"
        and "placeholder_content" not in flags
    ):
        flags.append("placeholder_content")
    for module in _declared_review_modules(manifest):
        _set_status(
            manifest,
            module,
            "changes_requested",
            reviewer.strip(),
            note=(
                "AlgorithmSpec changed. Review or regenerate this module before approval."
            ),
            action="spec_updated",
        )
    _write_json(record.path / "manifest.json", manifest)
    return load_draft(draft_key, drafts_root)


def save_experiment_design(
    draft_key: str,
    experiment_design: Mapping[str, Any] | None,
    reviewer: str,
    drafts_root: str | os.PathLike[str] | None = None,
) -> DraftRecord:
    if not reviewer.strip():
        raise DraftStateError("reviewer name is required")
    normalized = (
        validate_experiment_design(experiment_design)
        if experiment_design is not None
        else None
    )
    record = load_draft(draft_key, drafts_root)
    if record.manifest is None:
        raise DraftValidationError("draft manifest is invalid")
    manifest = _load_raw(record.path)
    algorithm = manifest["algorithm"]
    previous = algorithm.get("experiment_design")
    if previous == normalized:
        raise DraftStateError("experiment scenario has no saved changes")
    previous_hash = _algorithm_spec_sha256(manifest)
    if normalized is None:
        algorithm.pop("experiment_design", None)
        scenario_label = "No declarative map"
    else:
        algorithm["experiment_design"] = normalized
        scenario_label = normalized["provenance"]["label"]
    _validate_candidate(record.path, manifest)
    timestamp = utc_now()
    current_hash = _algorithm_spec_sha256(manifest)
    manifest["generation"]["algorithm_spec_confirmation"] = {
        "confirmed_by": reviewer.strip(),
        "confirmed_at": timestamp,
        "spec_sha256": current_hash,
        "note": "Experiment teaching scenario updated during draft review.",
    }
    manifest["generation"].setdefault("algorithm_spec_revisions", []).append(
        {
            "previous_sha256": previous_hash,
            "spec_sha256": current_hash,
            "reviewer": reviewer.strip(),
            "updated_at": timestamp,
            "affected_modules": ["experiment"],
        }
    )
    _set_status(
        manifest,
        "experiment",
        "changes_requested",
        reviewer.strip(),
        note=(
            f"Experiment scenario changed to {scenario_label}. Regenerate or "
            "replace only the Experiment module."
        ),
        action="experiment_design_updated",
    )
    _write_json(record.path / "manifest.json", manifest)
    return load_draft(draft_key, drafts_root)


def save_theory(
    draft_key: str,
    content: str,
    reviewer: str,
    drafts_root: str | os.PathLike[str] | None = None,
) -> DraftRecord:
    if not reviewer.strip():
        raise DraftStateError("reviewer name is required")
    if not content.strip():
        raise DraftValidationError("theory content cannot be empty")
    record = load_draft(draft_key, drafts_root)
    if record.manifest is None:
        raise DraftValidationError("draft manifest is invalid")
    manifest = _load_raw(record.path)
    if manifest["review"]["modules"]["theory"]["status"] == "approved":
        raise DraftStateError(
            "approved theory is locked; request changes before editing it"
        )
    theory_file = manifest["modules"]["theory"]["file"]
    _write_text(record.path / theory_file, content)
    report = validate_source_directory(record.path)
    _set_status(
        manifest,
        "theory",
        "awaiting_review" if report.valid else "validation_failed",
        reviewer,
        note="Theory content was edited.",
        action="edited",
    )
    _write_json(record.path / "manifest.json", manifest)
    return load_draft(draft_key, drafts_root)


def replace_module_file(
    draft_key: str,
    module: str,
    file_name: str,
    payload: bytes,
    reviewer: str,
    drafts_root: str | os.PathLike[str] | None = None,
) -> DraftRecord:
    if module not in {"notebook", "experiment"}:
        raise DraftStateError("only Notebook and Experiment files can be replaced")
    if not reviewer.strip():
        raise DraftStateError("reviewer name is required")
    if not payload:
        raise DraftValidationError("replacement file cannot be empty")
    expected_suffix = ".ipynb" if module == "notebook" else ".py"
    if Path(file_name).suffix.lower() != expected_suffix:
        raise DraftValidationError(
            f"{module} replacement must use the {expected_suffix} extension"
        )
    record = load_draft(draft_key, drafts_root)
    if record.manifest is None:
        raise DraftValidationError("draft manifest is invalid")
    manifest = _load_raw(record.path)
    if manifest["review"]["modules"][module]["status"] == "approved":
        raise DraftStateError(
            f"approved {module} is locked; request changes before regenerating it"
        )
    state = manifest["review"]["modules"][module]
    if state["status"] == "approved":
        raise DraftStateError(
            f"approved {module} is locked; request changes before replacing it"
        )
    declaration = manifest["modules"][module]
    relative_path = (
        declaration["file"] if module == "notebook" else declaration["module"]
    )
    _validate_candidate(record.path, manifest, {relative_path: payload})
    _atomic_write(record.path / relative_path, payload)
    _set_status(
        manifest,
        module,
        "awaiting_review",
        reviewer.strip(),
        note=f"Module file replaced from upload '{Path(file_name).name}'.",
        action="file_replaced",
    )
    _write_json(record.path / "manifest.json", manifest)
    return load_draft(draft_key, drafts_root)


def save_animation_module(
    draft_key: str,
    *,
    file_name: str | None,
    payload: bytes | None,
    formula: str,
    highlights: Iterable[str],
    derivation_steps: Iterable[Mapping[str, Any]],
    reviewer: str,
    concept_markdown: str = "",
    symbols: Iterable[Mapping[str, str]] = (),
    viewing_flow: Iterable[str] = (),
    drafts_root: str | os.PathLike[str] | None = None,
) -> DraftRecord:
    if not reviewer.strip():
        raise DraftStateError("reviewer name is required")
    cleaned_highlights = tuple(
        item.strip() for item in highlights if item.strip()
    )
    cleaned_viewing_flow = tuple(
        item.strip() for item in viewing_flow if item.strip()
    )
    cleaned_symbols = tuple(
        {
            "symbol": str(item.get("symbol", "")).strip(),
            "meaning": str(item.get("meaning", "")).strip(),
        }
        for item in symbols
    )
    cleaned_steps = tuple(dict(step) for step in derivation_steps)
    _validate_animation_metadata(
        concept_markdown,
        formula,
        cleaned_symbols,
        cleaned_highlights,
        cleaned_viewing_flow,
        cleaned_steps,
    )

    record = load_draft(draft_key, drafts_root)
    if record.manifest is None:
        raise DraftValidationError("draft manifest is invalid")
    manifest = _load_raw(record.path)
    existing = manifest["modules"].get("animation")
    if existing is None and payload is None:
        raise DraftValidationError("an MP4 file is required when adding Animation")
    if existing is not None:
        state = manifest["review"]["modules"]["animation"]
        if state["status"] == "approved":
            raise DraftStateError(
                "approved animation is locked; request changes before editing it"
            )
    if payload is not None:
        _validate_animation_payload(file_name or "", payload)

    manifest["modules"]["animation"] = {
        "file": "animation.mp4",
        "concept_markdown": concept_markdown.strip(),
        "formula": formula.strip(),
        "symbols": [dict(item) for item in cleaned_symbols],
        "highlights": list(cleaned_highlights),
        "viewing_flow": list(cleaned_viewing_flow),
        "derivation_steps": [dict(step) for step in cleaned_steps],
    }
    if existing is None:
        manifest["review"]["modules"]["animation"] = {
            "status": "draft",
            "reviewer": None,
            "note": "",
            "updated_at": utc_now(),
        }
    replacements = {"animation.mp4": payload} if payload is not None else {}
    _validate_candidate(record.path, manifest, replacements)
    if payload is not None:
        _atomic_write(record.path / "animation.mp4", payload)
    _set_status(
        manifest,
        "animation",
        "awaiting_review",
        reviewer.strip(),
        note=(
            f"Animation MP4 and metadata saved from '{Path(file_name or '').name}'."
            if payload is not None
            else "Animation metadata updated."
        ),
        action="animation_added" if existing is None else "animation_updated",
    )
    _write_json(record.path / "manifest.json", manifest)
    return load_draft(draft_key, drafts_root)


def remove_animation_module(
    draft_key: str,
    reviewer: str,
    drafts_root: str | os.PathLike[str] | None = None,
    trash_root: str | os.PathLike[str] | None = None,
) -> DraftRecord:
    if not reviewer.strip():
        raise DraftStateError("reviewer name is required")
    record = load_draft(draft_key, drafts_root)
    if record.manifest is None:
        raise DraftValidationError("draft manifest is invalid")
    manifest = _load_raw(record.path)
    animation = manifest["modules"].get("animation")
    if animation is None:
        raise DraftStateError("draft does not declare an Animation module")
    state = manifest["review"]["modules"]["animation"]
    if state["status"] == "approved":
        raise DraftStateError(
            "approved animation is locked; request changes before removing it"
        )

    animation_path = record.path / animation["file"]
    manifest["modules"].pop("animation")
    manifest["review"]["modules"].pop("animation")
    _append_history(
        manifest,
        module="animation",
        action="animation_removed",
        old_status=state["status"],
        new_status=None,
        reviewer=reviewer.strip(),
        note="Optional Animation removed from the draft.",
    )
    _validate_candidate(record.path, manifest)

    trash = (
        Path(trash_root).resolve()
        if trash_root is not None
        else default_packages_root() / ".trash" / "draft-assets"
    )
    trash.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = trash / f"{record.key}-{timestamp}-animation.mp4"
    os.replace(animation_path, destination)
    try:
        _write_json(record.path / "manifest.json", manifest)
    except Exception:
        os.replace(destination, animation_path)
        raise
    return load_draft(draft_key, drafts_root)


def resolve_placeholder_blocker(
    draft_key: str,
    reviewer: str,
    note: str,
    drafts_root: str | os.PathLike[str] | None = None,
) -> DraftRecord:
    if not reviewer.strip():
        raise DraftStateError("reviewer name is required")
    if not note.strip():
        raise DraftStateError("a completion note is required")
    record = load_draft(draft_key, drafts_root)
    if not record.report.valid or record.manifest is None:
        raise DraftStateError("draft validation must pass first")
    manifest = _load_raw(record.path)
    flags = manifest["generation"].get("blocking_flags", [])
    if "placeholder_content" not in flags:
        raise DraftStateError("draft has no placeholder_content blocker")
    missing = {
        module
        for module in CORE_MODULES
        if not module_content_ready(manifest, module)
    }
    if missing:
        labels = ", ".join(sorted(missing))
        raise DraftStateError(
            "content completion is incomplete; missing " + labels
        )
    manifest["generation"]["blocking_flags"] = [
        flag for flag in flags if flag != "placeholder_content"
    ]
    manifest["generation"]["manual_completion"] = {
        "reviewer": reviewer.strip(),
        "note": note.strip(),
        "completed_at": utc_now(),
    }
    _append_history(
        manifest,
        module=None,
        action="publication_blocker_resolved",
        old_status=None,
        new_status=None,
        reviewer=reviewer.strip(),
        note=note.strip(),
    )
    _write_json(record.path / "manifest.json", manifest)
    return load_draft(draft_key, drafts_root)


def regenerate_module(
    draft_key: str,
    module: str,
    reviewer: str,
    drafts_root: str | os.PathLike[str] | None = None,
) -> DraftRecord:
    if module not in CORE_MODULES:
        raise DraftStateError(f"unsupported module: {module}")
    if not reviewer.strip():
        raise DraftStateError("reviewer name is required")
    record = load_draft(draft_key, drafts_root)
    if record.manifest is None:
        raise DraftValidationError("draft manifest is invalid")
    manifest = _load_raw(record.path)
    if (
        manifest.get("generation", {}).get("revision_of")
        and manifest.get("generation", {}).get("mode") == "template"
    ):
        raise DraftStateError(
            "revision modules cannot be replaced with deterministic scaffolds; "
            "configure the module Agent or upload a validated replacement"
        )
    source = next(
        source for source in manifest["sources"] if source.get("type") == "file"
    )
    source_text = extract_source_text(
        source.get("name", source["path"]),
        (record.path / source["path"]).read_bytes(),
    )
    _write_generated_modules(record.path, manifest, source_text, module)
    report = validate_source_directory(record.path)
    _set_status(
        manifest,
        module,
        "awaiting_review" if report.valid else "validation_failed",
        reviewer,
        note="Module regenerated from the current AlgorithmSpec.",
        action="regenerated",
    )
    _write_json(record.path / "manifest.json", manifest)
    return load_draft(draft_key, drafts_root)


def generate_module_with_agent(
    draft_key: str,
    module: str,
    reviewer: str,
    review_note: str = "",
    drafts_root: str | os.PathLike[str] | None = None,
    *,
    chat_model: Any | None = None,
    environ: Mapping[str, str] | None = None,
) -> DraftRecord:
    from .module_agent import generate_module_content

    if module not in CORE_MODULES:
        raise DraftStateError(f"unsupported Agent module: {module}")
    if not reviewer.strip():
        raise DraftStateError("reviewer name is required")
    record = load_draft(draft_key, drafts_root)
    if record.manifest is None:
        raise DraftValidationError("draft manifest is invalid")
    manifest = _load_raw(record.path)
    state = manifest["review"]["modules"][module]
    if state["status"] == "approved":
        raise DraftStateError(
            f"approved {module} is locked; request changes before regenerating it"
        )
    source = next(
        (
            source
            for source in manifest["sources"]
            if source.get("type") == "file"
        ),
        None,
    )
    if source is None:
        raise DraftValidationError("draft has no local source for module generation")
    source_path = record.path / source["path"]
    source_text = extract_source_text(
        source.get("name", source_path.name),
        source_path.read_bytes(),
    )
    declaration = manifest["modules"][module]
    relative_path = (
        declaration["module"]
        if module == "experiment"
        else declaration["file"]
    )
    current_path = record.path / relative_path
    current_content = current_path.read_text(encoding="utf-8")
    effective_note = review_note.strip()
    if not effective_note and state["status"] == "changes_requested":
        effective_note = str(state.get("note", "")).strip()

    result = generate_module_content(
        module,
        _algorithm_spec_value(manifest),
        source_text,
        review_note=effective_note,
        current_content=current_content,
        chat_model=chat_model,
        environ=environ,
    )
    _validate_candidate(
        record.path,
        manifest,
        {relative_path: result.payload},
    )
    _atomic_write(current_path, result.payload)
    generation_history = manifest["generation"].setdefault(
        "module_generations", {}
    )
    generation_history.setdefault(module, []).append(dict(result.metadata))
    _set_status(
        manifest,
        module,
        "awaiting_review",
        reviewer.strip(),
        note=(
            "Module generated by Agent from the confirmed AlgorithmSpec."
            + (
                f" Reviewer feedback applied: {effective_note}"
                if effective_note
                else ""
            )
        ),
        action="agent_generated",
    )

    completed = all(generation_history.get(name) for name in CORE_MODULES)
    flags = manifest["generation"].get("blocking_flags", [])
    if completed and "placeholder_content" in flags:
        manifest["generation"]["blocking_flags"] = [
            flag for flag in flags if flag != "placeholder_content"
        ]
        manifest["generation"]["agent_completion"] = {
            "completed_at": utc_now(),
            "modules": list(CORE_MODULES),
            "note": (
                "All core scaffold files were replaced by validated Agent "
                "outputs. Human module approval is still required."
            ),
        }
        _append_history(
            manifest,
            module=None,
            action="placeholder_replaced_by_agents",
            old_status=None,
            new_status=None,
            reviewer=reviewer.strip(),
            note="All core modules now have validated Agent-generated drafts.",
        )
    _write_json(record.path / "manifest.json", manifest)
    return load_draft(draft_key, drafts_root)


def request_changes(
    draft_key: str,
    module: str,
    reviewer: str,
    note: str,
    drafts_root: str | os.PathLike[str] | None = None,
) -> DraftRecord:
    if not reviewer.strip():
        raise DraftStateError("reviewer name is required")
    if not note.strip():
        raise DraftStateError("a change request reason is required")
    record = load_draft(draft_key, drafts_root)
    if record.manifest is None:
        raise DraftValidationError("draft manifest is invalid")
    manifest = _load_raw(record.path)
    if module not in _declared_review_modules(manifest):
        raise DraftStateError(f"unsupported module: {module}")
    _set_status(
        manifest,
        module,
        "changes_requested",
        reviewer.strip(),
        note.strip(),
        action="changes_requested",
    )
    _write_json(record.path / "manifest.json", manifest)
    return load_draft(draft_key, drafts_root)


def cancel_change_request(
    draft_key: str,
    module: str,
    reviewer: str,
    note: str,
    drafts_root: str | os.PathLike[str] | None = None,
) -> DraftRecord:
    if not reviewer.strip():
        raise DraftStateError("reviewer name is required")
    if not note.strip():
        raise DraftStateError("a cancellation reason is required")
    record = load_draft(draft_key, drafts_root)
    if record.manifest is None:
        raise DraftValidationError("draft manifest is invalid")
    manifest = _load_raw(record.path)
    if module not in _declared_review_modules(manifest):
        raise DraftStateError(f"unsupported module: {module}")
    current = manifest["review"]["modules"][module]["status"]
    if current != "changes_requested":
        raise DraftStateError(
            f"module '{module}' does not have an active change request"
        )
    _set_status(
        manifest,
        module,
        "awaiting_review",
        reviewer.strip(),
        note.strip(),
        action="change_request_cancelled",
    )
    _write_json(record.path / "manifest.json", manifest)
    return load_draft(draft_key, drafts_root)


def approve_module(
    draft_key: str,
    module: str,
    reviewer: str,
    note: str = "",
    drafts_root: str | os.PathLike[str] | None = None,
) -> DraftRecord:
    record = load_draft(draft_key, drafts_root)
    manifest = _load_raw(record.path)
    if module not in _declared_review_modules(manifest):
        raise DraftStateError(f"unsupported module: {module}")
    if not reviewer.strip():
        raise DraftStateError("reviewer name is required")
    if not record.report.valid or record.manifest is None:
        raise DraftStateError("draft validation must pass before approval")
    flags = manifest["generation"].get("blocking_flags", [])
    non_placeholder_flags = [
        flag for flag in flags if flag != "placeholder_content"
    ]
    if non_placeholder_flags:
        raise DraftStateError(
            "draft has publication blockers: "
            + ", ".join(non_placeholder_flags)
        )
    if (
        "placeholder_content" in flags
        and not module_content_ready(manifest, module)
    ):
        raise DraftStateError(
            f"module '{module}' still contains scaffold content; generate, "
            "edit, or replace it before approval"
        )
    current = manifest["review"]["modules"][module]["status"]
    if current != "awaiting_review":
        raise DraftStateError(
            f"module '{module}' must be awaiting_review before approval"
        )
    _set_status(
        manifest,
        module,
        "approved",
        reviewer.strip(),
        note.strip(),
        action="approved",
    )
    _write_json(record.path / "manifest.json", manifest)
    return load_draft(draft_key, drafts_root)


def reject_draft(
    draft_key: str,
    reviewer: str,
    reason: str,
    drafts_root: str | os.PathLike[str] | None = None,
    rejected_root: str | os.PathLike[str] | None = None,
) -> Path:
    if not reviewer.strip():
        raise DraftStateError("reviewer name is required")
    if not reason.strip():
        raise DraftStateError("a rejection reason is required")
    record = load_draft(draft_key, drafts_root)
    manifest = _load_raw(record.path)
    _append_history(
        manifest,
        module=None,
        action="draft_rejected",
        old_status=None,
        new_status=None,
        reviewer=reviewer.strip(),
        note=reason.strip(),
    )
    _write_json(record.path / "manifest.json", manifest)
    root = (
        Path(rejected_root).resolve()
        if rejected_root is not None
        else default_rejected_root()
    )
    root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = root / f"{record.path.name}-{timestamp}"
    os.replace(record.path, destination)
    return destination


def restore_rejected_draft(
    rejected_key: str,
    reviewer: str,
    rejected_root: str | os.PathLike[str] | None = None,
    drafts_root: str | os.PathLike[str] | None = None,
) -> DraftRecord:
    if not reviewer.strip():
        raise DraftStateError("reviewer name is required")
    rejected = (
        Path(rejected_root).resolve()
        if rejected_root is not None
        else default_rejected_root()
    )
    record = load_draft(rejected_key, rejected)
    if record.manifest is None:
        raise DraftValidationError("rejected draft manifest is invalid")
    drafts = Path(drafts_root).resolve() if drafts_root else default_drafts_root()
    destination = drafts / (
        f"{record.manifest.algorithm_id}-{record.manifest.version}"
    )
    if destination.exists():
        raise DraftStateError(f"draft '{destination.name}' already exists")
    manifest = _load_raw(record.path)
    for module in _declared_review_modules(manifest):
        _set_status(
            manifest,
            module,
            "changes_requested",
            reviewer.strip(),
            note="Rejected draft restored for another review cycle.",
            action="draft_restored",
        )
    _append_history(
        manifest,
        module=None,
        action="draft_restored",
        old_status=None,
        new_status=None,
        reviewer=reviewer.strip(),
        note="Draft moved from rejected archive back to active review.",
    )
    _write_json(record.path / "manifest.json", manifest)
    drafts.mkdir(parents=True, exist_ok=True)
    os.replace(record.path, destination)
    return load_draft(destination.name, drafts)


def trash_rejected_draft(
    rejected_key: str,
    reviewer: str,
    rejected_root: str | os.PathLike[str] | None = None,
    trash_root: str | os.PathLike[str] | None = None,
) -> Path:
    if not reviewer.strip():
        raise DraftStateError("reviewer name is required")
    rejected = (
        Path(rejected_root).resolve()
        if rejected_root is not None
        else default_rejected_root()
    )
    record = load_draft(rejected_key, rejected)
    manifest = _load_raw(record.path)
    _append_history(
        manifest,
        module=None,
        action="rejected_draft_trashed",
        old_status=None,
        new_status=None,
        reviewer=reviewer.strip(),
        note="Rejected draft moved to recoverable trash.",
    )
    _write_json(record.path / "manifest.json", manifest)
    trash = (
        Path(trash_root).resolve()
        if trash_root is not None
        else default_packages_root() / ".trash" / "rejected"
    )
    trash.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = trash / f"{record.path.name}-{timestamp}"
    os.replace(record.path, destination)
    return destination


def _build_archive(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        destination,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
    ) as archive:
        for file_path in iter_package_files(source):
            archive.write(file_path, file_path.relative_to(source).as_posix())


def install_approved_draft(
    draft_key: str,
    reviewer: str,
    drafts_root: str | os.PathLike[str] | None = None,
    installed_root: str | os.PathLike[str] | None = None,
    artifact_root: str | os.PathLike[str] | None = None,
    trash_root: str | os.PathLike[str] | None = None,
) -> tuple[InstalledAlgorithm, Path]:
    if not reviewer.strip():
        raise DraftStateError("reviewer name is required")
    record = load_draft(draft_key, drafts_root)
    if not record.report.valid or record.manifest is None:
        raise DraftStateError("draft validation must pass before installation")
    manifest = _load_raw(record.path)
    flags = manifest["generation"].get("blocking_flags", [])
    if flags:
        raise DraftStateError(
            "draft has publication blockers: " + ", ".join(flags)
        )
    review_modules = _declared_review_modules(manifest)
    statuses = {
        module: manifest["review"]["modules"][module]["status"]
        for module in review_modules
    }
    if any(status != "approved" for status in statuses.values()):
        raise DraftStateError(
            "all declared modules must be approved before installation"
        )

    destination_root = (
        Path(installed_root).resolve()
        if installed_root is not None
        else default_installed_root()
    )
    if (destination_root / manifest["id"]).exists():
        raise DraftStateError(
            f"algorithm '{manifest['id']}' is already installed; remove it first"
        )
    output_root = (
        Path(artifact_root).resolve()
        if artifact_root is not None
        else project_root() / "dist"
    )
    archive_path = output_root / f"{manifest['id']}-{manifest['version']}.zip"

    output_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="approved-draft-") as temp_dir:
        final_source = Path(temp_dir) / "package"
        shutil.copytree(record.path, final_source)
        final_manifest = _load_raw(final_source)
        for module in review_modules:
            _set_status(
                final_manifest,
                module,
                "installed",
                reviewer.strip(),
                note="Approved draft published.",
                action="installed",
            )
        _write_json(final_source / "manifest.json", final_manifest)
        final_report = validate_source_directory(final_source)
        if not final_report.valid:
            raise DraftValidationError(
                "; ".join(issue.message for issue in final_report.errors)
            )
        archive_handle, archive_name = tempfile.mkstemp(
            prefix=f".{archive_path.name}.",
            suffix=".tmp",
            dir=output_root,
        )
        os.close(archive_handle)
        temporary_archive = Path(archive_name)
        temporary_archive.unlink()
        _build_archive(final_source, temporary_archive)
        try:
            os.replace(temporary_archive, archive_path)
            installed = install_package(archive_path, destination_root)
        finally:
            temporary_archive.unlink(missing_ok=True)

    archive_root = (
        Path(trash_root).resolve()
        if trash_root is not None
        else default_packages_root() / ".trash" / "drafts"
    )
    archive_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    archived_draft = archive_root / f"{record.path.name}-{timestamp}"
    os.replace(record.path, archived_draft)
    return installed, archive_path
