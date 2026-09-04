from __future__ import annotations

import ast
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import re
import shutil
import stat
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping

import nbformat
from packaging.requirements import InvalidRequirement, Requirement

from .models import (
    AlgorithmManifest,
    AlgorithmNotFoundError,
    DependencyStatus,
    DuplicateAlgorithmError,
    InstalledAlgorithm,
    PackageIssue,
    PackageValidationError,
    ValidationReport,
)
from .experiment_design import validate_experiment_design


SCHEMA_VERSION = 2
SUPPORTED_SCHEMA_VERSIONS = {1, 2}
REVIEW_STATUSES = {
    "not_generated",
    "generating",
    "draft",
    "validation_failed",
    "awaiting_review",
    "changes_requested",
    "approved",
    "installed",
}
CORE_MODULES = ("theory", "notebook", "experiment")
ALGORITHM_ID_PATTERN = re.compile(r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$")
VERSION_PATTERN = re.compile(r"^\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?$")
MAX_FILE_COUNT = 500
MAX_COMPRESSED_SIZE = 512 * 1024 * 1024
MAX_UNCOMPRESSED_SIZE = 1024 * 1024 * 1024
MAX_COMPRESSION_RATIO = 200


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_installed_root() -> Path:
    return project_root() / "algorithm_packages" / "installed"


class _IssueCollector:
    def __init__(self) -> None:
        self.issues: list[PackageIssue] = []

    def error(self, code: str, message: str) -> None:
        self.issues.append(PackageIssue("error", code, message))

    def warning(self, code: str, message: str) -> None:
        self.issues.append(PackageIssue("warning", code, message))


def _safe_relative_path(value: Any, field_name: str, issues: _IssueCollector) -> str | None:
    if not isinstance(value, str) or not value.strip():
        issues.error("invalid_path", f"{field_name} must be a non-empty relative path")
        return None
    normalized = value.replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute() or ".." in path.parts or normalized.startswith("/"):
        issues.error("unsafe_path", f"{field_name} must remain inside the package")
        return None
    return str(path)


def _required_string(
    data: Mapping[str, Any], field_name: str, issues: _IssueCollector
) -> str | None:
    value = data.get(field_name)
    if not isinstance(value, str) or not value.strip():
        issues.error("invalid_manifest", f"manifest field '{field_name}' is required")
        return None
    return value.strip()


def _string_list(
    value: Any,
    field_name: str,
    issues: _IssueCollector,
    *,
    required: bool = True,
) -> list[str]:
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        issues.error(
            "invalid_algorithm_spec",
            f"{field_name} must be a list of non-empty strings",
        )
        return []
    if required and not value:
        issues.error("invalid_algorithm_spec", f"{field_name} cannot be empty")
    return [item.strip() for item in value]


def _parse_modules(
    raw_modules: Any,
    issues: _IssueCollector,
    *,
    require_core: bool,
) -> tuple[
    str | None,
    Mapping[str, Any] | None,
    Mapping[str, Any] | None,
    Mapping[str, Any] | None,
    dict[str, Any],
]:
    if not isinstance(raw_modules, Mapping):
        issues.error("invalid_modules", "manifest modules must be an object")
        return None, None, None, None, {}

    modules = dict(raw_modules)
    if require_core:
        for module_name in CORE_MODULES:
            if module_name not in modules:
                issues.error(
                    "missing_module",
                    f"schema v2 requires the '{module_name}' module",
                )

    theory = modules.get("theory")
    theory_file = None
    if not isinstance(theory, Mapping):
        issues.error("missing_theory", "manifest theory module is required")
    else:
        theory_file = _safe_relative_path(theory.get("file"), "modules.theory.file", issues)
        modules["theory"] = dict(theory)
        if theory_file is not None:
            modules["theory"]["file"] = theory_file

    animation = modules.get("animation")
    if animation is not None:
        if not isinstance(animation, Mapping):
            issues.error("invalid_animation", "animation must be an object")
            animation = None
        else:
            animation_file = _safe_relative_path(
                animation.get("file"), "modules.animation.file", issues
            )
            animation = dict(animation)
            if animation_file is not None:
                animation["file"] = animation_file
                modules["animation"] = animation
            else:
                animation = None
            if animation is not None:
                for field_name in ("concept_markdown", "formula"):
                    value = animation.get(field_name)
                    if value is not None and not isinstance(value, str):
                        issues.error(
                            "invalid_animation",
                            f"animation.{field_name} must be a string",
                        )
                for field_name in ("highlights", "viewing_flow"):
                    value = animation.get(field_name)
                    if value is not None and (
                        not isinstance(value, list)
                        or not all(
                            isinstance(item, str) and item.strip()
                            for item in value
                        )
                    ):
                        issues.error(
                            "invalid_animation",
                            f"animation.{field_name} must be a list of non-empty strings",
                        )
                symbols = animation.get("symbols")
                if symbols is not None and (
                    not isinstance(symbols, list)
                    or not all(
                        isinstance(item, Mapping)
                        and set(item) == {"symbol", "meaning"}
                        and all(
                            isinstance(item[field], str) and item[field].strip()
                            for field in ("symbol", "meaning")
                        )
                        for item in symbols
                    )
                ):
                    issues.error(
                        "invalid_animation",
                        "animation.symbols must contain symbol/meaning objects",
                    )
                derivation_steps = animation.get("derivation_steps")
                if derivation_steps is not None:
                    if not isinstance(derivation_steps, list) or not all(
                        isinstance(item, Mapping) for item in derivation_steps
                    ):
                        issues.error(
                            "invalid_animation",
                            "animation.derivation_steps must be a list of objects",
                        )
                    else:
                        normalized_steps: list[dict[str, Any]] = []
                        for index, item in enumerate(derivation_steps):
                            step = dict(item)
                            if "name" in step or "content" in step:
                                step = {
                                    "title": step.get("title", step.get("name", "")),
                                    "text": step.get("text", step.get("content", "")),
                                    "latex": step.get("latex", ""),
                                }
                                issues.warning(
                                    "legacy_animation_step",
                                    f"animation.derivation_steps[{index}] uses legacy name/content fields",
                                )
                            unknown = set(step) - {"title", "text", "latex"}
                            if unknown:
                                issues.error(
                                    "invalid_animation",
                                    f"animation.derivation_steps[{index}] contains unsupported fields",
                                )
                            for field_name in ("title", "text"):
                                value = step.get(field_name)
                                if value is not None and not isinstance(value, str):
                                    issues.error(
                                        "invalid_animation",
                                        f"animation.derivation_steps[{index}].{field_name} must be a string",
                                    )
                            latex = step.get("latex")
                            if latex is not None and not (
                                isinstance(latex, str)
                                or (
                                    isinstance(latex, list)
                                    and all(
                                        isinstance(value, str) and value.strip()
                                        for value in latex
                                    )
                                )
                            ):
                                issues.error(
                                    "invalid_animation",
                                    f"animation.derivation_steps[{index}].latex is invalid",
                                )
                            if not any(
                                step.get(field_name)
                                for field_name in ("title", "text", "latex")
                            ):
                                issues.error(
                                    "invalid_animation",
                                    f"animation.derivation_steps[{index}] cannot be empty",
                                )
                            normalized_steps.append(step)
                        animation["derivation_steps"] = normalized_steps
                        modules["animation"] = animation

    notebook = modules.get("notebook")
    if notebook is not None:
        if not isinstance(notebook, Mapping):
            issues.error("invalid_notebook", "notebook must be an object")
            notebook = None
        else:
            notebook_file = _safe_relative_path(
                notebook.get("file"), "modules.notebook.file", issues
            )
            notebook = dict(notebook)
            if notebook_file is not None:
                notebook["file"] = notebook_file
                modules["notebook"] = notebook
            else:
                notebook = None

    experiment = modules.get("experiment")
    if experiment is not None:
        if not isinstance(experiment, Mapping):
            issues.error("invalid_experiment", "experiment must be an object")
            experiment = None
        else:
            experiment_module = _safe_relative_path(
                experiment.get("module"), "modules.experiment.module", issues
            )
            requirements = experiment.get("requirements", [])
            if not isinstance(requirements, list):
                issues.error(
                    "invalid_requirements", "experiment.requirements must be a list"
                )
            else:
                for index, requirement in enumerate(requirements):
                    if not isinstance(requirement, Mapping):
                        issues.error(
                            "invalid_requirements",
                            f"experiment requirement {index} must be an object",
                        )
                        continue
                    package = requirement.get("package")
                    import_name = requirement.get("import")
                    if not isinstance(package, str) or not package.strip():
                        issues.error(
                            "invalid_requirements",
                            f"experiment requirement {index} needs a package",
                        )
                    else:
                        try:
                            Requirement(package)
                        except InvalidRequirement as exc:
                            issues.error(
                                "invalid_requirements",
                                f"invalid requirement '{package}': {exc}",
                            )
                    if not isinstance(import_name, str) or not import_name.strip():
                        issues.error(
                            "invalid_requirements",
                            f"experiment requirement {index} needs an import name",
                        )
            experiment = dict(experiment)
            if experiment_module is not None:
                experiment["module"] = experiment_module
                modules["experiment"] = experiment
            else:
                experiment = None

    return theory_file, animation, notebook, experiment, modules


def _parse_manifest(
    package_root: Path, issues: _IssueCollector
) -> AlgorithmManifest | None:
    manifest_path = package_root / "manifest.json"
    if not manifest_path.is_file():
        issues.error("missing_manifest", "manifest.json is required")
        return None

    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        issues.error("invalid_manifest", f"manifest.json cannot be read: {exc}")
        return None

    if not isinstance(raw, Mapping):
        issues.error("invalid_manifest", "manifest.json must contain a JSON object")
        return None

    schema_version = raw.get("schema_version")
    if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        issues.error(
            "unsupported_schema",
            "schema_version must be one of "
            f"{sorted(SUPPORTED_SCHEMA_VERSIONS)}, got {schema_version!r}",
        )

    algorithm_id = _required_string(raw, "id", issues)
    name = _required_string(raw, "name", issues)
    version = _required_string(raw, "version", issues)
    summary = _required_string(raw, "summary", issues)
    category = _required_string(raw, "category", issues)

    if algorithm_id and not ALGORITHM_ID_PATTERN.fullmatch(algorithm_id):
        issues.error(
            "invalid_algorithm_id",
            "algorithm id must use lowercase letters, numbers, and single hyphens",
        )
    if version and not VERSION_PATTERN.fullmatch(version):
        issues.error("invalid_version", "version must follow semantic version format")

    sources: tuple[Mapping[str, Any], ...] = ()
    algorithm: Mapping[str, Any] = {}
    generation: Mapping[str, Any] = {}
    review: Mapping[str, Any] = {}

    if schema_version == 2:
        theory_file, animation, notebook, experiment, modules = _parse_modules(
            raw.get("modules"), issues, require_core=True
        )

        raw_sources = raw.get("sources")
        parsed_sources: list[Mapping[str, Any]] = []
        if not isinstance(raw_sources, list) or not raw_sources:
            issues.error("invalid_sources", "schema v2 sources must be a non-empty list")
        else:
            for index, source in enumerate(raw_sources):
                if not isinstance(source, Mapping):
                    issues.error(
                        "invalid_sources", f"source {index} must be an object"
                    )
                    continue
                source_type = source.get("type")
                if source_type == "file":
                    source_path = _safe_relative_path(
                        source.get("path"), f"sources[{index}].path", issues
                    )
                    digest = source.get("sha256")
                    if not isinstance(digest, str) or not re.fullmatch(
                        r"[0-9a-f]{64}", digest
                    ):
                        issues.error(
                            "invalid_sources",
                            f"sources[{index}].sha256 must be a lowercase SHA-256",
                        )
                    normalized_source = dict(source)
                    if source_path is not None:
                        normalized_source["path"] = source_path
                    parsed_sources.append(normalized_source)
                elif source_type == "url":
                    url = source.get("url")
                    if not isinstance(url, str) or not re.match(r"^https?://", url):
                        issues.error(
                            "invalid_sources",
                            f"sources[{index}].url must be an HTTP(S) URL",
                        )
                    parsed_sources.append(dict(source))
                else:
                    issues.error(
                        "invalid_sources",
                        f"sources[{index}].type must be 'file' or 'url'",
                    )
        sources = tuple(parsed_sources)

        raw_algorithm = raw.get("algorithm")
        if not isinstance(raw_algorithm, Mapping):
            issues.error("invalid_algorithm_spec", "algorithm must be an object")
        else:
            objective = raw_algorithm.get("objective")
            if not isinstance(objective, str) or not objective.strip():
                issues.error(
                    "invalid_algorithm_spec",
                    "algorithm.objective must be a non-empty string",
                )
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
                _string_list(
                    raw_algorithm.get(field_name),
                    f"algorithm.{field_name}",
                    issues,
                )
            if not isinstance(raw_algorithm.get("hyperparameters"), Mapping):
                issues.error(
                    "invalid_algorithm_spec",
                    "algorithm.hyperparameters must be an object",
                )
            experiment_design = raw_algorithm.get("experiment_design")
            if experiment_design is not None:
                try:
                    validate_experiment_design(experiment_design)
                except ValueError as exc:
                    issues.error("invalid_algorithm_spec", str(exc))
            algorithm = dict(raw_algorithm)

        raw_generation = raw.get("generation")
        if not isinstance(raw_generation, Mapping):
            issues.error("invalid_generation", "generation must be an object")
        else:
            for field_name in ("mode", "generator_version", "generated_at"):
                if not isinstance(raw_generation.get(field_name), str) or not str(
                    raw_generation.get(field_name)
                ).strip():
                    issues.error(
                        "invalid_generation",
                        f"generation.{field_name} must be a non-empty string",
                    )
            blocking_flags = raw_generation.get("blocking_flags", [])
            if not isinstance(blocking_flags, list) or not all(
                isinstance(item, str) and item.strip() for item in blocking_flags
            ):
                issues.error(
                    "invalid_generation",
                    "generation.blocking_flags must be a list of strings",
                )
            generation = dict(raw_generation)

        raw_review = raw.get("review")
        if not isinstance(raw_review, Mapping):
            issues.error("invalid_review", "review must be an object")
        else:
            review_modules = raw_review.get("modules")
            if not isinstance(review_modules, Mapping):
                issues.error("invalid_review", "review.modules must be an object")
            else:
                review_module_names = list(CORE_MODULES)
                if "animation" in modules:
                    review_module_names.append("animation")
                for module_name in review_module_names:
                    item = review_modules.get(module_name)
                    if not isinstance(item, Mapping):
                        issues.error(
                            "invalid_review",
                            f"review.modules.{module_name} must be an object",
                        )
                        continue
                    if item.get("status") not in REVIEW_STATUSES:
                        issues.error(
                            "invalid_review",
                            f"review.modules.{module_name}.status is invalid",
                        )
            history = raw_review.get("history")
            if not isinstance(history, list):
                issues.error("invalid_review", "review.history must be a list")
            review = dict(raw_review)
    else:
        theory_file, animation, notebook, experiment, modules = _parse_modules(
            {
                key: raw[key]
                for key in ("theory", "animation", "notebook", "experiment")
                if key in raw
            },
            issues,
            require_core=False,
        )

    if not all(
        [
            schema_version in SUPPORTED_SCHEMA_VERSIONS,
            algorithm_id,
            name,
            version,
            summary,
            category,
            theory_file,
        ]
    ):
        return None

    return AlgorithmManifest(
        schema_version=schema_version,
        algorithm_id=algorithm_id,
        name=name,
        version=version,
        summary=summary,
        category=category,
        theory_file=theory_file,
        animation=animation,
        notebook=notebook,
        experiment=experiment,
        sources=sources,
        algorithm=algorithm,
        modules=modules,
        generation=generation,
        review=review,
        raw=dict(raw),
    )


def _resolve_declared_file(
    package_root: Path,
    relative_path: str,
    field_name: str,
    issues: _IssueCollector,
) -> Path | None:
    candidate = (package_root / relative_path).resolve()
    try:
        candidate.relative_to(package_root.resolve())
    except ValueError:
        issues.error("unsafe_path", f"{field_name} resolves outside the package")
        return None
    if not candidate.is_file():
        issues.error("missing_file", f"{field_name} file does not exist: {relative_path}")
        return None
    return candidate


def _check_declared_files(
    package_root: Path, manifest: AlgorithmManifest, issues: _IssueCollector
) -> None:
    for index, source in enumerate(manifest.sources):
        if source.get("type") != "file" or not isinstance(source.get("path"), str):
            continue
        source_path = _resolve_declared_file(
            package_root,
            source["path"],
            f"sources[{index}]",
            issues,
        )
        if source_path is not None:
            try:
                digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
            except OSError as exc:
                issues.error(
                    "invalid_sources",
                    f"sources[{index}] cannot be read: {exc}",
                )
            else:
                if digest != source.get("sha256"):
                    issues.error(
                        "source_digest_mismatch",
                        f"sources[{index}] SHA-256 does not match its file",
                    )

    theory_path = _resolve_declared_file(
        package_root, manifest.theory_file, "theory", issues
    )
    if theory_path is not None:
        if theory_path.suffix.lower() != ".md":
            issues.error("invalid_theory", "theory file must use the .md extension")
        else:
            try:
                theory_content = theory_path.read_text(encoding="utf-8")
            except (OSError, UnicodeError) as exc:
                issues.error("invalid_theory", f"theory file cannot be read: {exc}")
            else:
                if not theory_content.strip():
                    issues.error("invalid_theory", "theory file cannot be empty")

    if manifest.animation is not None:
        animation_path = _resolve_declared_file(
            package_root, manifest.animation["file"], "animation", issues
        )
        if animation_path is not None:
            if animation_path.suffix.lower() != ".mp4":
                issues.error("invalid_animation", "animation file must be an MP4")
            elif animation_path.stat().st_size == 0:
                issues.error("invalid_animation", "animation file cannot be empty")
            elif (
                manifest.schema_version == 2
                and animation_path.stat().st_size > 200 * 1024 * 1024
            ):
                issues.error(
                    "invalid_animation",
                    "animation file exceeds the 200 MiB limit",
                )
            elif manifest.schema_version == 2:
                try:
                    with animation_path.open("rb") as animation_file:
                        header = animation_file.read(12)
                except OSError as exc:
                    issues.error(
                        "invalid_animation",
                        f"animation file cannot be read: {exc}",
                    )
                else:
                    if len(header) < 12 or header[4:8] != b"ftyp":
                        issues.error(
                            "invalid_animation",
                            "animation file has no recognizable MP4 header",
                        )

    if manifest.notebook is not None:
        notebook_path = _resolve_declared_file(
            package_root, manifest.notebook["file"], "notebook", issues
        )
        if notebook_path is not None:
            if notebook_path.suffix.lower() != ".ipynb":
                issues.error("invalid_notebook", "notebook file must use .ipynb")
            else:
                try:
                    notebook = nbformat.read(notebook_path, as_version=4)
                    nbformat.validate(notebook)
                except Exception as exc:
                    issues.error("invalid_notebook", f"notebook validation failed: {exc}")

    if manifest.experiment is not None:
        module_path = _resolve_declared_file(
            package_root, manifest.experiment["module"], "experiment", issues
        )
        if module_path is not None:
            if module_path.suffix.lower() != ".py":
                issues.error("invalid_experiment", "experiment module must be a Python file")
            else:
                _check_experiment_source(module_path, issues)


def _check_experiment_source(module_path: Path, issues: _IssueCollector) -> None:
    try:
        source = module_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(module_path))
    except (OSError, UnicodeError, SyntaxError) as exc:
        issues.error("invalid_experiment", f"experiment syntax validation failed: {exc}")
        return

    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    if "get_spec" not in functions:
        issues.error("invalid_experiment", "experiment must define get_spec()")
    elif isinstance(functions["get_spec"], ast.AsyncFunctionDef):
        issues.error("invalid_experiment", "get_spec() must be synchronous")
    elif functions["get_spec"].args.args:
        issues.error("invalid_experiment", "get_spec() must not accept positional arguments")

    if "run" not in functions:
        issues.error("invalid_experiment", "experiment must define run(parameters, reporter)")
    elif isinstance(functions["run"], ast.AsyncFunctionDef):
        issues.error("invalid_experiment", "run() must be synchronous")
    elif len(functions["run"].args.args) < 2:
        issues.error(
            "invalid_experiment", "run() must accept parameters and reporter arguments"
        )


def _check_dependencies(
    manifest: AlgorithmManifest, issues: _IssueCollector
) -> tuple[DependencyStatus, ...]:
    if manifest.experiment is None:
        return ()

    statuses: list[DependencyStatus] = []
    for item in manifest.experiment.get("requirements", []):
        if not isinstance(item, Mapping):
            continue
        requirement_text = item.get("package")
        import_name = item.get("import")
        if not isinstance(requirement_text, str) or not isinstance(import_name, str):
            continue
        try:
            requirement = Requirement(requirement_text)
        except InvalidRequirement:
            continue

        installed_version = None
        reason = None
        try:
            installed_version = importlib.metadata.version(requirement.name)
        except importlib.metadata.PackageNotFoundError:
            reason = f"distribution '{requirement.name}' is not installed"

        try:
            import_available = importlib.util.find_spec(import_name) is not None
        except (ModuleNotFoundError, ValueError):
            import_available = False
        if reason is None and not import_available:
            reason = f"module '{import_name}' cannot be imported"
        if (
            reason is None
            and installed_version is not None
            and requirement.specifier
            and installed_version not in requirement.specifier
        ):
            reason = (
                f"installed {installed_version} does not satisfy "
                f"{requirement.specifier}"
            )

        status = DependencyStatus(
            requirement=requirement_text,
            import_name=import_name,
            available=reason is None,
            installed_version=installed_version,
            reason=reason,
        )
        statuses.append(status)
        if not status.available:
            issues.warning(
                "missing_dependency",
                f"{reason}; install with: {status.install_hint}",
            )
    return tuple(statuses)


def _validate_directory(
    package_root: Path,
    source: Path,
    installed_root: Path | None,
    check_duplicate: bool,
) -> ValidationReport:
    issues = _IssueCollector()
    manifest = _parse_manifest(package_root, issues)
    dependencies: tuple[DependencyStatus, ...] = ()

    if manifest is not None:
        _check_declared_files(package_root, manifest, issues)
        dependencies = _check_dependencies(manifest, issues)
        if check_duplicate and installed_root is not None:
            destination = installed_root / manifest.algorithm_id
            if destination.exists():
                issues.error(
                    "duplicate_algorithm",
                    f"algorithm '{manifest.algorithm_id}' is already installed",
                )

    return ValidationReport(
        source=source,
        manifest=manifest,
        issues=tuple(issues.issues),
        dependencies=dependencies,
    )


def validate_source_directory(source: str | os.PathLike[str]) -> ValidationReport:
    path = Path(source).resolve()
    if not path.is_dir():
        issue = PackageIssue("error", "invalid_source", "package source must be a directory")
        return ValidationReport(path, None, (issue,))
    report = _validate_directory(path, path, None, check_duplicate=False)
    extra_issues = list(report.issues)
    files = list(path.rglob("*"))
    if len(files) > MAX_FILE_COUNT:
        extra_issues.append(
            PackageIssue(
                "error",
                "archive_too_large",
                f"package contains more than {MAX_FILE_COUNT} entries",
            )
        )
    total_size = 0
    for file_path in files:
        if file_path.is_symlink():
            extra_issues.append(
                PackageIssue(
                    "error",
                    "archive_symlink",
                    f"symbolic links are not allowed: {file_path.relative_to(path)}",
                )
            )
        elif file_path.is_file():
            total_size += file_path.stat().st_size
    if total_size > MAX_UNCOMPRESSED_SIZE:
        extra_issues.append(
            PackageIssue(
                "error",
                "archive_too_large",
                "package extracted size exceeds the limit",
            )
        )
    return ValidationReport(
        source=report.source,
        manifest=report.manifest,
        issues=tuple(extra_issues),
        dependencies=report.dependencies,
    )


def _inspect_zip(archive: zipfile.ZipFile, issues: _IssueCollector) -> None:
    infos = archive.infolist()
    if len(infos) > MAX_FILE_COUNT:
        issues.error("archive_too_large", f"archive contains more than {MAX_FILE_COUNT} files")

    total_compressed = sum(info.compress_size for info in infos)
    total_uncompressed = sum(info.file_size for info in infos)
    if total_compressed > MAX_COMPRESSED_SIZE:
        issues.error("archive_too_large", "archive compressed size exceeds the limit")
    if total_uncompressed > MAX_UNCOMPRESSED_SIZE:
        issues.error("archive_too_large", "archive extracted size exceeds the limit")

    for info in infos:
        normalized = info.filename.replace("\\", "/")
        path = PurePosixPath(normalized)
        if (
            path.is_absolute()
            or ".." in path.parts
            or normalized.startswith("/")
            or not normalized
        ):
            issues.error("unsafe_archive_path", f"unsafe archive path: {info.filename!r}")

        mode = info.external_attr >> 16
        if stat.S_ISLNK(mode):
            issues.error("archive_symlink", f"symbolic links are not allowed: {info.filename}")

        if (
            info.compress_size > 0
            and info.file_size / info.compress_size > MAX_COMPRESSION_RATIO
        ):
            issues.error(
                "suspicious_compression",
                f"file has a suspicious compression ratio: {info.filename}",
            )


def _find_package_root(extracted_root: Path) -> Path | None:
    if (extracted_root / "manifest.json").is_file():
        return extracted_root
    children = [path for path in extracted_root.iterdir() if path.name != "__MACOSX"]
    directories = [path for path in children if path.is_dir()]
    files = [path for path in children if path.is_file()]
    if not files and len(directories) == 1 and (directories[0] / "manifest.json").is_file():
        return directories[0]
    return None


def _extract_and_validate(
    archive_path: Path,
    temporary_root: Path,
    installed_root: Path | None,
    check_duplicate: bool,
) -> tuple[ValidationReport, Path | None]:
    issues = _IssueCollector()
    try:
        with zipfile.ZipFile(archive_path, "r") as archive:
            _inspect_zip(archive, issues)
            if any(issue.level == "error" for issue in issues.issues):
                return ValidationReport(archive_path, None, tuple(issues.issues)), None
            archive.extractall(temporary_root)
    except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
        issues.error("invalid_archive", f"ZIP archive cannot be read: {exc}")
        return ValidationReport(archive_path, None, tuple(issues.issues)), None

    package_root = _find_package_root(temporary_root)
    if package_root is None:
        issues.error(
            "invalid_archive_layout",
            "ZIP must contain manifest.json at its root or in one top-level directory",
        )
        return ValidationReport(archive_path, None, tuple(issues.issues)), None

    report = _validate_directory(
        package_root, archive_path, installed_root, check_duplicate
    )
    combined = ValidationReport(
        source=archive_path,
        manifest=report.manifest,
        issues=tuple(issues.issues) + report.issues,
        dependencies=report.dependencies,
    )
    return combined, package_root


def validate_package(
    zip_path: str | os.PathLike[str],
    installed_root: str | os.PathLike[str] | None = None,
    *,
    check_duplicate: bool = True,
) -> ValidationReport:
    archive_path = Path(zip_path).resolve()
    if not archive_path.is_file():
        issue = PackageIssue("error", "invalid_source", "ZIP package does not exist")
        return ValidationReport(archive_path, None, (issue,))
    if archive_path.suffix.lower() != ".zip":
        issue = PackageIssue("error", "invalid_source", "algorithm package must be a ZIP file")
        return ValidationReport(archive_path, None, (issue,))

    destination_root = (
        Path(installed_root).resolve()
        if installed_root is not None
        else default_installed_root()
    )
    with tempfile.TemporaryDirectory(prefix="algorithm-package-validate-") as temp_dir:
        report, _ = _extract_and_validate(
            archive_path,
            Path(temp_dir),
            destination_root,
            check_duplicate,
        )
        return report


def install_package(
    zip_path: str | os.PathLike[str],
    installed_root: str | os.PathLike[str] | None = None,
) -> InstalledAlgorithm:
    archive_path = Path(zip_path).resolve()
    destination_root = (
        Path(installed_root).resolve()
        if installed_root is not None
        else default_installed_root()
    )
    destination_root.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix=".algorithm-package-install-", dir=destination_root
    ) as temp_dir:
        report, package_root = _extract_and_validate(
            archive_path,
            Path(temp_dir),
            destination_root,
            check_duplicate=True,
        )
        if not report.valid or report.manifest is None or package_root is None:
            if any(issue.code == "duplicate_algorithm" for issue in report.errors):
                raise DuplicateAlgorithmError(
                    f"algorithm '{report.manifest.algorithm_id if report.manifest else '?'}' "
                    "is already installed"
                )
            raise PackageValidationError(report)

        if report.manifest.schema_version == 2:
            gate_issues: list[PackageIssue] = []
            blocking_flags = report.manifest.generation.get("blocking_flags", [])
            if blocking_flags:
                gate_issues.append(
                    PackageIssue(
                        "error",
                        "publication_blocked",
                        "schema v2 package has publication blockers: "
                        + ", ".join(blocking_flags),
                    )
                )
            review_modules = report.manifest.review.get("modules", {})
            review_module_names = list(CORE_MODULES)
            if report.manifest.animation is not None:
                review_module_names.append("animation")
            for module_name in review_module_names:
                state = review_modules.get(module_name, {})
                if state.get("status") not in {"approved", "installed"}:
                    gate_issues.append(
                        PackageIssue(
                            "error",
                            "module_not_approved",
                            f"schema v2 module '{module_name}' is not approved",
                        )
                    )
            if gate_issues:
                raise PackageValidationError(
                    ValidationReport(
                        source=report.source,
                        manifest=report.manifest,
                        issues=report.issues + tuple(gate_issues),
                        dependencies=report.dependencies,
                    )
                )

        destination = destination_root / report.manifest.algorithm_id
        if destination.exists():
            raise DuplicateAlgorithmError(
                f"algorithm '{report.manifest.algorithm_id}' is already installed"
            )

        staging = destination_root / f".{report.manifest.algorithm_id}.staging"
        if staging.exists():
            shutil.rmtree(staging)
        shutil.copytree(package_root, staging)
        os.replace(staging, destination)

    return InstalledAlgorithm(
        manifest=report.manifest,
        path=destination,
        dependencies=report.dependencies,
    )


def uninstall_package(
    algorithm_id: str,
    installed_root: str | os.PathLike[str] | None = None,
) -> Path:
    """Move an installed package to a local trash directory.

    The operation is intentionally recoverable. The returned path identifies
    the archived package directory.
    """
    if not ALGORITHM_ID_PATTERN.fullmatch(algorithm_id):
        raise AlgorithmNotFoundError(f"invalid algorithm id: {algorithm_id!r}")

    destination_root = (
        Path(installed_root).resolve()
        if installed_root is not None
        else default_installed_root()
    )
    installed_path = (destination_root / algorithm_id).resolve()
    try:
        installed_path.relative_to(destination_root)
    except ValueError as exc:
        raise AlgorithmNotFoundError("algorithm path escapes the installed root") from exc
    if not installed_path.is_dir():
        raise AlgorithmNotFoundError(f"algorithm '{algorithm_id}' is not installed")

    trash_root = destination_root.parent / ".trash"
    trash_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    trash_path = trash_root / f"{algorithm_id}-{timestamp}"
    os.replace(installed_path, trash_path)
    module_prefix = f"_algorithm_package_{algorithm_id.replace('-', '_')}_"
    for module_name in tuple(sys.modules):
        if module_name.startswith(module_prefix):
            sys.modules.pop(module_name, None)
    return trash_path


def validate_installed_directory(path: Path) -> ValidationReport:
    return _validate_directory(path, path, None, check_duplicate=False)


def iter_package_files(source: Path) -> Iterable[Path]:
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(source)
        if any(part in {"__pycache__", ".DS_Store"} for part in relative.parts):
            continue
        if path.suffix in {".pyc", ".pyo"}:
            continue
        yield path
