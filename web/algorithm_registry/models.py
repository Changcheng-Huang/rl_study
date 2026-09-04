from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class PackageIssue:
    level: str
    code: str
    message: str


@dataclass(frozen=True)
class DependencyStatus:
    requirement: str
    import_name: str
    available: bool
    installed_version: str | None = None
    reason: str | None = None

    @property
    def install_hint(self) -> str:
        return (
            "UV_CACHE_DIR=/private/tmp/animations-uv-cache "
            f'uv add "{self.requirement}"'
        )


@dataclass(frozen=True)
class AlgorithmManifest:
    schema_version: int
    algorithm_id: str
    name: str
    version: str
    summary: str
    category: str
    theory_file: str
    animation: Mapping[str, Any] | None = None
    notebook: Mapping[str, Any] | None = None
    experiment: Mapping[str, Any] | None = None
    sources: tuple[Mapping[str, Any], ...] = ()
    algorithm: Mapping[str, Any] = field(default_factory=dict)
    modules: Mapping[str, Any] = field(default_factory=dict)
    generation: Mapping[str, Any] = field(default_factory=dict)
    review: Mapping[str, Any] = field(default_factory=dict)
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False)


@dataclass(frozen=True)
class ValidationReport:
    source: Path
    manifest: AlgorithmManifest | None
    issues: tuple[PackageIssue, ...]
    dependencies: tuple[DependencyStatus, ...] = ()

    @property
    def valid(self) -> bool:
        return not any(issue.level == "error" for issue in self.issues)

    @property
    def errors(self) -> tuple[PackageIssue, ...]:
        return tuple(issue for issue in self.issues if issue.level == "error")

    @property
    def warnings(self) -> tuple[PackageIssue, ...]:
        return tuple(issue for issue in self.issues if issue.level == "warning")


@dataclass(frozen=True)
class InstalledAlgorithm:
    manifest: AlgorithmManifest
    path: Path
    dependencies: tuple[DependencyStatus, ...] = ()

    @property
    def experiment_available(self) -> bool:
        return self.manifest.experiment is not None and all(
            dependency.available for dependency in self.dependencies
        )


@dataclass(frozen=True)
class ProgressEvent:
    current: int
    total: int
    message: str | None = None


@dataclass(frozen=True)
class MetricEvent:
    name: str
    value: float
    step: int | None = None


class ExperimentReporter:
    """Collect progress events and optionally forward them to a UI callback."""

    def __init__(self, on_progress=None, on_metric=None):
        self.on_progress = on_progress
        self.on_metric = on_metric
        self.progress_events: list[ProgressEvent] = []
        self.metric_events: list[MetricEvent] = []

    def progress(self, current: int, total: int, message: str | None = None) -> None:
        if isinstance(current, bool) or not isinstance(current, int):
            raise ValueError("progress current must be an integer")
        if isinstance(total, bool) or not isinstance(total, int) or total <= 0:
            raise ValueError("progress total must be a positive integer")
        if current < 0 or current > total:
            raise ValueError("progress current must be between 0 and total")
        if message is not None and not isinstance(message, str):
            raise ValueError("progress message must be a string")

        event = ProgressEvent(current=current, total=total, message=message)
        self.progress_events.append(event)
        if self.on_progress is not None:
            self.on_progress(event)

    def metric(self, name: str, value: float, step: int | None = None) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("metric name must be a non-empty string")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("metric value must be numeric")
        if step is not None and (isinstance(step, bool) or not isinstance(step, int)):
            raise ValueError("metric step must be an integer")

        event = MetricEvent(name=name, value=float(value), step=step)
        self.metric_events.append(event)
        if self.on_metric is not None:
            self.on_metric(event)


@dataclass(frozen=True)
class LoadedExperiment:
    algorithm: InstalledAlgorithm
    spec: Mapping[str, Any]
    timeout_seconds: float = 120.0

    def run(
        self,
        parameters: Mapping[str, Any],
        reporter: ExperimentReporter | None = None,
    ) -> Mapping[str, Any]:
        from .contracts import normalize_parameters
        from .runtime import run_isolated_experiment

        normalized = normalize_parameters(self.spec, parameters)
        active_reporter = reporter or ExperimentReporter()
        return run_isolated_experiment(
            self.algorithm,
            normalized,
            active_reporter,
            spec=self.spec,
            timeout_seconds=self.timeout_seconds,
        )


class AlgorithmPackageError(Exception):
    """Base error for algorithm package operations."""


class PackageValidationError(AlgorithmPackageError):
    def __init__(self, report: ValidationReport):
        self.report = report
        details = "; ".join(issue.message for issue in report.errors)
        super().__init__(details or "algorithm package validation failed")


class DuplicateAlgorithmError(AlgorithmPackageError):
    pass


class AlgorithmNotFoundError(AlgorithmPackageError):
    pass


class ExperimentUnavailableError(AlgorithmPackageError):
    pass


class ExperimentTimeoutError(ExperimentUnavailableError):
    pass


class DraftError(AlgorithmPackageError):
    pass


class DraftNotFoundError(DraftError):
    pass


class DraftValidationError(DraftError):
    pass


class DraftStateError(DraftError):
    pass
