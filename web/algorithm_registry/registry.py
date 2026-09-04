from __future__ import annotations

from pathlib import Path

from .models import (
    AlgorithmNotFoundError,
    ExperimentUnavailableError,
    InstalledAlgorithm,
    LoadedExperiment,
)
from .package import (
    ALGORITHM_ID_PATTERN,
    default_installed_root,
    validate_installed_directory,
)
from .runtime import DEFAULT_EXPERIMENT_TIMEOUT, load_isolated_spec


def list_installed(
    installed_root: str | Path | None = None,
) -> tuple[InstalledAlgorithm, ...]:
    root = (
        Path(installed_root).resolve()
        if installed_root is not None
        else default_installed_root()
    )
    if not root.is_dir():
        return ()

    algorithms: list[InstalledAlgorithm] = []
    for path in sorted(root.iterdir()):
        if not path.is_dir() or path.name.startswith("."):
            continue
        report = validate_installed_directory(path)
        if report.valid and report.manifest is not None:
            algorithms.append(
                InstalledAlgorithm(
                    manifest=report.manifest,
                    path=path,
                    dependencies=report.dependencies,
                )
            )
    return tuple(algorithms)


def _find_installed(
    algorithm_id: str, installed_root: str | Path | None
) -> InstalledAlgorithm:
    if not ALGORITHM_ID_PATTERN.fullmatch(algorithm_id):
        raise AlgorithmNotFoundError(f"invalid algorithm id: {algorithm_id!r}")
    for algorithm in list_installed(installed_root):
        if algorithm.manifest.algorithm_id == algorithm_id:
            return algorithm
    raise AlgorithmNotFoundError(f"algorithm '{algorithm_id}' is not installed")


def _ensure_experiment_available(algorithm: InstalledAlgorithm) -> None:
    if algorithm.manifest.experiment is None:
        raise ExperimentUnavailableError(
            f"algorithm '{algorithm.manifest.algorithm_id}' has no experiment"
        )
    missing = [
        dependency for dependency in algorithm.dependencies if not dependency.available
    ]
    if missing:
        hints = "; ".join(dependency.install_hint for dependency in missing)
        raise ExperimentUnavailableError(
            f"algorithm experiment has missing dependencies; {hints}"
        )


def load_experiment(
    algorithm_id: str,
    installed_root: str | Path | None = None,
    *,
    timeout_seconds: float = DEFAULT_EXPERIMENT_TIMEOUT,
) -> LoadedExperiment:
    algorithm = _find_installed(algorithm_id, installed_root)
    _ensure_experiment_available(algorithm)
    spec = load_isolated_spec(algorithm, timeout_seconds=timeout_seconds)
    return LoadedExperiment(
        algorithm=algorithm,
        spec=spec,
        timeout_seconds=timeout_seconds,
    )
