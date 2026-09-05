from __future__ import annotations

from pathlib import Path
import json
from functools import lru_cache
from typing import Mapping

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


@lru_cache(maxsize=16)
def _list_installed_cached(root_text: str, fingerprint: tuple[tuple[str, int], ...]) -> tuple[InstalledAlgorithm, ...]:
    root = Path(root_text)
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
    fingerprint = tuple(
        (path.name, (path / "manifest.json").stat().st_mtime_ns)
        for path in sorted(root.iterdir())
        if path.is_dir() and not path.name.startswith(".") and (path / "manifest.json").is_file()
    )
    return _list_installed_cached(str(root), fingerprint)


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
    spec_file = algorithm.manifest.experiment.get("spec_file")
    if isinstance(spec_file, str):
        try:
            spec = json.loads((algorithm.path / spec_file).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ExperimentUnavailableError(
                f"experiment specification snapshot cannot be read: {exc}"
            ) from exc
    else:
        module_path = algorithm.path / algorithm.manifest.experiment["module"]
        cache_key = (
            str(algorithm.path),
            algorithm.manifest.algorithm_id,
            module_path.stat().st_mtime_ns,
            timeout_seconds,
        )
        cached = _LEGACY_SPEC_CACHE.get(cache_key)
        if cached is None:
            cached = load_isolated_spec(algorithm, timeout_seconds=timeout_seconds)
            _LEGACY_SPEC_CACHE.clear()
            _LEGACY_SPEC_CACHE[cache_key] = cached
        spec = cached
    return LoadedExperiment(
        algorithm=algorithm,
        spec=spec,
        timeout_seconds=timeout_seconds,
    )


_LEGACY_SPEC_CACHE: dict[tuple[object, ...], Mapping[str, object]] = {}
