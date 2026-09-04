from __future__ import annotations

import hashlib
import importlib.util
import multiprocessing
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping

from .contracts import validate_experiment_result, validate_experiment_spec
from .models import (
    ExperimentReporter,
    ExperimentTimeoutError,
    ExperimentUnavailableError,
    InstalledAlgorithm,
)


DEFAULT_EXPERIMENT_TIMEOUT = 120.0


def _load_module(package_path: str, algorithm_id: str, module_relative: str) -> ModuleType:
    root = Path(package_path)
    module_path = (root / module_relative).resolve()
    try:
        module_path.relative_to(root.resolve())
    except ValueError as exc:
        raise RuntimeError("experiment module escapes the package") from exc
    digest = hashlib.sha256(
        f"{root}:{module_path.stat().st_mtime_ns}:{multiprocessing.current_process().pid}".encode()
    ).hexdigest()[:12]
    package_name = f"_algorithm_package_{algorithm_id.replace('-', '_')}_{digest}"
    module_name = f"{package_name}.experiment"
    package_module = ModuleType(package_name)
    package_module.__path__ = [str(root)]
    package_module.__package__ = package_name
    sys.modules[package_name] = package_module
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to create experiment module loader")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    if not callable(getattr(module, "get_spec", None)):
        raise RuntimeError("experiment get_spec() is not callable")
    if not callable(getattr(module, "run", None)):
        raise RuntimeError("experiment run() is not callable")
    return module


class _WorkerReporter:
    def __init__(self, connection) -> None:
        self.connection = connection

    def progress(self, current: int, total: int, message: str | None = None) -> None:
        self.connection.send(
            {
                "type": "progress",
                "current": current,
                "total": total,
                "message": message,
            }
        )

    def metric(self, name: str, value: float, step: int | None = None) -> None:
        self.connection.send(
            {
                "type": "metric",
                "name": name,
                "value": value,
                "step": step,
            }
        )


def _worker(
    package_path: str,
    algorithm_id: str,
    module_relative: str,
    action: str,
    parameters: Mapping[str, Any] | None,
    connection,
) -> None:
    try:
        module = _load_module(package_path, algorithm_id, module_relative)
        if action == "spec":
            result = module.get_spec()
        elif action == "run":
            result = module.run(parameters or {}, _WorkerReporter(connection))
        else:
            raise RuntimeError(f"unknown worker action: {action}")
        connection.send({"type": "result", "payload": result})
    except BaseException as exc:
        try:
            connection.send(
                {
                    "type": "error",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            )
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        connection.close()


def _execute(
    algorithm: InstalledAlgorithm,
    action: str,
    parameters: Mapping[str, Any] | None,
    reporter: ExperimentReporter | None,
    timeout_seconds: float,
) -> Any:
    experiment = algorithm.manifest.experiment
    if experiment is None:
        raise ExperimentUnavailableError(
            f"algorithm '{algorithm.manifest.algorithm_id}' has no experiment"
        )
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")

    context = multiprocessing.get_context("spawn")
    receive, send = context.Pipe(duplex=False)
    process = context.Process(
        target=_worker,
        args=(
            str(algorithm.path),
            algorithm.manifest.algorithm_id,
            experiment["module"],
            action,
            dict(parameters or {}),
            send,
        ),
        daemon=True,
    )
    process.start()
    send.close()
    deadline = time.monotonic() + timeout_seconds
    result_received = False
    result: Any = None
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ExperimentTimeoutError(
                    f"experiment exceeded the {timeout_seconds:g} second timeout"
                )
            if receive.poll(min(remaining, 0.1)):
                try:
                    message = receive.recv()
                except EOFError:
                    message = None
                if not isinstance(message, Mapping):
                    raise ExperimentUnavailableError(
                        "experiment worker returned an invalid message"
                    )
                message_type = message.get("type")
                if message_type == "progress":
                    if reporter is not None:
                        reporter.progress(
                            message.get("current"),
                            message.get("total"),
                            message.get("message"),
                        )
                elif message_type == "metric":
                    if reporter is not None:
                        reporter.metric(
                            message.get("name"),
                            message.get("value"),
                            message.get("step"),
                        )
                elif message_type == "error":
                    error_type = message.get("error_type", "Error")
                    detail = message.get("message", "")
                    raise ExperimentUnavailableError(
                        f"experiment worker failed ({error_type}): {detail}"
                    )
                elif message_type == "result":
                    result_received = True
                    result = message.get("payload")
                    break
                else:
                    raise ExperimentUnavailableError(
                        "experiment worker returned an unknown message"
                    )
            elif not process.is_alive():
                break

        process.join(timeout=1)
        if not result_received:
            raise ExperimentUnavailableError(
                f"experiment worker exited without a result (exit code {process.exitcode})"
            )
        return result
    finally:
        receive.close()
        if process.is_alive():
            process.terminate()
        process.join(timeout=1)
        if process.is_alive() and hasattr(process, "kill"):
            process.kill()
            process.join(timeout=1)


def load_isolated_spec(
    algorithm: InstalledAlgorithm,
    *,
    timeout_seconds: float = DEFAULT_EXPERIMENT_TIMEOUT,
) -> Mapping[str, Any]:
    spec = _execute(algorithm, "spec", None, None, timeout_seconds)
    try:
        validate_experiment_spec(spec)
    except Exception as exc:
        raise ExperimentUnavailableError(f"experiment spec is invalid: {exc}") from exc
    return spec


def run_isolated_experiment(
    algorithm: InstalledAlgorithm,
    parameters: Mapping[str, Any],
    reporter: ExperimentReporter,
    *,
    spec: Mapping[str, Any] | None = None,
    timeout_seconds: float = DEFAULT_EXPERIMENT_TIMEOUT,
) -> Mapping[str, Any]:
    result = _execute(
        algorithm,
        "run",
        parameters,
        reporter,
        timeout_seconds,
    )
    try:
        validate_experiment_result(result, spec)
    except Exception as exc:
        raise ExperimentUnavailableError(
            f"experiment result is invalid: {exc}"
        ) from exc
    return result
