from __future__ import annotations

import json
import os
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import nbformat

from tools.build_algorithm_package import build_archive
from web.algorithm_registry import (
    ContractValidationError,
    DuplicateAlgorithmError,
    ExperimentReporter,
    ExperimentUnavailableError,
    install_package,
    list_installed,
    load_experiment,
    normalize_parameters,
    uninstall_package,
    validate_experiment_result,
    validate_experiment_spec,
    validate_package,
    validate_source_directory,
)


def manifest(
    algorithm_id: str = "example-algorithm",
    *,
    include_animation: bool = False,
    include_notebook: bool = False,
    include_experiment: bool = False,
    requirements: list[dict] | None = None,
) -> dict:
    data = {
        "schema_version": 1,
        "id": algorithm_id,
        "name": "Example Algorithm",
        "version": "1.0.0",
        "summary": "An algorithm package used by automated tests.",
        "category": "value-based",
        "theory": {"file": "theory.md"},
    }
    if include_animation:
        data["animation"] = {
            "file": "animation.mp4",
            "formula": "V(s)=0",
            "highlights": ["A deterministic test"],
        }
    if include_notebook:
        data["notebook"] = {"file": "notebook.ipynb"}
    if include_experiment:
        data["experiment"] = {
            "module": "experiment.py",
            "requirements": requirements or [],
        }
    return data


def write_source(
    root: Path,
    data: dict,
    *,
    experiment_source: str | None = None,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(
        json.dumps(data, indent=2), encoding="utf-8"
    )
    (root / "theory.md").write_text("# Example\n\nTheory content.", encoding="utf-8")

    if "animation" in data:
        (root / data["animation"]["file"]).write_bytes(b"test mp4 payload")
    if "notebook" in data:
        notebook = nbformat.v4.new_notebook()
        notebook.cells.append(nbformat.v4.new_markdown_cell("# Example notebook"))
        nbformat.write(notebook, root / data["notebook"]["file"])
    if "experiment" in data:
        source = experiment_source or """
from .helper import OFFSET

def get_spec():
    return {
        "parameters": {
            "episodes": {
                "type": "int",
                "default": 3,
                "min": 1,
                "max": 10,
                "step": 1
            }
        }
    }

def run(parameters, reporter):
    values = []
    for episode in range(parameters["episodes"]):
        value = float(episode + OFFSET)
        values.append(value)
        reporter.progress(episode + 1, parameters["episodes"])
        reporter.metric("reward", value, step=episode)
    return {
        "metrics": {"reward": values},
        "summary": {"final_reward": values[-1]},
        "artifacts": []
    }
"""
        (root / data["experiment"]["module"]).write_text(source, encoding="utf-8")
        (root / "helper.py").write_text("OFFSET = 1\n", encoding="utf-8")


def zip_source(source: Path, destination: Path, *, wrapped: bool = False) -> None:
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(source.rglob("*")):
            if path.is_file():
                relative = path.relative_to(source)
                archive_name = Path(source.name, relative) if wrapped else relative
                archive.write(path, archive_name.as_posix())


class PackageValidationTests(unittest.TestCase):
    def test_minimal_theory_package_is_valid(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source"
            write_source(source, manifest())
            archive = root / "package.zip"
            zip_source(source, archive)

            report = validate_package(archive, root / "installed")

            self.assertTrue(report.valid, report.errors)
            self.assertEqual(report.manifest.algorithm_id, "example-algorithm")
            self.assertEqual(report.dependencies, ())

    def test_full_wrapped_package_is_valid(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "full-package"
            data = manifest(
                include_animation=True,
                include_notebook=True,
                include_experiment=True,
                requirements=[{"package": "packaging>=24", "import": "packaging"}],
            )
            write_source(source, data)
            archive = root / "package.zip"
            zip_source(source, archive, wrapped=True)

            report = validate_package(archive, root / "installed")

            self.assertTrue(report.valid, report.errors)
            self.assertTrue(report.dependencies[0].available)

    def test_missing_dependency_is_a_warning(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source"
            data = manifest(
                include_experiment=True,
                requirements=[
                    {
                        "package": "definitely-missing-algorithm-package>=99",
                        "import": "definitely_missing_algorithm_package",
                    }
                ],
            )
            write_source(source, data)
            archive = root / "package.zip"
            zip_source(source, archive)

            report = validate_package(archive, root / "installed")

            self.assertTrue(report.valid)
            self.assertFalse(report.dependencies[0].available)
            self.assertEqual(report.warnings[0].code, "missing_dependency")

    def test_invalid_manifest_and_experiment_syntax_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source"
            data = manifest(algorithm_id="Invalid ID", include_experiment=True)
            write_source(source, data, experiment_source="def get_spec(:\n")
            archive = root / "package.zip"
            zip_source(source, archive)

            report = validate_package(archive, root / "installed")

            self.assertFalse(report.valid)
            codes = {issue.code for issue in report.errors}
            self.assertIn("invalid_algorithm_id", codes)
            self.assertIn("invalid_experiment", codes)

    def test_damaged_notebook_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source"
            data = manifest(include_notebook=True)
            write_source(source, data)
            (source / "notebook.ipynb").write_text("{not-json", encoding="utf-8")
            archive = root / "package.zip"
            zip_source(source, archive)

            report = validate_package(archive, root / "installed")

            self.assertFalse(report.valid)
            self.assertIn("invalid_notebook", {issue.code for issue in report.errors})

    def test_path_traversal_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "malicious.zip"
            with zipfile.ZipFile(archive, "w") as output:
                output.writestr("../outside.txt", "bad")

            report = validate_package(archive, root / "installed")

            self.assertFalse(report.valid)
            self.assertIn("unsafe_archive_path", {issue.code for issue in report.errors})
            self.assertFalse((root / "outside.txt").exists())

    def test_archive_symlink_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "symlink.zip"
            info = zipfile.ZipInfo("linked")
            info.create_system = 3
            info.external_attr = (0o120777 << 16)
            with zipfile.ZipFile(archive, "w") as output:
                output.writestr(info, "target")

            report = validate_package(archive, root / "installed")

            self.assertFalse(report.valid)
            self.assertIn("archive_symlink", {issue.code for issue in report.errors})

    def test_archive_size_limit_is_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source"
            write_source(source, manifest())
            archive = root / "package.zip"
            zip_source(source, archive)

            with mock.patch(
                "web.algorithm_registry.package.MAX_UNCOMPRESSED_SIZE", 10
            ):
                report = validate_package(archive, root / "installed")

            self.assertFalse(report.valid)
            self.assertIn("archive_too_large", {issue.code for issue in report.errors})

    def test_source_symlink_is_rejected(self) -> None:
        if not hasattr(os, "symlink"):
            self.skipTest("symbolic links are not supported")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source"
            write_source(source, manifest())
            os.symlink(source / "theory.md", source / "linked-theory.md")

            report = validate_source_directory(source)

            self.assertFalse(report.valid)
            self.assertIn("archive_symlink", {issue.code for issue in report.errors})


class InstallationAndRuntimeTests(unittest.TestCase):
    def test_install_list_load_and_run(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source"
            write_source(source, manifest(include_experiment=True))
            archive = root / "package.zip"
            zip_source(source, archive)
            installed_root = root / "installed"

            installed = install_package(archive, installed_root)
            algorithms = list_installed(installed_root)
            experiment = load_experiment("example-algorithm", installed_root)
            reporter = ExperimentReporter()
            result = experiment.run({"episodes": 2}, reporter)

            self.assertEqual(installed.manifest.algorithm_id, "example-algorithm")
            self.assertEqual(len(algorithms), 1)
            self.assertTrue(algorithms[0].experiment_available)
            self.assertEqual(result["metrics"]["reward"], [1.0, 2.0])
            self.assertEqual(len(reporter.progress_events), 2)
            self.assertEqual(len(reporter.metric_events), 2)

    def test_duplicate_install_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source"
            write_source(source, manifest())
            archive = root / "package.zip"
            zip_source(source, archive)
            installed_root = root / "installed"

            install_package(archive, installed_root)
            with self.assertRaises(DuplicateAlgorithmError):
                install_package(archive, installed_root)

    def test_missing_dependency_disables_only_experiment(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source"
            data = manifest(
                include_experiment=True,
                requirements=[
                    {
                        "package": "definitely-missing-algorithm-package>=99",
                        "import": "definitely_missing_algorithm_package",
                    }
                ],
            )
            write_source(source, data)
            archive = root / "package.zip"
            zip_source(source, archive)
            installed_root = root / "installed"

            installed = install_package(archive, installed_root)

            self.assertTrue(installed.path.joinpath("theory.md").is_file())
            self.assertFalse(installed.experiment_available)
            with self.assertRaises(ExperimentUnavailableError):
                load_experiment("example-algorithm", installed_root)

    def test_experiment_load_failure_is_reported_without_breaking_registry(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source"
            broken_source = """
def get_spec():
    raise RuntimeError("broken spec")

def run(parameters, reporter):
    return {"metrics": {}, "summary": {}, "artifacts": []}
"""
            write_source(
                source,
                manifest(include_experiment=True),
                experiment_source=broken_source,
            )
            archive = root / "package.zip"
            zip_source(source, archive)
            installed_root = root / "installed"
            install_package(archive, installed_root)

            with self.assertRaises(ExperimentUnavailableError):
                load_experiment("example-algorithm", installed_root)
            self.assertEqual(len(list_installed(installed_root)), 1)

    def test_build_tool_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source"
            output = root / "dist"
            write_source(source, manifest())

            archive = build_archive(source, output)
            report = validate_package(archive, root / "installed")

            self.assertEqual(archive.name, "example-algorithm-1.0.0.zip")
            self.assertTrue(report.valid, report.errors)

    def test_uninstall_moves_package_to_recoverable_trash(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source"
            write_source(source, manifest())
            archive = root / "package.zip"
            zip_source(source, archive)
            installed_root = root / "installed"
            install_package(archive, installed_root)

            trash_path = uninstall_package("example-algorithm", installed_root)

            self.assertFalse((installed_root / "example-algorithm").exists())
            self.assertTrue((trash_path / "manifest.json").is_file())
            self.assertEqual(list_installed(installed_root), ())

    def test_repository_example_builds_installs_and_runs(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        source = (
            project_root
            / "algorithm_packages"
            / "examples"
            / "monte_carlo_control"
        )
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = build_archive(source, root / "dist")
            installed_root = root / "installed"
            installed = install_package(archive, installed_root)
            experiment = load_experiment("monte-carlo-control", installed_root)

            result = experiment.run(
                {
                    "episodes": 100,
                    "gamma": 0.99,
                    "epsilon": 0.2,
                    "slippery": False,
                    "seed": 7,
                }
            )

            self.assertEqual(installed.manifest.name, "Monte Carlo Control")
            self.assertIsNotNone(installed.manifest.notebook)
            self.assertEqual(len(result["metrics"]["episode_reward"]), 100)
            self.assertEqual(result["summary"]["episodes"], 100)


class ExperimentContractTests(unittest.TestCase):
    def test_spec_defaults_and_result(self) -> None:
        spec = {
            "parameters": {
                "episodes": {
                    "type": "int",
                    "default": 5,
                    "min": 1,
                    "max": 10,
                },
                "slippery": {"type": "bool", "default": True},
            }
        }
        validate_experiment_spec(spec)

        normalized = normalize_parameters(spec, {"episodes": 3})
        self.assertEqual(normalized, {"episodes": 3, "slippery": True})

        result = {
            "metrics": {"reward": [0.0, 1.0]},
            "summary": {"success": True},
            "artifacts": [{"type": "image", "path": "assets/policy.png"}],
        }
        validate_experiment_result(result)

    def test_grid_presentation_and_policy_view_contract(self) -> None:
        spec = {
            "parameters": {},
            "presentation": {
                "task": {
                    "mission": "Reach the goal.",
                    "dynamics": ["Move one cell per step."],
                    "rewards": ["Goal gives +1."],
                },
                "environment_map": {
                    "kind": "grid",
                    "layout": ["SF", "HG"],
                    "legend": {
                        "S": {"label": "Start", "role": "start"},
                        "F": {"label": "Safe", "role": "normal"},
                        "H": {
                            "label": "Hole",
                            "role": "hazard",
                            "terminal": True,
                        },
                        "G": {
                            "label": "Goal",
                            "role": "goal",
                            "terminal": True,
                        },
                    },
                    "actions": {
                        "0": {"label": "Up", "arrow": "↑"},
                        "1": {"label": "Right", "arrow": "→"},
                    },
                },
            },
        }
        validate_experiment_spec(spec)
        validate_experiment_result(
            {
                "metrics": {"reward": [0.0, 1.0]},
                "summary": {"success": True},
                "artifacts": [],
                "views": {
                    "policy_grid": {
                        "state_values": [0.1, 0.5, None, 1.0],
                        "best_actions": [1, 1, None, None],
                    }
                },
            },
            spec,
        )

    def test_invalid_grid_and_policy_view_are_rejected(self) -> None:
        invalid_spec = {
            "parameters": {},
            "presentation": {
                "environment_map": {
                    "kind": "grid",
                    "layout": ["SF", "G"],
                    "legend": {},
                    "actions": {"0": {"label": "Up", "arrow": "↑"}},
                }
            },
        }
        with self.assertRaises(ContractValidationError):
            validate_experiment_spec(invalid_spec)

        spec = {
            "parameters": {},
            "presentation": {
                "environment_map": {
                    "kind": "grid",
                    "layout": ["SG"],
                    "legend": {
                        "S": {"label": "Start", "role": "start"},
                        "G": {"label": "Goal", "role": "goal"},
                    },
                    "actions": {"0": {"label": "Right", "arrow": "→"}},
                }
            },
        }
        validate_experiment_spec(spec)
        with self.assertRaises(ContractValidationError):
            validate_experiment_result(
                {
                    "views": {
                        "policy_grid": {
                            "state_values": [0.0, 1.0],
                            "best_actions": [9, None],
                        }
                    }
                },
                spec,
            )

    def test_invalid_spec_result_and_reporter_events(self) -> None:
        with self.assertRaises(ContractValidationError):
            validate_experiment_spec(
                {"parameters": {"rate": {"type": "float", "default": 2.0, "max": 1.0}}}
            )
        with self.assertRaises(ContractValidationError):
            validate_experiment_result({"metrics": {"loss": [float("nan")]}})
        with self.assertRaises(ContractValidationError):
            validate_experiment_result(
                {
                    "artifacts": [
                        {"type": "image", "path": "../outside-package.png"}
                    ]
                }
            )

        reporter = ExperimentReporter()
        with self.assertRaises(ValueError):
            reporter.progress(2, 1)
        with self.assertRaises(ValueError):
            reporter.metric("", 1.0)


if __name__ == "__main__":
    unittest.main()
