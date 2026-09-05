from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import nbformat

from web.algorithm_registry.latex import (
    double_q_learning_core_latex,
    normalize_latex,
    validate_latex,
)
from web.algorithm_registry.notebook_links import (
    builtin_colab_url,
    manual_notebook_relative_path,
    manual_publication_for,
)
from web.algorithm_registry.notebook_tools import normalize_and_validate_notebook
from web.algorithm_registry.registry import load_experiment
from web.algorithm_registry.theory_content import (
    presentation_from_markdown,
    validate_theory_presentation,
)


class LatexTests(unittest.TestCase):
    def test_legacy_formula_is_normalized_for_streamlit_latex(self) -> None:
        value = "$a^{*} = argmax_{a} Q_A(s',a); target_A = r + gamma * Q_B(s',a^*)$"
        normalized = validate_latex(value)
        self.assertNotIn("$", normalized)
        self.assertIn(r"\begin{aligned}", normalized)
        self.assertIn(r"\arg\max", normalized)
        self.assertIn(r"\gamma \cdot Q_B", normalized)
        self.assertIn("a^*", normalized)

    def test_double_q_formula_contains_symmetric_terminal_updates(self) -> None:
        value = double_q_learning_core_latex()
        self.assertIn(r"Q_A(s,a) &\leftarrow", value)
        self.assertIn(r"Q_B(s,a) &\leftarrow", value)
        self.assertEqual(value.count(r"\text{terminal}"), 2)
        self.assertEqual(validate_latex(value), value)


class TheoryPresentationTests(unittest.TestCase):
    def test_legacy_markdown_keeps_all_sections_and_preserves_checkpoint(self) -> None:
        markdown = """# Example

Intro.

## Intuition
First concept.

## Assumptions
Second concept.

## Mathematical update
Formula details.

## Pseudocode
Loop details.
"""
        value = presentation_from_markdown(
            markdown,
            "Example",
            preserve={
                "key_ideas": ["Keep me"],
                "checkpoint": [
                    {
                        "question": "Question?",
                        "options": ["A", "B"],
                        "answer": 0,
                        "explanation": "Because.",
                    }
                ],
            },
        )
        self.assertIn("First concept", value["concept_markdown"])
        self.assertIn("Second concept", value["concept_markdown"])
        self.assertIn("Formula details", value["math_markdown"])
        self.assertIn("Loop details", value["pseudocode_markdown"])
        self.assertEqual(value["key_ideas"], ["Keep me"])
        self.assertEqual(len(value["checkpoint"]), 1)

    def test_complete_checkpoint_is_validated(self) -> None:
        value = validate_theory_presentation(
            {
                "title": "Double Q-Learning",
                "concept_markdown": "Concept",
                "math_markdown": "Math",
                "pseudocode_markdown": "Pseudo",
                "key_ideas": ["Separate selection from evaluation"],
                "when_to_use": ["Noisy value estimates"],
                "checkpoint": [
                    {
                        "question": "Which table evaluates an action selected by Q_A?",
                        "options": ["Q_A", "Q_B"],
                        "answer": 1,
                        "explanation": "The second estimator performs cross-evaluation.",
                    }
                ],
            },
            algorithm_name="Double Q-Learning",
        )
        self.assertEqual(value["checkpoint"][0]["answer"], 1)

    def test_wrong_title_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "title must match"):
            validate_theory_presentation(
                {
                    "title": "Wrong",
                    "concept_markdown": "Concept",
                    "math_markdown": "Math",
                    "pseudocode_markdown": "Pseudo",
                },
                algorithm_name="Expected",
            )


class NotebookCompatibilityTests(unittest.TestCase):
    def _payload(self, source: str) -> bytes:
        notebook = nbformat.v4.new_notebook(
            cells=[
                nbformat.v4.new_markdown_cell("# Overview"),
                nbformat.v4.new_code_cell(source),
            ]
        )
        buffer = io.StringIO()
        nbformat.write(notebook, buffer)
        return buffer.getvalue().encode("utf-8")

    def test_metadata_and_controlled_install_cell_are_added(self) -> None:
        payload, requirements = normalize_and_validate_notebook(
            self._payload("import numpy as np\nprint(np.zeros(2))"),
            "double-q-learning",
            "1.0.1",
        )
        notebook = nbformat.read(io.StringIO(payload.decode()), as_version=4)
        self.assertEqual(notebook.metadata.kernelspec.name, "python3")
        self.assertEqual(notebook.metadata.rlae_validation, "static-only-not-executed")
        self.assertIn("%pip install -q numpy>=2.2", notebook.cells[0].source)
        self.assertEqual(requirements[0]["import"], "numpy")

    def test_network_import_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "network"):
            normalize_and_validate_notebook(
                self._payload("import requests"), "example", "1.0.0"
            )


class NotebookLinkTests(unittest.TestCase):
    def _algorithm(self, root: Path, content: bytes = b"notebook"):
        installed = root / "installed"
        installed.mkdir()
        (installed / "notebook.ipynb").write_bytes(content)
        manifest = SimpleNamespace(
            algorithm_id="double-q-learning",
            version="1.0.1",
            notebook={"file": "notebook.ipynb"},
        )
        return SimpleNamespace(manifest=manifest, path=installed)

    def test_builtin_colab_url_uses_current_repository_defaults(self) -> None:
        self.assertEqual(
            builtin_colab_url("q_learning.ipynb"),
            "https://colab.research.google.com/github/Changcheng-Huang/"
            "rl_study/blob/main/notebook/q_learning.ipynb",
        )

    def test_manual_copy_must_exist_and_match_installed_notebook(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            algorithm = self._algorithm(root)
            missing = manual_publication_for(algorithm, repository_root=root)
            self.assertEqual(missing.status, "missing")
            self.assertIsNone(missing.colab_url)

            destination = root / manual_notebook_relative_path(
                "double-q-learning", "1.0.1"
            )
            destination.parent.mkdir(parents=True)
            destination.write_bytes(b"different")
            stale = manual_publication_for(algorithm, repository_root=root)
            self.assertEqual(stale.status, "stale")
            self.assertIsNone(stale.colab_url)

            destination.write_bytes(b"notebook")
            ready = manual_publication_for(algorithm, repository_root=root)
            self.assertEqual(ready.status, "ready")
            self.assertIn(
                "notebook/imported/double-q-learning-1.0.1.ipynb",
                ready.colab_url,
            )


class StaticExperimentSpecTests(unittest.TestCase):
    def test_static_spec_avoids_worker_start_on_page_load(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = {"parameters": {}}
            (root / "experiment_spec.json").write_text(json.dumps(spec))
            manifest = SimpleNamespace(
                algorithm_id="example",
                experiment={"module": "experiment.py", "spec_file": "experiment_spec.json"},
            )
            algorithm = SimpleNamespace(manifest=manifest, path=root, dependencies=())
            with (
                mock.patch("web.algorithm_registry.registry._find_installed", return_value=algorithm),
                mock.patch("web.algorithm_registry.registry._ensure_experiment_available"),
                mock.patch("web.algorithm_registry.registry.load_isolated_spec") as worker,
            ):
                loaded = load_experiment("example")
            self.assertEqual(loaded.spec, spec)
            worker.assert_not_called()


if __name__ == "__main__":
    unittest.main()
