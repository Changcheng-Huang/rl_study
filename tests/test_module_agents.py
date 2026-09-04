from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import nbformat

from web.algorithm_registry import (
    GeneratedModule,
    create_draft,
    generate_module_content,
    generate_module_with_agent,
    get_module_agent_configuration,
)
from web.algorithm_registry.module_agent import (
    ExperimentModuleOutput,
    NotebookCellOutput,
    NotebookModuleOutput,
    TheoryModuleOutput,
)

from test_algorithm_workflow_v2 import draft_input


class FakeStructuredModel:
    def __init__(self, response):
        self.response = response
        self.messages = None

    def invoke(self, messages):
        self.messages = messages
        return self.response


class FakeChatModel:
    def __init__(self, response):
        self.response = response
        self.request = None
        self.structured_model = FakeStructuredModel(response)

    def with_structured_output(self, schema, **kwargs):
        self.request = {"schema": schema, **kwargs}
        return self.structured_model


def response(parsed):
    return {
        "raw": SimpleNamespace(
            id="module-run",
            usage_metadata={
                "input_tokens": 100,
                "output_tokens": 50,
                "total_tokens": 150,
            },
        ),
        "parsed": parsed,
        "parsing_error": None,
    }


def algorithm_spec():
    return {
        "id": "q-learning",
        "name": "Q-Learning",
        "version": "1.0.0",
        "category": "value-based",
        "summary": "Learn action values.",
        "algorithm": {
            "objective": "Learn a policy.",
            "assumptions": ["Finite actions"],
            "inputs": ["Transitions"],
            "outputs": ["Q table"],
            "states": ["State"],
            "actions": ["Action"],
            "hyperparameters": {"alpha": {"default": 0.1}},
            "core_equations": ["Q \\leftarrow Q + \\alpha \\delta"],
            "pseudocode": ["Observe", "Update"],
            "supported_environments": ["FrozenLake-v1"],
        },
    }


class ModuleAgentTests(unittest.TestCase):
    def test_role_specific_configuration_overrides_shared_model(self) -> None:
        configuration = get_module_agent_configuration(
            "notebook",
            {
                "ALGORITHM_AGENT_API_KEY": "test-key",
                "ALGORITHM_AGENT_MODEL": "general-model",
                "ALGORITHM_MODULE_AGENT_MODEL": "shared-module-model",
                "NOTEBOOK_AGENT_MODEL": "coder-model",
                "NOTEBOOK_AGENT_ENABLE_THINKING": "false",
            },
        )

        self.assertTrue(configuration.configured)
        self.assertEqual(configuration.model, "coder-model")
        self.assertFalse(configuration.enable_thinking)

    def test_qwen_module_generation_defaults_to_non_thinking(self) -> None:
        configuration = get_module_agent_configuration(
            "theory",
            {
                "ALGORITHM_AGENT_API_KEY": "test-key",
                "ALGORITHM_MODULE_AGENT_MODEL": "qwen3.7-plus",
            },
        )

        self.assertTrue(configuration.configured)
        self.assertFalse(configuration.enable_thinking)

    def test_theory_notebook_and_experiment_outputs_are_materialized(self) -> None:
        theory_model = FakeChatModel(
            response(
                TheoryModuleOutput(
                    markdown="# Q-Learning\n\n" + ("Teaching content. " * 20),
                    warnings=[],
                )
            )
        )
        theory = generate_module_content(
            "theory",
            algorithm_spec(),
            "Trusted source text.",
            chat_model=theory_model,
        )
        self.assertTrue(theory.payload.startswith(b"# Q-Learning"))
        self.assertEqual(theory.metadata["usage"]["total_tokens"], 150)

        notebook_model = FakeChatModel(
            response(
                NotebookModuleOutput(
                    cells=[
                        NotebookCellOutput(
                            cell_type="markdown", source="# Q-Learning"
                        ),
                        NotebookCellOutput(
                            cell_type="code", source="alpha = 0.1\nalpha"
                        ),
                        NotebookCellOutput(
                            cell_type="markdown", source="## Interpretation"
                        ),
                    ],
                    warnings=[],
                )
            )
        )
        notebook = generate_module_content(
            "notebook",
            algorithm_spec(),
            "Trusted source text.",
            chat_model=notebook_model,
        )
        parsed_notebook = nbformat.reads(
            notebook.payload.decode("utf-8"), as_version=4
        )
        self.assertEqual(len(parsed_notebook.cells), 3)
        self.assertEqual(parsed_notebook.metadata["algorithm_id"], "q-learning")

        experiment_model = FakeChatModel(
            response(
                ExperimentModuleOutput(
                    python_source=(
                        "def get_spec():\n"
                        "    return {'parameters': {}}\n\n"
                        "def run(parameters, reporter):\n"
                        "    reporter.progress(1, 1, 'done')\n"
                        "    reporter.metric('score', 1.0, step=1)\n"
                        "    return {'metrics': {'score': [1.0]}, "
                        "'summary': {'done': True}, 'artifacts': []}\n"
                        + ("\n# reviewed implementation" * 8)
                    ),
                    warnings=[],
                )
            )
        )
        experiment = generate_module_content(
            "experiment",
            algorithm_spec(),
            "Trusted source text.",
            review_note="Keep the result deterministic.",
            chat_model=experiment_model,
        )
        self.assertIn(b"def run(parameters, reporter)", experiment.payload)
        self.assertIsNotNone(experiment.metadata["review_note_sha256"])
        experiment_prompt = experiment_model.structured_model.messages[0][1]
        self.assertIn("presentation.environment_map", experiment_prompt)
        self.assertIn("views.policy_grid", experiment_prompt)
        self.assertIn(
            "write all generated human-readable content in English",
            experiment_prompt,
        )

    def test_three_valid_agent_modules_clear_placeholder_and_record_history(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            drafts_root = root / "drafts"
            draft = create_draft(
                draft_input(),
                drafts_root,
                root / "installed",
            )
            notebook = nbformat.v4.new_notebook(
                cells=[
                    nbformat.v4.new_markdown_cell("# Generated"),
                    nbformat.v4.new_code_cell("value = 1\nvalue"),
                ]
            )
            notebook_buffer = io.StringIO()
            nbformat.write(notebook, notebook_buffer)
            payloads = {
                "theory": b"# Generated theory\n\nReviewed Agent content.",
                "notebook": notebook_buffer.getvalue().encode("utf-8"),
                "experiment": (
                    b"def get_spec():\n"
                    b"    return {'parameters': {}}\n\n"
                    b"def run(parameters, reporter):\n"
                    b"    return {'metrics': {}, 'summary': {}, 'artifacts': []}\n"
                ),
            }

            def generated(module, *_args, **_kwargs):
                return GeneratedModule(
                    module=module,
                    payload=payloads[module],
                    metadata={
                        "model": f"{module}-model",
                        "generated_at": "2026-07-31T00:00:00+00:00",
                    },
                )

            with patch(
                "web.algorithm_registry.module_agent.generate_module_content",
                side_effect=generated,
            ):
                for module in ("theory", "notebook", "experiment"):
                    draft = generate_module_with_agent(
                        draft.key,
                        module,
                        "Ada",
                        "Apply the review feedback.",
                        drafts_root,
                    )

            self.assertEqual(
                draft.manifest.generation["blocking_flags"],
                [],
            )
            self.assertEqual(
                set(draft.manifest.generation["module_generations"]),
                {"theory", "notebook", "experiment"},
            )
            actions = [
                event["action"]
                for event in draft.manifest.review["history"]
            ]
            self.assertEqual(actions.count("agent_generated"), 3)
            self.assertIn("placeholder_replaced_by_agents", actions)
            self.assertTrue(
                all(
                    draft.manifest.review["modules"][module]["status"]
                    == "awaiting_review"
                    for module in ("theory", "notebook", "experiment")
                )
            )


if __name__ == "__main__":
    unittest.main()
