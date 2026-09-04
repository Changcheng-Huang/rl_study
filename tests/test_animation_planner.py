from __future__ import annotations

import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace

from web.algorithm_registry import (
    AnimationConceptOptionsOutput,
    AnimationGuidanceOutput,
    build_animation_creator_kit,
    create_draft,
    default_animation_guidance,
    default_animation_options,
    generate_animation_guidance_with_agent,
    generate_animation_options_with_agent,
    generate_default_animation_guidance,
    generate_default_animation_options,
    get_animation_planner_configuration,
    load_animation_guidance,
    load_animation_options,
    plan_animation_options,
    plan_animation_guidance,
    select_animation_option,
)

from test_algorithm_workflow_v2 import draft_input


def algorithm_spec() -> dict:
    return {
        "id": "q-learning",
        "name": "Q-Learning",
        "version": "1.0.0",
        "category": "value-based",
        "summary": "Learn action values from transitions.",
        "algorithm": {
            "objective": "Learn a policy that reaches a goal.",
            "assumptions": ["Finite actions"],
            "inputs": ["Transitions"],
            "outputs": ["Q table"],
            "states": ["Grid cell"],
            "actions": ["Left", "Down", "Right", "Up"],
            "hyperparameters": {"alpha": {"default": 0.1}},
            "core_equations": ["Q(s,a) \\leftarrow Q(s,a) + \\alpha \\delta"],
            "pseudocode": ["Observe", "Update"],
            "supported_environments": ["FrozenLake-v1"],
        },
    }


class FakeStructuredModel:
    def __init__(self, parsed):
        self.parsed = parsed
        self.messages = None

    def invoke(self, messages):
        self.messages = messages
        return {
            "raw": SimpleNamespace(
                id="animation-plan-run",
                usage_metadata={"input_tokens": 80, "output_tokens": 40},
            ),
            "parsed": self.parsed,
            "parsing_error": None,
        }


class FakeChatModel:
    def __init__(self, parsed):
        self.structured = FakeStructuredModel(parsed)
        self.options = None

    def with_structured_output(self, schema, **kwargs):
        self.options = {"schema": schema, **kwargs}
        return self.structured


class AnimationPlannerTests(unittest.TestCase):
    def test_deterministic_guidance_and_creator_kit(self) -> None:
        guidance = default_animation_guidance(algorithm_spec())

        self.assertGreaterEqual(len(guidance["scenes"]), 3)
        self.assertIn("Q(s,a)", guidance["metadata"]["formula"])
        archive = zipfile.ZipFile(io.BytesIO(build_animation_creator_kit(guidance)))
        self.assertEqual(
            set(archive.namelist()),
            {
                "START_HERE_video_brief.md",
                "01_storyboard.md",
                "02_narration_script.md",
                "03_formula_and_symbol_review.md",
                "04_delivery_checklist.md",
                "05_optional_tool_reference.md",
            },
        )
        guide = archive.read("START_HERE_video_brief.md").decode("utf-8")
        self.assertIn("website will not run", guide)
        self.assertIn("not required to read or edit JSON", guide)
        self.assertFalse(any(name.endswith(".json") for name in archive.namelist()))

    def test_exactly_three_animation_concepts_are_generated(self) -> None:
        options = default_animation_options(algorithm_spec())
        self.assertEqual(len(options), 3)
        self.assertEqual(len({item["option_id"] for item in options}), 3)

        output = AnimationConceptOptionsOutput.model_validate(
            {"options": options, "warnings": []}
        )
        model = FakeChatModel(output)
        result = plan_animation_options(
            algorithm_spec(), "Trusted source excerpt.", chat_model=model
        )
        self.assertEqual(len(result.options), 3)
        self.assertIn("exactly three", model.structured.messages[0][1])

    def test_openai_compatible_configuration_cascade(self) -> None:
        configuration = get_animation_planner_configuration(
            {
                "ALGORITHM_AGENT_API_KEY": "test-key",
                "ALGORITHM_AGENT_MODEL": "general-model",
                "ANIMATION_PLANNING_AGENT_MODEL": "animation-model",
                "ANIMATION_PLANNING_AGENT_ENABLE_THINKING": "false",
                "ANIMATION_PLANNING_AGENT_STRUCTURED_METHOD": "json_mode",
            }
        )

        self.assertTrue(configuration.configured)
        self.assertEqual(configuration.model, "animation-model")
        self.assertEqual(configuration.structured_output_method, "json_mode")
        self.assertFalse(configuration.enable_thinking)

    def test_agent_returns_structured_non_rendering_plan(self) -> None:
        output = AnimationGuidanceOutput.model_validate(
            default_animation_guidance(algorithm_spec())
        )
        model = FakeChatModel(output)

        result = plan_animation_guidance(
            algorithm_spec(),
            "Trusted source excerpt.",
            review_note="Keep it under two minutes.",
            chat_model=model,
        )

        self.assertEqual(result.metadata["response_id"], "animation-plan-run")
        self.assertEqual(result.metadata["usage"]["input_tokens"], 80)
        system_prompt = model.structured.messages[0][1]
        self.assertIn("Do not generate Python", system_prompt)
        self.assertIn("upload a finished MP4", system_prompt)
        self.assertIn(
            "write every generated human-readable field in English",
            system_prompt,
        )

    def test_guidance_persistence_does_not_create_or_reopen_animation(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            drafts_root = root / "drafts"
            draft = create_draft(
                draft_input(), drafts_root, root / "installed"
            )
            statuses_before = dict(draft.manifest.review["modules"])

            generate_default_animation_options(
                draft.key, "Provider Ada", drafts_root=drafts_root
            )
            select_animation_option(
                draft.key,
                "decision-walkthrough",
                "Provider Ada",
                drafts_root=drafts_root,
            )

            updated = generate_default_animation_guidance(
                draft.key, "Provider Ada", drafts_root=drafts_root
            )
            loaded = load_animation_guidance(draft.key, drafts_root)

            self.assertIsNotNone(loaded)
            self.assertNotIn("animation", updated.manifest.raw["modules"])
            self.assertEqual(updated.manifest.review["modules"], statuses_before)
            record = updated.manifest.generation["animation_guidance"]
            payload = (draft.path / record["file"]).read_bytes()
            self.assertEqual(record["sha256"], __import__("hashlib").sha256(payload).hexdigest())
            self.assertIn(
                "did not create or modify an MP4",
                updated.manifest.review["history"][-1]["note"],
            )

    def test_agent_guidance_records_run_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            drafts_root = root / "drafts"
            draft = create_draft(
                draft_input(), drafts_root, root / "installed"
            )
            model = FakeChatModel(
                AnimationGuidanceOutput.model_validate(
                    default_animation_guidance(algorithm_spec())
                )
            )

            generate_default_animation_options(
                draft.key, "Provider Ada", drafts_root=drafts_root
            )
            select_animation_option(
                draft.key,
                "comparison",
                "Provider Ada",
                drafts_root=drafts_root,
            )

            updated = generate_animation_guidance_with_agent(
                draft.key,
                "Provider Ada",
                "Emphasize the update.",
                chat_model=model,
                drafts_root=drafts_root,
            )

            record = updated.manifest.generation["animation_guidance"]
            self.assertEqual(record["source"], "Animation Planning Agent")
            self.assertEqual(record["agent_runs"][-1]["response_id"], "animation-plan-run")
            self.assertEqual(record["selected_option_id"], "comparison")

    def test_animation_option_selection_is_persisted(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            drafts_root = root / "drafts"
            draft = create_draft(draft_input(), drafts_root, root / "installed")
            model = FakeChatModel(
                AnimationConceptOptionsOutput.model_validate(
                    {"options": default_animation_options(algorithm_spec())}
                )
            )
            generated = generate_animation_options_with_agent(
                draft.key,
                "Provider Ada",
                chat_model=model,
                drafts_root=drafts_root,
            )
            selected = select_animation_option(
                generated.key,
                "learning-evolution",
                "Provider Ada",
                drafts_root=drafts_root,
            )
            loaded = load_animation_options(selected.key, drafts_root)
            self.assertEqual(loaded["selected_option_id"], "learning-evolution")
            self.assertEqual(
                loaded["record"]["selected_by"], "Provider Ada"
            )


if __name__ == "__main__":
    unittest.main()
