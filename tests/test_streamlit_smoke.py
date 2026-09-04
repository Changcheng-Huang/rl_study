from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "animations-matplotlib-test"),
)

from streamlit.testing.v1 import AppTest


class StreamlitSmokeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.packages_directory = tempfile.TemporaryDirectory()
        self.previous_packages_root = os.environ.get("ALGORITHM_PACKAGES_ROOT")
        os.environ["ALGORITHM_PACKAGES_ROOT"] = self.packages_directory.name
        from web.algorithm_registry import DraftInput, create_draft

        create_draft(
            DraftInput(
                algorithm_id="smoke-test-algorithm",
                name="Smoke Test Algorithm",
                version="1.0.0",
                category="model-free-control",
                summary="A temporary draft used only for interface testing.",
                objective="Verify the review workspace.",
                assumptions=("Finite state space",),
                inputs=("Environment",),
                outputs=("Policy",),
                states=("Environment state",),
                actions=("Available action",),
                hyperparameters={"gamma": {"default": 0.99}},
                core_equations=("G_t = R_{t+1} + gamma G_{t+1}",),
                pseudocode=("Generate an episode", "Update action values"),
                supported_environments=("FrozenLake-v1",),
                source_name="source.md",
                source_bytes=b"# Trusted source\n\nAlgorithm source material.",
            )
        )

    def tearDown(self) -> None:
        if self.previous_packages_root is None:
            os.environ.pop("ALGORITHM_PACKAGES_ROOT", None)
        else:
            os.environ["ALGORITHM_PACKAGES_ROOT"] = self.previous_packages_root
        self.packages_directory.cleanup()

    def test_manage_algorithms_page_loads_v2_workflow(self) -> None:
        app_path = Path(__file__).resolve().parents[1] / "web" / "app.py"
        app = AppTest.from_file(str(app_path)).run(timeout=30)
        self.assertFalse(app.exception)

        app.sidebar.radio[0].set_value("Manage Algorithms")
        app.run(timeout=30)

        self.assertFalse(app.exception)
        self.assertEqual(len(app.button_group), 1)
        self.assertEqual(
            [option.content for option in app.button_group[0].options],
            [
                "Create v2 Draft",
                "Review Drafts",
                "Rejected Drafts",
                "Legacy v1 ZIP",
                "Installed",
            ],
        )
        editable_labels = [
            element.label for element in [*app.text_input, *app.text_area]
        ]
        self.assertFalse(
            any("JSON" in label for label in editable_labels),
            editable_labels,
        )
        agent_button = next(
            button
            for button in app.button
            if button.label == "Suggest AlgorithmSpec with Agent"
        )
        self.assertTrue(agent_button.disabled)
        self.assertTrue(
            any(
                "ALGORITHM_AGENT_API_KEY" in item.value
                for item in app.info
            )
        )
        create_button = next(
            button
            for button in app.button
            if button.label
            == "Confirm AlgorithmSpec and Create Review Draft"
        )
        self.assertTrue(create_button.disabled)
        generation_profile = next(
            item for item in app.selectbox if item.label == "Generation profile"
        )
        generation_profile.set_value("Generic scaffold")
        app.button_group[0].set_value(["Create v2 Draft"])
        app.run(timeout=30)
        generic_algorithm_id = next(
            item for item in app.text_input if item.label == "Algorithm ID"
        )
        generic_name = next(item for item in app.text_input if item.label == "Name")
        self.assertEqual(generic_algorithm_id.value, "")
        self.assertEqual(generic_name.value, "")
        self.assertTrue(
            any("Generic workflow starts empty" in item.value for item in app.info)
        )
        generation_profile = next(
            item for item in app.selectbox if item.label == "Generation profile"
        )
        generation_profile.set_value("Monte Carlo Control preset")
        app.button_group[0].set_value(["Create v2 Draft"])
        app.run(timeout=30)
        # ButtonGroup's AppTest adapter requires its single value as a list.
        app.button_group[0].set_value(["Create v2 Draft"])
        app.sidebar.text_input[0].set_value("Provider Ada")
        apply_roles = next(
            button for button in app.button if button.label == "Apply role names"
        )
        apply_roles.click()
        app.button_group[0].set_value(["Create v2 Draft"])
        app.run(timeout=30)
        self.assertTrue(
            any(
                "Provider and Reviewer names were applied" in item.value
                for item in app.success
            )
        )
        self.assertTrue(
            any(item.value == "### Role names applied" for item in app.markdown)
        )
        create_button = next(
            button
            for button in app.button
            if button.label
            == "Confirm AlgorithmSpec and Create Review Draft"
        )
        # A checkbox inside st.form does not rerun the page. The submit button
        # must therefore validate confirmation on submit instead of deadlocking.
        self.assertFalse(create_button.disabled)

        app.button_group[0].set_value(["Review Drafts"])
        app.run(timeout=30)
        self.assertEqual(app.button_group[0].value, "Review Drafts")
        self.assertEqual(
            [option.content for option in app.button_group[1].options],
            [
                "Overview",
                "Theory · not generated",
                "Notebook · not generated",
                "Experiment · not generated",
                "Animation · not started",
                "Publish · blocked",
            ],
        )
        self.assertTrue(
            any(
                item.label == "How to read this AlgorithmSpec"
                for item in app.expander
            )
        )
        guide_path = app_path.parents[1] / "docs" / "algorithm_spec_user_guide.md"
        self.assertTrue(guide_path.is_file())
        self.assertIn("What the Settings table does", guide_path.read_text())

        app.button_group[0].set_value(["Review Drafts"])
        app.button_group[1].set_value(["Animation"])
        app.run(timeout=30)
        self.assertTrue(
            any(
                item.label == "How the Animation workflow works"
                for item in app.expander
            )
        )
        animation_guide = (
            app_path.parents[1] / "docs" / "animation_workflow_user_guide.md"
        )
        self.assertTrue(animation_guide.is_file())
        self.assertIn("Generate three concepts", animation_guide.read_text())
        starter_button = next(
            button
            for button in app.button
            if button.label == "Create three starter concepts"
        )
        starter_button.click()
        # Keep both ButtonGroup values in the list form expected by AppTest
        # while the click triggers a rerun with a new status-label signature.
        app.button_group[0].set_value(["Review Drafts"])
        app.button_group[1].set_value(["Animation"])
        app.run(timeout=30)
        self.assertEqual(app.button_group[1].value, "Animation")
        self.assertIn(
            "Animation · concepts ready",
            [option.content for option in app.button_group[1].options],
        )
        self.assertFalse(
            any(item.label == "Theory Markdown" for item in app.text_area)
        )

        app.button_group[0].set_value(["Review Drafts"])
        app.button_group[1].set_value(["Theory"])
        app.run(timeout=30)
        self.assertFalse(any(item.label == "Theory Markdown" for item in app.text_area))
        self.assertFalse(
            any(item.label == "Write Theory manually" for item in app.checkbox)
        )
        self.assertFalse(
            any(button.label == "Save Theory" for button in app.button)
        )
        approve_theory = next(button for button in app.button if button.label == "Approve")
        self.assertTrue(approve_theory.disabled)

        # Applying a role name must preserve the selected manager and module.
        app.sidebar.text_input[1].set_value("Reviewer Grace")
        apply_roles = next(
            button for button in app.button if button.label == "Apply role names"
        )
        apply_roles.click()
        app.button_group[0].set_value(["Review Drafts"])
        app.button_group[1].set_value(["Theory"])
        app.run(timeout=30)
        self.assertEqual(app.button_group[0].value, "Review Drafts")
        self.assertEqual(app.button_group[1].value, "Theory")

    def test_approved_theory_can_be_reopened_for_provider_edits(self) -> None:
        from web.algorithm_registry import approve_module, save_theory

        save_theory(
            "smoke-test-algorithm-1.0.0",
            "# Reviewed Theory\n\nA complete learner-facing explanation.",
            "Provider Ada",
        )
        approve_module(
            "smoke-test-algorithm-1.0.0",
            "theory",
            "Reviewer Grace",
        )

        app_path = Path(__file__).resolve().parents[1] / "web" / "app.py"
        app = AppTest.from_file(str(app_path)).run(timeout=30)
        app.sidebar.radio[0].set_value("Manage Algorithms")
        app.run(timeout=30)
        app.sidebar.text_input[0].set_value("Provider Ada")
        app.sidebar.text_input[1].set_value("Reviewer Grace")
        next(button for button in app.button if button.label == "Apply role names").click()
        app.button_group[0].set_value(["Create v2 Draft"])
        app.run(timeout=30)

        app.button_group[0].set_value(["Review Drafts"])
        app.run(timeout=30)
        app.button_group[0].set_value(["Review Drafts"])
        app.button_group[1].set_value(["Theory"])
        app.run(timeout=30)

        theory_editor = next(
            item for item in app.text_area if item.label == "Theory Markdown"
        )
        self.assertTrue(theory_editor.disabled)
        reopen = next(
            button for button in app.button if button.label == "Reopen for Changes"
        )
        self.assertFalse(reopen.disabled)

        review_note = next(
            item
            for item in app.text_area
            if item.label == "Review note / required change reason"
        )
        review_note.set_value("Clarify the terminal-state update.")
        reopen.click()
        app.button_group[0].set_value(["Review Drafts"])
        app.button_group[1].set_value(["Theory"])
        app.run(timeout=30)

        self.assertIn(
            "Theory · changes requested",
            [option.content for option in app.button_group[1].options],
        )
        theory_editor = next(
            item for item in app.text_area if item.label == "Theory Markdown"
        )
        self.assertFalse(theory_editor.disabled)


if __name__ == "__main__":
    unittest.main()
