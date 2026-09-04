from __future__ import annotations

import io
import json
import shutil
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import nbformat
from pypdf import PdfWriter
from pypdf.generic import (
    DecodedStreamObject,
    DictionaryObject,
    NameObject,
)

from web.algorithm_registry import (
    DraftInput,
    DraftStateError,
    DraftValidationError,
    ExperimentReporter,
    ExperimentTimeoutError,
    ExperimentUnavailableError,
    PackageValidationError,
    approve_module,
    cancel_change_request,
    create_draft,
    create_revision_draft,
    extract_source_text,
    get_experiment_design_preset,
    install_approved_draft,
    install_package,
    list_drafts,
    list_installed,
    list_rejected_drafts,
    load_experiment,
    reject_draft,
    regenerate_module,
    remove_animation_module,
    replace_module_file,
    resolve_placeholder_blocker,
    request_changes,
    restore_rejected_draft,
    save_algorithm_spec,
    save_animation_module,
    save_experiment_design,
    save_theory,
    trash_rejected_draft,
    uninstall_package,
    validate_source_directory,
)
from web.algorithm_registry.package import CORE_MODULES


def draft_input(
    *,
    algorithm_id: str = "example-algorithm",
    mode: str = "template",
) -> DraftInput:
    return DraftInput(
        algorithm_id=algorithm_id,
        name=(
            "Monte Carlo Control"
            if algorithm_id == "monte-carlo-control"
            else "Example Algorithm"
        ),
        version="1.0.0",
        category="model-free-control",
        summary="A v2 workflow test algorithm.",
        objective="Learn a policy from episodic experience.",
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
        reference_urls=("https://example.com/paper",),
        generation_mode=mode,
    )


def text_pdf() -> bytes:
    writer = PdfWriter()
    page = writer.add_blank_page(width=300, height=200)
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    resources = DictionaryObject(
        {
            NameObject("/Font"): DictionaryObject(
                {NameObject("/F1"): writer._add_object(font)}
            )
        }
    )
    page[NameObject("/Resources")] = resources
    stream = DecodedStreamObject()
    stream.set_data(b"BT /F1 12 Tf 50 100 Td (Algorithm source) Tj ET")
    page[NameObject("/Contents")] = writer._add_object(stream)
    output = io.BytesIO()
    writer.write(output)
    return output.getvalue()


def sample_mp4() -> bytes:
    return b"\x00\x00\x00\x18ftypisom\x00\x00\x02\x00isomiso2"


class AlgorithmSpecV2Tests(unittest.TestCase):
    def test_json_fragments_and_object_defaults_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with self.assertRaises(DraftValidationError):
                create_draft(
                    replace(
                        draft_input(),
                        core_equations=("{", '"latex": "Q(s,a)"', "}"),
                    ),
                    root / "drafts",
                    root / "installed",
                )
            with self.assertRaises(DraftValidationError):
                create_draft(
                    replace(
                        draft_input(),
                        hyperparameters={
                            "gamma": {"default": {"value": 0.99}}
                        },
                    ),
                    root / "drafts",
                    root / "installed",
                )

    def test_agent_confirmed_spec_requires_verified_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with self.assertRaises(DraftValidationError):
                create_draft(
                    replace(
                        draft_input(),
                        algorithm_spec_agent={
                            "provider": "langchain-openai-compatible",
                            "accepted_after_manual_review": True,
                            "evidence": [],
                        },
                    ),
                    root / "drafts",
                    root / "installed",
                )

    def test_platform_experiment_design_is_stored_in_algorithm_spec(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            design = get_experiment_design_preset("frozen-lake-4x4-v1")
            draft = create_draft(
                replace(draft_input(), experiment_design=design),
                root / "drafts",
                root / "installed",
            )

            stored = draft.manifest.algorithm["experiment_design"]
            self.assertEqual(stored["provenance"]["type"], "platform_preset")
            self.assertEqual(
                stored["environment_map"]["layout"],
                ["SFFF", "FHFH", "FFFH", "HFFG"],
            )

    def test_agent_suggestion_metadata_is_preserved_in_generation_record(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            draft = create_draft(
                replace(
                    draft_input(),
                    algorithm_spec_agent={
                        "provider": "langchain-openai-compatible",
                        "framework": "langchain",
                        "model": "test-model",
                        "structured_output_method": "function_calling",
                        "response_id": "run_test",
                        "accepted_after_manual_review": True,
                    },
                ),
                root / "drafts",
                root / "installed",
            )

            agent_record = draft.manifest.generation["algorithm_spec_agent"]
            self.assertEqual(
                agent_record["provider"],
                "langchain-openai-compatible",
            )
            self.assertEqual(agent_record["framework"], "langchain")
            self.assertEqual(agent_record["model"], "test-model")
            self.assertEqual(agent_record["response_id"], "run_test")

    def test_generic_draft_is_valid_but_publication_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            draft = create_draft(
                draft_input(),
                root / "drafts",
                root / "installed",
            )

            self.assertTrue(draft.report.valid, draft.report.errors)
            self.assertEqual(draft.manifest.schema_version, 2)
            self.assertEqual(
                draft.manifest.generation["blocking_flags"],
                ["placeholder_content"],
            )
            self.assertTrue(
                all(
                    draft.manifest.review["modules"][module]["status"]
                    == "not_generated"
                    for module in CORE_MODULES
                )
            )
            self.assertEqual(len(list_drafts(root / "drafts")), 1)
            with self.assertRaises(DraftStateError):
                approve_module(
                    draft.key,
                    "theory",
                    "reviewer",
                    drafts_root=root / "drafts",
                )
            archive = root / "generic.zip"
            import zipfile

            with zipfile.ZipFile(archive, "w") as output:
                for path in draft.path.rglob("*"):
                    if path.is_file():
                        output.write(path, path.relative_to(draft.path).as_posix())
            with self.assertRaises(PackageValidationError):
                install_package(archive, root / "installed")

    def test_completed_module_can_be_approved_before_other_scaffolds(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            drafts_root = root / "drafts"
            draft = create_draft(
                draft_input(),
                drafts_root,
                root / "installed",
            )
            with self.assertRaises(DraftStateError):
                approve_module(
                    draft.key,
                    "theory",
                    "Ada",
                    drafts_root=drafts_root,
                )

            save_theory(
                draft.key,
                "# Reviewed theory\n\nThis module is ready for review.",
                "Ada",
                drafts_root=drafts_root,
            )
            approved = approve_module(
                draft.key,
                "theory",
                "Ada",
                drafts_root=drafts_root,
            )

            self.assertEqual(
                approved.manifest.review["modules"]["theory"]["status"],
                "approved",
            )
            self.assertEqual(
                approved.manifest.review["modules"]["notebook"]["status"],
                "not_generated",
            )
            self.assertEqual(
                approved.manifest.generation["blocking_flags"],
                ["placeholder_content"],
            )

    def test_missing_v2_algorithm_field_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            draft = create_draft(
                draft_input(),
                root / "drafts",
                root / "installed",
            )
            manifest_path = draft.path / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            del manifest["algorithm"]["pseudocode"]
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            report = validate_source_directory(draft.path)

            self.assertFalse(report.valid)
            self.assertIn(
                "invalid_algorithm_spec",
                {issue.code for issue in report.errors},
            )

    def test_unsafe_v2_source_path_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            draft = create_draft(
                draft_input(),
                root / "drafts",
                root / "installed",
            )
            manifest_path = draft.path / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["sources"][0]["path"] = "../outside.md"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            report = validate_source_directory(draft.path)

            self.assertFalse(report.valid)
            self.assertIn("unsafe_path", {issue.code for issue in report.errors})

    def test_pdf_text_is_extracted_and_blank_pdf_is_rejected(self) -> None:
        extracted = extract_source_text("source.pdf", text_pdf())
        self.assertIn("Algorithm source", extracted)

        writer = PdfWriter()
        writer.add_blank_page(width=100, height=100)
        output = io.BytesIO()
        writer.write(output)
        with self.assertRaises(DraftValidationError):
            extract_source_text("blank.pdf", output.getvalue())

    def test_invalid_animation_upload_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with self.assertRaises(DraftValidationError):
                create_draft(
                    replace(
                        draft_input(),
                        animation_name="animation.mp4",
                        animation_bytes=b"not-an-mp4",
                    ),
                    root / "drafts",
                    root / "installed",
                )


class DraftReviewAndInstallationTests(unittest.TestCase):
    def test_change_request_can_be_cancelled_without_modifying_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            drafts_root = root / "drafts"
            draft = create_draft(
                draft_input(
                    algorithm_id="monte-carlo-control",
                    mode="monte-carlo-preset",
                ),
                drafts_root,
                root / "installed",
            )
            notebook_path = draft.path / draft.manifest.notebook["file"]
            original_notebook = notebook_path.read_bytes()
            request_changes(
                draft.key,
                "notebook",
                "Ada",
                "The requested plot is no longer needed.",
                drafts_root=drafts_root,
            )

            restored = cancel_change_request(
                draft.key,
                "notebook",
                "Ada",
                "Withdrawing the request after reviewing the lesson scope.",
                drafts_root=drafts_root,
            )

            self.assertEqual(notebook_path.read_bytes(), original_notebook)
            self.assertEqual(
                restored.manifest.review["modules"]["notebook"]["status"],
                "awaiting_review",
            )
            self.assertEqual(
                restored.manifest.review["history"][-1]["action"],
                "change_request_cancelled",
            )
            with self.assertRaises(DraftStateError):
                cancel_change_request(
                    draft.key,
                    "notebook",
                    "Ada",
                    "There is no active request now.",
                    drafts_root=drafts_root,
                )

    def test_optional_animation_can_be_added_edited_and_removed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            drafts_root = root / "drafts"
            draft = create_draft(
                draft_input(
                    algorithm_id="monte-carlo-control",
                    mode="monte-carlo-preset",
                ),
                drafts_root,
                root / "installed",
            )
            animated = save_animation_module(
                draft.key,
                file_name="lesson.mp4",
                payload=sample_mp4(),
                concept_markdown="## Episode returns",
                formula="G_t = R_{t+1} + gamma G_{t+1}",
                symbols=({"symbol": "γ", "meaning": "discount factor"},),
                highlights=("Complete episode", "Backward return"),
                viewing_flow=("Watch the episode finish", "Trace returns backward"),
                derivation_steps=({"title": "Return", "latex": "G_t"},),
                reviewer="Ada",
                drafts_root=drafts_root,
            )
            self.assertIsNotNone(animated.manifest.animation)
            self.assertEqual(
                animated.manifest.animation["symbols"][0]["symbol"], "γ"
            )
            self.assertEqual(
                animated.manifest.review["modules"]["animation"]["status"],
                "awaiting_review",
            )
            approve_module(
                draft.key,
                "animation",
                "Ada",
                drafts_root=drafts_root,
            )
            with self.assertRaises(DraftStateError):
                save_animation_module(
                    draft.key,
                    file_name=None,
                    payload=None,
                    formula="Locked",
                    highlights=(),
                    derivation_steps=(),
                    reviewer="Ada",
                    drafts_root=drafts_root,
                )
            request_changes(
                draft.key,
                "animation",
                "Ada",
                "Correct the formula label.",
                drafts_root=drafts_root,
            )
            save_animation_module(
                draft.key,
                file_name=None,
                payload=None,
                formula="Updated formula",
                highlights=("Updated teaching point",),
                derivation_steps=(),
                reviewer="Ada",
                drafts_root=drafts_root,
            )
            removed = remove_animation_module(
                draft.key,
                "Ada",
                drafts_root=drafts_root,
                trash_root=root / "trash" / "draft-assets",
            )
            self.assertIsNone(removed.manifest.animation)
            self.assertNotIn("animation", removed.manifest.review["modules"])
            self.assertEqual(
                len(list((root / "trash" / "draft-assets").glob("*.mp4"))),
                1,
            )

    def test_installed_v2_package_creates_patch_revision_draft(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            drafts_root = root / "drafts"
            installed_root = root / "installed"
            draft = create_draft(
                draft_input(
                    algorithm_id="monte-carlo-control",
                    mode="monte-carlo-preset",
                ),
                drafts_root,
                installed_root,
            )
            for module in CORE_MODULES:
                approve_module(
                    draft.key,
                    module,
                    "Ada",
                    drafts_root=drafts_root,
                )
            install_approved_draft(
                draft.key,
                "Ada",
                drafts_root=drafts_root,
                installed_root=installed_root,
                artifact_root=root / "dist",
                trash_root=root / "trash" / "drafts",
            )

            revision = create_revision_draft(
                "monte-carlo-control",
                "1.0.1",
                "Grace",
                drafts_root=drafts_root,
                installed_root=installed_root,
            )
            self.assertEqual(revision.manifest.version, "1.0.1")
            self.assertEqual(
                revision.manifest.generation["revision_of"]["version"],
                "1.0.0",
            )
            self.assertTrue(
                all(
                    state["status"] == "approved"
                    for state in revision.manifest.review["modules"].values()
                )
            )
            with self.assertRaises(DraftStateError):
                install_approved_draft(
                    revision.key,
                    "Grace",
                    drafts_root=drafts_root,
                    installed_root=installed_root,
                    artifact_root=root / "dist",
                    trash_root=root / "trash" / "drafts",
                )

    def test_template_revision_cannot_overwrite_module_with_scaffold(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            drafts_root = root / "drafts"
            installed_root = root / "installed"
            base = create_draft(
                draft_input(algorithm_id="q-learning-tabular"),
                drafts_root,
                installed_root,
            )
            save_theory(
                base.key,
                "# Q-Learning\n\nReviewed theory.",
                "Grace",
                drafts_root=drafts_root,
            )
            notebook = nbformat.v4.new_notebook(
                cells=[
                    nbformat.v4.new_markdown_cell("# Q-Learning"),
                    nbformat.v4.new_code_cell("value = 1\nvalue"),
                ]
            )
            notebook_buffer = io.StringIO()
            nbformat.write(notebook, notebook_buffer)
            replace_module_file(
                base.key,
                "notebook",
                "notebook.ipynb",
                notebook_buffer.getvalue().encode("utf-8"),
                "Grace",
                drafts_root=drafts_root,
            )
            replace_module_file(
                base.key,
                "experiment",
                "experiment.py",
                (
                    b"def get_spec():\n"
                    b"    return {'parameters': {}}\n\n"
                    b"def run(parameters, reporter):\n"
                    b"    return {'metrics': {}, 'summary': {}, 'artifacts': []}\n"
                ),
                "Grace",
                drafts_root=drafts_root,
            )
            resolve_placeholder_blocker(
                base.key,
                "Grace",
                "All core files were manually completed for the revision test.",
                drafts_root=drafts_root,
            )
            for module in CORE_MODULES:
                approve_module(
                    base.key, module, "Grace", drafts_root=drafts_root
                )
            install_approved_draft(
                base.key,
                "Grace",
                drafts_root=drafts_root,
                installed_root=installed_root,
                artifact_root=root / "dist",
                trash_root=root / "trash" / "drafts",
            )
            revision = create_revision_draft(
                "q-learning-tabular",
                "1.0.1",
                "Grace",
                drafts_root=drafts_root,
                installed_root=installed_root,
            )
            experiment_path = revision.path / revision.manifest.experiment["module"]
            original = experiment_path.read_bytes()

            updated = request_changes(
                revision.key,
                "experiment",
                "Grace",
                "Exercise the protected revision regeneration path.",
                drafts_root=drafts_root,
            )
            self.assertEqual(
                updated.manifest.review["modules"]["experiment"]["status"],
                "changes_requested",
            )
            self.assertEqual(
                updated.manifest.review["modules"]["theory"]["status"],
                "approved",
            )
            self.assertEqual(
                updated.manifest.review["modules"]["notebook"]["status"],
                "approved",
            )
            with self.assertRaisesRegex(
                DraftStateError,
                "cannot be replaced with deterministic scaffolds",
            ):
                regenerate_module(
                    revision.key,
                    "experiment",
                    "Grace",
                    drafts_root=drafts_root,
                )
            self.assertEqual(experiment_path.read_bytes(), original)

    def test_legacy_animation_step_fields_are_normalized(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = replace(
                draft_input(
                    algorithm_id="monte-carlo-control",
                    mode="monte-carlo-preset",
                ),
                animation_name="lesson.mp4",
                animation_bytes=sample_mp4(),
                animation_derivation_steps=(
                    {"title": "Canonical", "text": "Initial"},
                ),
            )
            draft = create_draft(data, root / "drafts", root / "installed")
            manifest_path = draft.path / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["modules"]["animation"]["derivation_steps"] = [
                {"name": "Legacy title", "content": "Legacy text"}
            ]
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            report = validate_source_directory(draft.path)
            self.assertTrue(report.valid)
            self.assertEqual(
                report.manifest.animation["derivation_steps"][0]["title"],
                "Legacy title",
            )
            self.assertEqual(
                report.manifest.animation["derivation_steps"][0]["text"],
                "Legacy text",
            )
            self.assertTrue(
                any(issue.code == "legacy_animation_step" for issue in report.warnings)
            )

    def test_declared_animation_must_be_approved_before_installation(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            drafts_root = root / "drafts"
            installed_root = root / "installed"
            draft = create_draft(
                replace(
                    draft_input(
                        algorithm_id="monte-carlo-control",
                        mode="monte-carlo-preset",
                    ),
                    animation_name="lesson.mp4",
                    animation_bytes=sample_mp4(),
                    animation_formula="Q(s,a)",
                    animation_highlights=("Episode return",),
                ),
                drafts_root,
                installed_root,
            )
            for module in CORE_MODULES:
                approve_module(
                    draft.key,
                    module,
                    "Ada",
                    drafts_root=drafts_root,
                )
            with self.assertRaises(DraftStateError):
                install_approved_draft(
                    draft.key,
                    "Ada",
                    drafts_root=drafts_root,
                    installed_root=installed_root,
                    artifact_root=root / "dist",
                    trash_root=root / "trash" / "drafts",
                )

            approve_module(
                draft.key,
                "animation",
                "Ada",
                drafts_root=drafts_root,
            )
            installed, _ = install_approved_draft(
                draft.key,
                "Ada",
                drafts_root=drafts_root,
                installed_root=installed_root,
                artifact_root=root / "dist",
                trash_root=root / "trash" / "drafts",
            )
            self.assertIsNotNone(installed.manifest.animation)
            self.assertEqual(
                installed.manifest.review["modules"]["animation"]["status"],
                "installed",
            )

    def test_review_changes_edit_and_reject_are_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            draft = create_draft(
                draft_input(
                    algorithm_id="monte-carlo-control",
                    mode="monte-carlo-preset",
                ),
                root / "drafts",
                root / "installed",
            )
            approve_module(
                draft.key,
                "theory",
                "Ada",
                drafts_root=root / "drafts",
            )
            request_changes(
                draft.key,
                "theory",
                "Ada",
                "Correct the theory wording.",
                drafts_root=root / "drafts",
            )
            edited = save_theory(
                draft.key,
                "# Corrected theory\n\nReviewed content.",
                "Ada",
                drafts_root=root / "drafts",
            )
            self.assertEqual(
                edited.manifest.review["modules"]["theory"]["status"],
                "awaiting_review",
            )
            changed = request_changes(
                draft.key,
                "notebook",
                "Ada",
                "Add a convergence plot.",
                drafts_root=root / "drafts",
            )
            self.assertEqual(
                changed.manifest.review["modules"]["notebook"]["status"],
                "changes_requested",
            )
            rejected = reject_draft(
                draft.key,
                "Ada",
                "Experiment assumptions are not ready.",
                drafts_root=root / "drafts",
                rejected_root=root / "rejected",
            )
            self.assertTrue(rejected.is_dir())
            self.assertEqual(list_drafts(root / "drafts"), ())

    def test_approved_content_is_locked_until_changes_are_requested(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            draft = create_draft(
                draft_input(
                    algorithm_id="monte-carlo-control",
                    mode="monte-carlo-preset",
                ),
                root / "drafts",
                root / "installed",
            )
            approve_module(
                draft.key,
                "theory",
                "Ada",
                drafts_root=root / "drafts",
            )
            with self.assertRaises(DraftStateError):
                save_theory(
                    draft.key,
                    "# Locked",
                    "Ada",
                    drafts_root=root / "drafts",
                )
            request_changes(
                draft.key,
                "theory",
                "Ada",
                "Correct the introduction.",
                drafts_root=root / "drafts",
            )
            edited = save_theory(
                draft.key,
                "# Revised\n\nCorrected introduction.",
                "Ada",
                drafts_root=root / "drafts",
            )
            self.assertEqual(
                edited.manifest.review["modules"]["theory"]["status"],
                "awaiting_review",
            )

    def test_spec_edit_and_module_replacement_reset_review(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            draft = create_draft(
                draft_input(
                    algorithm_id="monte-carlo-control",
                    mode="monte-carlo-preset",
                ),
                root / "drafts",
                root / "installed",
            )
            algorithm = dict(draft.manifest.algorithm)
            algorithm["objective"] = "A revised objective."
            updated = save_algorithm_spec(
                draft.key,
                name="Monte Carlo Control",
                category="model-free-control",
                summary="Updated summary.",
                algorithm=algorithm,
                reviewer="Ada",
                drafts_root=root / "drafts",
            )
            self.assertEqual(updated.manifest.summary, "Updated summary.")
            self.assertTrue(
                all(
                    updated.manifest.review["modules"][module]["status"]
                    == "changes_requested"
                    for module in CORE_MODULES
                )
            )

            original = (
                updated.path / updated.manifest.notebook["file"]
            ).read_bytes()
            with self.assertRaises(DraftValidationError):
                replace_module_file(
                    updated.key,
                    "notebook",
                    "broken.ipynb",
                    b"{not-json",
                    "Ada",
                    drafts_root=root / "drafts",
                )
            self.assertEqual(
                (updated.path / updated.manifest.notebook["file"]).read_bytes(),
                original,
            )
            notebook = nbformat.v4.new_notebook(
                cells=[nbformat.v4.new_markdown_cell("# Replacement")]
            )
            buffer = io.StringIO()
            nbformat.write(notebook, buffer)
            replaced = replace_module_file(
                updated.key,
                "notebook",
                "replacement.ipynb",
                buffer.getvalue().encode(),
                "Ada",
                drafts_root=root / "drafts",
            )
            self.assertEqual(
                replaced.manifest.review["modules"]["notebook"]["status"],
                "awaiting_review",
            )

    def test_generic_scaffold_can_be_manually_completed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            draft = create_draft(
                draft_input(),
                root / "drafts",
                root / "installed",
            )
            save_theory(
                draft.key,
                "# Reviewed theory\n\nManually completed.",
                "Ada",
                drafts_root=root / "drafts",
            )
            notebook = nbformat.v4.new_notebook(
                cells=[nbformat.v4.new_markdown_cell("# Reviewed notebook")]
            )
            notebook_buffer = io.StringIO()
            nbformat.write(notebook, notebook_buffer)
            replace_module_file(
                draft.key,
                "notebook",
                "reviewed.ipynb",
                notebook_buffer.getvalue().encode(),
                "Ada",
                drafts_root=root / "drafts",
            )
            with self.assertRaises(DraftStateError):
                resolve_placeholder_blocker(
                    draft.key,
                    "Ada",
                    "Implementation complete.",
                    drafts_root=root / "drafts",
                )
            experiment_source = b"""
def get_spec():
    return {"parameters": {}}

def run(parameters, reporter):
    return {"metrics": {}, "summary": {"reviewed": True}, "artifacts": []}
"""
            replace_module_file(
                draft.key,
                "experiment",
                "reviewed.py",
                experiment_source,
                "Ada",
                drafts_root=root / "drafts",
            )
            completed = resolve_placeholder_blocker(
                draft.key,
                "Ada",
                "Algorithm expert supplied and reviewed all three modules.",
                drafts_root=root / "drafts",
            )
            self.assertEqual(
                completed.manifest.generation["blocking_flags"],
                [],
            )

    def test_rejected_draft_can_be_restored_or_moved_to_trash(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            draft = create_draft(
                draft_input(
                    algorithm_id="monte-carlo-control",
                    mode="monte-carlo-preset",
                ),
                root / "drafts",
                root / "installed",
            )
            rejected_path = reject_draft(
                draft.key,
                "Ada",
                "Needs another review cycle.",
                drafts_root=root / "drafts",
                rejected_root=root / "rejected",
            )
            rejected = list_rejected_drafts(root / "rejected")
            self.assertEqual(len(rejected), 1)
            restored = restore_rejected_draft(
                rejected[0].key,
                "Ada",
                rejected_root=root / "rejected",
                drafts_root=root / "drafts",
            )
            self.assertEqual(restored.key, draft.key)
            self.assertFalse(rejected_path.exists())

            reject_draft(
                restored.key,
                "Ada",
                "Archive this attempt.",
                drafts_root=root / "drafts",
                rejected_root=root / "rejected",
            )
            rejected = list_rejected_drafts(root / "rejected")
            trashed = trash_rejected_draft(
                rejected[0].key,
                "Ada",
                rejected_root=root / "rejected",
                trash_root=root / "trash" / "rejected",
            )
            self.assertTrue(trashed.is_dir())
            self.assertEqual(list_rejected_drafts(root / "rejected"), ())

    def test_monte_carlo_draft_installs_runs_and_uninstalls(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            drafts_root = root / "drafts"
            installed_root = root / "installed"
            draft = create_draft(
                draft_input(
                    algorithm_id="monte-carlo-control",
                    mode="monte-carlo-preset",
                ),
                drafts_root,
                installed_root,
            )
            for module in CORE_MODULES:
                draft = approve_module(
                    draft.key,
                    module,
                    "Ada",
                    drafts_root=drafts_root,
                )

            installed, archive = install_approved_draft(
                draft.key,
                "Ada",
                drafts_root=drafts_root,
                installed_root=installed_root,
                artifact_root=root / "dist",
                trash_root=root / "trash" / "drafts",
            )

            self.assertEqual(installed.manifest.schema_version, 2)
            self.assertTrue(archive.is_file())
            self.assertEqual(list_drafts(drafts_root), ())
            self.assertEqual(len(list_installed(installed_root)), 1)
            experiment = load_experiment(
                "monte-carlo-control",
                installed_root,
                timeout_seconds=10,
            )
            result = experiment.run(
                {
                    "episodes": 100,
                    "gamma": 0.99,
                    "epsilon": 0.2,
                    "slippery": False,
                    "seed": 7,
                }
            )
            self.assertEqual(result["summary"]["episodes"], 100)
            trash_path = uninstall_package(
                "monte-carlo-control",
                installed_root,
            )
            self.assertTrue(trash_path.is_dir())
            self.assertEqual(list_installed(installed_root), ())

    def test_failed_publish_preserves_the_approved_draft(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            drafts_root = root / "drafts"
            installed_root = root / "installed"
            draft = create_draft(
                draft_input(
                    algorithm_id="monte-carlo-control",
                    mode="monte-carlo-preset",
                ),
                drafts_root,
                installed_root,
            )
            for module in CORE_MODULES:
                draft = approve_module(
                    draft.key,
                    module,
                    "Ada",
                    drafts_root=drafts_root,
                )

            with mock.patch(
                "web.algorithm_registry.drafts.install_package",
                side_effect=RuntimeError("simulated publish failure"),
            ):
                with self.assertRaises(RuntimeError):
                    install_approved_draft(
                        draft.key,
                        "Ada",
                        drafts_root=drafts_root,
                        installed_root=installed_root,
                        artifact_root=root / "dist",
                        trash_root=root / "trash" / "drafts",
                    )

            self.assertTrue(draft.path.is_dir())
            self.assertEqual(list_installed(installed_root), ())


class IsolatedRuntimeTests(unittest.TestCase):
    def _install_experiment(self, source: str, root: Path):
        package = root / "source"
        package.mkdir()
        manifest = {
            "schema_version": 1,
            "id": "isolated-test",
            "name": "Isolated Test",
            "version": "1.0.0",
            "summary": "Runtime isolation fixture.",
            "category": "test",
            "theory": {"file": "theory.md"},
            "experiment": {"module": "experiment.py", "requirements": []},
        }
        (package / "manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        (package / "theory.md").write_text("# Test", encoding="utf-8")
        (package / "experiment.py").write_text(source, encoding="utf-8")
        archive = root / "package.zip"
        import zipfile

        with zipfile.ZipFile(archive, "w") as output:
            for path in package.iterdir():
                output.write(path, path.name)
        install_package(archive, root / "installed")

    def test_progress_and_metrics_cross_process_boundary(self) -> None:
        source = """
def get_spec():
    return {"parameters": {"episodes": {"type": "int", "default": 2, "min": 1}}}

def run(parameters, reporter):
    reporter.progress(1, 2, "half")
    reporter.metric("reward", 1.0, step=1)
    reporter.progress(2, 2, "done")
    return {"metrics": {"reward": [1.0]}, "summary": {"ok": True}, "artifacts": []}
"""
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self._install_experiment(source, root)
            reporter = ExperimentReporter()
            result = load_experiment(
                "isolated-test", root / "installed", timeout_seconds=5
            ).run({"episodes": 2}, reporter)

            self.assertTrue(result["summary"]["ok"])
            self.assertEqual(len(reporter.progress_events), 2)
            self.assertEqual(len(reporter.metric_events), 1)

    def test_timeout_and_worker_crash_are_contained(self) -> None:
        timeout_source = """
import time
def get_spec():
    return {"parameters": {}}
def run(parameters, reporter):
    time.sleep(5)
    return {"metrics": {}, "summary": {}, "artifacts": []}
"""
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self._install_experiment(timeout_source, root)
            experiment = load_experiment(
                "isolated-test", root / "installed", timeout_seconds=0.2
            )
            with self.assertRaises(ExperimentTimeoutError):
                experiment.run({})

        crash_source = """
import os
def get_spec():
    return {"parameters": {}}
def run(parameters, reporter):
    os._exit(7)
"""
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self._install_experiment(crash_source, root)
            experiment = load_experiment(
                "isolated-test", root / "installed", timeout_seconds=2
            )
            with self.assertRaises(ExperimentUnavailableError):
                experiment.run({})


if __name__ == "__main__":
    unittest.main()
