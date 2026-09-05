from __future__ import annotations

import argparse
import io
import json
import re
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import nbformat


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = PROJECT_ROOT / "web"
if str(WEB_ROOT) not in sys.path:
    sys.path.insert(0, str(WEB_ROOT))

from algorithm_registry.latex import (  # noqa: E402
    double_q_learning_core_latex,
    validate_latex,
)
from algorithm_registry.models import InstalledAlgorithm  # noqa: E402
from algorithm_registry.notebook_tools import normalize_and_validate_notebook  # noqa: E402
from algorithm_registry.package import validate_source_directory  # noqa: E402
from algorithm_registry.runtime import load_isolated_spec  # noqa: E402
from algorithm_registry.theory_content import (  # noqa: E402
    encode_theory_presentation,
    presentation_from_markdown,
    validate_theory_presentation,
)


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def migrate(source: Path, drafts_root: Path, *, reviewer: str) -> Path:
    report = validate_source_directory(source)
    if not report.valid or report.manifest is None:
        raise ValueError("legacy source is not a valid package")
    if report.manifest.algorithm_id != "double-q-learning":
        raise ValueError("migration source must be Double Q-Learning")
    destination = drafts_root / "double-q-learning-1.0.1"
    if destination.exists():
        raise FileExistsError(f"destination already exists: {destination}")
    drafts_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".double-q-migration-", dir=drafts_root))
    try:
        shutil.copytree(source, temporary, dirs_exist_ok=True)
        manifest_path = temporary / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        old_version = str(manifest["version"])
        manifest["version"] = "1.0.1"
        manifest["algorithm"]["core_equations"] = [
            validate_latex(item) for item in manifest["algorithm"]["core_equations"]
        ]

        theory_path = temporary / manifest["modules"]["theory"]["file"]
        theory_text = theory_path.read_text(encoding="utf-8")
        theory_text = re.sub(
            r"(?m)^#\s+.*$", "# Double Q-Learning", theory_text, count=1
        ).replace(r"$\\-greedy$", "ε-greedy")
        theory_path.write_text(theory_text, encoding="utf-8")
        presentation = presentation_from_markdown(theory_text, "Double Q-Learning")
        presentation.update(
            {
                "title": "Double Q-Learning",
                "key_ideas": [
                    "Use one value table to select an action and the other to evaluate it.",
                    "Randomly alternate updates between Q_A and Q_B.",
                    "Combine both tables when deriving the behavior and final policy.",
                ],
                "when_to_use": [
                    "Tabular control problems where noisy maximum estimates create optimism."
                ],
                "checkpoint": [
                    {
                        "question": "When updating Q_A, which table evaluates the action selected by Q_A?",
                        "options": ["Q_A", "Q_B", "Neither table"],
                        "answer": 1,
                        "explanation": "Cross-evaluation uses Q_B to evaluate the action selected by Q_A.",
                    },
                    {
                        "question": "Why are ties in argmax broken randomly?",
                        "options": ["To avoid index bias", "To increase gamma", "To end the episode"],
                        "answer": 0,
                        "explanation": "Always selecting the first tied action would introduce systematic action-index bias.",
                    },
                    {
                        "question": "What is the target at a terminal transition?",
                        "options": ["The immediate reward", "Zero in every task", "A bootstrapped Q value"],
                        "answer": 0,
                        "explanation": "Terminal transitions do not bootstrap from a next-state estimate.",
                    },
                ],
            }
        )
        presentation = validate_theory_presentation(
            presentation, algorithm_name="Double Q-Learning"
        )
        presentation_file = "theory.presentation.json"
        (temporary / presentation_file).write_bytes(encode_theory_presentation(presentation))
        manifest["modules"]["theory"]["presentation_file"] = presentation_file

        notebook_path = temporary / manifest["modules"]["notebook"]["file"]
        notebook = nbformat.read(notebook_path, as_version=4)
        replacements = {
            0: "# Double Q-Learning\n\n## Overview\n",
            1: "## Algorithm\n\n",
            3: "## Setup\n\n",
            5: "## Implementation\n\n",
            7: "## Training\n\n## Visualization\n\n",
            9: "## Summary\n\n",
        }
        for index, prefix in replacements.items():
            notebook.cells[index].source = prefix + str(notebook.cells[index].source).lstrip("# \n")
        notebook.cells.insert(
            len(notebook.cells) - 1,
            nbformat.v4.new_markdown_cell(
                "## Exercises\n\n1. Compare fixed and decaying exploration.\n"
                "2. Measure the difference between Q_A and Q_B after training."
            ),
        )
        buffer = io.StringIO()
        nbformat.write(notebook, buffer)
        notebook_payload, requirements = normalize_and_validate_notebook(
            buffer.getvalue().encode("utf-8"),
            "double-q-learning",
            "1.0.1",
            require_template=True,
        )
        notebook_path.write_bytes(notebook_payload)
        manifest["modules"]["notebook"].update(
            {
                "requirements": requirements,
                "validation": "static-only-not-executed",
            }
        )

        animation = manifest["modules"].get("animation")
        if animation is not None:
            animation["formula"] = double_q_learning_core_latex()

        generation = manifest.setdefault("generation", {})
        generation["revision_of"] = {
            "id": "double-q-learning",
            "version": old_version,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "created_by": reviewer,
        }
        for module, state in manifest["review"]["modules"].items():
            state.update(
                {
                    "status": "awaiting_review",
                    "reviewer": reviewer,
                    "note": "Migrated to the unified imported learning experience.",
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }
            )
        _write_json(manifest_path, manifest)

        intermediate = validate_source_directory(temporary)
        if not intermediate.valid or intermediate.manifest is None:
            raise ValueError(
                "; ".join(issue.message for issue in intermediate.errors)
            )
        candidate = InstalledAlgorithm(
            manifest=intermediate.manifest,
            path=temporary,
            dependencies=intermediate.dependencies,
        )
        spec = load_isolated_spec(candidate)
        _write_json(temporary / "experiment_spec.json", spec)
        manifest["modules"]["experiment"]["spec_file"] = "experiment_spec.json"
        _write_json(manifest_path, manifest)

        final_report = validate_source_directory(temporary)
        if not final_report.valid:
            raise ValueError("; ".join(issue.message for issue in final_report.errors))
        temporary.replace(destination)
        return destination
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument(
        "--drafts-root",
        type=Path,
        default=PROJECT_ROOT / "algorithm_packages" / "drafts",
    )
    parser.add_argument("--reviewer", default="Codex migration")
    args = parser.parse_args()
    print(migrate(args.source.resolve(), args.drafts_root.resolve(), reviewer=args.reviewer))


if __name__ == "__main__":
    main()
