from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import streamlit as st

from algorithm_registry.theory_content import (
    presentation_from_markdown,
    validate_theory_presentation,
)
from .common import render_quiz, right_card


def load_theory_presentation(algorithm) -> tuple[dict[str, Any], bool]:
    theory_path = algorithm.path / algorithm.manifest.theory_file
    markdown = theory_path.read_text(encoding="utf-8")
    declaration = algorithm.manifest.modules.get("theory", {})
    relative = declaration.get("presentation_file")
    if isinstance(relative, str):
        presentation_path = algorithm.path / relative
        if presentation_path.is_file():
            value = json.loads(presentation_path.read_text(encoding="utf-8"))
            return validate_theory_presentation(
                value, algorithm_name=algorithm.manifest.name
            ), True
    fallback = presentation_from_markdown(markdown, algorithm.manifest.name)
    fallback["title"] = algorithm.manifest.name
    return fallback, False


def render_imported_theory(algorithm) -> None:
    presentation, structured = load_theory_presentation(algorithm)
    st.subheader(presentation["title"])
    st.caption(algorithm.manifest.summary)
    left, right = st.columns([1.6, 1.0], gap="large")
    with right:
        right_card(
            "Key ideas",
            bullets=presentation["key_ideas"] or ["See the Concept tab for the complete lesson."],
        )
        right_card(
            "When to use",
            bullets=presentation["when_to_use"] or ["Confirm the assumptions in the Concept tab."],
        )
        if not structured:
            st.caption("Legacy Markdown was adapted to the unified layout.")
    with left:
        concept_tab, math_tab, pseudocode_tab, checkpoint_tab = st.tabs(
            ["Concept", "Math", "Pseudocode", "Checkpoint"]
        )
        with concept_tab:
            st.markdown(presentation["concept_markdown"])
        with math_tab:
            st.markdown(presentation["math_markdown"])
        with pseudocode_tab:
            st.markdown(presentation["pseudocode_markdown"])
        with checkpoint_tab:
            questions = [
                {
                    "q": item["question"],
                    "options": item["options"],
                    "answer": item["answer"],
                    "explain": item["explanation"],
                }
                for item in presentation["checkpoint"]
            ]
            if questions:
                render_quiz(f"imported_{algorithm.manifest.algorithm_id}", questions)
            else:
                st.info("Checkpoint questions are not available for this legacy lesson yet.")
