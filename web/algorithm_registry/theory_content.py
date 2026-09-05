from __future__ import annotations

import json
import re
from typing import Any, Mapping


THEORY_PRESENTATION_VERSION = 1


def normalize_theory_title(markdown: str, algorithm_name: str) -> str:
    title = f"# {algorithm_name}"
    if re.search(r"(?m)^#\s+.+?\s*$", markdown):
        return re.sub(r"(?m)^#\s+.+?\s*$", title, markdown, count=1)
    return f"{title}\n\n{markdown.lstrip()}"


def _markdown_blocks(markdown: str) -> tuple[str, list[tuple[str, str]]]:
    headings = list(re.finditer(r"(?m)^##\s+(.+?)\s*$", markdown))
    first_heading = headings[0].start() if headings else len(markdown)
    preamble = re.sub(r"(?m)^#\s+.+?\s*$", "", markdown[:first_heading]).strip()
    blocks: list[tuple[str, str]] = []
    for index, match in enumerate(headings):
        end = (
            headings[index + 1].start()
            if index + 1 < len(headings)
            else len(markdown)
        )
        title = match.group(1).strip()
        body = markdown[match.end():end].strip()
        blocks.append((title, f"## {title}\n\n{body}".strip()))
    return preamble, blocks


def presentation_from_markdown(
    markdown: str,
    algorithm_name: str,
    *,
    preserve: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a conservative four-tab fallback for legacy theory Markdown."""

    title_match = re.search(r"(?m)^#\s+(.+?)\s*$", markdown)
    title = title_match.group(1).strip() if title_match else algorithm_name
    preamble, blocks = _markdown_blocks(markdown)
    concept_parts = [preamble] if preamble else []
    math_parts: list[str] = []
    pseudocode_parts: list[str] = []
    for heading, block in blocks:
        normalized_heading = heading.lower()
        if any(name in normalized_heading for name in ("equation", "math", "update")):
            math_parts.append(block)
        elif any(name in normalized_heading for name in ("pseudocode", "algorithm steps")):
            pseudocode_parts.append(block)
        else:
            concept_parts.append(block)
    concept = "\n\n".join(concept_parts).strip() or markdown.strip()
    result = {
        "schema_version": THEORY_PRESENTATION_VERSION,
        "title": title,
        "key_ideas": list(preserve.get("key_ideas", [])) if preserve else [],
        "when_to_use": list(preserve.get("when_to_use", [])) if preserve else [],
        "concept_markdown": concept,
        "math_markdown": "\n\n".join(math_parts).strip()
        or "Mathematical details are included in the lesson text.",
        "pseudocode_markdown": "\n\n".join(pseudocode_parts).strip()
        or "Pseudocode is not separately available.",
        "checkpoint": list(preserve.get("checkpoint", [])) if preserve else [],
    }
    return validate_theory_presentation(result, algorithm_name=algorithm_name)


def validate_theory_presentation(
    value: Mapping[str, Any], *, algorithm_name: str
) -> dict[str, Any]:
    required_strings = (
        "title",
        "concept_markdown",
        "math_markdown",
        "pseudocode_markdown",
    )
    normalized = dict(value)
    normalized["schema_version"] = THEORY_PRESENTATION_VERSION
    for field in required_strings:
        item = normalized.get(field)
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"theory presentation {field} must be a non-empty string")
        normalized[field] = item.strip()
    if normalized["title"].strip().lower() != algorithm_name.strip().lower():
        raise ValueError("theory presentation title must match the algorithm name")
    for field in ("key_ideas", "when_to_use"):
        items = normalized.get(field, [])
        if not isinstance(items, list) or not all(
            isinstance(item, str) and item.strip() for item in items
        ):
            raise ValueError(f"theory presentation {field} must be a list of strings")
        normalized[field] = [item.strip() for item in items]
    checkpoint = normalized.get("checkpoint", [])
    if not isinstance(checkpoint, list):
        raise ValueError("theory presentation checkpoint must be a list")
    checked_questions = []
    for index, question in enumerate(checkpoint):
        if not isinstance(question, Mapping):
            raise ValueError(f"checkpoint question {index} must be an object")
        prompt = question.get("question")
        options = question.get("options")
        answer = question.get("answer")
        explanation = question.get("explanation")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"checkpoint question {index} needs a question")
        if not isinstance(options, list) or len(options) < 2 or not all(
            isinstance(item, str) and item.strip() for item in options
        ):
            raise ValueError(f"checkpoint question {index} needs at least two options")
        if (
            isinstance(answer, bool)
            or not isinstance(answer, int)
            or not 0 <= answer < len(options)
        ):
            raise ValueError(f"checkpoint question {index} answer is out of range")
        if not isinstance(explanation, str) or not explanation.strip():
            raise ValueError(f"checkpoint question {index} needs an explanation")
        checked_questions.append(
            {
                "question": prompt.strip(),
                "options": [item.strip() for item in options],
                "answer": answer,
                "explanation": explanation.strip(),
            }
        )
    normalized["checkpoint"] = checked_questions
    return normalized


def encode_theory_presentation(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
