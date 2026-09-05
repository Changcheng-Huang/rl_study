from __future__ import annotations

import re


class LatexValidationError(ValueError):
    """Raised when learner-facing mathematics is not canonical LaTeX."""


_PLAIN_TOKENS = (
    (re.compile(r"(?<![\\A-Za-z])argmax(?![A-Za-z])"), r"\\arg\\max"),
    (re.compile(r"(?<![\\A-Za-z])argmin(?![A-Za-z])"), r"\\arg\\min"),
    (re.compile(r"(?<![\\A-Za-z])(alpha|beta|gamma|delta|epsilon|theta|lambda|pi)(?![A-Za-z])"), r"\\\1"),
)


def normalize_latex(value: str, *, multiline: bool = True) -> str:
    """Normalize common programming-style equations into st.latex-ready text.

    The stored representation deliberately excludes Markdown math delimiters.
    This is a compatibility normalizer for legacy package data; new Agent
    prompts and editors are expected to produce canonical LaTeX directly.
    """

    if not isinstance(value, str):
        raise LatexValidationError("formula must be a string")
    cleaned = value.strip()
    if cleaned.startswith("$$") and cleaned.endswith("$$") and len(cleaned) >= 4:
        cleaned = cleaned[2:-2].strip()
    elif cleaned.startswith("$") and cleaned.endswith("$") and len(cleaned) >= 2:
        cleaned = cleaned[1:-1].strip()
    cleaned = cleaned.replace("<-", r"\leftarrow")
    cleaned = re.sub(r"\s+\*\s+", r" \\cdot ", cleaned)
    for pattern, replacement in _PLAIN_TOKENS:
        cleaned = pattern.sub(replacement, cleaned)

    if multiline and ";" in cleaned and not cleaned.startswith(r"\begin{"):
        rows = [part.strip() for part in cleaned.split(";") if part.strip()]
        aligned: list[str] = []
        for row in rows:
            if r"\leftarrow" in row:
                row = row.replace(r"\leftarrow", r"&\leftarrow", 1)
            elif "=" in row:
                row = row.replace("=", "&=", 1)
            aligned.append(row)
        cleaned = r"\begin{aligned}" + " \\\\ ".join(aligned) + r"\end{aligned}"
    return cleaned


def validate_latex(value: str, *, allow_empty: bool = False) -> str:
    cleaned = normalize_latex(value)
    if not cleaned:
        if allow_empty:
            return ""
        raise LatexValidationError("formula must not be empty")
    if "$" in cleaned:
        raise LatexValidationError("formula must not contain $ delimiters")
    if "<-" in cleaned or re.search(r"\s+\*\s+", cleaned):
        raise LatexValidationError("formula contains programming-style operators")
    for word in ("argmax", "argmin", "alpha", "gamma", "epsilon"):
        if re.search(rf"(?<![\\A-Za-z]){word}(?![A-Za-z])", cleaned):
            raise LatexValidationError(f"formula contains non-LaTeX token '{word}'")
    return cleaned


def double_q_learning_core_latex() -> str:
    """Canonical symmetric Double Q-Learning update for the animation card."""

    return r"""\begin{aligned}
a^* &= \arg\max_{a'} Q_A(s',a') \\
y_A &= \begin{cases}r,&\text{terminal}\\r+\gamma Q_B(s',a^*),&\text{otherwise}\end{cases} \\
Q_A(s,a) &\leftarrow Q_A(s,a)+\alpha\bigl(y_A-Q_A(s,a)\bigr) \\
b^* &= \arg\max_{a'} Q_B(s',a') \\
y_B &= \begin{cases}r,&\text{terminal}\\r+\gamma Q_A(s',b^*),&\text{otherwise}\end{cases} \\
Q_B(s,a) &\leftarrow Q_B(s,a)+\alpha\bigl(y_B-Q_B(s,a)\bigr)
\end{aligned}"""
