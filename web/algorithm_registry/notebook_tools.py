from __future__ import annotations

import ast
import io
import re
from typing import Any

import nbformat


NOTEBOOK_DEPENDENCIES = {
    "numpy": "numpy>=2.2",
    "pandas": "pandas>=2.3",
    "matplotlib": "matplotlib>=3.10",
    "seaborn": "seaborn>=0.13",
    "gymnasium": "gymnasium>=1.2",
    "scipy": "scipy>=1.15",
}
_STDLIB = {
    "collections", "dataclasses", "functools", "itertools", "json", "math",
    "random", "statistics", "typing", "time", "copy", "enum",
}
_FORBIDDEN = re.compile(
    r"\b(requests|urllib|httpx|socket|subprocess|os\.system|pathlib\.Path\([^)]*\)\.(write|unlink)|open\s*\([^)]*,\s*['\"]?[wax+])"
)


def normalize_and_validate_notebook(
    payload: bytes,
    algorithm_id: str,
    version: str,
    *,
    require_template: bool = False,
) -> tuple[bytes, list[dict[str, str]]]:
    try:
        notebook = nbformat.read(io.StringIO(payload.decode("utf-8")), as_version=4)
    except Exception as exc:
        raise ValueError(f"notebook is not valid nbformat: {exc}") from exc

    if require_template and not 8 <= len(notebook.cells) <= 24:
        raise ValueError("generated notebook must contain 8-24 cells")
    if require_template:
        markdown = "\n".join(
            str(cell.source) for cell in notebook.cells if cell.cell_type == "markdown"
        ).lower()
        missing = [
            section
            for section in (
                "overview", "algorithm", "setup", "implementation", "training",
                "visualization", "exercises", "summary",
            )
            if not re.search(rf"(?m)^#+\s+.*\b{section}\b", markdown)
        ]
        if missing:
            raise ValueError("generated notebook is missing sections: " + ", ".join(missing))

    imported: set[str] = set()
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type != "code":
            continue
        source = str(cell.source)
        if "rlae-platform-dependencies" in cell.metadata.get("tags", []):
            continue
        if source.lstrip().startswith(("%", "!")):
            raise ValueError(f"notebook cell {index + 1} contains Agent-supplied magic or shell code")
        if _FORBIDDEN.search(source):
            raise ValueError(f"notebook cell {index + 1} contains network, process, or file-write code")
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            raise ValueError(f"notebook cell {index + 1} has invalid Python: {exc}") from exc
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
    unsupported = sorted(imported - _STDLIB - set(NOTEBOOK_DEPENDENCIES))
    if unsupported:
        raise ValueError("notebook imports unsupported modules: " + ", ".join(unsupported))

    dependencies = [
        {"package": NOTEBOOK_DEPENDENCIES[name], "import": name}
        for name in sorted(imported & set(NOTEBOOK_DEPENDENCIES))
    ]
    if dependencies:
        packages = " ".join(item["package"] for item in dependencies)
        install = nbformat.v4.new_code_cell(
            f"%pip install -q {packages}",
            metadata={"tags": ["rlae-platform-dependencies"]},
        )
        if notebook.cells and "rlae-platform-dependencies" in notebook.cells[0].metadata.get("tags", []):
            notebook.cells[0] = install
        else:
            notebook.cells.insert(0, install)
    notebook.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook.metadata["language_info"] = {"name": "python", "version": "3"}
    notebook.metadata["algorithm_id"] = algorithm_id
    notebook.metadata["algorithm_version"] = version
    notebook.metadata["rlae_validation"] = "static-only-not-executed"
    nbformat.validate(notebook)
    buffer = io.StringIO()
    nbformat.write(notebook, buffer)
    return buffer.getvalue().encode("utf-8"), dependencies
