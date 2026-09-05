from __future__ import annotations

import os
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


NOTEBOOK_PUBLICATION_NOTICE = """
- **Built-in notebooks** are read from this repository's `notebook/` directory
  and can open directly in Colab.
- **Imported or Agent-generated notebooks** stay local after review and
  installation. The platform does not upload them to GitHub automatically.
- To publish an imported notebook, download it, save the identical file under
  `notebook/imported/`, then commit and push it with Git. A Colab button appears
  only when the repository copy matches the installed notebook.
- Colab fetches the `.ipynb` file from GitHub. The repository copy must therefore
  be available on the configured public repository and branch.
""".strip()


@dataclass(frozen=True)
class GitHubNotebookConfiguration:
    owner: str = "Changcheng-Huang"
    repository: str = "rl_study"
    branch: str = "main"


@dataclass(frozen=True)
class ManualNotebookStatus:
    status: str
    relative_path: str
    local_path: Path
    colab_url: str | None
    message: str


def github_notebook_configuration(
    environ: Mapping[str, str] | None = None,
) -> GitHubNotebookConfiguration:
    values = os.environ if environ is None else environ
    return GitHubNotebookConfiguration(
        owner=values.get(
            "RLAE_NOTEBOOK_GITHUB_OWNER", "Changcheng-Huang"
        ).strip()
        or "Changcheng-Huang",
        repository=values.get("RLAE_NOTEBOOK_GITHUB_REPO", "rl_study").strip()
        or "rl_study",
        branch=values.get("RLAE_NOTEBOOK_GITHUB_BRANCH", "main").strip()
        or "main",
    )


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def colab_url(
    relative_path: str,
    *,
    environ: Mapping[str, str] | None = None,
) -> str:
    configuration = github_notebook_configuration(environ)
    encoded_path = urllib.parse.quote(relative_path.strip("/"), safe="/")
    branch = urllib.parse.quote(configuration.branch, safe="")
    return (
        f"https://colab.research.google.com/github/{configuration.owner}/"
        f"{configuration.repository}/blob/{branch}/{encoded_path}"
    )


def builtin_colab_url(
    file_name: str,
    *,
    environ: Mapping[str, str] | None = None,
) -> str:
    encoded_name = Path(file_name).name
    return colab_url(f"notebook/{encoded_name}", environ=environ)


def manual_notebook_relative_path(algorithm_id: str, version: str) -> str:
    return f"notebook/imported/{algorithm_id}-{version}.ipynb"


def manual_publication_for(
    algorithm,
    *,
    environ: Mapping[str, str] | None = None,
    repository_root: Path | None = None,
) -> ManualNotebookStatus:
    relative_path = manual_notebook_relative_path(
        algorithm.manifest.algorithm_id,
        algorithm.manifest.version,
    )
    local_path = (repository_root or project_root()) / relative_path
    if algorithm.manifest.notebook is None:
        return ManualNotebookStatus(
            status="not-applicable",
            relative_path=relative_path,
            local_path=local_path,
            colab_url=None,
            message="This algorithm has no notebook.",
        )
    generated_path = algorithm.path / algorithm.manifest.notebook["file"]
    if not local_path.is_file():
        return ManualNotebookStatus(
            status="missing",
            relative_path=relative_path,
            local_path=local_path,
            colab_url=None,
            message=(
                "Download the reviewed notebook, save it at the repository path "
                "shown below, then commit and push it."
            ),
        )
    try:
        matches = local_path.read_bytes() == generated_path.read_bytes()
    except OSError as exc:
        return ManualNotebookStatus(
            status="unreadable",
            relative_path=relative_path,
            local_path=local_path,
            colab_url=None,
            message=f"The manual publication copy cannot be read: {exc}",
        )
    if not matches:
        return ManualNotebookStatus(
            status="stale",
            relative_path=relative_path,
            local_path=local_path,
            colab_url=None,
            message=(
                "The repository copy differs from the installed notebook. Replace "
                "it with the reviewed download before committing."
            ),
        )
    return ManualNotebookStatus(
        status="ready",
        relative_path=relative_path,
        local_path=local_path,
        colab_url=colab_url(relative_path, environ=environ),
        message=(
            "The repository copy matches. Commit and push it if the Colab link "
            "has not been published yet."
        ),
    )
