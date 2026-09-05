from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


class NotebookPublishError(RuntimeError):
    pass


class NotebookPublishConflict(NotebookPublishError):
    pass


@dataclass(frozen=True)
class GitHubNotebookConfiguration:
    owner: str
    repository: str
    branch: str = "main"
    root: str = "notebooks"
    token: str = ""

    @property
    def configured(self) -> bool:
        return bool(self.owner and self.repository and self.token)


def github_notebook_configuration(
    environ: Mapping[str, str] | None = None,
) -> GitHubNotebookConfiguration:
    values = os.environ if environ is None else environ
    return GitHubNotebookConfiguration(
        owner=values.get("RLAE_NOTEBOOK_GITHUB_OWNER", "").strip(),
        repository=values.get("RLAE_NOTEBOOK_GITHUB_REPO", "").strip(),
        branch=values.get("RLAE_NOTEBOOK_GITHUB_BRANCH", "main").strip() or "main",
        root=values.get("RLAE_NOTEBOOK_GITHUB_ROOT", "notebooks").strip().strip("/") or "notebooks",
        token=values.get("RLAE_NOTEBOOK_GITHUB_TOKEN", "").strip(),
    )


def _request(
    url: str,
    configuration: GitHubNotebookConfiguration,
    *,
    payload: Mapping[str, Any] | None = None,
):
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        url,
        data=data,
        method="PUT" if payload is not None else "GET",
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {configuration.token}",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "rlae-notebook-publisher",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        return json.loads(response.read().decode("utf-8"))


def publish_notebook_bytes(
    *,
    algorithm_id: str,
    version: str,
    content: bytes,
    configuration: GitHubNotebookConfiguration,
) -> dict[str, str]:
    if not configuration.configured:
        raise NotebookPublishError("GitHub notebook publishing is not configured")
    relative = f"{configuration.root}/{algorithm_id}/{version}/notebook.ipynb"
    encoded_path = urllib.parse.quote(relative, safe="/")
    api_url = (
        f"https://api.github.com/repos/{configuration.owner}/"
        f"{configuration.repository}/contents/{encoded_path}"
    )
    existing = None
    try:
        existing = _request(
            api_url + "?ref=" + urllib.parse.quote(configuration.branch, safe=""),
            configuration,
        )
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            raise NotebookPublishError(f"GitHub lookup failed with HTTP {exc.code}") from exc
    except OSError as exc:
        raise NotebookPublishError(f"GitHub lookup failed: {exc}") from exc

    if isinstance(existing, Mapping):
        try:
            remote_content = base64.b64decode(str(existing.get("content", "")), validate=False)
        except Exception as exc:
            raise NotebookPublishError("GitHub returned invalid notebook content") from exc
        if remote_content != content:
            raise NotebookPublishConflict(
                "a different notebook already exists for this algorithm version"
            )
        html_url = str(existing.get("html_url") or "")
    else:
        try:
            result = _request(
                api_url,
                configuration,
                payload={
                    "message": f"Publish {algorithm_id} {version} notebook",
                    "content": base64.b64encode(content).decode("ascii"),
                    "branch": configuration.branch,
                },
            )
        except urllib.error.HTTPError as exc:
            raise NotebookPublishError(f"GitHub upload failed with HTTP {exc.code}") from exc
        except OSError as exc:
            raise NotebookPublishError(f"GitHub upload failed: {exc}") from exc
        html_url = str(result.get("content", {}).get("html_url") or "")

    github_url = html_url or (
        f"https://github.com/{configuration.owner}/{configuration.repository}/"
        f"blob/{configuration.branch}/{encoded_path}"
    )
    colab_url = (
        f"https://colab.research.google.com/github/{configuration.owner}/"
        f"{configuration.repository}/blob/{configuration.branch}/{encoded_path}"
    )
    return {"path": relative, "github_url": github_url, "colab_url": colab_url}


def default_publication_registry() -> Path:
    return Path(__file__).resolve().parents[2] / "algorithm_packages" / ".state" / "notebook_publications.json"


def load_publications(path: Path | None = None) -> dict[str, Any]:
    target = path or default_publication_registry()
    if not target.is_file():
        return {}
    try:
        value = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def record_publication(
    algorithm_id: str,
    version: str,
    value: Mapping[str, Any],
    *,
    path: Path | None = None,
) -> None:
    target = path or default_publication_registry()
    target.parent.mkdir(parents=True, exist_ok=True)
    records = load_publications(target)
    records[f"{algorithm_id}@{version}"] = {
        **dict(value),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(target)


def publication_for(
    algorithm_id: str,
    version: str,
    *,
    path: Path | None = None,
) -> Mapping[str, Any] | None:
    return load_publications(path).get(f"{algorithm_id}@{version}")


def publish_installed_notebook(
    algorithm,
    *,
    environ: Mapping[str, str] | None = None,
    registry_path: Path | None = None,
) -> Mapping[str, Any]:
    configuration = github_notebook_configuration(environ)
    if algorithm.manifest.notebook is None:
        result = {"status": "not-applicable", "message": "Algorithm has no notebook."}
    elif not configuration.configured:
        result = {"status": "not-configured", "message": "GitHub notebook publishing is not configured."}
    else:
        notebook_path = algorithm.path / algorithm.manifest.notebook["file"]
        try:
            urls = publish_notebook_bytes(
                algorithm_id=algorithm.manifest.algorithm_id,
                version=algorithm.manifest.version,
                content=notebook_path.read_bytes(),
                configuration=configuration,
            )
        except (NotebookPublishError, OSError) as exc:
            result = {"status": "failed", "message": str(exc)}
        else:
            result = {"status": "published", **urls}
    record_publication(
        algorithm.manifest.algorithm_id,
        algorithm.manifest.version,
        result,
        path=registry_path,
    )
    return result


def builtin_colab_url(file_name: str, *, environ: Mapping[str, str] | None = None) -> str | None:
    configuration = github_notebook_configuration(environ)
    if not configuration.owner or not configuration.repository:
        return None
    encoded = urllib.parse.quote(file_name, safe="")
    return (
        f"https://colab.research.google.com/github/{configuration.owner}/"
        f"{configuration.repository}/blob/{configuration.branch}/notebook/{encoded}"
    )
