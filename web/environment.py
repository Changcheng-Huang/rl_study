from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv


def load_project_environment(path: str | Path | None = None) -> bool:
    """Load the project .env without overriding process-level deployment values."""

    dotenv_path = (
        Path(path)
        if path is not None
        else Path(__file__).resolve().parents[1] / ".env"
    )
    return load_dotenv(dotenv_path=dotenv_path, override=False)
