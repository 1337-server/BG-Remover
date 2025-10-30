"""Shared filesystem paths used across every runtime variant."""
from __future__ import annotations

from pathlib import Path


def _discover_project_root() -> Path:
    """Return the repository root inferred from the current file location."""

    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    # Fallback to the package directory when a marker is not found.
    return current.parents[1]


PROJECT_ROOT = _discover_project_root()

INPUT_DIR = PROJECT_ROOT / "input"
OUTPUT_DIR = PROJECT_ROOT / "output"
CONFIG_DIR = PROJECT_ROOT / "config"
CONFIG_FILE = CONFIG_DIR / "config.json"

for directory in (INPUT_DIR, OUTPUT_DIR, CONFIG_DIR):
    directory.mkdir(parents=True, exist_ok=True)
