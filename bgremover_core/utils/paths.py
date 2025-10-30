"""Common path discovery utilities shared by all runtimes."""
from __future__ import annotations

import sys
from collections.abc import Iterable
from pathlib import Path


def _discover_base_dir() -> Path:
    """Return the base directory that stores runtime assets.

    When packaged with PyInstaller ``sys.frozen`` is present and the
    executable path should be treated as the runtime root. During normal
    development the repository root is used instead so that assets beside the
    source tree are discovered reliably.
    """

    if getattr(sys, "frozen", False):  # pragma: no cover - exercised in builds
        return Path(sys.executable).resolve().parent

    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    # Fallback to the package root when a project marker cannot be found.
    return current.parents[2]


BASE_DIR: Path = _discover_base_dir()
CONFIG_DIR: Path = BASE_DIR / "config"
INPUT_DIR: Path = BASE_DIR / "input"
OUTPUT_DIR: Path = BASE_DIR / "output"
MODELS_DIR: Path = BASE_DIR / "models"
CONFIG_FILE: Path = CONFIG_DIR / "config.json"

RUNTIME_DIRECTORIES: tuple[Path, ...] = (
    CONFIG_DIR,
    INPUT_DIR,
    OUTPUT_DIR,
    MODELS_DIR,
)


def ensure_runtime_directories(
    directories: Iterable[Path] = RUNTIME_DIRECTORIES,
) -> None:
    """Ensure the configured runtime directories in ``directories`` exist."""

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


ensure_runtime_directories()


__all__ = [
    "BASE_DIR",
    "CONFIG_DIR",
    "CONFIG_FILE",
    "INPUT_DIR",
    "MODELS_DIR",
    "OUTPUT_DIR",
    "RUNTIME_DIRECTORIES",
    "ensure_runtime_directories",
]

