"""Filesystem helpers shared by multiple runtimes."""
from __future__ import annotations

from pathlib import Path


def ensure_output_directory(path: Path | str) -> Path:
    """Ensure that ``path`` exists as a directory and return it."""

    path = Path(path).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_batch_output_dir(
    input_dir: Path | str, output_dir: Path | str | None = None
) -> Path:
    """Return the output directory for batch jobs following project conventions."""

    input_dir = Path(input_dir).expanduser()
    if output_dir is not None:
        return ensure_output_directory(output_dir)
    sibling = input_dir.parent / "output"
    return ensure_output_directory(sibling)


__all__ = ["ensure_output_directory", "resolve_batch_output_dir"]
