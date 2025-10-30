"""Backwards-compatible exports for runtime filesystem paths."""
from __future__ import annotations

from .utils.paths import (
    BASE_DIR,
    CONFIG_DIR,
    CONFIG_FILE,
    INPUT_DIR,
    MODELS_DIR,
    OUTPUT_DIR,
    RUNTIME_DIRECTORIES,
    ensure_runtime_directories,
)

# Retain the old name for compatibility with existing imports throughout the
# project. ``BASE_DIR`` reflects the canonical runtime directory location.
PROJECT_ROOT = BASE_DIR

# Ensure directories exist when importing this module directly. The helper call
# is idempotent, so repeated imports are safe.
ensure_runtime_directories(RUNTIME_DIRECTORIES)

__all__ = [
    "BASE_DIR",
    "CONFIG_DIR",
    "CONFIG_FILE",
    "INPUT_DIR",
    "MODELS_DIR",
    "OUTPUT_DIR",
    "PROJECT_ROOT",
    "RUNTIME_DIRECTORIES",
    "ensure_runtime_directories",
]
