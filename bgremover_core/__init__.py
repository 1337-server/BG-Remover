"""Shared public API for the background remover core package."""
from __future__ import annotations

from .config import Config, ConfigError, init_logging, load_config, persist_config
from .models.loader import BackgroundRemovalSession, detect_providers, get_session
from .models.specs import MODEL_SPECS, ModelSpec
from .processing.pipeline import (
    ProcessingDebugInfo,
    ProcessingResult,
    Report,
    ReportEntry,
    preprocess,
    process_folder,
    process_image,
    remove_background,
)

__all__ = [
    "BackgroundRemovalSession",
    "Config",
    "ConfigError",
    "MODEL_SPECS",
    "ModelSpec",
    "ProcessingDebugInfo",
    "ProcessingResult",
    "Report",
    "ReportEntry",
    "detect_providers",
    "get_session",
    "init_logging",
    "load_config",
    "persist_config",
    "preprocess",
    "process_folder",
    "process_image",
    "remove_background",
]
