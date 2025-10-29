"""Shared public API for the background remover core package."""
from __future__ import annotations

from .config import Config, ConfigError, init_logging, load_config, persist_config
from .models.loader import BackgroundRemovalSession, detect_providers, get_session
from .models.specs import MODEL_SPECS, ModelSpec
from .processing.pipeline import Report, ReportEntry, process_folder, remove_background

__all__ = [
    "BackgroundRemovalSession",
    "Config",
    "ConfigError",
    "MODEL_SPECS",
    "ModelSpec",
    "Report",
    "ReportEntry",
    "detect_providers",
    "get_session",
    "init_logging",
    "load_config",
    "persist_config",
    "process_folder",
    "remove_background",
]
