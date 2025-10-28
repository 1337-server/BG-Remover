"""Application factory for the background remover web service."""
from __future__ import annotations

import logging
import os
from typing import Any, Mapping, MutableMapping

from app.services import runtime_compat
from app.services.bg_remove import ensure_global_session

try:  # pragma: no cover - optional during testing
    from flask import Flask, request
except ModuleNotFoundError as exc:  # pragma: no cover - guard for test imports
    Flask = None  # type: ignore
    _FLASK_IMPORT_ERROR = exc
else:
    _FLASK_IMPORT_ERROR = None


LOGGER = logging.getLogger(__name__)


def _parse_bool(value: Any, default: bool) -> bool:
    """Return a ``bool`` for ``value`` with a fallback to ``default``."""

    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _read_env_int(name: str, default: int) -> int:
    """Return ``name`` converted to ``int`` or ``default`` when parsing fails."""

    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        return int(raw_value)
    except ValueError:
        LOGGER.warning("Environment variable %s is not a valid integer: %s", name, raw_value)
        return default


def _load_environment_config() -> MutableMapping[str, Any]:
    """Build accelerator-related configuration values from environment variables."""

    return {
        "BG_ACCELERATOR": os.getenv("BG_ACCELERATOR", "auto"),
        "BG_CUDA_DEVICE_ID": _read_env_int("BG_CUDA_DEVICE_ID", 0),
        "BG_WARN_ON_CPU": _parse_bool(os.getenv("BG_WARN_ON_CPU"), True),
    }


def create_app(config_overrides: Mapping[str, Any] | None = None) -> "Flask":
    """Create and configure the Flask application."""

    if Flask is None:  # pragma: no cover - executed only when Flask missing
        raise RuntimeError("Flask is required to create the web application.") from _FLASK_IMPORT_ERROR

    app = Flask(__name__)

    config_values: MutableMapping[str, Any] = _load_environment_config()
    if config_overrides:
        config_values.update(config_overrides)

    app.config.setdefault("BG_ACCELERATOR", str(config_values.get("BG_ACCELERATOR", "auto")))
    app.config.setdefault("BG_CUDA_DEVICE_ID", int(config_values.get("BG_CUDA_DEVICE_ID", 0)))
    app.config.setdefault("BG_WARN_ON_CPU", _parse_bool(config_values.get("BG_WARN_ON_CPU"), True))

    # Initialise the runtime stack before creating the rembg session. This
    # ensures NumPy/ONNXRuntime compatibility issues are surfaced early.
    runtime_compat.ensure_runtime_ready()
    # Initialise the global background removal session once at startup using the resolved config.
    ensure_global_session(config=app.config)

    from app.routes.image_converter import image_converter_bp

    app.register_blueprint(image_converter_bp)

    @app.after_request
    def add_static_cache_headers(response):
        """Add caching headers to static asset responses to improve load performance."""

        if request.path.startswith("/static/"):
            response.headers.setdefault("Cache-Control", "public, max-age=31536000, immutable")
        return response

    return app


