"""Application factory for the background remover web service."""
from __future__ import annotations

import logging
import os
from collections.abc import Mapping, MutableMapping
from typing import TYPE_CHECKING, Any

from app.services import model_registry, runtime_compat
from app.services.bg_remove import ensure_global_session

if TYPE_CHECKING:  # pragma: no cover - typing assistance only
    from flask import Flask


LOGGER = logging.getLogger(__name__)

_BG_CONFIG_KEYS = {"BG_ACCELERATOR", "BG_CUDA_DEVICE_ID", "BG_WARN_ON_CPU"}


def _parse_bool(value: Any, default: bool) -> bool:
    """Return ``value`` interpreted as a boolean."""

    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    try:
        return bool(int(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _parse_int(value: Any, default: int) -> int:
    """Return ``value`` interpreted as an integer."""

    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _normalise_mode(value: Any) -> str:
    """Return a lower-case accelerator mode string."""

    if value is None:
        return "auto"
    mode = str(value).strip().lower()
    return mode if mode in {"auto", "cuda", "cpu"} else "auto"


def _build_config(overrides: Mapping[str, Any] | None) -> MutableMapping[str, Any]:
    """Return the application configuration with accelerator settings applied."""

    config: MutableMapping[str, Any] = {
        "BG_ACCELERATOR": _normalise_mode(os.getenv("BG_ACCELERATOR", "auto")),
        "BG_CUDA_DEVICE_ID": _parse_int(os.getenv("BG_CUDA_DEVICE_ID"), 0),
        "BG_WARN_ON_CPU": _parse_bool(os.getenv("BG_WARN_ON_CPU"), True),
    }

    if overrides:
        for key, value in overrides.items():
            if key not in _BG_CONFIG_KEYS:
                config[key] = value
                continue
            if key == "BG_ACCELERATOR":
                config[key] = _normalise_mode(value)
            elif key == "BG_CUDA_DEVICE_ID":
                config[key] = _parse_int(value, config[key])
            elif key == "BG_WARN_ON_CPU":
                config[key] = _parse_bool(value, config[key])

    return config


def create_app(config_overrides: Mapping[str, Any] | None = None) -> Flask:
    """Create and configure the Flask application."""

    try:  # pragma: no cover - executed only when Flask missing
        from flask import Flask
        from flask import request as flask_request
    except ModuleNotFoundError as exc:
        raise RuntimeError("Flask is required to create the web application.") from exc

    app = Flask(__name__)
    app.config.from_mapping(_build_config(config_overrides))

    # Initialise the runtime stack before creating the rembg session. This
    # ensures NumPy/ONNXRuntime compatibility issues are surfaced early.
    runtime_compat.ensure_runtime_ready()
    # Warm up ONNX models to avoid cold-start latency before requests arrive.
    model_registry.preload_models(config=app.config)
    # Initialise the global background removal session once at startup.
    ensure_global_session(config=app.config)

    from app.routes.image_converter import image_converter_bp

    app.register_blueprint(image_converter_bp)

    @app.after_request
    def add_static_cache_headers(response):
        """Add caching headers to static asset responses to improve load performance."""

        if flask_request.path.startswith("/static/"):
            response.headers.setdefault("Cache-Control", "public, max-age=31536000, immutable")
        return response

    LOGGER.info(
        "Application configured with accelerator=%s (device %s, warn_on_cpu=%s)",
        app.config.get("BG_ACCELERATOR"),
        app.config.get("BG_CUDA_DEVICE_ID"),
        app.config.get("BG_WARN_ON_CPU"),
    )

    return app


