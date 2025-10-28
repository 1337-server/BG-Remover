"""Flask application factory for the background removal service."""
from __future__ import annotations

import logging
from collections.abc import Mapping, MutableMapping
from typing import TYPE_CHECKING

from app.services import model_registry, runtime_compat
from app.services.bg_remove import ensure_global_session

if TYPE_CHECKING:  # pragma: no cover - typing assistance only
    from flask import Flask


LOGGER = logging.getLogger(__name__)


def _build_config(overrides: Mapping[str, object] | None) -> MutableMapping[str, object]:
    """Return the application configuration mapping."""

    config: MutableMapping[str, object] = {}
    if overrides:
        config.update(overrides)
    return config


def _run_startup_tasks() -> None:
    """Prepare runtime dependencies and warm up background removal sessions."""

    runtime_compat.ensure_runtime_ready()
    model_registry.preload_models()
    ensure_global_session()


def create_app(
    config_overrides: Mapping[str, object] | None = None,
    *,
    run_startup_tasks: bool = True,
) -> Flask:
    """Create and configure the Flask application."""

    try:  # pragma: no cover - executed only when Flask missing
        from flask import Flask
        from flask import request as flask_request
    except ModuleNotFoundError as exc:
        raise RuntimeError("Flask is required to create the web application.") from exc

    app = Flask(__name__)
    app.config.from_mapping(_build_config(config_overrides))

    if run_startup_tasks:
        _run_startup_tasks()

    from app.routes.image_converter import image_converter_bp

    app.register_blueprint(image_converter_bp)

    @app.after_request
    def add_static_cache_headers(response):
        """Add caching headers to static asset responses to improve load performance."""

        if flask_request.path.startswith("/static/"):
            response.headers.setdefault("Cache-Control", "public, max-age=31536000, immutable")
        return response

    LOGGER.info("Application initialised for CPU-only background removal.")

    return app


def run_startup_tasks() -> None:
    """Public wrapper that performs the heavy-weight startup preparation."""

    _run_startup_tasks()
