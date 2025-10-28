"""Flask application factory for the background removal UI."""
from __future__ import annotations

import logging
from collections.abc import Mapping, MutableMapping
from typing import TYPE_CHECKING

from app.services.bg_remove import ensure_global_session

if TYPE_CHECKING:  # pragma: no cover - typing assistance only
    from flask import Flask

LOGGER = logging.getLogger(__name__)


def _build_config(overrides: Mapping[str, object] | None) -> MutableMapping[str, object]:
    """Return a configuration mapping populated with ``overrides``."""

    config: MutableMapping[str, object] = {}
    if overrides:
        config.update(overrides)
    return config


def _run_startup_tasks() -> None:
    """Perform lightweight initialisation ahead of serving requests."""

    ensure_global_session()


def create_app(
    config_overrides: Mapping[str, object] | None = None,
    *,
    run_startup_tasks: bool = True,
) -> Flask:
    """Create and configure the Flask application instance."""

    from flask import Flask
    from flask import request as flask_request

    app = Flask(__name__)
    app.config.from_mapping(_build_config(config_overrides))

    if run_startup_tasks:
        _run_startup_tasks()

    from app.routes.image_converter import image_converter_bp

    app.register_blueprint(image_converter_bp)

    @app.after_request
    def add_static_cache_headers(response):
        """Add caching headers for static assets to improve load performance."""

        if flask_request.path.startswith("/static/"):
            response.headers.setdefault("Cache-Control", "public, max-age=31536000, immutable")
        return response

    LOGGER.info("Background removal application initialised.")
    return app


def run_startup_tasks() -> None:
    """Expose startup preparation for external callers."""

    _run_startup_tasks()
