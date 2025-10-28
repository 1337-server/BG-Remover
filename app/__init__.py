"""Application factory for the background remover web service."""
from __future__ import annotations

from typing import TYPE_CHECKING

from app.services.bg_remove import ensure_global_session

if TYPE_CHECKING:  # pragma: no cover - hints only
    from flask import Flask


def _import_flask() -> tuple["Flask", object]:
    """Return Flask's factory and the request proxy or raise a helpful error."""

    try:  # pragma: no cover - optional during testing
        from flask import Flask, request
    except ModuleNotFoundError as exc:  # pragma: no cover - guard for test imports
        raise RuntimeError("Flask is required to create the web application.") from exc
    return Flask, request


def create_app() -> "Flask":
    """Create and configure the Flask application."""

    Flask, flask_request = _import_flask()
    app = Flask(__name__)

    # Initialise the global background removal session once at startup.
    ensure_global_session()

    from app.extensions import socketio
    from app.routes.image_converter import image_converter_bp

    with app.app_context():
        socketio.init_app(app)
        app.register_blueprint(image_converter_bp)

    @app.after_request
    def add_static_cache_headers(response):
        """Add caching headers to static asset responses to improve load performance."""

        if flask_request.path.startswith("/static/"):
            response.headers.setdefault("Cache-Control", "public, max-age=31536000, immutable")
        return response

    return app


