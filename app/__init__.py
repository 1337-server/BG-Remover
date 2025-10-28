"""Application factory for the background remover web service."""
from __future__ import annotations

from app.services import runtime_compat
from app.services.bg_remove import ensure_global_session

try:  # pragma: no cover - optional during testing
    from flask import Flask, request
except ModuleNotFoundError as exc:  # pragma: no cover - guard for test imports
    Flask = None  # type: ignore
    _FLASK_IMPORT_ERROR = exc
else:
    _FLASK_IMPORT_ERROR = None


def create_app() -> "Flask":
    """Create and configure the Flask application."""

    if Flask is None:  # pragma: no cover - executed only when Flask missing
        raise RuntimeError("Flask is required to create the web application.") from _FLASK_IMPORT_ERROR

    app = Flask(__name__)

    # Initialise the runtime stack before creating the rembg session. This
    # ensures NumPy/ONNXRuntime compatibility issues are surfaced early.
    runtime_compat.ensure_runtime_ready()
    # Initialise the global background removal session once at startup.
    ensure_global_session()

    from app.extensions import socketio
    from app.routes.image_converter import image_converter_bp

    socketio.init_app(app)
    app.register_blueprint(image_converter_bp)

    @app.after_request
    def add_static_cache_headers(response):
        """Add caching headers to static asset responses to improve load performance."""

        if request.path.startswith("/static/"):
            response.headers.setdefault("Cache-Control", "public, max-age=31536000, immutable")
        return response

    return app


