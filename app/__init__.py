"""Application factory for the background remover web service."""
from __future__ import annotations

from app.services.bg_remove import ensure_global_session

try:  # pragma: no cover - optional during testing
    from flask import Flask
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

    # Initialise the global background removal session once at startup.
    ensure_global_session()

    from app.routes.image_converter import image_converter_bp

    app.register_blueprint(image_converter_bp)
    return app


