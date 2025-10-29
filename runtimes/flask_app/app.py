"""Flask application factory for the background remover web runtime."""
from __future__ import annotations

from flask import Flask

from bgremover_core import init_logging, load_config

from .routes import bp


def create_app(config_overrides: dict[str, object] | None = None) -> Flask:
    """Return a configured Flask application."""

    config = load_config()
    init_logging(config.log_level)
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.secret_key = "bgremover-secret"  # Required for flashing messages.
    app.config["BGR_CONFIG"] = config
    if config_overrides:
        app.config.update(config_overrides)
    app.register_blueprint(bp)
    return app


__all__ = ["create_app"]
