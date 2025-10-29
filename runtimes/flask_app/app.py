"""Flask application factory for the background remover web runtime."""
from __future__ import annotations

from typing import Any

from flask import Flask

from bgremover_core import init_logging, load_config

from .routes import bp


def create_app(config_overrides: dict[str, object] | None = None) -> Flask:
    """Return a configured Flask application ready for registration or running."""

    config = load_config()
    init_logging(config.log_level)
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.secret_key = "bgremover-secret"  # Required for flashing messages.
    app.config["BGR_CONFIG"] = config
    if config_overrides:
        app.config.update(config_overrides)
    app.register_blueprint(bp)
    return app


def main(**kwargs: Any) -> None:
    """Start a development Flask server when executing the module directly."""

    app = create_app()
    app.run(host="127.0.0.1", port=5000, debug=False, **kwargs)


if __name__ == "__main__":  # pragma: no cover - manual execution convenience.
    main()


__all__ = ["create_app"]
