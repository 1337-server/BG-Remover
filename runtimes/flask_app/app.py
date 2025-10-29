"""Flask application factory for the background remover web runtime."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from flask import Flask

from bgremover_core import init_logging, load_config

from .routes import webui
from .services import ResultStore

_DEFAULT_SECRET = "bgremover-secret"


def _create_result_store(app: Flask) -> ResultStore:
    """Return a :class:`ResultStore` initialised for ``app``."""

    instance_dir = Path(app.instance_path)
    instance_dir.mkdir(parents=True, exist_ok=True)
    default_output = instance_dir / "results"
    configured = Path(app.config.setdefault("OUTPUT_DIR", default_output))
    output_dir = configured
    output_dir.mkdir(parents=True, exist_ok=True)
    history_limit = int(app.config.get("HISTORY_LIMIT", 50))
    return ResultStore(output_dir, history_limit=history_limit)


def create_app(config_overrides: dict[str, object] | None = None) -> Flask:
    """Return a configured Flask application ready for registration or running."""

    config = load_config()
    init_logging(config.log_level)
    app = Flask(__name__, template_folder="templates", static_folder="static")
    secret_value = (
        config_overrides.get("SECRET_KEY", _DEFAULT_SECRET)
        if config_overrides
        else _DEFAULT_SECRET
    )
    app.secret_key = secret_value
    app.config.setdefault("MAX_CONTENT_LENGTH", 64 * 1024 * 1024)  # 64 MiB uploads
    app.config.setdefault("BGR_CONFIG", config)
    if config_overrides:
        app.config.update(config_overrides)

    app.extensions["executor"] = ThreadPoolExecutor(max_workers=4)
    app.extensions["result_store"] = _create_result_store(app)

    app.register_blueprint(webui)
    return app


def main(**kwargs: Any) -> None:
    """Start a development Flask server when executing the module directly."""

    app = create_app()
    app.run(host="127.0.0.1", port=5000, debug=False, **kwargs)


if __name__ == "__main__":  # pragma: no cover - manual execution convenience.
    main()


__all__ = ["create_app"]
