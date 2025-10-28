"""Development server entry point for the Flask web interface."""
from __future__ import annotations

import argparse
import json
import logging
import os
import threading
from collections.abc import Mapping, Sequence

try:  # pragma: no cover - eventlet optional during tests
    import eventlet

    eventlet.monkey_patch(thread=False)
except ModuleNotFoundError:  # pragma: no cover - fallback to standard library
    eventlet = None  # type: ignore[assignment]

from app import create_app, run_startup_tasks


def _configure_logging() -> None:
    """Initialise logging so structured JSON events are emitted to stdout."""

    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(level=logging.INFO)
    root_logger.setLevel(logging.INFO)


_configure_logging()

_LOGGER = logging.getLogger(__name__)

_ENV_PORT_KEYS = ("PORT", "FLASK_RUN_PORT")
_ENV_HOST_KEYS = ("HOST", "FLASK_RUN_HOST")


def _log_json(level: int, event: str, **fields: object) -> None:
    """Emit structured JSON logs for development server events."""

    payload = {"event": event, **fields}
    message = json.dumps(payload, sort_keys=True)
    _LOGGER.log(level, message)


def _spawn_startup_thread(config: Mapping[str, object] | None = None) -> threading.Event:
    """Execute startup tasks asynchronously to keep the Flask boot path responsive."""

    completion = threading.Event()

    def _worker() -> None:
        _log_json(
            logging.INFO,
            "startup_tasks_begin",
            config_keys=sorted(config.keys()) if config else [],
        )
        try:
            run_startup_tasks()
            _log_json(logging.INFO, "startup_tasks_complete")
        except Exception as exc:  # pragma: no cover - surfaced via logs in production
            _LOGGER.exception("Startup task execution failed")
            _log_json(logging.ERROR, "startup_tasks_failed", error=str(exc))
        finally:
            completion.set()

    thread = threading.Thread(target=_worker, name="startup-initialiser", daemon=True)
    thread.start()
    return completion


def _env_int(*keys: str) -> int | None:
    """Return the first integer environment variable for ``keys``."""

    for key in keys:
        raw_value = os.getenv(key)
        if raw_value is None:
            continue
        try:
            return int(raw_value)
        except ValueError:
            _LOGGER.warning("Environment variable %s is not a valid integer: %s", key, raw_value)
    return None


def _env_str(*keys: str) -> str | None:
    """Return the first non-empty string environment variable for ``keys``."""

    for key in keys:
        raw_value = os.getenv(key)
        if raw_value:
            return raw_value
    return None


def _build_parser() -> argparse.ArgumentParser:
    """Create an argument parser for the development server."""

    parser = argparse.ArgumentParser(description="Run the Flask development server")
    parser.add_argument(
        "--host",
        default=None,
        help="Host interface to bind (overrides HOST/FLASK_RUN_HOST)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port to bind (overrides PORT/FLASK_RUN_PORT)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable Flask debug mode regardless of FLASK_DEBUG.",
    )
    return parser


def _should_run_startup_tasks(debug: bool) -> bool:
    """Return ``True`` when heavy startup tasks should execute in this process.

    Flask's development reloader executes the module twice: once in the
    supervisor process that watches for file changes and once in the serving
    process that handles incoming requests. Loading ONNX models in both
    processes doubles the startup time and exhausts VRAM unnecessarily. The
    reloader child process sets the ``WERKZEUG_RUN_MAIN`` environment variable,
    which we can use to limit heavy initialisation to the real serving process.
    """

    if not debug:
        return True
    return os.getenv("WERKZEUG_RUN_MAIN") == "true"


def main(argv: Sequence[str] | None = None) -> None:
    """Start the Flask development server."""

    parser = _build_parser()
    args = parser.parse_args(argv)

    env_host = _env_str(*_ENV_HOST_KEYS)
    env_port = _env_int(*_ENV_PORT_KEYS)

    host = args.host or env_host or "0.0.0.0"
    port = args.port if args.port is not None else env_port or 5000
    debug = args.debug or os.getenv("FLASK_DEBUG") == "1"

    _log_json(logging.INFO, "initialising_models")
    app = create_app(config_overrides=None, run_startup_tasks=False)

    startup_event: threading.Event | None = None
    if _should_run_startup_tasks(debug):
        startup_event = _spawn_startup_thread(app.config)
    else:
        _log_json(logging.INFO, "startup_tasks_deferred", reason="reloader_supervisor")

    _log_json(
        logging.INFO,
        "server_start",
        host=host,
        port=port,
        eventlet=bool(eventlet),
        debug=bool(debug),
        startup_async=True,
        startup_event_set=bool(startup_event and startup_event.is_set()),
    )
    _log_json(logging.INFO, "starting_flask", host=host, port=port)
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    main()

