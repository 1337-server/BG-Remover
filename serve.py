"""Development server entry point for the Flask web interface."""
from __future__ import annotations

import argparse
import json
import logging
import os
from collections.abc import Sequence
from typing import Any

try:  # pragma: no cover - eventlet optional during tests
    import eventlet

    eventlet.monkey_patch()
except ModuleNotFoundError:  # pragma: no cover - fallback to standard library
    eventlet = None  # type: ignore[assignment]

from app import create_app

_LOGGER = logging.getLogger(__name__)

_ENV_PORT_KEYS = ("PORT", "FLASK_RUN_PORT")
_ENV_HOST_KEYS = ("HOST", "FLASK_RUN_HOST")


def _log_json(level: int, event: str, **fields: object) -> None:
    """Emit structured JSON logs for development server events."""

    payload = {"event": event, **fields}
    message = json.dumps(payload, sort_keys=True)
    _LOGGER.log(level, message)


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
    parser.add_argument(
        "--accelerator",
        choices=["auto", "cuda", "cpu"],
        default=None,
        help="Select the execution accelerator (overrides BG_ACCELERATOR).",
    )
    parser.add_argument(
        "--cuda-device-id",
        type=int,
        default=None,
        help="Select the CUDA device id when using GPU acceleration.",
    )
    parser.add_argument(
        "--no-warn-on-cpu",
        action="store_true",
        help="Disable CPU fallback warnings for this process.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Start the Flask development server."""

    parser = _build_parser()
    args = parser.parse_args(argv)

    env_host = _env_str(*_ENV_HOST_KEYS)
    env_port = _env_int(*_ENV_PORT_KEYS)

    host = args.host or env_host or "0.0.0.0"
    port = args.port if args.port is not None else env_port or 5000
    debug = args.debug or os.getenv("FLASK_DEBUG") == "1"

    config_overrides: dict[str, Any] = {}
    if args.accelerator:
        config_overrides["BG_ACCELERATOR"] = args.accelerator
    if args.cuda_device_id is not None:
        config_overrides["BG_CUDA_DEVICE_ID"] = args.cuda_device_id
    if args.no_warn_on_cpu:
        config_overrides["BG_WARN_ON_CPU"] = False

    app = create_app(config_overrides=config_overrides or None)
    _log_json(
        logging.INFO,
        "server_start",
        host=host,
        port=port,
        eventlet=bool(eventlet),
        debug=bool(debug),
        config_overrides=config_overrides or {},
    )
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    main()

