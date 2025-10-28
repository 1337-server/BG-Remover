"""Development server entry point for the Flask-SocketIO web UI."""
from __future__ import annotations

import argparse
import logging
import os
from typing import Sequence

from app.socketio_utils import ensure_eventlet_monkey_patched, validate_eventlet_patch

# Eventlet must monkey patch the standard library before importing Flask, SocketIO,
# or other networking-heavy modules. Trigger the cooperative patch immediately to
# ensure consistent behaviour across entry points.
ensure_eventlet_monkey_patched()

from app.runner import run_socketio_server

_LOGGER = logging.getLogger(__name__)

_ENV_PORT_KEYS = ("SOCKETIO_PORT", "PORT")
_ENV_HOST_KEYS = ("SOCKETIO_HOST", "HOST")


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
    """Return an argument parser for the development server entry point."""

    parser = argparse.ArgumentParser(description="Run the Flask-SocketIO development server")
    parser.add_argument(
        "--host",
        default=None,
        help="Host interface to bind (overrides SOCKETIO_HOST/HOST)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port to bind (overrides SOCKETIO_PORT/PORT)",
    )
    parser.add_argument(
        "--strict-port",
        action="store_true",
        help="Fail instead of automatically picking the next available port.",
    )
    parser.add_argument(
        "--port-scan-limit",
        type=int,
        default=None,
        help="Maximum number of additional ports to try when the desired port is busy.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Start the Socket.IO-enabled development server."""

    parser = _build_parser()
    args = parser.parse_args(argv)

    env_host = _env_str(*_ENV_HOST_KEYS)
    env_port = _env_int(*_ENV_PORT_KEYS)

    host = args.host or env_host or "0.0.0.0"
    port = args.port if args.port is not None else env_port or 5000

    validate_eventlet_patch()
    run_socketio_server(
        host=host,
        port=port,
        allow_port_fallback=not args.strict_port,
        port_scan_limit=args.port_scan_limit,
    )


if __name__ == "__main__":
    main()
