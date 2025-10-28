"""Helper script for running the Flask development server with accelerator overrides."""
from __future__ import annotations

import argparse
import logging
import os
from typing import Any, Dict, Sequence

from app import create_app
from app.services.bg_remove import get_accelerator_status

_LOGGER = logging.getLogger(__name__)

_ENV_PORT_KEYS = ("PORT", "FLASK_RUN_PORT")
_ENV_HOST_KEYS = ("HOST", "FLASK_RUN_HOST")


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
        help="Override BG_ACCELERATOR for this process.",
    )
    parser.add_argument(
        "--cuda-device-id",
        type=int,
        default=None,
        help="Override BG_CUDA_DEVICE_ID for this process.",
    )
    parser.add_argument(
        "--no-warn-on-cpu",
        action="store_true",
        help="Suppress CPU fallback warnings regardless of BG_WARN_ON_CPU.",
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

    config_overrides: Dict[str, Any] = {}
    if args.accelerator:
        config_overrides["BG_ACCELERATOR"] = args.accelerator
    if args.cuda_device_id is not None:
        config_overrides["BG_CUDA_DEVICE_ID"] = args.cuda_device_id
    if args.no_warn_on_cpu:
        config_overrides["BG_WARN_ON_CPU"] = False

    app = create_app(config_overrides if config_overrides else None)
    status = get_accelerator_status()
    runtime_label = "cuda" if status.provider == "cuda" else "cpu"
    _LOGGER.info(
        "Selected execution provider: %s (runtime=%s, gpu=%s)",
        status.provider_description,
        runtime_label,
        status.gpu_name or "n/a",
    )
    if status.warning and app.config.get("BG_WARN_ON_CPU", True):
        _LOGGER.warning("%s", status.warning)

    _LOGGER.info("Starting Flask development server on http://%s:%s", host, port)
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    main()
