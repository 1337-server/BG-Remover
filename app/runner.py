"""Helpers for launching the Flask-SocketIO web server."""

from __future__ import annotations

import logging
import socket
import time
from contextlib import closing
from typing import Any

from app.socketio_utils import ensure_eventlet_monkey_patched, validate_eventlet_patch

# Ensure cooperative sockets and threading primitives are in place before any
# Flask or Socket.IO modules are imported.
ensure_eventlet_monkey_patched()

from app.extensions import socketio

_LOGGER = logging.getLogger(__name__)

_DEFAULT_PORT_SCAN_LIMIT = 20
_PORT_RELEASE_RETRIES = 5
_PORT_RELEASE_DELAY_SECONDS = 0.3


def _normalise_probe_host(host: str) -> str:
    """Return a host that can be used for connectivity checks."""

    if host in {"0.0.0.0", "::", ""}:
        return "127.0.0.1"
    return host


def _is_port_in_use(host: str, port: int) -> bool:
    """Return ``True`` if ``port`` appears to be occupied on ``host``.

    The probe opens a TCP socket and attempts to connect to the provided host.
    A successful connection indicates another process is already bound to the
    port, while a connection failure suggests the port is available for use.
    """

    probe_host = _normalise_probe_host(host)
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((probe_host, port)) == 0


def _wait_for_port_release(host: str, port: int) -> bool:
    """Wait briefly for Windows to release a port after a recent shutdown."""

    for attempt in range(1, _PORT_RELEASE_RETRIES + 1):
        if not _is_port_in_use(host, port):
            return True
        _LOGGER.debug(
            "Port %s still appears busy (attempt %s/%s). Waiting %.1fs for release.",
            port,
            attempt,
            _PORT_RELEASE_RETRIES,
            _PORT_RELEASE_DELAY_SECONDS,
        )
        time.sleep(_PORT_RELEASE_DELAY_SECONDS)
    return not _is_port_in_use(host, port)


def _pick_available_port(host: str, port: int, allow_fallback: bool, scan_limit: int) -> int:
    """Return an available port, optionally scanning forward when busy."""

    if _wait_for_port_release(host, port):
        return port

    if not allow_fallback:
        message = f"Port {port} is already in use. Try another port."
        _LOGGER.error(message)
        raise OSError(message)

    for offset in range(1, scan_limit + 1):
        candidate = port + offset
        if _wait_for_port_release(host, candidate):
            _LOGGER.warning(
                "Port %s is busy; falling back to port %s.",
                port,
                candidate,
            )
            return candidate

    message = (
        f"Unable to locate a free port in the range {port}-{port + scan_limit}. "
        "Adjust the port manually and try again."
    )
    _LOGGER.error(message)
    raise OSError(message)


def run_socketio_server(
    host: str = "0.0.0.0",
    port: int = 5000,
    *,
    allow_port_fallback: bool = True,
    port_scan_limit: int | None = None,
    **kwargs: Any,
) -> None:
    """Start the application using Socket.IO's configured async mode.

    The helper ensures Eventlet's cooperative monkey patching is applied when
    available before loading the Flask application factory. Additional keyword
    arguments are forwarded to :meth:`flask_socketio.SocketIO.run` so callers
    can enable debug mode or tweak SSL parameters when needed.
    """

    from app import create_app

    validate_eventlet_patch()

    scan_limit_raw = _DEFAULT_PORT_SCAN_LIMIT if port_scan_limit is None else port_scan_limit
    scan_limit = max(0, scan_limit_raw)

    selected_port = _pick_available_port(
        host,
        port,
        allow_fallback=allow_port_fallback,
        scan_limit=scan_limit,
    )

    kwargs.setdefault("use_reloader", False)

    app = create_app()
    display_host = "127.0.0.1" if host in {"0.0.0.0", ""} else host
    _LOGGER.info(
        "Starting Flask-SocketIO server on http://%s:%s (eventlet mode)",
        display_host,
        selected_port,
    )

    try:
        socketio.run(app, host=host, port=selected_port, **kwargs)
    except OSError:
        raise
    except Exception:  # pragma: no cover - defensive guard for unexpected errors
        _LOGGER.exception("The Socket.IO server terminated unexpectedly.")
        raise
    finally:
        try:
            socketio.stop()
        except Exception:  # pragma: no cover - best effort cleanup
            _LOGGER.debug("Socket.IO stop() raised during cleanup.", exc_info=True)
