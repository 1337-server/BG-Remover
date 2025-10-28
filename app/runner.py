"""Utilities for starting the Flask-SocketIO web server."""
from __future__ import annotations

import logging
from typing import Any

_EVENTLET_PATCH_ERROR: Exception | None = None
_EVENTLET_PATCHED = False

try:  # pragma: no cover - import side effect
    import eventlet as _eventlet  # type: ignore
except ModuleNotFoundError as exc:  # pragma: no cover - runtime optional dependency
    _EVENTLET_PATCH_ERROR = exc
else:
    try:
        _eventlet.monkey_patch()
    except Exception as exc:  # pragma: no cover - defensive guard
        _EVENTLET_PATCH_ERROR = exc
    else:
        _EVENTLET_PATCHED = True

from app.socketio_utils import ensure_eventlet_monkey_patched, mark_eventlet_monkey_patched  # noqa: E402

_LOGGER = logging.getLogger(__name__)

if _EVENTLET_PATCHED:
    mark_eventlet_monkey_patched()
elif _EVENTLET_PATCH_ERROR is not None:
    if not isinstance(_EVENTLET_PATCH_ERROR, ModuleNotFoundError):
        _LOGGER.warning(
            "Eventlet monkey patching failed during startup: %s", _EVENTLET_PATCH_ERROR
        )
    ensure_eventlet_monkey_patched()
else:
    ensure_eventlet_monkey_patched()

from app.extensions import socketio  # noqa: E402


def run_socketio_server(host: str = "0.0.0.0", port: int = 5000, **kwargs: Any) -> None:
    """Start the application using Socket.IO's configured async mode.

    The helper ensures Eventlet's cooperative monkey patching is applied when
    available before loading the Flask application factory. Additional keyword
    arguments are forwarded to :meth:`flask_socketio.SocketIO.run` so callers
    can enable debug mode or tweak SSL parameters when needed.
    """

    from app import create_app

    app = create_app()
    socketio.run(app, host=host, port=port, **kwargs)
