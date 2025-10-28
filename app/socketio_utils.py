"""Utilities for optional Eventlet integration with Flask-SocketIO."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - hints only
    from flask_socketio import SocketIO

_LOGGER = logging.getLogger(__name__)
_PATCH_RESULT: bool | None = None


def ensure_eventlet_monkey_patched(socketio: "SocketIO") -> None:
    """Apply Eventlet's monkey patching when the async mode requires it.

    The helper logs a warning whenever Eventlet is unavailable and the
    application therefore needs to rely on the fallback async mode configured
    by :func:`app.extensions.create_socketio`. When Eventlet is present, the
    standard library is monkey patched to enable cooperative scheduling.
    Subsequent calls become no-ops so importing multiple entrypoints does not
    reapply the patch or spam the logs.
    """

    global _PATCH_RESULT
    if _PATCH_RESULT is not None:
        return

    if getattr(socketio, "async_mode", None) != "eventlet":
        _LOGGER.warning(
            "Eventlet is not available. Running Socket.IO with '%s' async mode instead.",
            socketio.async_mode,
        )
        _PATCH_RESULT = False
        return

    try:
        import eventlet  # type: ignore
    except ModuleNotFoundError:
        _LOGGER.warning(
            "Eventlet async mode requested but the dependency is not installed. "
            "Continuing with '%s' async mode.",
            socketio.async_mode,
        )
        _PATCH_RESULT = False
        return
    except Exception:  # pragma: no cover - defensive guard
        _LOGGER.exception(
            "Unexpected error importing Eventlet; continuing without monkey patching.",
        )
        _PATCH_RESULT = False
        return

    try:
        eventlet.monkey_patch()
    except Exception:  # pragma: no cover - defensive guard
        _LOGGER.exception(
            "Eventlet monkey patching failed; continuing without cooperative sockets.",
        )
        _PATCH_RESULT = False
    else:
        _LOGGER.debug("Eventlet monkey patching applied successfully.")
        _PATCH_RESULT = True
