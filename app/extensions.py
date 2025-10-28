"""Application-wide extensions such as Socket.IO instances."""
from __future__ import annotations

from typing import Optional

try:  # pragma: no cover - optional dependency during import time
    from flask_socketio import SocketIO
except ModuleNotFoundError as exc:  # pragma: no cover - raised when dependency missing
    SocketIO = None  # type: ignore
    _SOCKETIO_IMPORT_ERROR = exc
else:
    _SOCKETIO_IMPORT_ERROR: Optional[Exception] = None


def create_socketio() -> "SocketIO":
    """Return a configured :class:`~flask_socketio.SocketIO` instance.

    The helper defers raising an informative error until Socket.IO is used.
    This keeps module imports lightweight during testing environments where the
    optional dependency may not be installed.
    """

    if SocketIO is None:  # pragma: no cover - executed only when dependency missing
        raise RuntimeError(
            "Flask-SocketIO is required for real-time preview support."
        ) from _SOCKETIO_IMPORT_ERROR

    return SocketIO(async_mode="threading", cors_allowed_origins="*")


socketio = create_socketio()

