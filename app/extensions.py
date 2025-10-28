"""Application-wide extensions such as Socket.IO instances."""
from __future__ import annotations

import warnings
from typing import Callable, Optional

from app.socketio_utils import ensure_eventlet_monkey_patched

_EVENTLET_PATCHED = ensure_eventlet_monkey_patched()

try:  # pragma: no cover - optional dependency during import time
    from flask_socketio import SocketIO
except ModuleNotFoundError as exc:  # pragma: no cover - raised when dependency missing
    _SOCKETIO_IMPORT_ERROR = exc

    class _StubSocketIO:
        """Lightweight fallback used when Flask-SocketIO is unavailable."""

        def __init__(self, async_mode: str = "threading", **_: object) -> None:
            self.async_mode = async_mode

        def init_app(self, app: object, **__: object) -> None:
            """Initialise the stub against ``app`` without real socket support."""

        def emit(self, *_: object, **__: object) -> None:
            """Silently drop emitted events when running without Socket.IO."""

        def start_background_task(
            self, target: Callable[..., None], *args: object, **kwargs: object
        ) -> None:
            """Raise a clear error when background tasks cannot be scheduled."""

            raise RuntimeError(
                "Flask-SocketIO is required to schedule background tasks."
                " Install the 'flask-socketio' extra to enable live previews."
            ) from _SOCKETIO_IMPORT_ERROR

        def on(self, *_: object, **__: object) -> Callable[[Callable[..., object]], Callable[..., object]]:
            """Return a decorator that passes through the provided function."""

            def decorator(func: Callable[..., object]) -> Callable[..., object]:
                return func

            return decorator

        def run(self, *_: object, **__: object) -> None:
            """Raise a helpful error when attempting to start the server without the dependency."""

            raise RuntimeError(
                "Flask-SocketIO is required for the live preview server."
            ) from _SOCKETIO_IMPORT_ERROR

    SocketIO = _StubSocketIO  # type: ignore[assignment]
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

    async_mode = "eventlet" if _EVENTLET_PATCHED else "threading"
    if async_mode != "eventlet":  # pragma: no cover - executed only when dependency missing
        warnings.warn(
            "Eventlet is not available. Falling back to threading mode; WebSocket support will be limited.",
            RuntimeWarning,
            stacklevel=2,
        )

    return SocketIO(async_mode=async_mode, cors_allowed_origins="*", cors_credentials=True)


socketio = create_socketio()

