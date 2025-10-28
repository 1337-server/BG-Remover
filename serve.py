"""Development server entry point for the Flask-SocketIO web UI."""
from __future__ import annotations

try:  # pragma: no cover - import side effect
    import eventlet as _eventlet  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - runtime optional dependency
    _eventlet = None  # type: ignore[assignment]
else:
    try:
        _eventlet.monkey_patch()
    except Exception:  # pragma: no cover - defensive guard
        _eventlet = None  # type: ignore[assignment]

from app.runner import run_socketio_server


def main() -> None:
    """Start the Socket.IO-enabled development server."""

    run_socketio_server()


if __name__ == "__main__":
    main()
