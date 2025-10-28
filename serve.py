"""Development server entry point for the Flask-SocketIO web UI."""
from __future__ import annotations

# Eventlet must monkey patch the standard library before importing Flask, SocketIO,
# or other networking-heavy modules. Importing and patching here keeps the
# application compatible regardless of whether ``serve.py`` is executed directly
# or via ``python -m app``.
try:  # pragma: no cover - eventlet is optional in some environments
    import eventlet  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - fallback when eventlet missing
    eventlet = None  # type: ignore[assignment]
else:  # pragma: no cover - import side effect only
    eventlet.monkey_patch()

from app.runner import run_socketio_server


def main() -> None:
    """Start the Socket.IO-enabled development server."""

    run_socketio_server()


if __name__ == "__main__":
    main()
