"""Development server entry point for the Flask-SocketIO web UI."""
from __future__ import annotations

from app.socketio_utils import ensure_eventlet_monkey_patched, validate_eventlet_patch

# Eventlet must monkey patch the standard library before importing Flask, SocketIO,
# or other networking-heavy modules. Trigger the cooperative patch immediately to
# ensure consistent behaviour across entry points.
ensure_eventlet_monkey_patched()

from app.runner import run_socketio_server


def main() -> None:
    """Start the Socket.IO-enabled development server."""

    validate_eventlet_patch()
    run_socketio_server()


if __name__ == "__main__":
    main()
