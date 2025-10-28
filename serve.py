"""Development server entry point for the Flask-SocketIO web UI."""
from __future__ import annotations

from app.runner import run_socketio_server


def main() -> None:
    """Start the Socket.IO-enabled development server."""

    run_socketio_server()


if __name__ == "__main__":
    main()
