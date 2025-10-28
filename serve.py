"""Development server entry point for the Flask-SocketIO web UI."""
from __future__ import annotations

import eventlet  # type: ignore


def main() -> None:
    """Start the Socket.IO-enabled development server."""

    eventlet.monkey_patch()
    from app import create_app
    from app.extensions import socketio

    app = create_app()
    socketio.run(app, host="0.0.0.0", port=5000)


if __name__ == "__main__":
    main()
