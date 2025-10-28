"""Socket.IO entrypoint for running the Flask application with WebSocket support."""
from __future__ import annotations

import os

import eventlet
from flask import Flask
from flask_socketio import SocketIO

from app import create_app

# Eventlet is required so that Flask-SocketIO can handle long-lived WebSocket
# connections efficiently. The monkey patching call must happen before most
# other imports to ensure the standard library behaves cooperatively.
eventlet.monkey_patch()


def create_socketio_app() -> tuple[Flask, SocketIO]:
    """Initialise and return the Flask application and Socket.IO server."""

    flask_app = create_app()
    socketio = SocketIO(
        flask_app,
        cors_allowed_origins="*",
        async_mode="eventlet",
        logger=True,
        engineio_logger=True,
    )
    return flask_app, socketio


def main() -> None:
    """Run the Flask application using Flask-SocketIO."""

    app, socketio = create_socketio_app()
    port = int(os.getenv("PORT", "5000"))
    socketio.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
