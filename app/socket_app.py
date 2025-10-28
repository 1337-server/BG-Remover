"""Socket.IO entry point for running the Flask application with WebSocket support."""
from __future__ import annotations

import os
from typing import TYPE_CHECKING

import eventlet
from flask import Flask

from app import create_app
from app.extensions import socketio

if TYPE_CHECKING:  # pragma: no cover - used only for type checkers
    from flask_socketio import SocketIO

# Eventlet is required so that Flask-SocketIO can handle long-lived WebSocket
# connections efficiently. The monkey patching call must happen before most
# other imports to ensure the standard library behaves cooperatively.
eventlet.monkey_patch()


def create_socketio_app() -> tuple[Flask, "SocketIO"]:
    """Initialise and return the Flask application and shared Socket.IO server.

    Returning the globally configured :class:`~flask_socketio.SocketIO` instance
    keeps background tasks and server-emitted events consistent across
    environments (development or Docker) because every request handler interacts
    with the same Socket.IO object that ultimately runs the server.
    """

    flask_app = create_app()
    return flask_app, socketio


def main() -> None:
    """Run the Flask application using the shared Flask-SocketIO instance."""

    app, socketio_server = create_socketio_app()
    port = int(os.getenv("PORT", "5000"))
    socketio_server.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
