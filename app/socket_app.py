"""Socket.IO entry point for running the Flask application with WebSocket support."""
from __future__ import annotations

import os
from typing import TYPE_CHECKING

from app.socketio_utils import ensure_eventlet_monkey_patched

ensure_eventlet_monkey_patched()

from flask import Flask

from app.extensions import socketio

if TYPE_CHECKING:  # pragma: no cover - hints only
    from flask_socketio import SocketIO


def create_socketio_app() -> tuple[Flask, "SocketIO"]:
    """Initialise and return the Flask application and Socket.IO server."""

    from app import create_app

    flask_app = create_app()
    return flask_app, socketio


def main() -> None:
    """Run the Flask application using the shared Flask-SocketIO instance."""

    app, socketio_server = create_socketio_app()
    port = int(os.getenv("PORT", "5000"))
    socketio_server.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
