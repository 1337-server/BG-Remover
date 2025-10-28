"""Socket.IO entry point for running the Flask application with WebSocket support."""
from __future__ import annotations

# ``eventlet.monkey_patch`` must execute before importing any Flask, Socket.IO or
# Torch modules. This placement keeps Docker and direct module execution paths
# consistent.
try:  # pragma: no cover - eventlet is optional when running tests
    import eventlet  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - tolerate environments without eventlet
    eventlet = None  # type: ignore[assignment]
else:  # pragma: no cover - import side effect only
    eventlet.monkey_patch()

import os
from typing import TYPE_CHECKING

from app.extensions import socketio

if TYPE_CHECKING:  # pragma: no cover - hints only
    from flask import Flask
    from flask_socketio import SocketIO


def create_socketio_app() -> tuple["Flask", "SocketIO"]:
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
