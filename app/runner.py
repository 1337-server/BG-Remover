"""Utilities for starting the Flask-SocketIO web server."""
from __future__ import annotations

from typing import Any

import eventlet  # type: ignore


def run_socketio_server(host: str = "0.0.0.0", port: int = 5000, **kwargs: Any) -> None:
    """Start the application using Socket.IO's eventlet-powered server.

    The helper performs ``eventlet.monkey_patch`` before importing the Flask
    application factory. This ensures that standard library networking modules
    cooperate with eventlet's green threads, which is required for true
    WebSocket support in Flask-SocketIO. Additional keyword arguments are
    forwarded to :meth:`flask_socketio.SocketIO.run` so callers can enable
    debug mode or tweak SSL parameters when needed.
    """

    eventlet.monkey_patch()

    from app import create_app
    from app.extensions import socketio

    app = create_app()
    socketio.run(app, host=host, port=port, **kwargs)
