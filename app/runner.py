"""Utilities for starting the Flask-SocketIO web server."""
from __future__ import annotations

from typing import Any

from app.extensions import socketio
from app.socketio_utils import ensure_eventlet_monkey_patched


def run_socketio_server(host: str = "0.0.0.0", port: int = 5000, **kwargs: Any) -> None:
    """Start the application using Socket.IO's configured async mode.

    The helper ensures Eventlet's cooperative monkey patching is applied when
    available before loading the Flask application factory. Additional keyword
    arguments are forwarded to :meth:`flask_socketio.SocketIO.run` so callers
    can enable debug mode or tweak SSL parameters when needed.
    """

    ensure_eventlet_monkey_patched(socketio)

    from app import create_app

    app = create_app()
    socketio.run(app, host=host, port=port, **kwargs)
