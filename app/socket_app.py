"""Socket.IO entry point for running the Flask application with WebSocket support."""
from __future__ import annotations

import os
from typing import TYPE_CHECKING

try:  # pragma: no cover - import side effect
    import eventlet as _eventlet  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - runtime optional dependency
    _eventlet = None  # type: ignore[assignment]
else:
    try:
        _eventlet.monkey_patch()
    except Exception:  # pragma: no cover - defensive guard
        _eventlet = None  # type: ignore[assignment]

from app.socketio_utils import ensure_eventlet_monkey_patched, mark_eventlet_monkey_patched

if _eventlet is not None:
    mark_eventlet_monkey_patched()
else:
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
