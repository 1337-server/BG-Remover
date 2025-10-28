"""Module entry point for ``python -m app``."""
from __future__ import annotations

try:  # pragma: no cover - import side effect
    import eventlet as _eventlet  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - runtime optional dependency
    _eventlet = None  # type: ignore[assignment]
else:
    try:
        _eventlet.monkey_patch()
    except Exception:  # pragma: no cover - defensive guard
        _eventlet = None  # type: ignore[assignment]

from app.runner import run_socketio_server


def main() -> None:
    """Execute the Socket.IO web server when the package is run as a module."""

    run_socketio_server()


if __name__ == "__main__":
    main()
