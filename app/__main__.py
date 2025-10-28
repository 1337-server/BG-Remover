"""Module entry point for ``python -m app``."""
from __future__ import annotations

# Eventlet must monkey patch cooperative versions of the standard library before
# any Flask, Socket.IO, or Torch imports. Applying the patch here ensures the
# runtime environment is correctly prepared for the rest of the application
# initialisation sequence.
try:  # pragma: no cover - optional dependency during unit tests
    import eventlet  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - eventlet not installed in some envs
    eventlet = None  # type: ignore[assignment]
else:  # pragma: no cover - import side effect only
    eventlet.monkey_patch()

from app.runner import run_socketio_server


def main() -> None:
    """Execute the Socket.IO web server when the package is run as a module."""

    run_socketio_server()


if __name__ == "__main__":
    main()
