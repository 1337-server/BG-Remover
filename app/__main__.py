"""Module entry point for ``python -m app``."""
from __future__ import annotations

from app.runner import run_socketio_server


def main() -> None:
    """Execute the Socket.IO web server when the package is run as a module."""

    run_socketio_server()


if __name__ == "__main__":
    main()
