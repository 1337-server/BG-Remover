"""Module entry point for ``python -m app``."""
from __future__ import annotations

import os

from app import create_app


def main() -> None:
    """Start the Flask development server when the package is executed."""

    app = create_app()
    host = os.environ.get("FLASK_RUN_HOST", "0.0.0.0")
    port = int(os.environ.get("FLASK_RUN_PORT", "5000"))
    app.run(host=host, port=port)


if __name__ == "__main__":
    main()
