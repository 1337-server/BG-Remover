"""Flask runtime package entry point."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - only used for type checkers.
    from .app import create_app as _create_app  # noqa: F401

__all__ = ["create_app", "routes"]


def create_app(*args: Any, **kwargs: Any):
    """Defer to :func:`runtimes.flask_app.app.create_app` at call time.

    Importing lazily avoids double-import runtime warnings when running the
    module via ``python -m runtimes.flask_app.app`` while keeping the public API
    unchanged for callers that expect ``runtimes.flask_app.create_app``.
    """

    from .app import create_app as _create_app

    return _create_app(*args, **kwargs)


def __getattr__(name: str):
    """Provide lazy attribute access for optional re-exports."""

    if name == "routes":
        from . import routes as routes_module

        return routes_module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
