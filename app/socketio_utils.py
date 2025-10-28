"""Utilities for managing Eventlet monkey patching."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - hints only
    from collections.abc import Callable

_LOGGER = logging.getLogger(__name__)
_PATCH_RESULT: bool | None = None


def _is_eventlet_monkey_patched(patcher: "Callable[[str], bool]") -> bool:
    """Return ``True`` when Eventlet's monkey patching has already run."""

    try:
        # Checking the socket module is a reliable proxy for Eventlet's patch.
        return bool(patcher("socket"))
    except Exception:  # pragma: no cover - defensive guard
        _LOGGER.debug("Unable to determine Eventlet monkey patch status.", exc_info=True)
        return False


def ensure_eventlet_monkey_patched() -> bool:
    """Apply Eventlet's monkey patching when available.

    Eventlet must patch the standard library before importing Flask, Socket.IO
    or other networking heavy modules. The helper is idempotent so that multiple
    entry points can safely request monkey patching without reapplying it.

    Returns
    -------
    bool
        ``True`` when the standard library was already patched or patching
        succeeds, ``False`` otherwise.
    """

    global _PATCH_RESULT
    if _PATCH_RESULT is not None:
        return _PATCH_RESULT

    try:
        import eventlet  # type: ignore
        from eventlet import patcher  # type: ignore
    except ModuleNotFoundError:
        _LOGGER.warning(
            "Eventlet is not installed. Running without cooperative sockets.",
        )
        _PATCH_RESULT = False
        return False
    except Exception:  # pragma: no cover - defensive guard
        _LOGGER.exception(
            "Unexpected error importing Eventlet; continuing without monkey patching.",
        )
        _PATCH_RESULT = False
        return False

    if _is_eventlet_monkey_patched(patcher.is_monkey_patched):
        _LOGGER.debug("Eventlet monkey patching already active.")
        _PATCH_RESULT = True
        return True

    try:
        eventlet.monkey_patch()
    except Exception:  # pragma: no cover - defensive guard
        _LOGGER.exception(
            "Eventlet monkey patching failed; continuing without cooperative sockets.",
        )
        _PATCH_RESULT = False
        return False

    _LOGGER.debug("Eventlet monkey patching applied successfully.")
    _PATCH_RESULT = True
    return True
