"""Utilities for managing Eventlet monkey patching."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - type hints only
    from collections.abc import Callable

_LOGGER = logging.getLogger(__name__)
_PATCH_STATE: bool | None = None


def _set_patch_state(value: bool) -> bool:
    """Persist ``value`` as the cached monkey patch status and return it."""

    global _PATCH_STATE
    _PATCH_STATE = value
    return value


def _import_eventlet() -> tuple["Callable", "Callable"]:
    """Return the Eventlet module and its patcher helper."""

    import eventlet  # type: ignore
    from eventlet import patcher  # type: ignore

    return eventlet, patcher


def _is_eventlet_monkey_patched(patcher: "Callable[[str], bool]") -> bool:
    """Return ``True`` when Eventlet's monkey patching has already run."""

    try:
        return bool(patcher("socket"))
    except Exception:  # pragma: no cover - defensive guard
        _LOGGER.debug("Unable to determine Eventlet monkey patch status.", exc_info=True)
        return False


def is_eventlet_monkey_patched() -> bool:
    """Return ``True`` when cooperative Eventlet sockets are active."""

    if _PATCH_STATE is True:
        return True

    try:
        _, patcher = _import_eventlet()
    except ModuleNotFoundError:
        return _set_patch_state(False)
    except Exception:  # pragma: no cover - defensive guard
        _LOGGER.debug("Unable to import Eventlet while checking patch state.", exc_info=True)
        return bool(_PATCH_STATE)

    patched = _is_eventlet_monkey_patched(patcher.is_monkey_patched)
    if patched:
        _set_patch_state(True)
    return patched


def mark_eventlet_monkey_patched() -> None:
    """Record that Eventlet's monkey patching has been applied externally."""

    _set_patch_state(True)


def ensure_eventlet_monkey_patched() -> bool:
    """Apply Eventlet's monkey patching when available and return the status."""

    if is_eventlet_monkey_patched():
        return True

    try:
        eventlet, patcher = _import_eventlet()
    except ModuleNotFoundError:
        _LOGGER.warning(
            "Eventlet is not installed. Running without cooperative sockets.",
        )
        return _set_patch_state(False)
    except Exception:  # pragma: no cover - defensive guard
        _LOGGER.exception(
            "Unexpected error importing Eventlet; continuing without monkey patching.",
        )
        return _set_patch_state(False)

    if _is_eventlet_monkey_patched(patcher.is_monkey_patched):
        _LOGGER.debug("Eventlet monkey patching already active.")
        return _set_patch_state(True)

    try:
        eventlet.monkey_patch()
    except Exception:  # pragma: no cover - defensive guard
        _LOGGER.exception(
            "Eventlet monkey patching failed; continuing without cooperative sockets.",
        )
        return _set_patch_state(False)

    _LOGGER.debug("Eventlet monkey patching applied successfully.")
    return _set_patch_state(True)
