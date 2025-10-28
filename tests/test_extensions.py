"""Tests covering application extensions for missing dependencies."""
from __future__ import annotations

import pytest

from app import extensions


def test_stub_socketio_start_background_task_raises_runtime_error() -> None:
    """The stub implementation should raise a helpful error when used."""

    if extensions._SOCKETIO_IMPORT_ERROR is None:  # pragma: no cover - depends on env
        pytest.skip("Flask-SocketIO is available; stub is not used.")

    stub = extensions.SocketIO()

    with pytest.raises(RuntimeError) as excinfo:
        stub.start_background_task(lambda: None)

    assert "Flask-SocketIO is required" in str(excinfo.value)
