"""Tests for the simplified development server entry point."""
from __future__ import annotations

import pytest

import app as app_module


def test_parse_args_defaults() -> None:
    args = app_module.parse_server_args([])
    assert args.host == "0.0.0.0"
    assert args.port == 5000
    assert args.debug is False


def test_main_invokes_flask_run(monkeypatch: pytest.MonkeyPatch) -> None:
    called: dict[str, object] = {}

    class DummyApp:
        def run(self, *, host: str, port: int, debug: bool) -> None:  # pragma: no cover - executed in test
            called.update({"host": host, "port": port, "debug": debug})
            raise SystemExit

    monkeypatch.setattr(app_module, "create_app", lambda: DummyApp())

    with pytest.raises(SystemExit):
        app_module.run_dev_server(["--host", "127.0.0.1", "--port", "1234", "--debug"])

    assert called == {"host": "127.0.0.1", "port": 1234, "debug": True}
