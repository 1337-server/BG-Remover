"""Regression tests for the ``serve`` module entry point."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

import serve


def test_should_run_startup_tasks_without_debug(monkeypatch: pytest.MonkeyPatch) -> None:
    """Startup tasks should run when debug mode is disabled."""

    monkeypatch.delenv("WERKZEUG_RUN_MAIN", raising=False)
    assert serve._should_run_startup_tasks(debug=False)


def test_should_run_startup_tasks_in_reloader_child(monkeypatch: pytest.MonkeyPatch) -> None:
    """The reloader child process must execute the heavy startup tasks."""

    monkeypatch.setenv("WERKZEUG_RUN_MAIN", "true")
    assert serve._should_run_startup_tasks(debug=True)


def test_should_skip_startup_tasks_in_reloader_supervisor(monkeypatch: pytest.MonkeyPatch) -> None:
    """The reloader supervisor should defer heavy startup initialisation."""

    monkeypatch.delenv("WERKZEUG_RUN_MAIN", raising=False)
    assert not serve._should_run_startup_tasks(debug=True)


@dataclass(slots=True)
class _DummyEvent:
    """Simple stand-in for a threading event used during testing."""

    is_set_result: bool = False

    def is_set(self) -> bool:
        """Return the configured result to mirror :class:`threading.Event`."""

        return self.is_set_result


class _DummyApp:
    """Minimal Flask-like object that stops execution during tests."""

    def __init__(self) -> None:
        self.config: dict[str, Any] = {}

    def run(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover - deliberate exit point
        raise SystemExit("stop server")


def _dummy_create_app(*, config_overrides: Any, run_startup_tasks: bool) -> _DummyApp:
    """Return a fake Flask application for invoking :func:`serve.main`."""

    assert run_startup_tasks is False
    return _DummyApp()


def test_main_skips_startup_tasks_when_supervisor(monkeypatch: pytest.MonkeyPatch) -> None:
    """The reloader supervisor process must not trigger model loading."""

    monkeypatch.delenv("WERKZEUG_RUN_MAIN", raising=False)
    monkeypatch.setattr(serve, "create_app", _dummy_create_app)
    spawn_calls: list[Any] = []

    def _capture_spawn(config: Any) -> _DummyEvent:
        spawn_calls.append(config)
        return _DummyEvent()

    monkeypatch.setattr(serve, "_spawn_startup_thread", _capture_spawn)
    monkeypatch.setattr(serve, "_log_json", lambda *args, **kwargs: None)

    with pytest.raises(SystemExit):
        serve.main(["--debug"])

    assert spawn_calls == []


def test_main_invokes_startup_tasks_in_child(monkeypatch: pytest.MonkeyPatch) -> None:
    """The serving process should spawn the startup worker exactly once."""

    monkeypatch.setenv("WERKZEUG_RUN_MAIN", "true")
    monkeypatch.setattr(serve, "create_app", _dummy_create_app)
    spawn_calls: list[Any] = []

    def _capture_spawn(config: Any) -> _DummyEvent:
        spawn_calls.append(config)
        return _DummyEvent()

    monkeypatch.setattr(serve, "_spawn_startup_thread", _capture_spawn)
    monkeypatch.setattr(serve, "_log_json", lambda *args, **kwargs: None)

    with pytest.raises(SystemExit):
        serve.main(["--debug"])

    assert len(spawn_calls) == 1
