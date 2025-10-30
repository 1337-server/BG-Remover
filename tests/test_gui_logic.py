"""Headless-safe tests for the GUI helpers."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from bgremover_core.config import Config
from runtimes.gui.bg_remover_gui import BackgroundRemoverApp


def test_active_config_uses_override(tmp_path: Path) -> None:
    app = BackgroundRemoverApp.__new__(BackgroundRemoverApp)
    app.app_config = Config(model_dir=tmp_path)
    app.model_dir_var = SimpleNamespace(get=lambda: str(tmp_path / "custom"))
    result = app._active_config()
    assert result.model_dir == tmp_path / "custom"


def test_log_appends_messages() -> None:
    messages: list[tuple[str, str | None]] = []

    def insert(_end, message, tag=None):
        messages.append((message.strip(), tag))

    app = BackgroundRemoverApp.__new__(BackgroundRemoverApp)
    app.log_widget = SimpleNamespace(insert=insert, see=lambda *_args, **_kwargs: None)
    app._log("Hello world")
    app._log("Something failed", error=True)
    assert messages[0] == ("Hello world", None)
    assert messages[1] == ("Something failed", "error")
