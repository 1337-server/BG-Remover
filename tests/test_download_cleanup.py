"""Tests ensuring generated ZIP downloads are cleaned up automatically."""
from __future__ import annotations

import time
from pathlib import Path

import pytest

import app as app_module
import bg_removal as bg_remove


@pytest.fixture()
def configured_app() -> app_module.Flask:
    """Return a Flask application configured for testing the download registry."""

    flask_app = app_module.create_app(run_startup_tasks=False)
    flask_app.config["SERVER_NAME"] = "example.test"
    return flask_app


@pytest.fixture(autouse=True)
def clear_zip_registry() -> None:
    """Ensure the ZIP registry is empty before and after each test case."""

    app_module._drain_zip_registry()
    yield
    app_module._drain_zip_registry()


def _write_dummy_result(target: Path) -> bg_remove.RemovalResult:
    """Return a successful ``RemovalResult`` pointing to ``target``."""

    target.write_bytes(b"dummy")
    format_spec = bg_remove.get_output_format_spec(".png")
    return bg_remove.RemovalResult(
        image=None,
        format_spec=format_spec,
        elapsed_ms=0.0,
        path_out=target,
    )


def test_zip_entry_expires_when_download_not_requested(
    configured_app: app_module.Flask, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify stale ZIP entries are evicted automatically when never downloaded."""

    monkeypatch.setattr(app_module, "ZIP_REGISTRY_TTL_SECONDS", 0.1, raising=False)
    result = _write_dummy_result(tmp_path / "image.png")

    with configured_app.test_request_context("/", base_url="http://example.test"):
        token, _ = app_module._create_zip([result])

    with app_module._ZIP_REGISTRY_LOCK:
        registry_item = app_module._ZIP_REGISTRY.get(token)
    assert registry_item is not None
    zip_path = registry_item.path
    temp_dir = registry_item.temp_dir
    assert zip_path.exists()
    assert temp_dir.exists()

    time.sleep(0.25)
    for _ in range(10):
        if not zip_path.exists() and not temp_dir.exists():
            break
        time.sleep(0.05)

    with app_module._ZIP_REGISTRY_LOCK:
        assert token not in app_module._ZIP_REGISTRY
    assert not zip_path.exists()
    assert not temp_dir.exists()
