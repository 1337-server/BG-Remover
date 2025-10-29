"""Tests for the Flask runtime."""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from bgremover_core.config import Config
from runtimes.flask_app import routes
from runtimes.flask_app.app import create_app


@pytest.fixture()
def flask_app(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setattr(
        routes,
        "detect_providers",
        lambda _hints=None: ["CPUExecutionProvider"],
    )
    monkeypatch.setattr(routes, "load_config", lambda: Config(model_dir=tmp_path))
    app = create_app()
    app.config.update(TESTING=True, SECRET_KEY="test")
    return app


def test_index_renders(flask_app) -> None:
    client = flask_app.test_client()
    response = client.get("/")
    assert response.status_code == 200
    assert b"Background Remover" in response.data


def test_upload_success(monkeypatch: pytest.MonkeyPatch, flask_app, tmp_path: Path) -> None:
    client = flask_app.test_client()
    monkeypatch.setattr(
        routes,
        "remove_background",
        lambda *_args, **_kwargs: np.zeros((4, 4, 4), dtype=np.uint8),
    )
    buffer = io.BytesIO()
    Image.new("RGBA", (4, 4), color=(255, 255, 255, 255)).save(buffer, format="PNG")
    buffer.seek(0)
    data = {
        "image": (buffer, "test.png"),
        "model_key": "isnet-general-use",
        "feather_radius": "3",
    }
    response = client.post("/", data=data, content_type="multipart/form-data")
    assert response.status_code == 200
    assert response.headers["Content-Type"].startswith("image/png")


def test_upload_missing_file(flask_app) -> None:
    client = flask_app.test_client()
    response = client.post("/", data={}, follow_redirects=True)
    assert b"Please choose an image" in response.data
