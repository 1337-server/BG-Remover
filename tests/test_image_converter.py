"""Tests for the Flask route providing the background removal UI."""
from __future__ import annotations

import io

import pytest
from flask.testing import FlaskClient
from PIL import Image

from app import create_app
from app.services import bg_remove


@pytest.fixture()
def client() -> FlaskClient:
    app = create_app(run_startup_tasks=False)
    app.config["SECRET_KEY"] = "testing"
    return app.test_client()


def _make_upload() -> tuple[io.BytesIO, str]:
    image = Image.new("RGBA", (4, 4), color=(0, 0, 0, 0))
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    buffer.seek(0)
    return buffer, "sample.png"


def test_get_request_returns_form(client: FlaskClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert b"Remove background" in response.data


def test_post_without_file_returns_error(client: FlaskClient) -> None:
    response = client.post("/", data={}, follow_redirects=True)
    assert response.status_code == 400
    assert b"Please choose an image" in response.data


def test_post_with_image_displays_result(
    client: FlaskClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(bg_remove, "_rembg_remove", lambda data, session: data)
    upload = _make_upload()
    response = client.post(
        "/",
        data={"image": upload, "output_format": "png"},
        content_type="multipart/form-data",
        follow_redirects=True,
    )
    assert response.status_code == 200
    assert b"Background removed successfully" in response.data
    assert b"data:image/png;base64" in response.data
