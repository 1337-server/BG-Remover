"""Tests for the Flask route providing the background removal UI."""
from __future__ import annotations

import io

import pytest
from flask.testing import FlaskClient
from PIL import Image

import bg_removal as bg_remove
from app import create_app


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
    response = client.post("/?json=1", data={}, follow_redirects=False)
    assert response.status_code == 400
    payload = response.get_json()
    assert payload is not None
    assert payload["error"].startswith("Please choose an image")


def test_post_with_image_displays_result(
    client: FlaskClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    class DummySession:
        providers_available = ("CPUExecutionProvider",)

    monkeypatch.setattr(bg_remove, "_load_session", lambda model_name: DummySession())
    monkeypatch.setattr(
        bg_remove,
        "_predict_mask",
        lambda image, session: Image.new("L", image.size, color=255),
    )
    upload = _make_upload()
    response = client.post(
        "/?json=1",
        data={"image_file": upload, "output_format": "png"},
        content_type="multipart/form-data",
        follow_redirects=False,
    )
    assert response.status_code == 200
    payload = response.get_json()
    assert payload is not None
    assert payload["mime_type"] == "image/png"
    assert payload["result"]["success"] is True
    assert payload["image_base64"].startswith("data:image/png;base64,")


def test_post_uses_requested_model_session(
    client: FlaskClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ensure only the requested model session is initialised for POST submissions."""

    calls: list[str | None] = []

    def fake_ensure(model_name: str | None = None) -> None:
        calls.append(model_name)

    runtime_call_count = 0

    def fake_runtime_payload() -> dict[str, object]:
        nonlocal runtime_call_count
        runtime_call_count += 1
        return {"providers_available": ["CPUExecutionProvider"]}

    monkeypatch.setattr("app.ensure_global_session", fake_ensure)
    monkeypatch.setattr("app.get_runtime_payload", fake_runtime_payload)

    response = client.post(
        "/?json=1",
        data={"removal_model": "general_high_quality"},
        follow_redirects=False,
    )

    assert response.status_code == 400
    assert calls == ["briaai/RMBG-2.0"]
    assert runtime_call_count == 1
