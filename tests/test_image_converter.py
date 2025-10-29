"""Tests for the Flask route providing the background removal UI."""
from __future__ import annotations

import base64
import io
from pathlib import Path

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
        lambda image, session, model_options=None: Image.new("L", image.size, color=255),
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


def test_post_with_json_image_payload(
    client: FlaskClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    class DummySession:
        providers_available = ("CPUExecutionProvider",)

    monkeypatch.setattr(bg_remove, "_load_session", lambda model_name: DummySession())
    monkeypatch.setattr(
        bg_remove,
        "_predict_mask",
        lambda image, session, model_options=None: Image.new("L", image.size, color=255),
    )

    buffer, filename = _make_upload()
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")

    response = client.post(
        "/?json=1",
        json={"image_base64": encoded, "filename": filename, "output_format": "png"},
        follow_redirects=False,
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload is not None
    assert payload["result"]["success"] is True
    assert payload["mime_type"] == "image/png"
    assert payload["download_name"].endswith(".png")


def test_json_request_without_output_dir_avoids_disk_usage(
    client: FlaskClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ensure in-memory JSON responses do not persist temporary files."""

    class DummySession:
        providers_available = ("CPUExecutionProvider",)

    monkeypatch.setattr(bg_remove, "_load_session", lambda model_name: DummySession())

    spec = bg_remove.get_output_format_spec("png")
    call_args: dict[str, object] = {}

    def fake_remove_bg_file(
        input_path: Path | str,
        output: Path | str | None,
        *,
        save_to_disk: bool = True,
        retain_image: bool = False,
        **kwargs: object,
    ) -> bg_remove.RemovalResult:
        call_args["save_to_disk"] = save_to_disk
        call_args["output"] = output
        call_args["retain_image"] = retain_image
        assert save_to_disk is False
        assert output is None
        assert retain_image is True
        image = Image.new("RGBA", (2, 2), color=(255, 0, 0, 255))
        return bg_remove.RemovalResult(
            image=image,
            format_spec=spec,
            elapsed_ms=1.0,
            path_in=Path(input_path),
            path_out=None,
        )

    monkeypatch.setattr("app.remove_bg_file", fake_remove_bg_file)

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
    assert payload["result"]["path_out"] is None
    assert call_args["save_to_disk"] is False
    assert call_args["output"] is None


def test_json_request_with_output_dir_persists_file(
    client: FlaskClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ensure JSON downloads honour requested persistent output directories."""

    class DummySession:
        providers_available = ("CPUExecutionProvider",)

    monkeypatch.setattr(bg_remove, "_load_session", lambda model_name: DummySession())
    monkeypatch.setattr(
        bg_remove,
        "_predict_mask",
        lambda image, session, model_options=None: Image.new("L", image.size, color=255),
    )

    original_remove_bg_file = bg_remove.remove_bg_file
    call_args: dict[str, object] = {}

    def tracking_remove_bg_file(*args: object, **kwargs: object) -> bg_remove.RemovalResult:
        call_args["save_to_disk"] = kwargs.get("save_to_disk")
        result = original_remove_bg_file(*args, **kwargs)
        call_args["path_out"] = result.path_out
        return result

    monkeypatch.setattr("app.remove_bg_file", tracking_remove_bg_file)

    upload = _make_upload()
    output_dir = tmp_path / "exports"
    response = client.post(
        "/?json=1",
        data={
            "image_file": upload,
            "output_format": "png",
            "single_output_dir": str(output_dir),
        },
        content_type="multipart/form-data",
        follow_redirects=False,
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload is not None
    assert call_args["save_to_disk"] is True
    saved_path = call_args["path_out"]
    assert isinstance(saved_path, Path)
    assert saved_path.exists()
    assert payload["result"]["path_out"] == str(saved_path)


def test_invalid_model_option_returns_error(client: FlaskClient) -> None:
    buffer, filename = _make_upload()
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")

    response = client.post(
        "/?json=1",
        json={
            "image_base64": encoded,
            "filename": filename,
            "output_format": "png",
            "removal_model": "general_high_quality",
            "model_options": {"input_size": 9000},
        },
        follow_redirects=False,
    )

    assert response.status_code == 400
    payload = response.get_json()
    assert payload is not None
    assert "input_size" in payload["error"]


def test_model_options_passed_to_remove_bg_file(
    client: FlaskClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    class DummySession:
        providers_available = ("CPUExecutionProvider",)

    monkeypatch.setattr("app.ensure_global_session", lambda model_name=None: DummySession())

    spec = bg_remove.get_output_format_spec("png")
    captured: dict[str, object] = {}

    def fake_remove_bg_file(
        input_path: Path | str,
        output: Path | str | None,
        **kwargs: object,
    ) -> bg_remove.RemovalResult:
        captured["model_options"] = kwargs.get("model_options")
        image = Image.new("RGBA", (2, 2), color=(255, 0, 0, 255))
        return bg_remove.RemovalResult(
            image=image,
            format_spec=spec,
            elapsed_ms=1.0,
            path_in=Path(input_path),
            path_out=None,
        )

    monkeypatch.setattr("app.remove_bg_file", fake_remove_bg_file)

    buffer, filename = _make_upload()
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")

    response = client.post(
        "/?json=1",
        json={
            "image_base64": encoded,
            "filename": filename,
            "output_format": "png",
            "removal_model": "complex_scene",
            "model_options": {
                "precision": "fp32",
                "segmentation_points": 12,
                "prompt_mode": "foreground",
            },
        },
        follow_redirects=False,
    )

    assert response.status_code == 200
    assert captured["model_options"] == {
        "precision": "fp32",
        "segmentation_points": 12,
        "prompt_mode": "foreground",
    }


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
