"""Integration tests for the rebuilt Flask runtime."""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from bgremover_core.config import Config
from bgremover_core.processing.pipeline import PipelineError, Report, ReportEntry
from runtimes.flask_app import routes
from runtimes.flask_app.app import create_app


@pytest.fixture()
def flask_app(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Return a configured Flask app with mocked dependencies."""

    config = Config(model_dir=tmp_path)
    monkeypatch.setattr(routes, "detect_providers", lambda _hints=None: ["CPUExecutionProvider"])
    monkeypatch.setattr(routes, "load_config", lambda: config)
    monkeypatch.setattr(routes, "persist_config", lambda *_args, **_kwargs: tmp_path / "config.json")

    def fake_remove_background(image_array, model_key, **_kwargs):  # pragma: no cover - patched per test
        return np.zeros_like(image_array)

    monkeypatch.setattr(routes, "remove_background", fake_remove_background)

    def fake_process_folder(input_dir, output_dir, **_kwargs):
        output_root = Path(output_dir)
        output_root.mkdir(parents=True, exist_ok=True)
        dummy_input = Path(input_dir) / "dummy.png"
        dummy_input.parent.mkdir(parents=True, exist_ok=True)
        dummy_input.touch()
        output_path = output_root / "batch.png"
        Image.new("RGBA", (2, 2), color=(255, 255, 255, 255)).save(output_path)
        entry = ReportEntry(
            path_in=dummy_input,
            path_out=output_path,
            success=True,
            elapsed_ms=1.0,
            error=None,
        )
        return Report([entry])

    monkeypatch.setattr(routes, "process_folder", fake_process_folder)

    app = create_app({"TESTING": True, "SECRET_KEY": "test-secret", "OUTPUT_DIR": tmp_path / "web"})
    return app


@pytest.fixture()
def client(flask_app):
    return flask_app.test_client()


def _image_bytes(color: tuple[int, int, int, int] = (255, 255, 255, 255)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGBA", (4, 4), color=color).save(buffer, format="PNG")
    return buffer.getvalue()


def test_index_renders(client) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert b"Upload images" in response.data
    assert b"Processing history" in response.data
    html = response.data.decode("utf-8")
    assert '"model_specs"' in html
    assert '"isnet-general-use"' in html
    assert "Input size" in html


def test_process_images_success(monkeypatch: pytest.MonkeyPatch, client) -> None:
    captured = {}

    def recorder(array, model_key, **kwargs):
        captured.update(kwargs)
        captured["model_key"] = model_key
        return np.zeros_like(array)

    monkeypatch.setattr(routes, "remove_background", recorder)

    image_buffer = io.BytesIO(_image_bytes())
    data = {
        "images": (image_buffer, "example.png"),
        "model_key": "isnet-general-use",
        "feather_radius": "5",
        "background_color": "#00ff00",
        "transparent": "false",
        "alpha_matting": "true",
        "mask_blur": "7",
        "output_format": "PNG",
    }
    response = client.post("/process", data=data, content_type="multipart/form-data")
    assert response.status_code == 200
    payload = response.get_json()
    assert len(payload["results"]) == 1
    result = payload["results"][0]
    assert result["result_name"].endswith(".png")
    assert result["options"]["background_color"] == "#00ff00"
    assert result["options"]["alpha_matting"] is True
    assert result["options"]["mask_blur"] == 7
    assert captured["model_key"] == "isnet-general-use"
    assert captured["feather_radius"] == 5
    assert captured["alpha_matting"] is True
    assert captured["mask_blur"] == pytest.approx(7.0)
    assert captured["background_color"] == (0, 255, 0)


def test_process_images_failure(monkeypatch: pytest.MonkeyPatch, client) -> None:
    def failing(*_args, **_kwargs):
        raise PipelineError("boom")

    monkeypatch.setattr(routes, "remove_background", failing)
    image_buffer = io.BytesIO(_image_bytes())
    response = client.post(
        "/process",
        data={"images": (image_buffer, "bad.png"), "model_key": "isnet-general-use"},
        content_type="multipart/form-data",
    )
    assert response.status_code == 422
    payload = response.get_json()
    assert payload["error"] == "boom"


def test_batch_processing_success(monkeypatch: pytest.MonkeyPatch, client) -> None:
    captured = {}

    def recorder(input_dir, output_dir, **kwargs):
        captured.update(kwargs)
        output_root = Path(output_dir)
        output_root.mkdir(parents=True, exist_ok=True)
        dummy_input = Path(input_dir) / "dummy.png"
        dummy_input.parent.mkdir(parents=True, exist_ok=True)
        dummy_input.touch()
        output_path = output_root / "batch.png"
        Image.new("RGBA", (2, 2), color=(255, 255, 255, 255)).save(output_path)
        entry = ReportEntry(
            path_in=dummy_input,
            path_out=output_path,
            success=True,
            elapsed_ms=1.0,
            error=None,
        )
        return Report([entry])

    monkeypatch.setattr(routes, "process_folder", recorder)

    archive_buffer = io.BytesIO()
    with zipfile.ZipFile(archive_buffer, "w") as archive:
        archive.writestr("sample.png", _image_bytes((120, 40, 255, 255)))
    archive_buffer.seek(0)

    response = client.post(
        "/batch",
        data={
            "archive": (archive_buffer, "folder.zip"),
            "model_key": "isnet-general-use",
            "alpha_matting": "true",
            "mask_blur": "3",
        },
        content_type="multipart/form-data",
    )
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["summary"]["success"] == 1
    download_response = client.get(payload["download_url"])
    assert download_response.status_code == 200
    assert download_response.headers["Content-Type"] == "application/zip"
    assert captured["alpha_matting"] is True
    assert captured["mask_blur"] == pytest.approx(3.0)


def test_history_endpoint_updates_after_processing(client) -> None:
    image_buffer = io.BytesIO(_image_bytes())
    client.post(
        "/process",
        data={"images": (image_buffer, "history.png"), "model_key": "isnet-general-use"},
        content_type="multipart/form-data",
    )
    history_response = client.get("/history")
    assert history_response.status_code == 200
    payload = history_response.get_json()
    assert payload["history"]
    assert payload["history"][0]["original_name"] == "history.png"
