from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from bgremover_core.config import Config
from bgremover_core.processing.pipeline import Report, ReportEntry
from runtimes.flask_app import routes
from runtimes.flask_app.app import create_app


@pytest.fixture()
def flask_app(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Return a configured Flask app with mocked dependencies."""

    config = Config(model_dir=tmp_path)
    monkeypatch.setattr(routes, "detect_providers", lambda _hints=None: ["CPUExecutionProvider"])
    monkeypatch.setattr(routes, "load_config", lambda: config)
    monkeypatch.setattr(routes, "persist_config", lambda *_args, **_kwargs: tmp_path / "config.json")
    monkeypatch.setattr(routes, "get_session", lambda *args, **kwargs: None)

    def fake_remove_background(image_array, model_key, **_kwargs):  # pragma: no cover - patched per test
        return np.zeros_like(image_array)

    monkeypatch.setattr(routes, "remove_background", fake_remove_background)

    def fake_process_folder(input_dir, output_dir, **kwargs):
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
        progress = kwargs.get("progress_callback")
        if progress:
            progress(entry)
        return Report([entry])

    monkeypatch.setattr(routes, "process_folder", fake_process_folder)

    app = create_app({"TESTING": True, "SECRET_KEY": "test-secret", "OUTPUT_DIR": tmp_path / "web"})

    class ImmediateExecutor:
        def submit(self, func, *args, **kwargs):
            func(*args, **kwargs)

    app.extensions["executor"] = ImmediateExecutor()
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
    html = response.data.decode("utf-8")
    assert "Upload images" in html
    assert "Advanced settings" in html


def test_api_models(client) -> None:
    response = client.get("/api/models")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["default_model"]
    assert payload["models"]


def test_api_help_options(client) -> None:
    response = client.get("/api/help/options")
    assert response.status_code == 200
    payload = response.get_json()
    assert "feather_radius" in payload["options"]


def test_process_single_success(tmp_path: Path, client) -> None:
    output_dir = tmp_path / "exports"
    image_buffer = io.BytesIO(_image_bytes())
    advanced = json.dumps(
        {
            "feather_radius": 5,
            "background_color": "#00ff00",
            "transparent": False,
            "output_format": "PNG",
        }
    )
    data = {
        "file": (image_buffer, "example.png"),
        "removal_model": "isnet-general-use",
        "output_dir": str(output_dir),
        "advanced": advanced,
    }
    response = client.post("/api/process/single", data=data, content_type="multipart/form-data")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["status"] == "ok"
    assert payload["model_used"] == "isnet-general-use"
    saved_path = Path(payload["output_path"])
    assert saved_path.exists()
    assert payload["timings"]["total_ms"] >= 0


def test_process_single_validation_error(tmp_path: Path, client) -> None:
    advanced = json.dumps({"feather_radius": 999})
    image_buffer = io.BytesIO(_image_bytes())
    response = client.post(
        "/api/process/single",
        data={
            "file": (image_buffer, "invalid.png"),
            "removal_model": "isnet-general-use",
            "advanced": advanced,
            "output_dir": str(tmp_path / "out"),
        },
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
    payload = response.get_json()
    assert payload["code"] == "VALIDATION_ERROR"


def test_process_batch_success(tmp_path: Path, client) -> None:
    archive_buffer = io.BytesIO()
    with zipfile.ZipFile(archive_buffer, "w") as archive:
        archive.writestr("sample.png", _image_bytes((120, 40, 255, 255)))
    archive_buffer.seek(0)

    advanced = json.dumps({"output_format": "PNG"})
    response = client.post(
        "/api/process/batch",
        data={
            "file": (archive_buffer, "folder.zip"),
            "removal_model": "isnet-general-use",
            "advanced": advanced,
            "output_dir": str(tmp_path / "batch-output"),
        },
        content_type="multipart/form-data",
    )
    assert response.status_code == 202
    payload = response.get_json()
    job_id = payload["job_id"]
    summary_response = client.get(f"/api/jobs/{job_id}/summary")
    assert summary_response.status_code == 200
    summary_payload = summary_response.get_json()
    assert summary_payload["status"] == "ok"
    job = summary_payload["job"]
    assert job["counts"]["total"] == 1
    assert Path(job["output_dir"]).exists()
