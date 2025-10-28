"""Tests for the background removal Flask blueprint."""
from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pytest

pytest.importorskip("flask")

from flask import Flask

from app.routes import image_converter
from app.services.bg_remove import AcceleratorStatus, RemovalResult


def _create_app() -> Flask:
    """Create a Flask application configured with the image converter blueprint."""

    app = Flask(__name__)
    app.register_blueprint(image_converter.image_converter_bp)
    app.config.update(TESTING=True)
    return app


def test_serialise_results_provides_preview_links(tmp_path: Path) -> None:
    """Successful folder conversions should expose downloadable and previewable URLs."""

    app = _create_app()

    image_converter._FILE_REGISTRY.clear()
    image_converter._PREVIEW_REGISTRY.clear()

    input_path = tmp_path / "input.jpg"
    output_path = tmp_path / "output.png"
    input_path.write_bytes(b"input")
    output_path.write_bytes(b"output")

    result = RemovalResult(
        path_in=input_path,
        path_out=output_path,
        success=True,
        error=None,
        timing_ms=12.5,
    )

    with app.test_request_context():
        payload = image_converter._serialise_results([result])

    entry = payload["results"][0]
    assert entry["download_url"].startswith("/image/remove-bg/file/")
    assert entry["preview_url"].startswith("/image/remove-bg/preview/")
    assert entry["mime_type"] == "image/png"
    assert entry["format"] == "png"

    client = app.test_client()

    preview_response = client.get(entry["preview_url"])
    assert preview_response.status_code == 200
    assert preview_response.mimetype == "image/png"
    assert "attachment" not in preview_response.headers.get("Content-Disposition", "").lower()

    download_response = client.get(entry["download_url"])
    assert download_response.status_code == 200
    assert download_response.mimetype == "image/png"
    assert "attachment" in download_response.headers.get("Content-Disposition", "").lower()


def test_single_image_post_returns_json(monkeypatch: pytest.MonkeyPatch) -> None:
    """Submitting a single image should return the final processed payload."""

    app = _create_app()

    def fake_remove_bg_file(input_path, output_path, **_: object) -> RemovalResult:  # type: ignore[override]
        destination = Path(output_path)
        destination.write_bytes(b"png")
        return RemovalResult(Path(input_path), destination, True, None, 8.4)

    monkeypatch.setattr(image_converter, "remove_bg_file", fake_remove_bg_file)

    client = app.test_client()
    data = {
        "image_file": (BytesIO(b"fake image"), "sample.jpg"),
        "output_format": "png",
    }
    response = client.post(
        "/image/remove-bg?json=1",
        data=data,
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["mime_type"] == "image/png"
    assert payload["format"] == "png"
    assert payload["download_name"].endswith(".png")
    assert payload["result"]["success"] is True
    assert payload["image_base64"] == "cG5n"
    assert payload["accelerator"]["runtime"] in {"cpu", "cuda"}
    assert "providers" in payload["accelerator"]


def test_single_image_post_requires_file() -> None:
    """The single-image endpoint should reject submissions without a file."""

    app = _create_app()
    client = app.test_client()

    response = client.post(
        "/image/remove-bg?json=1",
        data={"output_format": "png"},
        content_type="multipart/form-data",
    )

    assert response.status_code == 400
    payload = response.get_json()
    assert payload["error"] == "Please upload an image or provide a folder path."
    assert payload["accelerator"]["runtime"] in {"cpu", "cuda"}


def test_accelerator_warning_included_in_json(monkeypatch: pytest.MonkeyPatch) -> None:
    """JSON responses should surface accelerator warnings when provided."""

    app = _create_app()

    warning_status = AcceleratorStatus(
        provider="cpu",
        provider_options={},
        available_providers=("CPUExecutionProvider",),
        gpu_name=None,
        rtx_50_series=False,
        warning="Running on CPU due to missing CUDAExecutionProvider.",
        requested_mode="auto",
        provider_description="CPUExecutionProvider",
    )

    def fake_remove_bg_file(input_path, output_path, **_: object) -> RemovalResult:  # type: ignore[override]
        destination = Path(output_path)
        destination.write_bytes(b"png")
        return RemovalResult(Path(input_path), destination, True, None, 5.2)

    monkeypatch.setattr(image_converter, "get_accelerator_status", lambda: warning_status)
    monkeypatch.setattr(image_converter, "remove_bg_file", fake_remove_bg_file)

    client = app.test_client()
    data = {
        "image_file": (BytesIO(b"fake image"), "sample.jpg"),
        "output_format": "png",
    }
    response = client.post(
        "/image/remove-bg?json=1",
        data=data,
        content_type="multipart/form-data",
    )

    payload = response.get_json()
    assert payload["accelerator"]["warning"] == warning_status.warning


def test_accelerator_health_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """The accelerator health endpoint should expose provider metadata."""

    app = _create_app()

    status = AcceleratorStatus(
        provider="cuda",
        provider_options={"device_id": 0},
        available_providers=("CUDAExecutionProvider", "CPUExecutionProvider"),
        gpu_name="NVIDIA GeForce RTX 5090",
        rtx_50_series=True,
        warning=None,
        requested_mode="auto",
        provider_description="CUDAExecutionProvider (device 0)",
    )

    monkeypatch.setattr(image_converter, "get_accelerator_status", lambda: status)

    client = app.test_client()
    response = client.get("/health/accelerator")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["selected"] == "cuda"
    assert payload["rtx_50_series"] is True
    assert payload["gpu_name"] == status.gpu_name
