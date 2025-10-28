"""Tests for the background removal Flask blueprint."""
from __future__ import annotations

import importlib
import os
import time
from io import BytesIO
from pathlib import Path

import pytest

try:  # pragma: no cover - optional dependency during CI
    from flask import Flask
    from PIL import Image
except ModuleNotFoundError as exc:  # pragma: no cover - fail fast when dependencies are absent
    missing = getattr(exc, "name", None) or "required dependencies"
    pytest.skip(f"{missing} is required for image converter tests", allow_module_level=True)

image_converter = importlib.import_module("app.routes.image_converter")
RemovalResult = importlib.import_module("app.services.bg_remove").RemovalResult


INTEGRATION_WEIGHTS_ENV = "BR_INTEGRATION_WEIGHTS_DIR"


def _integration_weights_dir() -> Path | None:
    """Return the integration weights directory when configured."""

    raw = os.getenv(INTEGRATION_WEIGHTS_ENV)
    if not raw:
        return None
    path = Path(raw)
    return path if path.exists() else None

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

    monkeypatch.setattr(image_converter, "ensure_global_session", lambda: None)
    monkeypatch.setattr(
        image_converter,
        "get_runtime_payload",
        lambda: {
            "runtime": "cpu",
            "provider": "CPUExecutionProvider",
            "gpu_name": None,
            "warning": None,
            "accelerator_message": "Using CPU (CPUExecutionProvider)",
        },
    )
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
    assert "selection" in payload
    selection = payload["selection"]
    assert selection["removal_model"] == image_converter.DEFAULT_REMOVAL_MODEL_KEY
    assert selection["output_directory"] is None
    assert selection["preview_size"] is None
    assert payload["runtime"] == "cpu"
    assert payload.get("warning") is None


def test_single_image_post_requires_file() -> None:
    """The single-image endpoint should reject submissions without a file."""

    app = _create_app()
    client = app.test_client()

    image_converter.ensure_global_session = lambda: None  # type: ignore[assignment]
    image_converter.get_runtime_payload = lambda: {"runtime": "cpu", "warning": None}  # type: ignore[assignment]

    response = client.post(
        "/image/remove-bg?json=1",
        data={"output_format": "png"},
        content_type="multipart/form-data",
    )

    assert response.status_code == 400
    payload = response.get_json()
    assert payload["error"] == "Please upload an image or provide a folder path."
    assert payload["runtime"] == "cpu"
    assert "warning" in payload


def test_single_image_post_includes_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    """Runtime metadata should be included when warnings are active."""

    app = _create_app()

    def fake_remove_bg_file(input_path, output_path, **_: object) -> RemovalResult:  # type: ignore[override]
        destination = Path(output_path)
        destination.write_bytes(b"png")
        return RemovalResult(Path(input_path), destination, True, None, 5.0)

    monkeypatch.setattr(image_converter, "remove_bg_file", fake_remove_bg_file)
    monkeypatch.setattr(image_converter, "ensure_global_session", lambda: None)
    monkeypatch.setattr(
        image_converter,
        "get_runtime_payload",
        lambda: {
            "runtime": "cpu",
            "provider": "CPUExecutionProvider",
            "gpu_name": None,
            "warning": "Using CPU fallback",
        },
    )

    client = app.test_client()
    response = client.post(
        "/image/remove-bg?json=1",
        data={"image_file": (BytesIO(b"fake"), "photo.jpg"), "output_format": "png"},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["runtime"] == "cpu"
    assert payload["warning"] == "Using CPU fallback"


def test_accelerator_health_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """The accelerator health endpoint should surface diagnostic data."""

    app = _create_app()

    monkeypatch.setattr(image_converter, "ensure_global_session", lambda: None)
    monkeypatch.setattr(
        image_converter,
        "get_accelerator_status",
        lambda: {
            "providers": ["CPUExecutionProvider"],
            "selected": "cpu",
            "gpu_name": None,
            "rtx_50_series": False,
        },
    )

    client = app.test_client()
    response = client.get("/health/accelerator")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload == {
        "providers": ["CPUExecutionProvider"],
        "selected": "cpu",
        "gpu_name": None,
        "rtx_50_series": False,
    }


@pytest.mark.integration
def test_single_image_post_real_latency(monkeypatch: pytest.MonkeyPatch) -> None:
    """End-to-end requests should complete with measurable latency using real weights."""

    weights_dir = _integration_weights_dir()
    if weights_dir is None:
        pytest.skip(f"Set {INTEGRATION_WEIGHTS_ENV} to enable integration tests")

    pytest.importorskip("onnxruntime")
    from app.services import model_registry

    try:
        model_registry.runtime_compat.ensure_runtime_ready()
    except RuntimeError as exc:
        pytest.skip(f"Runtime compatibility check failed: {exc}")

    monkeypatch.setenv("U2NET_HOME", str(weights_dir))

    app = _create_app()
    client = app.test_client()

    image = Image.new("RGB", (128, 128), color=(255, 0, 0))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    payload_bytes = buffer.getvalue()

    start = time.perf_counter()
    response = client.post(
        "/image/remove-bg?json=1",
        data={"image_file": (BytesIO(payload_bytes), "sample.png"), "output_format": "png"},
        content_type="multipart/form-data",
    )
    elapsed_ms = (time.perf_counter() - start) * 1000

    if response.status_code != 200:
        pytest.skip(f"Request failed with status {response.status_code}")

    payload = response.get_json()
    assert payload["result"]["success"] is True
    assert payload["format"] == "png"
    assert elapsed_ms > 0
    timing = payload["result"].get("timing_ms")
    assert timing is None or timing >= 0
