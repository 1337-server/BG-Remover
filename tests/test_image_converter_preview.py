"""Tests covering preview URLs returned by the folder conversion endpoint."""
from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pytest

pytest.importorskip("flask")

from flask import Flask

from app.routes import image_converter
from app.services.bg_remove import RemovalResult
from PIL import Image


def test_serialise_results_provides_preview_links(tmp_path: Path) -> None:
    """Successful folder conversions should expose downloadable and previewable URLs."""

    app = Flask(__name__)
    app.register_blueprint(image_converter.image_converter_bp)

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

    client = app.test_client()

    preview_response = client.get(entry["preview_url"])
    assert preview_response.status_code == 200
    assert preview_response.mimetype == "image/png"
    assert "attachment" not in preview_response.headers.get("Content-Disposition", "").lower()

    download_response = client.get(entry["download_url"])
    assert download_response.status_code == 200
    assert download_response.mimetype == "image/png"
    assert "attachment" in download_response.headers.get("Content-Disposition", "").lower()


def test_remove_bg_live_triggers_background_processing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The live endpoint should enqueue work and emit websocket messages."""

    app = Flask(__name__)
    app.register_blueprint(image_converter.image_converter_bp)
    app.config.update(TESTING=True)

    class DummySocket:
        def __init__(self) -> None:
            self.events: list[tuple[str, dict]] = []

        def emit(self, event: str, payload: dict, namespace: str | None = None, to: str | None = None) -> None:
            self.events.append((event, payload))

        def start_background_task(self, target, *args, **kwargs):  # type: ignore[no-untyped-def]
            target(*args, **kwargs)

    dummy_socket = DummySocket()
    monkeypatch.setattr(image_converter, "socketio", dummy_socket)

    def fake_remove_bg_file(*args, **kwargs):  # type: ignore[override]
        input_path = Path(args[0])
        output_path = Path(args[1])
        output_path.write_bytes(b"png")
        progress_callback = kwargs.get("progress_callback")
        preview_callback = kwargs.get("preview_callback")
        if callable(progress_callback):
            progress_callback("mask", 50.0)
        if callable(preview_callback):
            preview_callback(Image.new("RGBA", (1, 1), (255, 255, 255, 128)), "initial")
        return RemovalResult(input_path, output_path, True, None, 10.0)

    monkeypatch.setattr(image_converter, "remove_bg_file", fake_remove_bg_file)

    client = app.test_client()
    data = {
        "socket_id": "abc123",
        "image_file": (BytesIO(b"fake image"), "sample.jpg"),
    }
    response = client.post(
        "/image/remove-bg/live",
        data=data,
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["status"] == "processing"
    assert isinstance(payload.get("job_id"), str)

    emitted_events = [event for event, _ in dummy_socket.events]
    assert "progress" in emitted_events
    assert "preview" in emitted_events
    assert "completed" in emitted_events


def test_remove_bg_live_returns_service_unavailable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The live endpoint should surface informative errors when Socket.IO is missing."""

    app = Flask(__name__)
    app.register_blueprint(image_converter.image_converter_bp)
    app.config.update(TESTING=True)

    class FailingSocket:
        def emit(self, *_: object, **__: object) -> None:
            """The stub emit simply records the attempt for compatibility."""

        def start_background_task(self, *_: object, **__: object) -> None:
            """Simulate the behaviour of the stub that cannot spawn tasks."""

            raise RuntimeError("Background tasks are unavailable in stub mode.")

    monkeypatch.setattr(image_converter, "socketio", FailingSocket())

    client = app.test_client()
    data = {
        "socket_id": "abc123",
        "image_file": (BytesIO(b"fake image"), "sample.jpg"),
    }

    response = client.post(
        "/image/remove-bg/live",
        data=data,
        content_type="multipart/form-data",
    )

    assert response.status_code == 503
    payload = response.get_json()
    assert payload == {
        "error": "Background tasks are unavailable in stub mode."
    }

