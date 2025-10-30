"""Integration tests for the rebuilt Flask runtime."""
from __future__ import annotations

import io
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from werkzeug.datastructures import MultiDict

from bgremover_core.config import Config
from bgremover_core.processing.pipeline import PipelineError
from runtimes.flask import routes
from runtimes.flask.app import create_app


def _mock_successful_batch_executor(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch the batch executor to synchronously generate placeholder outputs."""

    def fake_execute_batch_job(
        app,
        job,
        *,
        batch_source,
        output_dir,
        options,
        config,
        recursive,
        remember_preferences,
    ):
        with app.app_context():
            store = routes._get_store()
            output_dir.mkdir(parents=True, exist_ok=True)
            job.emit(
                'started',
                {
                    "total": job.total_items,
                    "label": batch_source.label,
                    "recursive": recursive,
                },
            )
            generated: list[Path] = []
            for candidate in batch_source.candidates:
                dest_parent = (
                    output_dir / candidate.relative_path.parent
                    if recursive and len(candidate.relative_path.parts) > 1
                    else output_dir
                )
                dest_parent.mkdir(parents=True, exist_ok=True)
                destination = dest_parent / f"{candidate.relative_path.stem}.png"
                Image.new("RGBA", (2, 2), color=(255, 255, 255, 255)).save(destination)
                generated.append(destination)
                job.success_count += 1
                job.size_bytes += destination.stat().st_size
                job.emit(
                    'item_success',
                    {
                        'input': str(candidate.relative_path),
                        'output': str(destination.relative_to(output_dir)),
                    },
                )
            record_options = routes._serialise_options({"batch": True, **options})
            record_options["output_directory"] = str(output_dir)
            zip_path = output_dir / f"batch_{job.identifier}.zip"
            with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for file_path in generated:
                    archive.write(file_path, arcname=str(file_path.relative_to(output_dir)))
            job.zip_path = zip_path
            store.add_record(
                routes.ResultRecord(
                    identifier=job.identifier,
                    original_name=batch_source.label,
                    result_name=zip_path.name,
                    path=zip_path,
                    mime_type="application/zip",
                    created_at=datetime.now(UTC),
                    options=record_options,
                    size_bytes=zip_path.stat().st_size,
                )
            )
            summary = {
                "total": job.total_items,
                "success": job.success_count,
                "failed": job.total_items - job.success_count,
                "size_bytes": job.size_bytes,
            }
            job.emit(
                'finished',
                {
                    'status': 'finished',
                    'summary': summary,
                    'download_url': f"/download/{job.identifier}",
                    'output_dir': str(output_dir),
                },
            )
            job.mark_finished()

    monkeypatch.setattr(routes, "_execute_batch_job", fake_execute_batch_job)


@pytest.fixture()
def flask_app(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Return a configured Flask app with mocked dependencies."""

    config = Config(model_dir=tmp_path)
    monkeypatch.setattr(routes, "detect_providers", lambda _hints=None: ["CPUExecutionProvider"])
    monkeypatch.setattr(routes, "load_config", lambda *_, **__: config)
    monkeypatch.setattr(routes, "persist_config", lambda *_args, **_kwargs: tmp_path / "config.json")

    def fake_remove_background(image_array, model_key, **_kwargs):  # pragma: no cover - patched per test
        return np.zeros_like(image_array)

    monkeypatch.setattr(routes, "remove_background", fake_remove_background)

    app = create_app({"TESTING": True, "SECRET_KEY": "test-secret", "OUTPUT_DIR": tmp_path / "web"})
    return app


@pytest.fixture()
def client(flask_app):
    return flask_app.test_client()


def _image_bytes(color: tuple[int, int, int, int] = (255, 255, 255, 255)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGBA", (4, 4), color=color).save(buffer, format="PNG")
    return buffer.getvalue()


def test_home_redirects(client) -> None:
    response = client.get("/", follow_redirects=False)
    assert response.status_code == 302
    assert response.headers["Location"].endswith("/single")


def test_single_page_renders(client) -> None:
    response = client.get("/single")
    assert response.status_code == 200
    assert b"Upload images" in response.data
    assert b"Advanced settings" in response.data
    html = response.data.decode("utf-8")
    assert '"model_specs"' in html
    assert '"isnet-general-use"' in html
    assert "Input size" in html


def test_batch_page_renders(client) -> None:
    response = client.get("/batch")
    assert response.status_code == 200
    assert b"Batch job log" in response.data
    assert b"Advanced settings" in response.data


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


def test_batch_processing_success(monkeypatch: pytest.MonkeyPatch, client, flask_app) -> None:
    _mock_successful_batch_executor(monkeypatch)

    folder_files = [
        (io.BytesIO(_image_bytes()), "animals/cat.png"),
        (io.BytesIO(_image_bytes((120, 40, 255, 255))), "animals/dogs/dog.png"),
    ]
    data = MultiDict(
        [("folder_files", file_tuple) for file_tuple in folder_files]
        + [
            ("folder_path", "animals"),
            ("model_key", "isnet-general-use"),
            ("recursive", "true"),
        ]
    )

    response = client.post(
        "/api/process/batch",
        data=data,
        content_type="multipart/form-data",
    )
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["status"] == "accepted"
    job_id = payload["job_id"]
    assert job_id

    manager = flask_app.extensions["batch_manager"]
    job = manager.get(job_id)
    assert job is not None
    assert job._future is not None
    job._future.result(timeout=2)

    files_response = client.get(f"/api/process/batch/{job_id}/files")
    assert files_response.status_code == 200
    files_payload = files_response.get_json()
    files_list = set(files_payload["files"])
    assert "animals/cat.png" in files_list
    assert "animals/dogs/dog.png" in files_list

    download_response = client.get(f"/download/{job_id}")
    assert download_response.status_code == 200
    assert download_response.headers["Content-Type"] == "application/zip"


def test_batch_processing_server_folder(
    monkeypatch: pytest.MonkeyPatch,
    client,
    flask_app,
    tmp_path: Path,
) -> None:
    _mock_successful_batch_executor(monkeypatch)

    server_folder = tmp_path / "server-input"
    (server_folder / "cats").mkdir(parents=True, exist_ok=True)
    (server_folder / "dogs" / "working").mkdir(parents=True, exist_ok=True)
    Image.new("RGBA", (2, 2), color=(255, 255, 255, 255)).save(server_folder / "cats" / "cat.png")
    Image.new("RGBA", (2, 2), color=(0, 0, 255, 255)).save(server_folder / "dogs" / "working" / "dog.png")

    data = {
        "folder_path": str(server_folder),
        "model_key": "isnet-general-use",
        "recursive": "true",
    }

    response = client.post(
        "/api/process/batch",
        data=data,
        content_type="multipart/form-data",
    )
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["status"] == "accepted"
    job_id = payload["job_id"]

    manager = flask_app.extensions["batch_manager"]
    job = manager.get(job_id)
    assert job is not None
    assert job._future is not None
    job._future.result(timeout=2)

    files_response = client.get(f"/api/process/batch/{job_id}/files")
    assert files_response.status_code == 200
    files_payload = files_response.get_json()
    files_list = set(files_payload["files"])
    assert "cats/cat.png" in files_list
    assert "dogs/working/dog.png" in files_list

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
