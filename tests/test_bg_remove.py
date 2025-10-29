"""Unit tests for the simplified background removal helpers."""
from __future__ import annotations

import io
from pathlib import Path

import pytest
from PIL import Image

import bg_removal as bg_remove


def _make_image_bytes(color: tuple[int, int, int, int] = (255, 0, 0, 255)) -> bytes:
    image = Image.new("RGBA", (4, 4), color=color)
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    return buffer.getvalue()


def test_get_output_format_spec_handles_extensions() -> None:
    spec = bg_remove.get_output_format_spec(".PNG")
    assert spec.key == "png"
    assert spec.mime_type == "image/png"


def test_remove_background_bytes_returns_result(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummySession:
        providers_available = ("CPUExecutionProvider",)

    call_count = 0

    def fake_predict(image: Image.Image, session: DummySession) -> Image.Image:
        nonlocal call_count
        call_count += 1
        mask = Image.new("L", image.size, color=255)
        return mask

    monkeypatch.setattr(bg_remove, "_load_session", lambda model_name: DummySession())
    monkeypatch.setattr(bg_remove, "_predict_mask", fake_predict)

    result = bg_remove.remove_background_bytes(_make_image_bytes())
    assert result.success
    assert result.format_spec.key == "png"
    assert call_count == 1
    restored = Image.open(io.BytesIO(result.as_bytes()))
    assert restored.mode in {"RGBA", "RGB"}


def test_remove_bg_file_writes_to_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "source.png"
    Image.new("RGBA", (2, 2), color=(0, 128, 255, 255)).save(source, "PNG")

    class DummySession:
        providers_available = ("CPUExecutionProvider",)

    def fake_predict(image: Image.Image, session: DummySession) -> Image.Image:
        return Image.new("L", image.size, color=255)

    monkeypatch.setattr(bg_remove, "_load_session", lambda model_name: DummySession())
    monkeypatch.setattr(bg_remove, "_predict_mask", fake_predict)

    result = bg_remove.remove_bg_file(source, tmp_path)
    assert result.path_out is not None
    assert result.path_out.exists()
    assert result.path_out.suffix == ".png"


def test_encode_result_image_returns_data_url(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummySession:
        providers_available = ("CPUExecutionProvider",)

    monkeypatch.setattr(bg_remove, "_load_session", lambda model_name: DummySession())
    monkeypatch.setattr(
        bg_remove,
        "_predict_mask",
        lambda image, session: Image.new("L", image.size, color=255),
    )
    result = bg_remove.remove_background_bytes(_make_image_bytes())
    data_url = bg_remove.encode_result_image(result)
    assert data_url.startswith("data:image/png;base64,")


def test_download_model_from_huggingface(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Ensure Hugging Face hosted models download through the hub helper."""

    spec = bg_remove.MODEL_SPECS["matting-by-generation"]
    monkeypatch.setattr(bg_remove, "MODEL_DOWNLOAD_ROOT", tmp_path)

    assert spec.huggingface_filename is not None

    def fake_download(
        spec_arg: bg_remove.ModelSpec, url: str, destination: Path, headers: dict[str, str]
    ) -> Path:
        assert spec_arg is spec
        assert "Authorization" not in headers
        destination.write_bytes(b"onnx")
        return destination

    monkeypatch.setattr(bg_remove, "_download_model_via_http", fake_download)
    path = bg_remove._download_model(spec)
    expected_path = tmp_path / f"{spec.key}.onnx"
    assert path == expected_path
    assert expected_path.exists()
    assert expected_path.read_bytes() == b"onnx"


def test_download_model_from_huggingface_uses_token(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Attach Hugging Face tokens from the environment when provided."""

    spec = bg_remove.MODEL_SPECS["sam_segmentation_model"]
    monkeypatch.setattr(bg_remove, "MODEL_DOWNLOAD_ROOT", tmp_path)
    monkeypatch.setenv("HUGGINGFACEHUB_API_TOKEN", "secret")

    captured_headers: dict[str, str] = {}

    def fake_download(
        spec_arg: bg_remove.ModelSpec, url: str, destination: Path, headers: dict[str, str]
    ) -> Path:
        captured_headers.update(headers)
        destination.write_bytes(b"onnx")
        return destination

    monkeypatch.setattr(bg_remove, "_download_model_via_http", fake_download)
    path = bg_remove._download_model(spec)
    expected_path = tmp_path / f"{spec.key}.onnx"
    assert path == expected_path
    assert captured_headers.get("Authorization") == "Bearer secret"


def test_download_model_from_huggingface_reports_auth_issue(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Surface a helpful message when Hugging Face access is denied."""

    spec = bg_remove.MODEL_SPECS["briaai/RMBG-2.0"]
    monkeypatch.setattr(bg_remove, "MODEL_DOWNLOAD_ROOT", tmp_path)

    def fake_download(
        spec_arg: bg_remove.ModelSpec, url: str, destination: Path, headers: dict[str, str]
    ) -> Path:
        raise RuntimeError("HTTP Error 401: Unauthorized")

    monkeypatch.setattr(bg_remove, "_download_model_via_http", fake_download)

    with pytest.raises(RuntimeError) as excinfo:
        bg_remove._download_model(spec)

    assert "authentication" in str(excinfo.value).lower()


def test_all_removal_models_have_specs() -> None:
    """Ensure every configured UI model maps to a downloadable spec."""

    import importlib

    app_module = importlib.import_module("app")
    configured_keys = {option["model_name"] for option in app_module.REMOVAL_MODEL_OPTIONS}
    assert configured_keys.issubset(bg_remove.MODEL_SPECS)


def test_resolve_huggingface_token_reads_cached_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Load a cached Hugging Face token when environment variables are missing."""

    for variable in ("HUGGINGFACEHUB_API_TOKEN", "HF_API_TOKEN"):
        monkeypatch.delenv(variable, raising=False)

    home_dir = tmp_path / "home"
    huggingface_dir = home_dir / ".huggingface"
    huggingface_dir.mkdir(parents=True)
    (huggingface_dir / "token").write_text("hf_secret_token\n", encoding="utf-8")

    monkeypatch.setattr(bg_remove.Path, "home", classmethod(lambda cls: home_dir))

    token = bg_remove._resolve_huggingface_token()
    assert token == "hf_secret_token"


def test_ensure_models_downloaded_defers_until_requested(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Ensure downloads are not triggered until a session requests a model."""

    monkeypatch.setattr(bg_remove, "MODELS_DIRECTORY", tmp_path)
    monkeypatch.setattr(bg_remove, "MODEL_DOWNLOAD_ROOT", tmp_path)

    def raise_prefetch(spec: bg_remove.ModelSpec) -> None:
        raise AssertionError("Prefetch should not run during deferral test")

    monkeypatch.setattr(bg_remove, "_schedule_prefetch", raise_prefetch)

    download_called = False

    def fail_download(spec: bg_remove.ModelSpec, **_: object) -> Path:
        nonlocal download_called
        download_called = True
        raise AssertionError("Download should be deferred")

    monkeypatch.setattr(bg_remove, "_download_model", fail_download)

    statuses = bg_remove.ensure_models_downloaded(prefetch=False)

    assert not download_called
    assert statuses[bg_remove.DEFAULT_MODEL_NAME].state is bg_remove.DownloadState.PENDING


def test_ensure_models_downloaded_prefetches_requested_models(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Schedule background downloads for explicitly requested models."""

    monkeypatch.setattr(bg_remove, "MODELS_DIRECTORY", tmp_path)
    monkeypatch.setattr(bg_remove, "MODEL_DOWNLOAD_ROOT", tmp_path)

    scheduled: list[str] = []

    def record_prefetch(spec: bg_remove.ModelSpec) -> None:
        scheduled.append(spec.key)

    monkeypatch.setattr(bg_remove, "_schedule_prefetch", record_prefetch)

    statuses = bg_remove.ensure_models_downloaded(prefetch={"u2net"})

    assert "u2net" in scheduled
    assert statuses["u2net"].state is bg_remove.DownloadState.PENDING


def test_download_model_retries_transient_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Retry downloads with exponential backoff when transient errors occur."""

    spec = bg_remove.ModelSpec(
        key="example-transient",
        input_size=(1, 1),
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
        url="https://example.invalid/model.onnx",
        checksum_md5=None,
    )

    monkeypatch.setattr(bg_remove, "MODEL_DOWNLOAD_ROOT", tmp_path)
    monkeypatch.setattr(bg_remove, "MODELS_DIRECTORY", tmp_path)
    monkeypatch.setattr(bg_remove, "_verify_md5", lambda path, checksum: path.exists())

    attempts: list[int] = []

    def flaky_download(
        spec_arg: bg_remove.ModelSpec, destination: Path
    ) -> Path:
        attempts.append(1)
        if len(attempts) < 2:
            raise RuntimeError("temporary network issue")
        destination.write_bytes(b"onnx")
        return destination

    monkeypatch.setattr(bg_remove, "_download_model_from_url", flaky_download)

    delays: list[float] = []
    monkeypatch.setattr(bg_remove.time, "sleep", lambda value: delays.append(value))

    path = bg_remove._download_model(spec, max_attempts=3, backoff_base=0.1)

    assert path.exists()
    assert len(attempts) == 2
    assert delays == [0.1]

    status = bg_remove.get_download_statuses()[spec.key]
    assert status.state is bg_remove.DownloadState.AVAILABLE
    assert status.attempts == 2


def test_download_model_records_failure_after_retries(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Expose failure information when all download retries are exhausted."""

    spec = bg_remove.ModelSpec(
        key="example-failure",
        input_size=(1, 1),
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
        url="https://example.invalid/model.onnx",
        checksum_md5=None,
    )

    monkeypatch.setattr(bg_remove, "MODEL_DOWNLOAD_ROOT", tmp_path)
    monkeypatch.setattr(bg_remove, "MODELS_DIRECTORY", tmp_path)
    monkeypatch.setattr(bg_remove, "_verify_md5", lambda path, checksum: False)
    monkeypatch.setattr(bg_remove.time, "sleep", lambda value: None)

    def always_fail(
        spec_arg: bg_remove.ModelSpec, destination: Path
    ) -> Path:
        raise RuntimeError("permanent outage")

    monkeypatch.setattr(bg_remove, "_download_model_from_url", always_fail)

    with pytest.raises(RuntimeError):
        bg_remove._download_model(spec, max_attempts=2, backoff_base=0.0)

    status = bg_remove.get_download_statuses()[spec.key]
    assert status.state is bg_remove.DownloadState.FAILED
    assert status.error is not None
    assert status.attempts == 2
