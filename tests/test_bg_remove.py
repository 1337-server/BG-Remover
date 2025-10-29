"""Unit tests for the simplified background removal helpers."""
from __future__ import annotations

import io
import time
from pathlib import Path

import pytest
from PIL import Image

import bg_removal as bg_remove


@pytest.fixture()
def large_rgba_image(tmp_path: Path) -> Path:
    """Return a sizeable RGBA image saved to disk for streaming tests."""

    path = tmp_path / "large.png"
    Image.new("RGBA", (2048, 2048), color=(0, 255, 0, 255)).save(path, "PNG")
    return path


@pytest.fixture()
def large_rgba_bytes(large_rgba_image: Path) -> bytes:
    """Return the encoded bytes for the ``large_rgba_image`` fixture."""

    return large_rgba_image.read_bytes()


def _make_image_bytes(color: tuple[int, int, int, int] = (255, 0, 0, 255)) -> bytes:
    image = Image.new("RGBA", (4, 4), color=color)
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    return buffer.getvalue()


@pytest.mark.parametrize(
    ("color", "expected"),
    (
        (
            (255, 255, 255),
            tuple(
                (1.0 - mean) / std
                for mean, std in zip((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), strict=False)
            ),
        ),
        (
            (10, 10, 10),
            tuple(
                ((10 / 255.0) - mean) / std
                for mean, std in zip((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), strict=False)
            ),
        ),
    ),
)
def test_normalise_image_uses_fixed_scale(color: tuple[int, int, int], expected: tuple[float, ...]) -> None:
    """Verify that normalisation divides by the configured scale for bright and dark inputs."""

    spec = bg_remove.ModelSpec(
        key="test",
        input_size=(4, 4),
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
    )
    image = Image.new("RGB", (2, 2), color=color)
    tensor = bg_remove._normalise_image(image, spec)
    assert tensor.shape == (1, 3, 4, 4)
    for index, expected_value in enumerate(expected):
        assert tensor[0, index, 0, 0] == pytest.approx(expected_value, rel=1e-5)


def test_normalise_image_honours_custom_scale() -> None:
    """Ensure models that override ``normalisation_scale`` divide by the bespoke constant."""

    spec = bg_remove.ModelSpec(
        key="custom-scale",
        input_size=(2, 2),
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
        normalisation_scale=1.0,
    )
    image = Image.new("RGB", (2, 2), color=(128, 64, 32))
    tensor = bg_remove._normalise_image(image, spec)
    assert tensor.shape == (1, 3, 2, 2)
    expected_values = (128.0, 64.0, 32.0)
    for index, expected_value in enumerate(expected_values):
        assert tensor[0, index, 0, 0] == pytest.approx(expected_value, rel=1e-6)


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


def test_remove_background_stream_handles_large_image(
    large_rgba_image: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ensure streaming removal works for large files without eager byte copies."""

    class DummySession:
        providers_available = ("CPUExecutionProvider",)

    def fake_predict(image: Image.Image, session: DummySession) -> Image.Image:
        return Image.new("L", image.size, color=255)

    monkeypatch.setattr(bg_remove, "_load_session", lambda model_name: DummySession())
    monkeypatch.setattr(bg_remove, "_predict_mask", fake_predict)

    with large_rgba_image.open("rb") as stream:
        result = bg_remove.remove_background_stream(stream)

    assert result.success
    assert result.image is not None
    assert result.image.size == (2048, 2048)
    result.image.close()


def test_large_stream_and_byte_inputs_produce_identical_outputs(
    large_rgba_image: Path,
    large_rgba_bytes: bytes,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The byte-based and streaming workflows should yield matching results."""

    class DummySession:
        providers_available = ("CPUExecutionProvider",)

    def fake_predict(image: Image.Image, session: DummySession) -> Image.Image:
        return Image.new("L", image.size, color=255)

    monkeypatch.setattr(bg_remove, "_load_session", lambda model_name: DummySession())
    monkeypatch.setattr(bg_remove, "_predict_mask", fake_predict)

    with large_rgba_image.open("rb") as stream:
        stream_result = bg_remove.remove_background_stream(stream)
    bytes_result = bg_remove.remove_background_bytes(large_rgba_bytes)

    assert stream_result.success
    assert bytes_result.success
    assert stream_result.as_bytes() == bytes_result.as_bytes()
    if stream_result.image:
        stream_result.image.close()
    if bytes_result.image:
        bytes_result.image.close()


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


def test_remove_bg_folder_parallel_preserves_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure batch processing returns results matching the discovery order."""

    source_dir = tmp_path / "input"
    source_dir.mkdir()
    output_dir = tmp_path / "output"

    for index in range(3):
        path = source_dir / f"image_{index}.png"
        Image.new("RGBA", (index + 2, index + 2), color=(255, index, 0, 255)).save(path, "PNG")

    class DummySession:
        providers_available = ("CPUExecutionProvider",)

    session_calls: set[int] = set()

    def fake_predict(image: Image.Image, session: DummySession) -> Image.Image:
        session_calls.add(id(session))
        if image.size == (4, 4):
            time.sleep(0.05)
        return Image.new("L", image.size, color=255)

    monkeypatch.setattr(bg_remove, "_load_session", lambda model_name: DummySession())
    monkeypatch.setattr(bg_remove, "_predict_mask", fake_predict)

    results = bg_remove.remove_bg_folder(source_dir, output_dir)

    expected_order = [path.name for path in bg_remove._iter_input_files(source_dir, recursive=False)]
    assert [result.path_in.name for result in results] == expected_order
    assert len(session_calls) == 1
    for result in results:
        assert result.path_out is not None
        assert result.path_out.exists()


def test_remove_bg_folder_parallel_propagates_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify batch failures bubble up while successful work persists."""

    source_dir = tmp_path / "input"
    source_dir.mkdir()
    output_dir = tmp_path / "output"

    success_path = source_dir / "success.png"
    failure_path = source_dir / "failure.png"
    Image.new("RGBA", (2, 2), color=(0, 255, 0, 255)).save(success_path, "PNG")
    Image.new("RGBA", (3, 3), color=(255, 0, 0, 255)).save(failure_path, "PNG")

    class DummySession:
        providers_available = ("CPUExecutionProvider",)

    def fake_predict(image: Image.Image, session: DummySession) -> Image.Image:
        if image.size == (3, 3):
            raise RuntimeError("boom")
        return Image.new("L", image.size, color=255)

    monkeypatch.setattr(bg_remove, "_load_session", lambda model_name: DummySession())
    monkeypatch.setattr(bg_remove, "_predict_mask", fake_predict)

    results = bg_remove.remove_bg_folder(source_dir, output_dir)

    assert (output_dir / success_path.name).exists()

    success_result = next(result for result in results if result.path_in == success_path)
    failure_result = next(result for result in results if result.path_in == failure_path)

    assert success_result.success
    assert failure_result.error is not None
    assert "boom" in failure_result.error
    assert failure_result.path_out is None


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
