"""Tests covering the ONNX model loader and provider configuration."""

from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from bgremover_core.models import loader
from bgremover_core.models.loader import (
    _DEFAULT_CUDA_LIMIT_MIB,
    DOWNLOAD_AVAILABLE,
    DOWNLOAD_PENDING,
    BackgroundRemovalSession,
    DownloadStatus,
    _update_download_status,
    detect_providers,
    get_session,
)
from bgremover_core.models.specs import MODEL_SPECS, ModelSpec


def test_detect_providers_prefers_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    """``detect_providers`` should prefer CUDA providers when available."""

    monkeypatch.setattr(
        loader.ort,
        "get_available_providers",
        lambda: ["CPUExecutionProvider", "CUDAExecutionProvider"],
    )
    providers = detect_providers(["cpu"])
    assert providers[0] == "CUDAExecutionProvider"
    assert "CPUExecutionProvider" in providers


class DummySession:
    def __init__(self) -> None:
        self.inputs = [SimpleNamespace(name="input")]

    def get_inputs(self):  # pragma: no cover - called by loader
        return self.inputs

    def get_providers(self):  # pragma: no cover - called by loader
        return ["CPUExecutionProvider"]

    def run(self, *_args, **_kwargs):  # pragma: no cover - defensive
        return [np.ones((1, 1, 1), dtype=np.float32)]


def test_get_session_uses_custom_model_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Sessions should respect custom model directories without touching the network."""

    spec = next(iter(MODEL_SPECS.values()))
    model_path = tmp_path / "custom" / f"{spec.key}.onnx"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_bytes(b"dummy")

    def fake_download(model_spec, model_dir):
        assert model_dir == model_path.parent
        return model_path

    monkeypatch.setattr(loader, "_download_model", fake_download)
    monkeypatch.setattr(loader.ort, "SessionOptions", lambda: SimpleNamespace())
    monkeypatch.setattr(loader.ort, "InferenceSession", lambda *args, **kwargs: DummySession())

    session = get_session(spec.key, model_dir=model_path.parent)
    assert isinstance(session, BackgroundRemovalSession)
    assert session.spec.key == spec.key
    assert session.inner.get_providers() == ["CPUExecutionProvider"]


def test_cuda_provider_uses_configured_memory_limit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """CUDA providers should be configured with a conservative memory limit."""

    spec = next(iter(MODEL_SPECS.values()))
    model_path = tmp_path / f"{spec.key}.onnx"
    model_path.write_bytes(b"dummy")

    monkeypatch.setenv("BGR_CUDA_MEM_LIMIT_MB", "512")
    monkeypatch.setattr(loader, "_SESSION_CACHE", {})

    def fake_download(model_spec, model_dir):
        assert model_dir == tmp_path
        return model_path

    captured: dict[str, object] = {}

    def fake_inference_session(path, sess_options, providers):  # pragma: no cover - stub
        del path, sess_options
        captured["providers"] = providers
        return DummySession()

    monkeypatch.setattr(loader, "_download_model", fake_download)
    monkeypatch.setattr(loader.ort, "SessionOptions", lambda: SimpleNamespace())
    monkeypatch.setattr(loader.ort, "InferenceSession", fake_inference_session)

    session = get_session(
        spec.key,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        model_dir=tmp_path,
    )

    assert isinstance(session, BackgroundRemovalSession)
    assert "providers" in captured
    provider_entries = captured["providers"]
    assert isinstance(provider_entries, list)
    assert provider_entries[0][0] == "CUDAExecutionProvider"
    options = provider_entries[0][1]
    assert options["gpu_mem_limit"] == str(512 * 1024 * 1024)
    assert options["arena_extend_strategy"] == "kSameAsRequested"
    assert provider_entries[1] == "CPUExecutionProvider"


def test_cuda_provider_uses_default_memory_limit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """CUDA providers should adopt the built-in default memory ceiling."""

    spec = next(iter(MODEL_SPECS.values()))
    model_path = tmp_path / f"{spec.key}.onnx"
    model_path.write_bytes(b"dummy")

    monkeypatch.delenv("BGR_CUDA_MEM_LIMIT_MB", raising=False)
    monkeypatch.setattr(loader, "_SESSION_CACHE", {})
    monkeypatch.setattr(loader, "_CUDA_LIMIT_LOGGED", False)

    def fake_download(model_spec, model_dir):
        assert model_dir == tmp_path
        return model_path

    captured: dict[str, object] = {}

    def fake_inference_session(path, sess_options, providers):  # pragma: no cover - stub
        del path, sess_options
        captured["providers"] = providers
        return DummySession()

    monkeypatch.setattr(loader, "_download_model", fake_download)
    monkeypatch.setattr(loader.ort, "SessionOptions", lambda: SimpleNamespace())
    monkeypatch.setattr(loader.ort, "InferenceSession", fake_inference_session)

    session = get_session(
        spec.key,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        model_dir=tmp_path,
    )

    assert isinstance(session, BackgroundRemovalSession)
    provider_entries = captured["providers"]
    assert isinstance(provider_entries, list)
    assert provider_entries[0][0] == "CUDAExecutionProvider"
    options = provider_entries[0][1]
    assert options["gpu_mem_limit"] == str(_DEFAULT_CUDA_LIMIT_MIB * 1024 * 1024)
    assert options["arena_extend_strategy"] == "kSameAsRequested"
    assert provider_entries[1] == "CPUExecutionProvider"


def test_download_with_nested_local_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Model downloads should succeed when local filenames include subdirectories."""

    spec = ModelSpec(
        key="dummy-model",
        input_size=(1, 1),
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
        url="https://example.com/dummy.onnx",
        local_filename="nested/model.onnx",
    )

    class DummyResponse:
        """Minimal response object emulating ``requests`` streaming downloads."""

        status_code = 200

        def __enter__(self) -> DummyResponse:
            return self

        def __exit__(self, *_args) -> None:
            return None

        def raise_for_status(self) -> None:
            return None

        def iter_content(self, chunk_size: int = 1024):
            yield b"dummy-weights"

    monkeypatch.setattr(loader.requests, "get", lambda *_args, **_kwargs: DummyResponse())

    model_dir = tmp_path / "models"
    path = loader._download_model(spec, model_dir)

    assert path == model_dir / "nested/model.onnx"
    assert path.exists()
    assert path.read_bytes() == b"dummy-weights"


def test_download_via_http_creates_nested_directories(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``_download_model_via_http`` should create parents for nested destinations."""

    spec = ModelSpec(
        key="dummy-model",
        input_size=(1, 1),
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
        url="https://example.com/dummy.onnx",
        local_filename="nested/model.onnx",
    )

    class DummyResponse:
        """Stand-in for a streaming ``requests`` response."""

        status_code = 200

        def __enter__(self) -> DummyResponse:  # pragma: no cover - protocol behaviour
            return self

        def __exit__(self, *_args) -> None:  # pragma: no cover - protocol behaviour
            return None

        def raise_for_status(self) -> None:
            return None

        def iter_content(self, chunk_size: int = 1024):
            del chunk_size
            yield b"dummy-weights"

    monkeypatch.setattr(loader.requests, "get", lambda *_args, **_kwargs: DummyResponse())

    model_dir = tmp_path / "models"
    destination = model_dir / loader._build_model_filename(spec)

    assert not destination.parent.exists()

    path = loader._download_model_via_http(spec, spec.url or "", destination, headers={})

    assert path == destination
    assert path.exists()
    assert path.read_bytes() == b"dummy-weights"


def test_update_download_status_refreshes_timestamp(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Ensure repeated status updates produce monotonically increasing timestamps."""

    monkeypatch.setattr(loader, "_DOWNLOAD_STATUSES", {})

    initial_status = _update_download_status(tmp_path, "model", state=DOWNLOAD_PENDING)
    time.sleep(0.01)
    refreshed_status = _update_download_status(tmp_path, "model", state=DOWNLOAD_AVAILABLE)

    assert refreshed_status.updated_at > initial_status.updated_at


def test_download_status_uses_runtime_timestamp() -> None:
    """Direct :class:`DownloadStatus` instantiation should record the current time."""

    first_status = DownloadStatus(key="model", state=DOWNLOAD_PENDING)
    time.sleep(0.01)
    second_status = DownloadStatus(key="model", state=DOWNLOAD_AVAILABLE)

    assert second_status.updated_at > first_status.updated_at
