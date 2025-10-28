import os
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from app.services import model_registry

INTEGRATION_WEIGHTS_ENV = "BR_INTEGRATION_WEIGHTS_DIR"


def _integration_weights_dir() -> Path | None:
    """Return the integration weights directory when configured."""

    raw = os.getenv(INTEGRATION_WEIGHTS_ENV)
    if not raw:
        return None
    path = Path(raw)
    return path if path.exists() else None


class _FakeSession:
    """Simple stub mimicking ``onnxruntime.InferenceSession``."""

    def __init__(self, providers: list[str]) -> None:
        self._providers = providers
        self.run_calls = 0

    def get_inputs(self) -> list[SimpleNamespace]:
        return [SimpleNamespace(name="input", shape=[1, 3, 320, 320], type="tensor(float)")]

    def get_providers(self) -> list[str]:
        return list(self._providers)

    def run(self, *_args, **_kwargs) -> list[np.ndarray]:
        self.run_calls += 1
        return [np.zeros((1, 1, 320, 320), dtype=np.float32)]


@pytest.fixture(autouse=True)
def reset_registry() -> None:
    """Ensure the registry starts empty for every test."""

    model_registry.reset()
    yield
    model_registry.reset()


def test_preload_models_initialises_sessions(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Model preload should create and warm ONNX Runtime sessions."""

    session_store: dict[str, _FakeSession] = {}

    def fake_inference_session(path: str, providers, provider_options):  # type: ignore[override]
        assert Path(path).exists()
        provider_names = [entry if isinstance(entry, str) else entry[0] for entry in providers]
        session = _FakeSession(provider_names)
        session_store[path] = session
        assert provider_options
        cuda_options = provider_options[0]
        assert cuda_options.get("arena_extend_strategy") == "kSameAsRequested"
        assert cuda_options.get("cudnn_conv_algo_search") == "EXHAUSTIVE"
        assert cuda_options.get("do_copy_in_default_stream") is True
        return session

    fake_ort = SimpleNamespace(
        InferenceSession=fake_inference_session,
        get_available_providers=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    monkeypatch.setattr(model_registry, "ort", fake_ort)
    monkeypatch.setattr(model_registry.runtime_compat, "ensure_runtime_ready", lambda: None)
    monkeypatch.setattr(
        model_registry.accelerator,
        "onnx_providers_available",
        lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    model_dir = tmp_path
    (model_dir / "u2net.onnx").write_bytes(b"fake")

    sessions = model_registry.preload_models(
        model_dir=model_dir,
        model_names=["u2net"],
        config={"BG_ACCELERATOR": "cuda"},
    )

    assert "u2net" in sessions
    fake_session = sessions["u2net"]
    assert isinstance(fake_session, _FakeSession)
    assert fake_session.run_calls >= 1
    assert model_registry.list_models() == ["u2net"]


def test_preload_models_handles_missing_files(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Missing ONNX files should be logged and skipped without raising."""

    def _fail_session(*_args, **_kwargs):  # pragma: no cover - defensive guard
        raise RuntimeError("should not load")

    fake_ort = SimpleNamespace(
        InferenceSession=_fail_session,
        get_available_providers=lambda: ["CPUExecutionProvider"],
    )

    monkeypatch.setattr(model_registry, "ort", fake_ort)
    monkeypatch.setattr(model_registry.runtime_compat, "ensure_runtime_ready", lambda: None)
    monkeypatch.setattr(
        model_registry.accelerator,
        "onnx_providers_available",
        lambda: ["CPUExecutionProvider"],
    )

    sessions = model_registry.preload_models(model_dir=tmp_path, model_names=["u2net"])

    assert sessions == {}
    assert model_registry.list_models() == []


def test_preload_models_is_cached(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Repeated preload calls with the same signature should reuse cached sessions."""

    call_count = 0

    def fake_inference_session(path: str, providers, provider_options):  # type: ignore[override]
        nonlocal call_count
        call_count += 1
        return _FakeSession(["CPUExecutionProvider"])

    fake_ort = SimpleNamespace(
        InferenceSession=fake_inference_session,
        get_available_providers=lambda: ["CPUExecutionProvider"],
    )

    monkeypatch.setattr(model_registry, "ort", fake_ort)
    monkeypatch.setattr(model_registry.runtime_compat, "ensure_runtime_ready", lambda: None)
    monkeypatch.setattr(
        model_registry.accelerator,
        "onnx_providers_available",
        lambda: ["CPUExecutionProvider"],
    )

    model_file = tmp_path / "u2net.onnx"
    model_file.write_bytes(b"fake")

    first = model_registry.preload_models(model_dir=tmp_path, model_names=["u2net"], warm=True)
    second = model_registry.preload_models(model_dir=tmp_path, model_names=["u2net"], warm=True)

    assert call_count == 1
    assert first["u2net"] is second["u2net"]


def test_rebuild_session_with_cuda_updates_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Rebuilding with CUDA should replace the cached session and return it."""

    created_sessions: list[_FakeSession] = []

    def fake_inference_session(path: str, providers, provider_options):  # type: ignore[override]
        assert Path(path).exists()
        assert providers == ["CUDAExecutionProvider", "CPUExecutionProvider"]
        assert provider_options[0]["device_id"] == 0
        session = _FakeSession(["CUDAExecutionProvider", "CPUExecutionProvider"])
        created_sessions.append(session)
        return session

    fake_ort = SimpleNamespace(InferenceSession=fake_inference_session)

    monkeypatch.setattr(model_registry, "ort", fake_ort)

    model_path = tmp_path / "u2net.onnx"
    model_path.write_bytes(b"fake")

    rebuilt = model_registry.rebuild_session_with_cuda("u2net", device_id=0, model_dir=tmp_path)

    assert rebuilt is not None
    assert rebuilt in created_sessions
    assert model_registry.get_session("u2net") is rebuilt


def test_rebuild_session_with_cuda_requires_weights(tmp_path: Path) -> None:
    """Rebuilding with CUDA should return ``None`` when weights are missing."""

    rebuilt = model_registry.rebuild_session_with_cuda("u2net", device_id=0, model_dir=tmp_path)

    assert rebuilt is None


def test_provider_priority_prefers_tensorrt(monkeypatch: pytest.MonkeyPatch) -> None:
    """RTX 50-series GPUs should prioritise TensorRT when CUDA 12.9 is available."""

    monkeypatch.setattr(model_registry.accelerator, "onnx_providers_available", lambda: [
        "TensorrtExecutionProvider",
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ])
    monkeypatch.setattr(model_registry.accelerator, "detect_gpu_name", lambda: "NVIDIA RTX 5090")
    monkeypatch.setattr(model_registry.accelerator, "is_rtx_50xx", lambda name: True)
    monkeypatch.setattr(model_registry.runtime_compat, "supports_tensorrt_cuda_129", lambda: True)

    providers, options = model_registry._provider_priority("cuda", 0)

    assert providers[0] == "TensorrtExecutionProvider"
    assert options[0]["device_id"] == 0
    assert options[0]["trt_engine_cache_enable"] is True
    assert options[0]["trt_fp16_enable"] is True


@pytest.mark.integration
def test_preload_models_real_weights_latency(monkeypatch: pytest.MonkeyPatch) -> None:
    """Real ONNX weights should load and provide measurable latency."""

    weights_dir = _integration_weights_dir()
    if weights_dir is None:
        pytest.skip(f"Set {INTEGRATION_WEIGHTS_ENV} to enable integration tests")
    if model_registry.ort is None:
        pytest.skip("onnxruntime is not installed")

    try:
        model_registry.runtime_compat.ensure_runtime_ready()
    except RuntimeError as exc:
        pytest.skip(f"Runtime compatibility check failed: {exc}")

    start = time.perf_counter()
    sessions = model_registry.preload_models(
        model_dir=weights_dir,
        model_names=["u2net"],
        warm=True,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000

    assert "u2net" in sessions
    assert elapsed_ms > 0
