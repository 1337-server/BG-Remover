from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from app.services import model_registry


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
