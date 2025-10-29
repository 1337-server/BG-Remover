"""Tests for the model loader utilities."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from bgremover_core.models import loader
from bgremover_core.models.loader import BackgroundRemovalSession, detect_providers, get_session
from bgremover_core.models.specs import MODEL_SPECS


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
