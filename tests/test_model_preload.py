"""Tests for the ONNX Runtime model preloader utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, MutableMapping

import numpy as np
import pytest

from app.services import model_preload


class _FakeInput:
    """Simple container replicating the interface of ONNX input metadata."""

    def __init__(self, name: str, shape: List[int], tensor_type: str = "tensor(float)") -> None:
        self.name = name
        self.shape = shape
        self.type = tensor_type


class _FakeSession:
    """Stub ``InferenceSession`` tracking warm-up invocations for assertions."""

    def __init__(
        self,
        path: str,
        *,
        providers: List[str] | None = None,
        provider_options: List[MutableMapping[str, int]] | None = None,
    ) -> None:
        self.path = Path(path)
        self.providers = providers or []
        self.provider_options = provider_options or []
        self._inputs = [_FakeInput("input", [1, 3, 320, 320])]
        self.run_calls = 0
        self.last_feed: Dict[str, np.ndarray] | None = None

    def get_inputs(self) -> List[_FakeInput]:
        """Return deterministic input metadata for dummy invocations."""

        return list(self._inputs)

    def run(self, _: object, feed_dict: Dict[str, np.ndarray]) -> List[str]:
        """Record warm-up invocations and inputs for verification."""

        self.run_calls += 1
        self.last_feed = feed_dict
        return ["ok"]


class _FakeOrt:
    """Namespace exposing ``InferenceSession`` compatible with monkeypatching."""

    def __init__(self, session_factory: type[_FakeSession]) -> None:
        self.session_factory = session_factory
        self.created: List[_FakeSession] = []

    def InferenceSession(self, *args, **kwargs) -> _FakeSession:  # noqa: N802 - mimic API
        """Return a tracked fake session instance."""

        session = self.session_factory(*args, **kwargs)
        self.created.append(session)
        return session


@pytest.fixture(autouse=True)
def reset_preloader_state() -> None:
    """Ensure global caches are cleared before each test."""

    model_preload.reset_preloaded_sessions()


def test_preload_skips_when_onnxruntime_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """The loader should exit early when ONNX Runtime is unavailable."""

    monkeypatch.setattr(model_preload, "ort", None)
    sessions = model_preload.preload_all_models(model_dir="/nonexistent")
    assert sessions == {}


def test_preload_warms_models_and_caches(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Preloading should warm up each session once and reuse cached instances."""

    fake_ort = _FakeOrt(_FakeSession)
    monkeypatch.setattr(model_preload, "ort", fake_ort)
    monkeypatch.setattr(model_preload.accelerator, "onnx_providers_available", lambda: [
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ])
    model_file = tmp_path / "u2net.onnx"
    model_file.write_bytes(b"fake-onnx")

    sessions = model_preload.preload_all_models(
        model_dir=tmp_path,
        config={"BG_ACCELERATOR": "cuda", "BG_CUDA_DEVICE_ID": 3},
        model_names=["u2net"],
    )

    assert set(sessions) == {"u2net"}
    session = sessions["u2net"]
    assert isinstance(session, _FakeSession)
    assert session.run_calls == 1
    assert session.providers == ["CUDAExecutionProvider", "CPUExecutionProvider"]
    assert session.provider_options[0]["device_id"] == 3
    assert session.last_feed is not None
    dummy = session.last_feed["input"]
    assert isinstance(dummy, np.ndarray)
    assert tuple(dummy.shape) == (1, 3, 320, 320)

    cached = model_preload.preload_all_models(
        model_dir=tmp_path,
        config={"BG_ACCELERATOR": "cuda", "BG_CUDA_DEVICE_ID": 3},
        model_names=["u2net"],
    )
    assert cached["u2net"] is session
    assert len(fake_ort.created) == 1


def test_preload_falls_back_to_cpu_when_cuda_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """When CUDA is unavailable the loader should warm only the CPU provider."""

    fake_ort = _FakeOrt(_FakeSession)
    monkeypatch.setattr(model_preload, "ort", fake_ort)
    monkeypatch.setattr(model_preload.accelerator, "onnx_providers_available", lambda: [
        "CPUExecutionProvider",
    ])

    model_file = tmp_path / "u2net.onnx"
    model_file.write_bytes(b"fake-onnx")

    sessions = model_preload.preload_all_models(
        model_dir=tmp_path,
        config={"BG_ACCELERATOR": "cuda", "BG_CUDA_DEVICE_ID": 1},
        model_names=["u2net"],
    )

    session = sessions["u2net"]
    assert session.providers == ["CPUExecutionProvider"]
    assert session.provider_options == [{}]
