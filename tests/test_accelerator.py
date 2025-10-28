"""Tests for the accelerator selection utilities."""
from __future__ import annotations

from typing import List

import pytest

from app.services import accelerator


def test_pick_execution_provider_prefers_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    """CUDA should be selected when the provider is available."""

    def fake_providers() -> List[str]:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]

    monkeypatch.setattr(accelerator, "onnx_providers_available", fake_providers)

    provider, options = accelerator.pick_execution_provider("auto", 2)
    assert provider == "cuda"
    assert options == {"device_id": 2}


def test_pick_execution_provider_cpu_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """CPU mode should be returned when CUDA providers are unavailable."""

    monkeypatch.setattr(accelerator, "onnx_providers_available", lambda: ["CPUExecutionProvider"])

    provider, options = accelerator.pick_execution_provider("auto", 0)
    assert provider == "cpu"
    assert options == {}


@pytest.mark.parametrize(
    "name,expected",
    [
        ("NVIDIA GeForce RTX 5090", True),
        ("rtx 50 pro", True),
        ("RTX 4090", False),
        ("Some Other GPU", False),
    ],
)
def test_is_rtx_50xx_detection(name: str, expected: bool) -> None:
    """RTX 50-series detection should match known model names."""

    assert accelerator.is_rtx_50xx(name) is expected
