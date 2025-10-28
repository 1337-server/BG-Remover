"""Tests for the accelerator helper functions."""
from __future__ import annotations

import pytest

from app.services import accelerator


def test_pick_execution_provider_prefers_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    """GPU providers should be selected when CUDAExecutionProvider is available."""

    monkeypatch.setattr(
        accelerator,
        "onnx_providers_available",
        lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    provider, options = accelerator.pick_execution_provider("auto", 1)
    assert provider == "cuda"
    assert options == {"device_id": 1}


def test_pick_execution_provider_cpu_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """CPU execution should be selected when CUDA providers are absent."""

    monkeypatch.setattr(accelerator, "onnx_providers_available", lambda: ["CPUExecutionProvider"])
    provider, options = accelerator.pick_execution_provider("auto", 0)
    assert provider == "cpu"
    assert options == {}


@pytest.mark.parametrize(
    "name,expected",
    [
        ("NVIDIA GeForce RTX 5090", True),
        ("RTX 4080", False),
        ("rtx 50 laptop gpu", True),
        ("Tesla V100", False),
        ("", False),
    ],
)
def test_is_rtx_50xx(name: str, expected: bool) -> None:
    """Ensure RTX 50-series detection logic matches the expected names."""

    assert accelerator.is_rtx_50xx(name) is expected
