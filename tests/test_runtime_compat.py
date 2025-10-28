"""Unit tests for the runtime compatibility helpers."""
from __future__ import annotations

from types import SimpleNamespace

from app.services import runtime_compat


def test_is_force_cpu_enabled(monkeypatch) -> None:
    """The environment variable should enable or disable the CPU override."""

    monkeypatch.delenv("BR_FORCE_CPU", raising=False)
    assert runtime_compat.is_force_cpu_enabled() is False

    monkeypatch.setenv("BR_FORCE_CPU", "1")
    assert runtime_compat.is_force_cpu_enabled() is True

    monkeypatch.setenv("BR_FORCE_CPU", "false")
    assert runtime_compat.is_force_cpu_enabled() is False


def test_has_cuda_support_with_mocked_torch(monkeypatch) -> None:
    """CUDA detection should rely on PyTorch and respect the force-CPU flag."""

    monkeypatch.setattr(runtime_compat, "_torch", None, raising=False)
    monkeypatch.delenv("BR_FORCE_CPU", raising=False)
    monkeypatch.setenv("BR_SKIP_RUNTIME_CHECKS", "0")
    assert runtime_compat.has_cuda_support() is False

    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True))
    monkeypatch.setattr(runtime_compat, "_torch", fake_torch, raising=False)
    assert runtime_compat.has_cuda_support() is True

    monkeypatch.setenv("BR_FORCE_CPU", "yes")
    assert runtime_compat.has_cuda_support() is False
