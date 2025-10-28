"""Unit tests for the runtime compatibility helpers."""
from __future__ import annotations

from types import SimpleNamespace

from app.services import runtime_compat


def test_verify_runtime_compatibility_skips_when_requested(monkeypatch) -> None:
    """The BR_SKIP_RUNTIME_CHECKS flag should bypass real imports."""

    monkeypatch.setenv("BR_SKIP_RUNTIME_CHECKS", "1")
    module = runtime_compat.verify_runtime_compatibility()
    assert module.get_available_providers() == ["CPUExecutionProvider"]


def test_ensure_runtime_ready_caches_module(monkeypatch) -> None:
    """ensure_runtime_ready should import onnxruntime only once."""

    monkeypatch.setenv("BR_SKIP_RUNTIME_CHECKS", "0")
    fake_module = SimpleNamespace(get_available_providers=lambda: ["CPUExecutionProvider"])
    call_count = 0

    def fake_verify() -> SimpleNamespace:
        nonlocal call_count
        call_count += 1
        return fake_module

    monkeypatch.setattr(runtime_compat, "verify_runtime_compatibility", fake_verify)
    try:
        first = runtime_compat.ensure_runtime_ready()
        second = runtime_compat.ensure_runtime_ready()
    finally:
        runtime_compat._CACHED_ONNXRUNTIME = None  # type: ignore[attr-defined]

    assert first is fake_module
    assert second is fake_module
    assert call_count == 1
