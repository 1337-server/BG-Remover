"""Unit tests for the accelerator helper functions."""
from __future__ import annotations

import sys
from types import SimpleNamespace

from app.services import accelerator


def test_onnx_providers_available_returns_list(monkeypatch) -> None:
    """The helper should proxy provider lists from onnxruntime."""

    stub = SimpleNamespace(get_available_providers=lambda: ["CPUExecutionProvider"])
    monkeypatch.setitem(sys.modules, "onnxruntime", stub)
    try:
        assert accelerator.onnx_providers_available() == ["CPUExecutionProvider"]
    finally:
        monkeypatch.delitem(sys.modules, "onnxruntime", raising=False)


def test_onnx_providers_available_handles_errors(monkeypatch) -> None:
    """Failures during provider queries should return an empty list."""

    def raise_error() -> list[str]:
        raise RuntimeError("test error")

    stub = SimpleNamespace(get_available_providers=raise_error)
    monkeypatch.setitem(sys.modules, "onnxruntime", stub)
    try:
        assert accelerator.onnx_providers_available() == []
    finally:
        monkeypatch.delitem(sys.modules, "onnxruntime", raising=False)
