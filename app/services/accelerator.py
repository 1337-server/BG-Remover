"""Minimal ONNX Runtime helpers for the CPU-only build."""
from __future__ import annotations

import logging

LOGGER = logging.getLogger(__name__)


def onnx_providers_available() -> list[str]:
    """Return the execution providers exposed by the local ONNX Runtime build.

    The project targets CPU-only execution, but the helper remains useful for
    diagnostic endpoints that surface which providers are compiled into the
    installed ``onnxruntime`` wheel. When the module is missing or the provider
    query fails a defensive empty list is returned.
    """

    try:
        import onnxruntime as ort
    except ModuleNotFoundError:
        LOGGER.debug("onnxruntime is not installed; assuming no providers are available")
        return []

    try:
        providers = list(ort.get_available_providers())
    except Exception:  # pragma: no cover - defensive guard for unusual builds
        LOGGER.exception("Failed to query ONNX Runtime providers")
        return []
    return providers
