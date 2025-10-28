"""Runtime compatibility helpers for the CPU-only background removal stack."""
from __future__ import annotations

import importlib
import importlib.metadata
import logging
import os
from types import ModuleType, SimpleNamespace

try:  # pragma: no cover - optional dependency when Eventlet is unavailable
    from eventlet.green import threading as cooperative_threading  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - Eventlet not installed in some environments
    import threading as cooperative_threading  # type: ignore


LOGGER = logging.getLogger(__name__)

_STUB_ONNXRUNTIME = SimpleNamespace(get_available_providers=lambda: ["CPUExecutionProvider"])

try:  # pragma: no cover - optional dependency during testing
    import numpy as _np
except ModuleNotFoundError:  # pragma: no cover - handled by runtime checks
    _np = None  # type: ignore[assignment]

_RUNTIME_LOCK = cooperative_threading.Lock()
_CACHED_ONNXRUNTIME: ModuleType | None = None


def _read_numpy_version() -> str | None:
    """Return the installed NumPy version or ``None`` if unavailable."""

    if _np is None:
        return None
    return getattr(_np, "__version__", None)


def _normalise_bool(value: str) -> bool:
    """Return ``True`` when ``value`` represents a truthy flag."""

    return value.strip().lower() in {"1", "true", "yes", "on"}


def _should_skip_runtime_checks() -> bool:
    """Return ``True`` when runtime compatibility checks should be skipped."""

    override = os.getenv("BR_SKIP_RUNTIME_CHECKS", "")
    return bool(override and _normalise_bool(override))


def _format_dependency_note(version: str | None) -> str:
    """Return a human-readable description of the NumPy dependency set."""

    if not version:
        return "NumPy unavailable"
    major = version.split(".", 1)[0]
    label = "NumPy 1.x" if major == "1" else "NumPy 2.x"
    return f"{label} (v{version})"


def _distribution_installed(name: str) -> bool:
    """Return ``True`` when the given distribution name is importable."""

    try:
        importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return False
    return True


def verify_runtime_compatibility() -> ModuleType:
    """Import ONNX Runtime and verify that it is available for CPU inference."""

    if _should_skip_runtime_checks():
        LOGGER.info("Skipping ONNX Runtime compatibility checks (test mode)")
        return _STUB_ONNXRUNTIME  # type: ignore[return-value]

    numpy_version = _read_numpy_version()
    LOGGER.info("Initialising rembg runtime using %s", _format_dependency_note(numpy_version))

    if not _distribution_installed("onnxruntime"):
        raise RuntimeError(
            "Unable to import ONNX Runtime. Install the CPU-only wheel with ``pip install onnxruntime``."
        )

    try:
        module = importlib.import_module("onnxruntime")
    except Exception as exc:  # pragma: no cover - propagate to caller on demand
        raise RuntimeError(
            "Failed to import the onnxruntime module. This typically indicates a binary "
            "compatibility issue between NumPy and ONNX Runtime."
        ) from exc

    try:
        providers = list(module.get_available_providers())
    except Exception:  # pragma: no cover - provider lookup is best effort
        providers = ["unknown"]

    LOGGER.info("Loaded onnxruntime (providers: %s)", ", ".join(providers))
    return module


def ensure_runtime_ready() -> ModuleType:
    """Ensure the ONNX Runtime module is imported exactly once and cached."""

    global _CACHED_ONNXRUNTIME
    with _RUNTIME_LOCK:
        if _CACHED_ONNXRUNTIME is None:
            _CACHED_ONNXRUNTIME = verify_runtime_compatibility()
    assert _CACHED_ONNXRUNTIME is not None
    return _CACHED_ONNXRUNTIME
