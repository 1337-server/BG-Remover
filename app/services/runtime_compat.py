"""Runtime compatibility helpers for NumPy, ONNXRuntime, and rembg."""
from __future__ import annotations

import importlib
import importlib.metadata
import logging
import os
from types import ModuleType, SimpleNamespace
from typing import Iterable, Optional

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

try:  # pragma: no cover - optional dependency during testing
    import torch as _torch
except ModuleNotFoundError:  # pragma: no cover - handled by runtime checks
    _torch = None  # type: ignore[assignment]

_RUNTIME_LOCK = cooperative_threading.Lock()
_CACHED_ONNXRUNTIME: Optional[ModuleType] = None


def _read_numpy_version() -> Optional[str]:
    """Return the installed NumPy version or ``None`` if unavailable."""

    if _np is None:
        return None
    return getattr(_np, "__version__", None)


def _normalise_bool(value: str) -> bool:
    """Return ``True`` when ``value`` represents a truthy flag."""

    return value.strip().lower() in {"1", "true", "yes", "on"}


def _should_skip_runtime_checks() -> bool:
    """Return ``True`` when runtime compatibility checks should be skipped."""

    override = os.environ.get("BR_SKIP_RUNTIME_CHECKS", "")
    return bool(override and _normalise_bool(override))


def is_force_cpu_enabled() -> bool:
    """Return ``True`` when CPU execution is explicitly requested."""

    override = os.environ.get("BR_FORCE_CPU", "")
    return bool(override and _normalise_bool(override))


def has_cuda_support() -> bool:
    """Return ``True`` when CUDA appears to be available via PyTorch."""

    if _should_skip_runtime_checks():
        return False

    if _torch is None or is_force_cpu_enabled():
        return False
    try:
        return bool(_torch.cuda.is_available())
    except Exception as exc:  # pragma: no cover - defensive guard
        LOGGER.warning("Unable to query CUDA availability via torch: %s", exc)
        return False


def _iter_candidate_distributions(prefer_gpu: bool) -> Iterable[str]:
    """Yield distribution names to probe based on GPU preferences."""

    prefer_gpu = prefer_gpu and not is_force_cpu_enabled()
    if prefer_gpu:
        yield "onnxruntime-gpu"
        yield "onnxruntime"
    else:
        yield "onnxruntime"
        yield "onnxruntime-gpu"


def _distribution_installed(name: str) -> bool:
    """Return ``True`` when the given distribution name is importable."""

    try:
        importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return False
    return True


def _import_onnxruntime(distribution: str) -> ModuleType:
    """Import and return the ``onnxruntime`` module for ``distribution``."""

    try:
        module = importlib.import_module("onnxruntime")
    except Exception as exc:  # pragma: no cover - passthrough to higher-level handler
        raise RuntimeError(
            "Failed to import the onnxruntime module. This typically indicates a "
            "binary compatibility issue between NumPy and ONNXRuntime."
        ) from exc

    providers = []
    try:
        providers = list(module.get_available_providers())
    except Exception as exc:  # pragma: no cover - defensive guard
        LOGGER.debug("Failed to query ONNXRuntime providers: %s", exc)

    LOGGER.info(
        "Loaded %s (providers: %s)",
        distribution,
        ", ".join(providers) if providers else "unknown",
    )
    return module


def _format_dependency_note(version: Optional[str]) -> str:
    """Return a human-readable description of the NumPy dependency set."""

    if not version:
        return "NumPy unavailable"
    major = version.split(".", 1)[0]
    label = "NumPy 1.x" if major == "1" else "NumPy 2.x"
    return f"{label} (v{version})"


def verify_runtime_compatibility() -> ModuleType:
    """Import ONNXRuntime and verify that it is compatible with NumPy."""

    if _should_skip_runtime_checks():
        LOGGER.info("Skipping ONNXRuntime compatibility checks (test mode)")
        return _STUB_ONNXRUNTIME  # type: ignore[return-value]

    numpy_version = _read_numpy_version()
    LOGGER.info("Initialising rembg runtime using %s", _format_dependency_note(numpy_version))

    prefer_gpu = has_cuda_support()
    last_error: Optional[BaseException] = None

    for distribution in _iter_candidate_distributions(prefer_gpu):
        if not _distribution_installed(distribution):
            LOGGER.debug("Distribution %s is not installed", distribution)
            continue
        try:
            module = _import_onnxruntime(distribution)
        except Exception as exc:  # pragma: no cover - exception is re-raised
            last_error = exc
            LOGGER.error("Failed to load %s: %s", distribution, exc)
            continue
        return module

    message = (
        "Unable to import ONNXRuntime. This usually means NumPy and ONNXRuntime were "
        "built against incompatible ABIs. Reinstall dependencies with "
        "`pip install --force-reinstall -r requirements.txt` after clearing any "
        "existing virtual environments."
    )
    if last_error is not None:
        raise RuntimeError(message) from last_error
    raise RuntimeError(message)


def ensure_runtime_ready() -> ModuleType:
    """Ensure the ONNXRuntime module is imported exactly once and cached."""

    global _CACHED_ONNXRUNTIME
    with _RUNTIME_LOCK:
        if _CACHED_ONNXRUNTIME is None:
            _CACHED_ONNXRUNTIME = verify_runtime_compatibility()
    assert _CACHED_ONNXRUNTIME is not None
    return _CACHED_ONNXRUNTIME

