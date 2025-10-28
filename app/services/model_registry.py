"""Shared ONNX Runtime session registry for CPU inference."""
from __future__ import annotations

import logging
import os
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np

from app.services import accelerator, runtime_compat

LOGGER = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency when running unit tests
    import onnxruntime as ort
except ModuleNotFoundError:  # pragma: no cover - gracefully handled by preload
    ort = None  # type: ignore[assignment]


DEFAULT_MODEL_NAMES: tuple[str, ...] = (
    "u2net",
    "u2netp",
    "isnet-general-use",
    "isnet-anime",
    "u2net_human_seg",
)

_PRELOADED_SESSIONS: dict[str, ort.InferenceSession] = {}
_PRELOAD_SIGNATURE: tuple[str, tuple[str, ...]] | None = None
_AVAILABLE_PROVIDERS: list[str] = []
_PRELOAD_LOCK = Lock()


def _resolve_model_dir(model_dir: str | os.PathLike[str] | None) -> Path:
    """Return the directory containing cached ONNX model weights."""

    if model_dir is not None:
        directory = Path(model_dir)
    else:
        directory = Path(os.environ.get("U2NET_HOME", Path.home() / ".u2net"))
    return directory.expanduser()


def _dummy_feed(session: ort.InferenceSession) -> Mapping[str, np.ndarray]:
    """Return a dictionary of zeroed tensors matching the session inputs."""

    tensors: dict[str, np.ndarray] = {}
    for input_meta in session.get_inputs():
        shape: list[int] = []
        for dimension in input_meta.shape:
            if isinstance(dimension, int):
                resolved = max(dimension, 1)
            elif dimension is None:
                resolved = 1
            else:
                token = str(dimension).lower()
                if "batch" in token or token in {"n", "b"}:
                    resolved = 1
                elif token.startswith("c") or "channel" in token:
                    resolved = 3
                else:
                    resolved = 320
            shape.append(resolved)

        tensor_type = getattr(input_meta, "type", "tensor(float)")
        dtype = {
            "tensor(float16)": np.float16,
            "tensor(float)": np.float32,
            "tensor(double)": np.float64,
            "tensor(int64)": np.int64,
            "tensor(int32)": np.int32,
            "tensor(uint8)": np.uint8,
        }.get(tensor_type, np.float32)

        tensors[input_meta.name] = np.zeros(shape, dtype=dtype)
    return tensors


def _warm_session(session: ort.InferenceSession) -> None:
    """Execute a dummy inference to prime internal caches."""

    try:
        feed_dict = _dummy_feed(session)
        if feed_dict:
            session.run(None, feed_dict)
    except Exception:  # pragma: no cover - hardware specific behaviour
        LOGGER.exception("Failed to warm ONNX Runtime session", exc_info=True)


def preload_models(
    *,
    model_dir: str | os.PathLike[str] | None = None,
    config: Mapping[str, object] | None = None,
    model_names: Iterable[str] | None = None,
    warm: bool = True,
) -> dict[str, ort.InferenceSession]:
    """Preload ONNX Runtime sessions for the requested models."""

    del config  # GPU-era configuration arguments are ignored in the CPU build.

    runtime_compat.ensure_runtime_ready()

    if ort is None:
        LOGGER.info("onnxruntime is not installed; skipping eager model preload.")
        return {}

    resolved_dir = _resolve_model_dir(model_dir)
    requested_names = tuple(sorted(set(model_names or DEFAULT_MODEL_NAMES)))
    signature = (str(resolved_dir), requested_names)

    with _PRELOAD_LOCK:
        global _PRELOAD_SIGNATURE, _AVAILABLE_PROVIDERS

        if _PRELOAD_SIGNATURE == signature and _PRELOADED_SESSIONS:
            return dict(_PRELOADED_SESSIONS)

        _AVAILABLE_PROVIDERS = accelerator.onnx_providers_available()
        LOGGER.info(
            "Available ONNX Runtime providers: %s",
            ", ".join(_AVAILABLE_PROVIDERS) or "none",
        )

        loaded: dict[str, ort.InferenceSession] = {}

        for model_name in requested_names:
            model_path = resolved_dir / f"{model_name}.onnx"
            if not model_path.exists():
                LOGGER.debug("Skipping preload for %s; %s not found.", model_name, model_path)
                continue

            start_time = time.perf_counter()
            try:
                session_options = None
                if hasattr(ort, "SessionOptions"):
                    try:
                        session_options = ort.SessionOptions()
                        session_options.log_severity_level = 3
                    except Exception:  # pragma: no cover - depends on onnxruntime build
                        LOGGER.debug(
                            "Unable to configure ONNX Runtime session options for %s",
                            model_name,
                            exc_info=True,
                        )
                init_kwargs: dict[str, Any] = {"providers": ["CPUExecutionProvider"]}
                if session_options is not None:
                    init_kwargs["sess_options"] = session_options

                session = ort.InferenceSession(  # type: ignore[attr-defined]
                    str(model_path),
                    **init_kwargs,
                )
            except Exception:
                LOGGER.exception("Failed to initialise ONNX Runtime session for %s", model_name)
                continue

            if warm:
                _warm_session(session)

            loaded[model_name] = session
            elapsed = time.perf_counter() - start_time
            session_providers = list(session.get_providers())
            LOGGER.info(
                "Loaded %s in %.2f s (providers=%s)",
                model_name,
                elapsed,
                ", ".join(session_providers),
            )

        _PRELOADED_SESSIONS.clear()
        _PRELOADED_SESSIONS.update(loaded)
        _PRELOAD_SIGNATURE = signature

    if not loaded:
        LOGGER.warning(
            "No ONNX Runtime sessions were preloaded. Ensure model weights exist in %s.",
            resolved_dir,
        )

    return dict(_PRELOADED_SESSIONS)


def get_session(model_name: str) -> ort.InferenceSession | None:
    """Return a preloaded session for ``model_name`` if one is available."""

    return _PRELOADED_SESSIONS.get(model_name)


def list_models() -> list[str]:
    """Return the list of preloaded model names."""

    return sorted(_PRELOADED_SESSIONS)


def get_available_providers() -> list[str]:
    """Return the providers observed during the last preload call."""

    return list(_AVAILABLE_PROVIDERS)


def reset() -> None:
    """Clear cached sessions. Primarily intended for test suites."""

    with _PRELOAD_LOCK:
        _PRELOADED_SESSIONS.clear()
        global _PRELOAD_SIGNATURE
        _PRELOAD_SIGNATURE = None
        _AVAILABLE_PROVIDERS.clear()
