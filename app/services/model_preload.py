"""Helpers for eagerly loading ONNX Runtime background removal models."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from threading import Lock
from typing import Dict, Iterable, Mapping, MutableMapping

import numpy as np

from app.services import accelerator, runtime_compat

LOGGER = logging.getLogger(__name__)


try:  # pragma: no cover - optional dependency during unit tests
    import onnxruntime as ort
except ModuleNotFoundError:  # pragma: no cover - gracefully handled by loader
    ort = None  # type: ignore


DEFAULT_MODEL_NAMES: tuple[str, ...] = (
    "u2net",
    "u2netp",
    "isnet-general-use",
    "isnet-anime",
)

PRELOADED_SESSIONS: Dict[str, "ort.InferenceSession"] = {}

_PRELOAD_LOCK = Lock()
_PRELOAD_SIGNATURE: tuple[str, tuple[str, ...], str, int] | None = None


def _resolve_model_directory(model_dir: str | os.PathLike[str] | None) -> Path:
    """Return the directory containing the ONNX model weights."""

    if model_dir is not None:
        directory = Path(model_dir).expanduser()
    else:
        directory = Path(os.environ.get("U2NET_HOME", Path.home() / ".u2net")).expanduser()
    return directory


def _normalise_mode(value: str | None) -> str:
    """Return a normalised accelerator mode value."""

    if value is None:
        return "auto"
    mode = value.strip().lower()
    return mode if mode in {"auto", "cuda", "cpu"} else "auto"


def _extract_accelerator_config(config: Mapping[str, object] | None) -> tuple[str, int]:
    """Return the desired accelerator mode and CUDA device id."""

    settings = config or {}
    mode = _normalise_mode(str(settings.get("BG_ACCELERATOR", "auto")))
    if runtime_compat.is_force_cpu_enabled():
        mode = "cpu"
    try:
        device_id = int(settings.get("BG_CUDA_DEVICE_ID", 0))
    except (TypeError, ValueError):
        device_id = 0
    return mode, device_id


def _resolve_providers(mode: str, device_id: int) -> tuple[list[str], list[MutableMapping[str, int]]]:
    """Return provider names and options with graceful CPU fallback."""

    available = accelerator.onnx_providers_available()
    providers: list[str] = []
    provider_options: list[MutableMapping[str, int]] = []

    if mode != "cpu":
        if "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
            provider_options.append({"device_id": int(device_id)})
        elif mode == "cuda":
            LOGGER.warning(
                "CUDA execution was requested but CUDAExecutionProvider is unavailable; falling back to CPU."
            )

    providers.append("CPUExecutionProvider")
    provider_options.append({})
    return providers, provider_options


def _dummy_input_value(session: "ort.InferenceSession") -> Dict[str, np.ndarray]:
    """Return a feed dictionary containing dummy tensors for warm-up runs."""

    feed: Dict[str, np.ndarray] = {}
    for input_meta in session.get_inputs():
        shape: list[int] = []
        for dimension in input_meta.shape:
            if isinstance(dimension, int):
                resolved = max(dimension, 1)
            elif dimension is None:
                resolved = 1
            else:
                text = str(dimension).lower()
                if "batch" in text or text in {"n", "b"}:
                    resolved = 1
                elif "channel" in text or text.startswith("c"):
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

        feed[input_meta.name] = np.zeros(shape, dtype=dtype)
    return feed


def _warm_up_session(session: "ort.InferenceSession") -> None:
    """Execute a dummy inference to ensure kernels and weights are cached."""

    try:
        dummy_feed = _dummy_input_value(session)
        if dummy_feed:
            session.run(None, dummy_feed)
    except Exception:  # pragma: no cover - hardware/runtime specific failures
        LOGGER.exception("Failed to warm up ONNX Runtime session", exc_info=True)


def _log_vram_consumption() -> None:
    """Log the total GPU memory usage when available."""

    try:  # pragma: no cover - requires CUDA runtime
        import torch

        if torch.cuda.is_available():  # type: ignore[truthy-bool]
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            used_mib = (total_bytes - free_bytes) / (1024 ** 2)
            total_mib = total_bytes / (1024 ** 2)
            LOGGER.info(
                "Preloaded models are occupying approximately %.1f MiB of %.1f MiB available VRAM.",
                used_mib,
                total_mib,
            )
            return
    except ModuleNotFoundError:
        torch = None  # type: ignore
    except Exception:
        LOGGER.debug("Unable to query VRAM usage via torch", exc_info=True)

    try:  # pragma: no cover - requires pynvml and NVIDIA hardware
        import pynvml  # type: ignore

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        pynvml.nvmlShutdown()
    except ModuleNotFoundError:
        LOGGER.info(
            "Model preload complete; VRAM usage metrics are unavailable because pynvml is not installed."
        )
    except Exception:
        LOGGER.debug("Unable to query VRAM usage via pynvml", exc_info=True)
    else:
        used_mib = info.used / (1024 ** 2)
        total_mib = info.total / (1024 ** 2)
        LOGGER.info(
            "Preloaded models are occupying approximately %.1f MiB of %.1f MiB available VRAM.",
            used_mib,
            total_mib,
        )


def preload_all_models(
    *,
    model_dir: str | os.PathLike[str] | None = None,
    config: Mapping[str, object] | None = None,
    model_names: Iterable[str] = DEFAULT_MODEL_NAMES,
) -> Dict[str, "ort.InferenceSession"]:
    """Load the configured models into memory and return the cached sessions."""

    if ort is None:
        LOGGER.info(
            "onnxruntime is not installed; skipping eager model preload. Install onnxruntime-gpu to enable GPU warm-up."
        )
        return {}

    resolved_dir = _resolve_model_directory(model_dir)
    mode, device_id = _extract_accelerator_config(config)
    signature = (str(resolved_dir), tuple(sorted(set(model_names))), mode, device_id)

    with _PRELOAD_LOCK:
        global _PRELOAD_SIGNATURE
        if _PRELOAD_SIGNATURE == signature and PRELOADED_SESSIONS:
            return dict(PRELOADED_SESSIONS)

        providers, provider_options = _resolve_providers(mode, device_id)
        loaded: Dict[str, "ort.InferenceSession"] = {}

        for model_name in model_names:
            path = resolved_dir / f"{model_name}.onnx"
            if not path.exists():
                LOGGER.debug("Skipping preload for %s; %s was not found.", model_name, path)
                continue

            try:
                session = ort.InferenceSession(
                    str(path),
                    providers=list(providers),
                    provider_options=list(provider_options),
                )
            except Exception:
                LOGGER.exception("Failed to initialise ONNX Runtime session for %s", model_name)
                continue

            _warm_up_session(session)
            loaded[model_name] = session
            LOGGER.info("Preloaded ONNX Runtime session for %s", model_name)

        PRELOADED_SESSIONS.clear()
        PRELOADED_SESSIONS.update(loaded)
        _PRELOAD_SIGNATURE = signature

    if loaded:
        _log_vram_consumption()
    else:
        LOGGER.info("No ONNX models were preloaded; ensure model files are available in %s.", resolved_dir)
    return dict(PRELOADED_SESSIONS)


def get_preloaded_session(model_name: str) -> "ort.InferenceSession" | None:
    """Return a preloaded session for ``model_name`` if one was initialised."""

    return PRELOADED_SESSIONS.get(model_name)


def reset_preloaded_sessions() -> None:
    """Clear cached sessions; primarily intended for use in tests."""

    with _PRELOAD_LOCK:
        PRELOADED_SESSIONS.clear()
        global _PRELOAD_SIGNATURE
        _PRELOAD_SIGNATURE = None
