"""Shared ONNX Runtime session registry with GPU warm-up support."""
from __future__ import annotations

import logging
import os
import time
from collections.abc import Iterable, Mapping, MutableMapping
from pathlib import Path
from threading import Lock
from typing import Any, cast

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

PROVIDERS_PRIORITY: tuple[str, ...] = (
    "CUDAExecutionProvider",
    "TensorrtExecutionProvider",
    "CPUExecutionProvider",
)

_PRELOADED_SESSIONS: dict[str, ort.InferenceSession] = {}
_PRELOAD_SIGNATURE: tuple[str, tuple[str, ...], str, int] | None = None
_AVAILABLE_PROVIDERS: list[str] = []
_PRELOAD_LOCK = Lock()

_CACHE_DIR = Path(
    os.environ.get(
        "BG_ONNXRUNTIME_CACHE_DIR",
        os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache") / "br-remover" / "onnxruntime",
    )
).expanduser()
_TIMING_CACHE_PATH = _CACHE_DIR / "timing_cache"
_CACHE_READY: bool | None = None


def _ensure_cache_dirs() -> bool:
    """Ensure that the ONNX runtime cache directories exist and are writable."""

    global _CACHE_READY
    if _CACHE_READY is not None:
        return _CACHE_READY

    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        LOGGER.warning(
            "Unable to initialise ONNX runtime cache directory; GPU caching disabled",
            extra={"cache_dir": str(_CACHE_DIR), "error": str(exc)},
        )
        _CACHE_READY = False
    else:
        _CACHE_READY = True
    return _CACHE_READY


def _tensorrt_provider_options(device_id: int) -> MutableMapping[str, Any]:
    """Return tuned TensorRT execution provider options for fast warm starts."""

    cache_available = _ensure_cache_dirs()
    options: dict[str, Any] = {
        "device_id": int(device_id),
        "trt_fp16_enable": True,
        "trt_force_sequential_engine_build": False,
        "trt_int8_enable": False,
        "trt_dla_enable": False,
        "trt_cuda_graph_enable": True,
        # Keep TensorRT quiet during engine builds while still surfacing errors.
        "trt_detailed_build_log": False,
        "trt_logger_severity": "kERROR",
    }

    if cache_available:
        options.update(
            {
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": str(_CACHE_DIR),
                "trt_timing_cache_enable": True,
                "trt_timing_cache_path": str(_TIMING_CACHE_PATH),
            }
        )
    else:
        options.update(
            {
                "trt_engine_cache_enable": False,
                "trt_timing_cache_enable": False,
            }
        )
    return options


def _cuda_provider_options(device_id: int) -> MutableMapping[str, Any]:
    """Return CUDA execution provider options tuned for aggressive caching."""

    return {
        "device_id": int(device_id),
        "arena_extend_strategy": "kSameAsRequested",
        "gpu_mem_limit": 0,
        "cudnn_conv_algo_search": "EXHAUSTIVE",
        "do_copy_in_default_stream": True,
    }


def _resolve_model_dir(model_dir: str | os.PathLike[str] | None) -> Path:
    """Return the directory containing cached ONNX model weights."""

    if model_dir is not None:
        directory = Path(model_dir)
    else:
        directory = Path(os.environ.get("U2NET_HOME", Path.home() / ".u2net"))
    return directory.expanduser()


def _normalise_mode(mode: object | None) -> str:
    """Return a normalised accelerator mode string."""

    if mode is None:
        return "auto"
    value = str(mode).strip().lower()
    return value if value in {"auto", "cuda", "cpu"} else "auto"


def _extract_accelerator_config(config: Mapping[str, object] | None) -> tuple[str, int]:
    """Return the desired accelerator mode and CUDA device identifier."""

    settings = config or {}
    mode = _normalise_mode(settings.get("BG_ACCELERATOR"))
    if runtime_compat.is_force_cpu_enabled():
        mode = "cpu"
    raw_device = settings.get("BG_CUDA_DEVICE_ID", 0)
    try:
        device_id = int(cast(Any, raw_device))
    except (TypeError, ValueError):
        device_id = 0
    return mode, device_id


def _provider_priority(mode: str, device_id: int) -> tuple[list[str], list[MutableMapping[str, Any]]]:
    """Return provider names and configuration dictionaries for ONNXRuntime."""

    available = accelerator.onnx_providers_available()
    providers: list[str] = []
    provider_options: list[MutableMapping[str, Any]] = []

    gpu_requested = mode != "cpu"
    prefer_tensorrt = False
    if gpu_requested and "TensorrtExecutionProvider" in available:
        gpu_name = accelerator.detect_gpu_name()
        if gpu_name and accelerator.is_rtx_50xx(gpu_name):
            prefer_tensorrt = runtime_compat.supports_tensorrt_cuda_129()
            if prefer_tensorrt:
                LOGGER.info(
                    "TensorRT execution provider enabled for RTX 50-series GPU", extra={"gpu_name": gpu_name}
                )
            else:
                LOGGER.debug("TensorRT provider available but CUDA 12.9 support not detected")

    provider_candidates = list(PROVIDERS_PRIORITY[:-1])
    if prefer_tensorrt:
        provider_candidates = ["TensorrtExecutionProvider", "CUDAExecutionProvider"]

    if gpu_requested:
        for provider in provider_candidates:
            if provider in available:
                if provider == "TensorrtExecutionProvider":
                    options = _tensorrt_provider_options(device_id)
                elif provider == "CUDAExecutionProvider":
                    options = _cuda_provider_options(device_id)
                else:
                    options = {"device_id": int(device_id)}
                providers.append(provider)
                provider_options.append(options)

    providers.append("CPUExecutionProvider")
    provider_options.append({})
    return providers, provider_options


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
    """Execute a dummy inference to prime CUDA kernels and memory allocations."""

    try:
        feed_dict = _dummy_feed(session)
        if feed_dict:
            session.run(None, feed_dict)
    except Exception:  # pragma: no cover - hardware specific behaviour
        LOGGER.exception("Failed to warm ONNX Runtime session", exc_info=True)


def _log_vram_usage() -> None:
    """Log estimated GPU memory usage after preloading sessions."""

    try:  # pragma: no cover - requires torch with CUDA support
        import torch

        if torch.cuda.is_available():  # type: ignore[truthy-bool]
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            used_mib = (total_bytes - free_bytes) / (1024 ** 2)
            total_mib = total_bytes / (1024 ** 2)
            LOGGER.info(
                "Preloaded models occupy approximately %.1f MiB of %.1f MiB VRAM.",
                used_mib,
                total_mib,
            )
            return
    except ModuleNotFoundError:
        torch = None  # type: ignore[assignment]
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
            "Model preload complete; VRAM metrics unavailable because pynvml is not installed.",
        )
    except Exception:
        LOGGER.debug("Unable to query VRAM usage via pynvml", exc_info=True)
    else:
        used_mib = info.used / (1024 ** 2)
        total_mib = info.total / (1024 ** 2)
        LOGGER.info(
            "Preloaded models occupy approximately %.1f MiB of %.1f MiB VRAM.",
            used_mib,
            total_mib,
        )


def preload_models(
    *,
    model_dir: str | os.PathLike[str] | None = None,
    config: Mapping[str, object] | None = None,
    model_names: Iterable[str] | None = None,
    warm: bool = True,
) -> dict[str, ort.InferenceSession]:
    """Preload ONNX Runtime sessions for the requested models.

    The returned mapping is a copy of the registry cache to avoid callers
    mutating the internal dictionaries by accident.
    """

    runtime_compat.ensure_runtime_ready()

    if ort is None:
        LOGGER.info(
            "onnxruntime is not installed; skipping eager model preload."
            " Install onnxruntime-gpu to enable GPU acceleration.",
        )
        return {}

    resolved_dir = _resolve_model_dir(model_dir)
    requested_names = tuple(sorted(set(model_names or DEFAULT_MODEL_NAMES)))
    mode, device_id = _extract_accelerator_config(config)
    signature = (str(resolved_dir), requested_names, mode, device_id)

    with _PRELOAD_LOCK:
        global _PRELOAD_SIGNATURE, _AVAILABLE_PROVIDERS

        if _PRELOAD_SIGNATURE == signature and _PRELOADED_SESSIONS:
            return dict(_PRELOADED_SESSIONS)

        providers, provider_options = _provider_priority(mode, device_id)
        _AVAILABLE_PROVIDERS = accelerator.onnx_providers_available()
        LOGGER.info("Available ONNX Runtime providers: %s", ", ".join(_AVAILABLE_PROVIDERS) or "none")

        loaded: dict[str, ort.InferenceSession] = {}

        for model_name in requested_names:
            model_path = resolved_dir / f"{model_name}.onnx"
            if not model_path.exists():
                LOGGER.debug("Skipping preload for %s; %s not found.", model_name, model_path)
                continue

            start_time = time.perf_counter()
            try:
                session = ort.InferenceSession(  # type: ignore[attr-defined]
                    str(model_path),
                    providers=list(providers),
                    provider_options=list(provider_options),
                )
            except Exception:
                LOGGER.exception("Failed to initialise ONNX Runtime session for %s", model_name)
                continue

            if warm:
                _warm_session(session)

            loaded[model_name] = session
            elapsed = time.perf_counter() - start_time
            LOGGER.info(
                "Loaded %s in %.2f s (providers=%s)",
                model_name,
                elapsed,
                ", ".join(session.get_providers()),
            )

        _PRELOADED_SESSIONS.clear()
        _PRELOADED_SESSIONS.update(loaded)
        _PRELOAD_SIGNATURE = signature

    if loaded:
        _log_vram_usage()
    else:
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

