"""Helpers for selecting the optimal ONNX Runtime execution provider."""
from __future__ import annotations

from typing import Dict, List, Tuple

import logging

LOGGER = logging.getLogger(__name__)


def detect_gpu_name() -> str | None:
    """Return the name of the first available NVIDIA GPU, if any."""

    try:
        import pynvml  # type: ignore
    except ModuleNotFoundError:
        pynvml = None  # type: ignore

    if pynvml is not None:  # pragma: no cover - requires GPU hardware
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            name = pynvml.nvmlDeviceGetName(handle)
        except Exception:  # pragma: no cover - hardware specific behaviour
            LOGGER.debug("Unable to query GPU name via pynvml", exc_info=True)
        else:
            if isinstance(name, bytes):
                try:
                    return name.decode("utf-8")
                except Exception:  # pragma: no cover - unexpected encoding
                    return name.decode("latin-1", errors="ignore")
            return str(name)
        finally:
            try:
                pynvml.nvmlShutdown()
            except Exception:  # pragma: no cover - shutdown best-effort
                LOGGER.debug("Failed to shutdown pynvml cleanly", exc_info=True)

    try:
        import torch
    except ModuleNotFoundError:
        torch = None  # type: ignore

    if torch is not None:
        try:
            if torch.cuda.is_available():  # type: ignore[operator]
                return torch.cuda.get_device_name(0)
        except Exception:
            LOGGER.debug("Unable to query GPU name via torch", exc_info=True)

    return None


def onnx_providers_available() -> List[str]:
    """Return the list of ONNX Runtime execution providers available."""

    try:
        import onnxruntime as ort
    except ModuleNotFoundError:
        LOGGER.debug("onnxruntime is not installed; assuming no providers")
        return []

    try:
        providers = list(ort.get_available_providers())
    except Exception:
        LOGGER.exception("Failed to query ONNX Runtime providers")
        return []
    return providers


def pick_execution_provider(mode: str, device_id: int) -> Tuple[str, Dict[str, int]]:
    """Select the best available execution provider for ``mode``.

    The provider selection honours the ONNX Runtime priority order of
    CUDA, TensorRT, and finally the CPU execution provider. When GPU support
    is requested and a compatible GPU provider is present, a dictionary of
    provider options including the requested ``device_id`` is returned.
    """

    normalised_mode = (mode or "auto").strip().lower()
    if normalised_mode == "cpu":
        return "CPUExecutionProvider", {}

    providers = onnx_providers_available()
    preferred_gpu_providers = ["CUDAExecutionProvider", "TensorrtExecutionProvider"]

    if normalised_mode in {"cuda", "auto"}:
        for provider_name in preferred_gpu_providers:
            if provider_name in providers:
                return provider_name, {"device_id": int(device_id)}

    return "CPUExecutionProvider", {}


def is_rtx_50xx(name: str) -> bool:
    """Return ``True`` when ``name`` refers to an RTX 50-series GPU."""

    name_lower = name.lower()
    return any(token in name_lower for token in ("rtx 5090", "rtx 5080", "rtx 5070", "rtx 5060", "rtx 50"))
