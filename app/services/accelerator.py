"""Helpers for discovering and configuring ONNXRuntime execution providers."""
from __future__ import annotations

import logging
from typing import List, Mapping, Tuple

LOGGER = logging.getLogger(__name__)


def _decode_bytes(value: bytes | str | None) -> str | None:
    """Return ``value`` decoded to ``str`` when necessary."""

    if value is None:
        return None
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except Exception:  # pragma: no cover - extremely defensive
            return value.decode("latin-1", errors="ignore")
    return value


def detect_gpu_name() -> str | None:
    """Return the name of the first CUDA device, or ``None`` if unavailable."""

    try:
        import pynvml  # type: ignore
    except ModuleNotFoundError:
        pynvml = None  # type: ignore[assignment]
    except Exception as exc:  # pragma: no cover - defensive guard
        LOGGER.debug("NVML import failed: %s", exc)
        return None

    if pynvml is not None:
        try:
            pynvml.nvmlInit()  # type: ignore[attr-defined]
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # type: ignore[attr-defined]
            name = pynvml.nvmlDeviceGetName(handle)  # type: ignore[attr-defined]
            return _decode_bytes(name)
        except Exception as exc:  # pragma: no cover - hardware-specific
            LOGGER.debug("Unable to query GPU name via NVML: %s", exc)
        finally:
            try:
                pynvml.nvmlShutdown()  # type: ignore[attr-defined]
            except Exception:  # pragma: no cover - defensive cleanup
                pass

    try:
        import torch  # type: ignore
    except ModuleNotFoundError:
        return None
    except Exception as exc:  # pragma: no cover - defensive guard
        LOGGER.debug("Torch import failed while detecting GPUs: %s", exc)
        return None

    try:
        if torch.cuda.is_available():  # type: ignore[attr-defined]
            name = torch.cuda.get_device_name(0)  # type: ignore[attr-defined]
            return _decode_bytes(name)
    except Exception as exc:  # pragma: no cover - hardware-specific
        LOGGER.debug("Unable to query GPU name via torch: %s", exc)
    return None


def onnx_providers_available() -> List[str]:
    """Return a list of ONNXRuntime execution providers available at runtime."""

    try:
        import onnxruntime as ort  # type: ignore
    except ModuleNotFoundError:
        LOGGER.debug("onnxruntime is not installed; assuming CPU execution only")
        return ["CPUExecutionProvider"]
    except Exception as exc:  # pragma: no cover - defensive guard
        LOGGER.warning("Failed to import onnxruntime: %s", exc)
        return ["CPUExecutionProvider"]

    try:
        providers = list(ort.get_available_providers())
    except Exception as exc:  # pragma: no cover - defensive guard
        LOGGER.warning("Unable to query ONNXRuntime providers: %s", exc)
        return ["CPUExecutionProvider"]
    return providers


def pick_execution_provider(mode: str, device_id: int) -> Tuple[str, Mapping[str, int]]:
    """Return the preferred execution provider based on ``mode`` and ``device_id``."""

    normalized_mode = (mode or "auto").strip().lower()
    providers = onnx_providers_available()

    if normalized_mode == "cpu":
        return "cpu", {}

    if normalized_mode in {"cuda", "auto"} and "CUDAExecutionProvider" in providers:
        return "cuda", {"device_id": int(device_id)}

    return "cpu", {}


_RTX_50XX_MARKERS = ("rtx 5090", "rtx 5080", "rtx 5070", "rtx 5060", "rtx 50")


def is_rtx_50xx(name: str) -> bool:
    """Return ``True`` when ``name`` appears to describe an RTX 50-series GPU."""

    if not name:
        return False
    lower = name.lower()
    return any(marker in lower for marker in _RTX_50XX_MARKERS)


def describe_selected_provider(provider: str, options: Mapping[str, int]) -> str:
    """Return a human-readable description of the selected execution provider."""

    if provider == "cuda":
        device_id = options.get("device_id", 0)
        return f"CUDAExecutionProvider (device {device_id})"
    return "CPUExecutionProvider"
