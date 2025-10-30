"""Utility helpers for GPU memory-aware worker scaling."""
from __future__ import annotations

import logging
from dataclasses import dataclass

try:
    import pynvml  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - optional dependency
    pynvml = None

LOGGER = logging.getLogger(__name__)

# Estimate of VRAM consumed per image inference.
GPU_MEMORY_PER_IMAGE_BYTES: int = 2 * 1024 * 1024 * 1024


@dataclass(slots=True, frozen=True)
class GpuMemorySnapshot:
    """Capture the total and free VRAM reported by NVML."""

    total: int
    free: int


def _initialise_nvml() -> bool:
    """Initialise NVML, returning ``True`` when ready for use."""

    if pynvml is None:
        LOGGER.debug("pynvml not available; GPU memory cannot be queried.")
        return False
    try:
        pynvml.nvmlInit()
    except Exception as error:  # pragma: no cover - defensive logging
        LOGGER.debug("Failed to initialise NVML: %s", error)
        return False
    return True


def _shutdown_nvml() -> None:
    """Attempt to shut down NVML gracefully."""

    if pynvml is None:
        return
    try:
        pynvml.nvmlShutdown()
    except Exception:  # pragma: no cover - NVML can fail to shut down cleanly
        pass


def query_gpu_memory() -> GpuMemorySnapshot | None:
    """Return the per-device GPU memory statistics or ``None`` if unavailable."""

    if not _initialise_nvml():
        return None
    try:
        device_count = pynvml.nvmlDeviceGetCount()
        if device_count <= 0:
            LOGGER.debug("No GPU devices reported by NVML.")
            return None

        min_total_bytes: int | None = None
        min_free_bytes: int | None = None
        for index in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            device_total = int(memory.total)
            device_free = int(memory.free)
            if min_total_bytes is None or device_total < min_total_bytes:
                min_total_bytes = device_total
            if min_free_bytes is None or device_free < min_free_bytes:
                min_free_bytes = device_free

        if min_total_bytes is None or min_free_bytes is None:
            LOGGER.debug("Failed to determine per-device GPU memory statistics.")
            return None

        return GpuMemorySnapshot(total=min_total_bytes, free=min_free_bytes)
    except Exception as error:  # pragma: no cover - defensive logging
        LOGGER.debug("Failed to query NVML memory info: %s", error)
        return None
    finally:
        _shutdown_nvml()


def recommend_worker_count(
    requested_workers: int,
    *,
    per_image_bytes: int = GPU_MEMORY_PER_IMAGE_BYTES,
    snapshot: GpuMemorySnapshot | None = None,
) -> int:
    """Return a safe worker count based on VRAM availability."""

    requested = max(1, int(requested_workers))
    if requested == 1:
        return 1

    active_snapshot = snapshot or query_gpu_memory()
    if active_snapshot is None or active_snapshot.total <= 0:
        LOGGER.info(
            "GPU memory information unavailable; falling back to a single worker for safety.",
        )
        return 1

    per_image = max(per_image_bytes, 1)
    available_bytes = min(active_snapshot.total, active_snapshot.free)
    capacity = max(1, available_bytes // per_image)
    return max(1, min(requested, int(capacity)))


def has_enough_memory(
    *,
    per_image_bytes: int = GPU_MEMORY_PER_IMAGE_BYTES,
) -> bool:
    """Return ``True`` when the reported free VRAM can handle another image."""

    snapshot = query_gpu_memory()
    if snapshot is None:
        return True
    return snapshot.free >= max(per_image_bytes, 1)


__all__ = [
    "GPU_MEMORY_PER_IMAGE_BYTES",
    "GpuMemorySnapshot",
    "has_enough_memory",
    "query_gpu_memory",
    "recommend_worker_count",
]
