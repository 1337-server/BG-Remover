"""Image conversion helpers that avoid runtime-specific dependencies."""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image


def decode_image_bytes(data: bytes) -> Image.Image:
    """Return a :class:`PIL.Image.Image` created from ``data``."""

    return Image.open(io.BytesIO(data)).convert("RGBA")


def load_image_from_array(array: np.ndarray) -> Image.Image:
    """Return a :class:`PIL.Image.Image` created from a numpy ``array``."""

    if array.ndim == 3 and array.shape[2] == 4:
        mode = "RGBA"
    elif array.ndim == 3 and array.shape[2] == 3:
        mode = "RGB"
    else:
        raise ValueError("Input array must be HxWx3 or HxWx4")
    return Image.fromarray(array.astype("uint8"), mode=mode)


def save_image_to_path(image: Image.Image, path: Path, *, format_hint: str | None = None) -> Path:
    """Persist ``image`` to ``path`` using ``format_hint`` when provided."""

    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs = {"optimize": True}
    image.save(path, format_hint or image.format or "PNG", **save_kwargs)
    return path


__all__ = ["decode_image_bytes", "load_image_from_array", "save_image_to_path"]
