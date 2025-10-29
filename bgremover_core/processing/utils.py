"""Helper utilities shared by the processing pipeline."""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from bgremover_core.models.specs import ModelSpec

SUPPORTED_EXTENSIONS: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff")
MAX_FEATHER_RADIUS = 50


def normalise_image(image: Image.Image, spec: ModelSpec) -> np.ndarray:
    """Return a model-ready tensor for ``image`` according to ``spec``."""

    rgb_image = image.convert("RGB").resize(spec.input_size, Image.Resampling.LANCZOS)
    rgb_array = np.asarray(rgb_image, dtype=np.float32)
    scale = spec.normalisation_scale if spec.normalisation_scale > 0 else 1.0
    rgb_array /= scale
    normalised = np.zeros_like(rgb_array, dtype=np.float32)
    for index in range(3):
        normalised[:, :, index] = (rgb_array[:, :, index] - spec.mean[index]) / spec.std[index]
    normalised = normalised.transpose((2, 0, 1))
    return np.expand_dims(normalised, 0).astype(np.float32)


def compute_mask_image(array: np.ndarray, original_size: tuple[int, int]) -> Image.Image:
    """Convert an ONNX output ``array`` into a resized mask image."""

    mask = array
    while mask.ndim > 2:
        mask = mask[0]
    max_value = float(mask.max())
    min_value = float(mask.min())
    if max_value - min_value > 1e-5:
        mask = (mask - min_value) / (max_value - min_value)
    else:
        mask = np.zeros_like(mask)
    mask = (mask * 255).clip(0, 255).astype("uint8")
    image = Image.fromarray(mask, mode="L")
    if image.size != original_size:
        image = image.resize(original_size, Image.Resampling.LANCZOS)
    return image


def apply_mask_to_image(
    image: Image.Image,
    mask: Image.Image,
    *,
    feather_radius: int = 3,
) -> Image.Image:
    """Return an RGBA image with ``mask`` applied as the alpha channel."""

    alpha = np.asarray(mask, dtype=np.uint8)
    if feather_radius > 0:
        radius = min(int(feather_radius), MAX_FEATHER_RADIUS)
        if radius > 0:
            alpha_image = Image.fromarray(alpha, mode="L")
            alpha = np.asarray(alpha_image.filter(ImageFilter.GaussianBlur(radius=radius)), dtype=np.uint8)
    rgba = image.convert("RGBA")
    rgba.putalpha(Image.fromarray(alpha, mode="L"))
    return rgba


def iter_image_files(directory: Path, recursive: bool = False) -> Iterator[Path]:
    """Yield image files from ``directory`` respecting :data:`SUPPORTED_EXTENSIONS`."""

    directory = directory.expanduser()
    if recursive:
        for path in directory.rglob("*"):
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
                yield path
    else:
        for path in directory.iterdir():
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
                yield path


__all__ = [
    "SUPPORTED_EXTENSIONS",
    "MAX_FEATHER_RADIUS",
    "apply_mask_to_image",
    "compute_mask_image",
    "iter_image_files",
    "normalise_image",
]
