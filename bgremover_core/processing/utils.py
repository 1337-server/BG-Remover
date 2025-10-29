"""Helper utilities shared by the processing pipeline."""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageOps

from bgremover_core.models.specs import ModelSpec

SUPPORTED_EXTENSIONS: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff")
MAX_FEATHER_RADIUS = 50


def _resize_for_mode(image: Image.Image, size: tuple[int, int], mode: str) -> Image.Image:
    """Return ``image`` resized to ``size`` using the requested ``mode``."""

    mode = (mode or "stretch").lower()
    if mode == "stretch":
        return image.resize(size, Image.Resampling.LANCZOS)
    if mode == "keep-aspect":
        return ImageOps.pad(image, size, method=Image.Resampling.LANCZOS, color=None, centering=(0.5, 0.5))
    if mode == "crop":
        return ImageOps.fit(image, size, method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))
    if mode == "auto":
        width, height = image.size
        target_width, target_height = size
        if height == 0 or target_height == 0:
            return image.resize(size, Image.Resampling.LANCZOS)
        aspect_ratio = width / height
        target_ratio = target_width / target_height
        if abs(aspect_ratio - target_ratio) <= 0.1:
            return image.resize(size, Image.Resampling.LANCZOS)
        if aspect_ratio > target_ratio:
            return ImageOps.fit(image, size, method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))
        return ImageOps.pad(image, size, method=Image.Resampling.LANCZOS, color=None, centering=(0.5, 0.5))
    if mode == "stretch":  # pragma: no cover - defensive fallback
        return image.resize(size, Image.Resampling.LANCZOS)
    return image.resize(size, Image.Resampling.LANCZOS)


def normalise_image(
    image: Image.Image,
    spec: ModelSpec,
    *,
    resize_mode: str = "stretch",
) -> np.ndarray:
    """Return a model-ready tensor for ``image`` according to ``spec`` and ``resize_mode``."""

    resized = _resize_for_mode(image.convert("RGB"), spec.input_size, resize_mode)
    rgb_array = np.asarray(resized, dtype=np.float32)
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


def refine_mask(
    mask: Image.Image,
    image: Image.Image,
    *,
    alpha_matting: bool = False,
    foreground_threshold: int = 240,
    background_threshold: int = 10,
    erode_size: int = 10,
    smoothing: float = 0.0,
    edge_refinement: bool = False,
) -> Image.Image:
    """Return ``mask`` refined according to advanced settings."""

    refined = mask.convert("L")
    if alpha_matting:
        refined = _apply_alpha_matting(
            refined,
            image.convert("RGB"),
            foreground_threshold=foreground_threshold,
            background_threshold=background_threshold,
            erode_size=erode_size,
        )
    if smoothing > 0:
        radius = max(0.0, min(float(smoothing), 1.0)) * 8.0
        if radius > 0:
            refined = refined.filter(ImageFilter.GaussianBlur(radius=radius))
    if edge_refinement:
        refined = refined.filter(ImageFilter.UnsharpMask(radius=2, percent=160, threshold=3))
    return refined


def _apply_alpha_matting(
    mask: Image.Image,
    image: Image.Image,
    *,
    foreground_threshold: int,
    background_threshold: int,
    erode_size: int,
) -> Image.Image:
    """Return a mask refined using simple alpha matting heuristics."""

    fg = int(max(0, min(255, foreground_threshold)))
    bg = int(max(0, min(255, background_threshold)))
    erode = int(max(0, min(30, erode_size)))

    mask_array = np.asarray(mask, dtype=np.uint8)
    # Ensure the mask array can be modified in place for thresholding operations.
    if not mask_array.flags.writeable:
        mask_array = mask_array.copy()
    luminance = np.asarray(image.convert("L"), dtype=np.uint8)
    mask_array[luminance >= fg] = 255
    mask_array[luminance <= bg] = 0

    refined = Image.fromarray(mask_array, mode="L")
    if erode > 0:
        size = max(3, erode * 2 + 1)
        refined = refined.filter(ImageFilter.MinFilter(size=size))
        refined = refined.filter(ImageFilter.MaxFilter(size=size))
    return refined


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
    "refine_mask",
]
