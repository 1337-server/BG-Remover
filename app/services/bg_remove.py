"""Utilities for removing image backgrounds using ``rembg`` sessions."""
from __future__ import annotations

import base64
import io
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional

import numpy as np
from PIL import Image, ImageFilter, ImageOps

try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    cv2 = None  # type: ignore

try:  # pragma: no cover - optional dependency during testing
    from rembg import new_session as _rembg_new_session
    from rembg import remove as _rembg_remove
except ModuleNotFoundError as exc:  # pragma: no cover - propagated at runtime
    _rembg_remove = None
    _rembg_new_session = None
    _REMBG_IMPORT_ERROR = exc
else:
    _REMBG_IMPORT_ERROR = None

Session = Any

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
MAX_WORK_DIMENSION = 8000

_SESSION_SINGLETON: Optional[Session] = None
_SESSION_LOCK = threading.Lock()


@dataclass
class RemovalResult:
    """Represents the outcome of processing a single file."""

    path_in: Path
    path_out: Optional[Path]
    success: bool
    error: Optional[str]
    timing_ms: float

    def to_dict(self) -> dict:
        """Return a serialisable representation."""

        return {
            "path_in": str(self.path_in),
            "path_out": str(self.path_out) if self.path_out else None,
            "success": self.success,
            "error": self.error,
            "timing_ms": self.timing_ms,
        }


def create_session(model_name: str = "u2net") -> Session:
    """Create a new ``rembg`` session for the desired model."""

    if _rembg_new_session is None:
        raise RuntimeError("rembg is required to create a background removal session.") from _REMBG_IMPORT_ERROR
    return _rembg_new_session(model_name)


def ensure_global_session(model_name: str = "u2net") -> Session:
    """Initialise and cache a global ``rembg`` session."""

    global _SESSION_SINGLETON
    with _SESSION_LOCK:
        if _SESSION_SINGLETON is None:
            _SESSION_SINGLETON = create_session(model_name)
    assert _SESSION_SINGLETON is not None
    return _SESSION_SINGLETON


def _get_session(session: Optional[Session] = None) -> Session:
    """Return the provided session or the cached singleton."""

    if session is not None:
        return session
    return ensure_global_session()


def build_colorkey_mask(image: Image.Image, tolerance: int = 14) -> Optional[np.ndarray]:
    """Return a colour-key alpha mask when a near-solid background is detected."""

    if tolerance <= 0:
        return None

    rgb_image = image.convert("RGB")
    np_image = np.asarray(rgb_image, dtype=np.uint8)
    height, width, _ = np_image.shape
    patch = max(5, min(height, width) // 10)
    patch = min(patch, height, width)
    if patch <= 0:
        return None

    corners = [
        np_image[0:patch, 0:patch],
        np_image[0:patch, width - patch : width],
        np_image[height - patch : height, 0:patch],
        np_image[height - patch : height, width - patch : width],
    ]
    corner_means = np.array([corner.reshape(-1, 3).mean(axis=0) for corner in corners])
    background_colour = corner_means.mean(axis=0)
    deviations = [np.abs(corner - background_colour).max() for corner in corners]
    if max(deviations) > tolerance * 1.5:
        return None

    delta = np.max(np.abs(np_image.astype(np.int16) - background_colour.astype(np.int16)), axis=2)
    mask = np.where(delta <= tolerance, 0, 255).astype(np.uint8)
    return mask


def _feather_alpha(alpha: np.ndarray, radius: int) -> np.ndarray:
    """Apply Gaussian blur to soften mask edges when OpenCV is available."""

    if radius <= 0 or cv2 is None:
        return alpha
    kernel = radius * 2 + 1
    try:
        blurred = cv2.GaussianBlur(alpha, (kernel, kernel), 0)  # type: ignore[arg-type]
    except Exception:
        return alpha
    return blurred.astype(np.uint8)


def _run_rembg(image: Image.Image, session: Session) -> Image.Image:
    """Execute ``rembg.remove`` and return an RGBA mask image."""

    if _rembg_remove is None:
        raise RuntimeError("rembg is required to process images.") from _REMBG_IMPORT_ERROR

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    result_bytes = _rembg_remove(buffer.read(), session=session)
    result_stream = io.BytesIO(result_bytes)
    result_image = Image.open(result_stream).convert("RGBA")
    result_image.load()
    return result_image


def _apply_alpha_matting(
    alpha: np.ndarray,
    *,
    foreground_threshold: int,
    background_threshold: int,
    erode_size: int,
) -> np.ndarray:
    """Apply basic alpha refinement inspired by legacy rembg settings."""

    fg = int(np.clip(foreground_threshold, 0, 255))
    bg = int(np.clip(background_threshold, 0, 254))
    if fg <= bg:
        fg = min(255, bg + 1)

    normalized = alpha.astype(np.float32)
    normalized = np.clip(normalized - bg, 0, None)
    scale = max(fg - bg, 1)
    normalized = np.clip(normalized * (255.0 / scale), 0, 255)
    refined = normalized.astype(np.uint8)

    if erode_size > 0:
        if cv2 is not None:
            kernel = np.ones((erode_size, erode_size), dtype=np.uint8)
            refined = cv2.erode(refined, kernel, iterations=1)
        else:
            filter_size = max(3, (erode_size // 2) * 2 + 1)
            mask_image = Image.fromarray(refined)
            for _ in range(max(1, erode_size // 3 + 1)):
                mask_image = mask_image.filter(ImageFilter.MinFilter(filter_size))
            refined = np.asarray(mask_image, dtype=np.uint8)

    return refined


def remove_bg_file(
    input_path: str | Path,
    output_path: Optional[str | Path] = None,
    *,
    session: Optional[Session] = None,
    alpha_matting: bool = False,
    am_foreground: int = 240,
    am_background: int = 10,
    am_erode: int = 10,
    use_colorkey_fallback: bool = True,
    colorkey_tolerance: int = 14,
    feather_radius: int = 3,
) -> RemovalResult:
    """Remove the background from a single file and write a PNG with alpha."""

    session = _get_session(session)
    source = Path(input_path)
    if output_path is None:
        destination = source.with_suffix(".png")
    else:
        destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    error: Optional[str] = None

    try:
        with Image.open(source) as raw_image:
            oriented = ImageOps.exif_transpose(raw_image)
            rgba_source = oriented.convert("RGBA")
            work_image = rgba_source.convert("RGB")

            if max(work_image.size) > MAX_WORK_DIMENSION:
                resized = work_image.copy()
                resized.thumbnail((MAX_WORK_DIMENSION, MAX_WORK_DIMENSION), Image.Resampling.LANCZOS)
                mask_image = _run_rembg(resized, session)
                alpha_channel = mask_image.split()[-1]
                alpha_channel = alpha_channel.resize(rgba_source.size, Image.Resampling.LANCZOS)
            else:
                mask_image = _run_rembg(work_image, session)
                alpha_channel = mask_image.split()[-1]

            alpha_np = np.asarray(alpha_channel, dtype=np.uint8)

            if alpha_matting:
                alpha_np = _apply_alpha_matting(
                    alpha_np,
                    foreground_threshold=am_foreground,
                    background_threshold=am_background,
                    erode_size=am_erode,
                )

            if use_colorkey_fallback:
                fallback_mask = build_colorkey_mask(rgba_source, tolerance=colorkey_tolerance)
                if fallback_mask is not None:
                    alpha_np = np.maximum(alpha_np, fallback_mask)

            alpha_np = _feather_alpha(alpha_np, feather_radius)
            final_alpha = Image.fromarray(alpha_np, mode="L")
            output_image = rgba_source.copy()
            output_image.putalpha(final_alpha)
            output_image.save(destination, format="PNG")

            # Explicitly release large arrays to limit memory pressure.
            del alpha_np
            del final_alpha
            del output_image
    except Exception as exc:  # pragma: no cover - depends on external files
        error = str(exc)

    elapsed_ms = (time.perf_counter() - start) * 1000
    success = error is None
    return RemovalResult(source, destination if success else None, success, error, elapsed_ms)


def _iter_input_files(input_dir: Path, recursive: bool) -> Iterable[Path]:
    """Yield input files matching supported extensions."""

    if recursive:
        iterator: Iterable[Path] = input_dir.rglob("*")
    else:
        iterator = input_dir.iterdir()
    for candidate in iterator:
        if candidate.is_file() and candidate.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield candidate


def remove_bg_folder(
    input_dir: str | Path,
    output_dir: Optional[str | Path] = None,
    *,
    session: Optional[Session] = None,
    recursive: bool = False,
    alpha_matting: bool = False,
    am_foreground: int = 240,
    am_background: int = 10,
    am_erode: int = 10,
    use_colorkey_fallback: bool = True,
    colorkey_tolerance: int = 14,
    feather_radius: int = 3,
) -> List[RemovalResult]:
    """Process every supported image in ``input_dir`` sequentially."""

    session = _get_session(session)
    input_path = Path(input_dir)
    if not input_path.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {input_path}")

    if output_dir is None:
        output_path = input_path.parent / f"{input_path.name}_no_bg"
    else:
        output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results: List[RemovalResult] = []
    for source in _iter_input_files(input_path, recursive):
        relative = source.relative_to(input_path) if recursive else Path(source.name)
        destination = (output_path / relative).with_suffix(".png")
        destination.parent.mkdir(parents=True, exist_ok=True)
        result = remove_bg_file(
            source,
            destination,
            session=session,
            alpha_matting=alpha_matting,
            am_foreground=am_foreground,
            am_background=am_background,
            am_erode=am_erode,
            use_colorkey_fallback=use_colorkey_fallback,
            colorkey_tolerance=colorkey_tolerance,
            feather_radius=feather_radius,
        )
        results.append(result)
    return results


def encode_result_image(path: Path) -> str:
    """Return a base64-encoded representation of an output image."""

    with path.open("rb") as file_obj:
        data = file_obj.read()
    return base64.b64encode(data).decode("ascii")
