"""High level background removal pipelines shared across runtimes."""
from __future__ import annotations

import fnmatch
import logging
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from bgremover_core.config import Config, load_config
from bgremover_core.io.image_io import load_image_from_array, save_image_to_path
from bgremover_core.io.paths import resolve_batch_output_dir
from bgremover_core.models.loader import BackgroundRemovalSession, detect_providers, get_session
from bgremover_core.processing.utils import apply_mask_to_image, iter_image_files, refine_mask

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class ReportEntry:
    """Structured result describing the outcome for a single file."""

    path_in: Path
    path_out: Path | None
    success: bool
    elapsed_ms: float
    error: str | None = None


@dataclass(slots=True)
class Report:
    """Summary object returned by :func:`process_folder`."""

    entries: list[ReportEntry]

    @property
    def total(self) -> int:
        """Return the total number of processed entries."""

        return len(self.entries)

    @property
    def successes(self) -> int:
        """Return the number of successful entries."""

        return sum(1 for entry in self.entries if entry.success)

    @property
    def failures(self) -> int:
        """Return the number of failed entries."""

        return sum(1 for entry in self.entries if not entry.success)

    def to_rows(self) -> list[dict[str, str | float | None]]:
        """Return a serialisable representation suited for table rendering."""

        rows: list[dict[str, str | float | None]] = []
        for entry in self.entries:
            rows.append(
                {
                    "input": str(entry.path_in),
                    "output": str(entry.path_out) if entry.path_out else None,
                    "success": entry.success,
                    "elapsed_ms": entry.elapsed_ms,
                    "error": entry.error,
                }
            )
        return rows


class PipelineError(RuntimeError):
    """Raised when the pipeline cannot process an input image."""


@dataclass(slots=True)
class ProcessingOptions:
    """Container describing advanced image processing preferences.

    The ``mask_blur`` attribute represents a direct Gaussian blur radius applied
    to the alpha mask. When zero the pipeline falls back to the ``smoothing``
    ratio for backwards compatibility with earlier front-ends.
    """

    resize_mode: str = "stretch"
    feather_radius: int = 3
    alpha_matting: bool = False
    foreground_threshold: int = 240
    background_threshold: int = 10
    erode_size: int = 10
    smoothing: float = 0.0
    mask_blur: float = 0.0
    edge_refinement: bool = False
    post_process_mask: bool = True
    only_mask: bool = False
    mask_threshold: float = 0.0
    cut_out_mode: str = "object"
    background_mode: str = "clear"
    background_color: tuple[int, int, int] | None = None
    output_format: str = "PNG"
    preserve_names: bool = False
    max_workers: int = 1

    @classmethod
    def from_kwargs(cls, feather_radius: int = 3, **kwargs: object) -> ProcessingOptions:
        """Return a :class:`ProcessingOptions` parsed from ``kwargs``."""

        resize_mode = str(kwargs.get("resize_mode") or kwargs.get("input_resize") or "stretch")
        alpha_matting = bool(kwargs.get("alpha_matting", False))
        smoothing = _coerce_float(kwargs.get("smoothing"), default=0.0, minimum=0.0, maximum=1.0)
        mask_blur = _coerce_float(
            kwargs.get("mask_blur") or kwargs.get("mask_blur_radius"),
            default=0.0,
            minimum=0.0,
            maximum=25.0,
        )
        edge_refinement = bool(kwargs.get("edge_refinement", False))
        foreground_threshold = _coerce_int(
            kwargs.get("foreground_threshold")
            or kwargs.get("alpha_matting_foreground_threshold")
            or kwargs.get("alpha_foreground_threshold"),
            default=240,
            minimum=0,
            maximum=255,
        )
        background_threshold = _coerce_int(
            kwargs.get("background_threshold")
            or kwargs.get("alpha_matting_background_threshold")
            or kwargs.get("alpha_background_threshold"),
            default=10,
            minimum=0,
            maximum=255,
        )
        erode_size = _coerce_int(
            kwargs.get("erode_size")
            or kwargs.get("alpha_matting_erode_size")
            or kwargs.get("alpha_erode_size"),
            default=10,
            minimum=0,
            maximum=30,
        )
        background_color = _coerce_color(kwargs.get("background_color"))
        output_format = str(kwargs.get("output_format") or "PNG").upper()
        preserve_names = bool(kwargs.get("preserve_names", False))
        max_workers = _coerce_int(
            kwargs.get("max_workers") or kwargs.get("parallel_threads"),
            default=1,
            minimum=1,
            maximum=16,
        )
        post_process_mask = bool(kwargs.get("post_process_mask", True))
        only_mask = bool(kwargs.get("only_mask", False))
        mask_threshold = _coerce_float(
            kwargs.get("mask_threshold"),
            default=0.0,
            minimum=0.0,
            maximum=1.0,
        )
        cut_out_mode = str(kwargs.get("cut_out_mode") or "object").strip().lower()
        if cut_out_mode not in {"object", "mask", "bbox"}:
            cut_out_mode = "object"
        background_mode = str(kwargs.get("background_mode") or "clear").strip().lower()
        if background_mode not in {"none", "fill", "clear"}:
            background_mode = "clear"
        resolved_feather = _coerce_int(feather_radius, default=3, minimum=0, maximum=50)
        return cls(
            resize_mode=resize_mode,
            feather_radius=resolved_feather,
            alpha_matting=alpha_matting,
            foreground_threshold=foreground_threshold,
            background_threshold=background_threshold,
            erode_size=erode_size,
            smoothing=smoothing,
            mask_blur=mask_blur,
            edge_refinement=edge_refinement,
            post_process_mask=post_process_mask,
            only_mask=only_mask,
            mask_threshold=mask_threshold,
            cut_out_mode=cut_out_mode,
            background_mode=background_mode,
            background_color=background_color,
            output_format=output_format,
            preserve_names=preserve_names,
            max_workers=max_workers,
        )


@dataclass(slots=True, frozen=True)
class ProcessingDebugInfo:
    """Snapshot of intermediate tensors collected during processing."""

    pre_shape: tuple[int, ...]
    pre_min: float
    pre_max: float
    logits_shape: tuple[int, ...]
    logits_min: float
    logits_max: float
    mask_pre_min: float
    mask_pre_max: float
    mask_post_min: float
    mask_post_max: float


@dataclass(slots=True)
class ProcessingResult:
    """Aggregate result returned by :func:`process_image`."""

    image: Image.Image
    mask: np.ndarray
    alpha: np.ndarray
    debug: ProcessingDebugInfo


def _coerce_int(
    value: object,
    *,
    default: int,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    """Return ``value`` coerced to ``int`` within optional bounds."""

    try:
        result = int(value) if value is not None else int(default)
    except (TypeError, ValueError):
        result = int(default)
    if minimum is not None:
        result = max(minimum, result)
    if maximum is not None:
        result = min(maximum, result)
    return result


def _coerce_float(
    value: object,
    *,
    default: float,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    """Return ``value`` coerced to ``float`` within optional bounds."""

    try:
        result = float(value) if value is not None else float(default)
    except (TypeError, ValueError):
        result = float(default)
    if minimum is not None:
        result = max(minimum, result)
    if maximum is not None:
        result = min(maximum, result)
    return result


def _coerce_color(value: object) -> tuple[int, int, int] | None:
    """Return ``value`` coerced to an RGB tuple when possible."""

    if value is None:
        return None
    if isinstance(value, tuple) and len(value) == 3:
        return tuple(_coerce_int(component, default=0, minimum=0, maximum=255) for component in value)
    if isinstance(value, str):
        text = value.strip().lstrip("#")
        if len(text) == 6:
            try:
                return (int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16))
            except ValueError:  # pragma: no cover - defensive branch
                return None
    return None


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
    LOGGER.warning("Unknown resize mode %s; falling back to 'stretch'", mode)
    return image.resize(size, Image.Resampling.LANCZOS)


def _apply_background(image: Image.Image, color: tuple[int, int, int] | None) -> Image.Image:
    """Return ``image`` composited onto ``color`` when provided."""

    if not color:
        return image
    background = Image.new("RGBA", image.size, (*color, 255))
    return Image.alpha_composite(background, image)


def _crop_to_mask_bounds(
    image: Image.Image,
    mask: Image.Image,
    mask_array: np.ndarray,
) -> tuple[Image.Image, Image.Image, np.ndarray]:
    """Return ``image`` and mask data cropped to the non-zero mask bounding box."""

    bbox = mask.getbbox()
    if not bbox:
        return image, mask, mask_array
    left, upper, right, lower = bbox
    cropped_image = image.crop((left, upper, right, lower))
    cropped_mask = mask.crop((left, upper, right, lower))
    cropped_array = mask_array[upper:lower, left:right]
    return cropped_image, cropped_mask, cropped_array


def _infer_output_suffix(format_name: str) -> tuple[str, str]:
    """Return the Pillow format hint and extension for ``format_name``."""

    match format_name.upper():
        case "JPEG" | "JPG":
            return "JPEG", "jpg"
        case "WEBP":
            return "WEBP", "webp"
        case _:
            return "PNG", "png"


def _prepare_session(
    model_key: str,
    *,
    config: Config,
    providers: Sequence[str] | None = None,
) -> BackgroundRemovalSession:
    providers = list(providers or detect_providers(config.provider_hints))
    return get_session(model_key, providers=providers, model_dir=config.resolved_model_dir())


def preprocess(img_rgb: np.ndarray, spec: Any) -> np.ndarray:
    """Return a contiguous CHW float32 tensor according to ``spec``."""

    if img_rgb.ndim != 3 or img_rgb.shape[2] != 3:
        raise ValueError("preprocess expects an RGB image with shape HxWx3")
    if img_rgb.dtype != np.uint8:
        raise ValueError("preprocess expects uint8 input data")
    if not hasattr(spec, "input_size") or not hasattr(spec, "mean") or not hasattr(spec, "std"):
        raise ValueError("spec must expose input_size, mean, and std attributes")

    width, height = spec.input_size
    resized = cv2.resize(img_rgb, (width, height), interpolation=cv2.INTER_LINEAR)
    tensor = resized.astype(np.float32) / 255.0
    mean = np.asarray(spec.mean, dtype=np.float32)
    std = np.asarray(spec.std, dtype=np.float32)
    tensor = (tensor - mean) / std
    tensor = np.transpose(tensor, (2, 0, 1))[None, ...].astype(np.float32)
    tensor = np.ascontiguousarray(tensor)
    if tensor.dtype != np.float32:
        raise AssertionError("Preprocess output must be float32")
    if not tensor.flags.c_contiguous:
        raise AssertionError("Preprocess output must be contiguous in memory")
    return tensor


def _ensure_rgb(pil_image: Image.Image) -> np.ndarray:
    """Return a contiguous RGB numpy array from ``pil_image``."""

    rgb_image = pil_image.convert("RGB")
    array = np.asarray(rgb_image, dtype=np.uint8)
    if array.ndim != 3 or array.shape[2] != 3:
        raise PipelineError("Input image must be RGB")
    return np.ascontiguousarray(array)


def _run_session(session: BackgroundRemovalSession, tensor: np.ndarray) -> np.ndarray:
    """Execute ``session`` using ``tensor`` while logging inference context."""

    providers = getattr(session, "providers_available", ())
    input_name = getattr(session, "input_name", None)
    LOGGER.info("RUN providers=%s input=%s", providers, input_name)
    if input_name and hasattr(session, "inner"):
        outputs = session.inner.run(None, {input_name: tensor})
    else:
        outputs = session.run(tensor)
    prediction = np.asarray(outputs[0], dtype=np.float32)
    LOGGER.info("RUN output_shape=%s", prediction.shape)
    return prediction


def _sigmoid(array: np.ndarray) -> np.ndarray:
    """Return ``array`` squashed into the ``0`` – ``1`` range using sigmoid."""

    return 1.0 / (1.0 + np.exp(-array))


def _format_unique_values(alpha: np.ndarray) -> str:
    """Return a compact textual summary of unique alpha values in ``alpha``."""

    uniques = np.unique(alpha)
    preview = ", ".join(str(value) for value in uniques[:10])
    if uniques.size > 10:
        preview = f"{preview}, …"
    return f"count={uniques.size} values=[{preview}]"


def _prepare_debug(
    tensor: np.ndarray,
    logits: np.ndarray,
    mask_pre: np.ndarray,
    mask_post: np.ndarray,
) -> ProcessingDebugInfo:
    """Return populated :class:`ProcessingDebugInfo` for the provided tensors."""

    return ProcessingDebugInfo(
        pre_shape=tuple(int(value) for value in tensor.shape),
        pre_min=float(np.min(tensor)),
        pre_max=float(np.max(tensor)),
        logits_shape=tuple(int(value) for value in logits.shape),
        logits_min=float(np.min(logits)),
        logits_max=float(np.max(logits)),
        mask_pre_min=float(np.min(mask_pre)),
        mask_pre_max=float(np.max(mask_pre)),
        mask_post_min=float(np.min(mask_post)),
        mask_post_max=float(np.max(mask_post)),
    )


def _process_loaded_image(
    image: Image.Image,
    *,
    session: BackgroundRemovalSession,
    options: ProcessingOptions,
) -> ProcessingResult:
    """Execute the inference pipeline for ``image`` using ``session``."""

    source_rgba = image.convert("RGBA")
    model_input = _resize_for_mode(source_rgba.convert("RGB"), session.spec.input_size, options.resize_mode)
    rgb_array = _ensure_rgb(model_input)
    tensor = preprocess(rgb_array, session.spec)
    LOGGER.info(
        "PRE shape=%s dtype=%s min=%.6f max=%.6f",
        tensor.shape,
        tensor.dtype,
        float(tensor.min()),
        float(tensor.max()),
    )
    logits = _run_session(session, tensor)
    mask_pre = np.squeeze(_sigmoid(logits)).astype(np.float32)
    LOGGER.info("POST1 min=%.6f max=%.6f", float(mask_pre.min()), float(mask_pre.max()))
    mask_pre_stats = mask_pre.copy()
    mask_min, mask_max = float(mask_pre.min()), float(mask_pre.max())
    if mask_max - mask_min > 1e-6:
        mask_pre = (mask_pre - mask_min) / (mask_max - mask_min)
    original_width, original_height = source_rgba.size
    resized_mask = cv2.resize(mask_pre, (original_width, original_height), interpolation=cv2.INTER_LINEAR)
    resized_mask = np.clip(resized_mask, 0.0, 1.0).astype(np.float32)
    mask_image = Image.fromarray((resized_mask * 255.0).round().astype(np.uint8), mode="L")
    if options.post_process_mask:
        refined = refine_mask(
            mask_image,
            source_rgba,
            alpha_matting=options.alpha_matting,
            foreground_threshold=options.foreground_threshold,
            background_threshold=options.background_threshold,
            erode_size=options.erode_size,
            smoothing=options.smoothing,
            mask_blur=options.mask_blur,
            edge_refinement=options.edge_refinement,
        )
    else:
        refined = mask_image
    refined_array = np.asarray(refined, dtype=np.float32) / 255.0
    refined_array = np.clip(refined_array, 0.0, 1.0).astype(np.float32)
    if options.mask_threshold > 0.0:
        threshold = max(0.0, min(float(options.mask_threshold), 1.0))
        refined_array = np.where(refined_array >= threshold, 1.0, 0.0).astype(np.float32)
    LOGGER.info("POST2 min=%.6f max=%.6f", float(refined_array.min()), float(refined_array.max()))
    if np.any((refined_array < 0.0) | (refined_array > 1.0)):
        raise PipelineError("Mask values must be within [0, 1] before composing alpha")
    mask_to_apply = Image.fromarray((refined_array * 255.0).round().astype(np.uint8), mode="L")
    if options.only_mask or options.cut_out_mode == "mask":
        result_image = Image.merge("RGBA", (mask_to_apply, mask_to_apply, mask_to_apply, mask_to_apply))
    elif options.background_mode == "none":
        result_image = source_rgba.copy()
        if options.cut_out_mode == "bbox":
            result_image, mask_to_apply, refined_array = _crop_to_mask_bounds(
                result_image, mask_to_apply, refined_array
            )
    else:
        result_image = apply_mask_to_image(source_rgba, mask_to_apply, feather_radius=options.feather_radius)
        if options.cut_out_mode == "bbox":
            result_image, mask_to_apply, refined_array = _crop_to_mask_bounds(
                result_image, mask_to_apply, refined_array
            )
        fill_color = options.background_color if options.background_mode == "fill" else None
        result_image = _apply_background(result_image, fill_color)
    alpha_channel = np.asarray(result_image.getchannel("A"), dtype=np.uint8)
    LOGGER.info("OUT alpha_unique=%s", _format_unique_values(alpha_channel))
    debug = _prepare_debug(tensor, logits, mask_pre_stats, refined_array)
    return ProcessingResult(
        image=result_image,
        mask=np.ascontiguousarray(refined_array),
        alpha=np.ascontiguousarray(alpha_channel),
        debug=debug,
    )


def remove_background(
    image: np.ndarray,
    model_key: str,
    *,
    config: Config | None = None,
    feather_radius: int = 3,
    **advanced_options: object,
) -> np.ndarray:
    """Return ``image`` with its background removed using ``model_key``."""

    result = process_image(
        image,
        model_key=model_key,
        config=config,
        feather_radius=feather_radius,
        **advanced_options,
    )
    return np.asarray(result.image)


def process_image(
    image: np.ndarray,
    *,
    model_key: str | None = None,
    config: Config | None = None,
    feather_radius: int = 3,
    providers: Sequence[str] | None = None,
    **advanced_options: object,
) -> ProcessingResult:
    """Return a :class:`ProcessingResult` for the supplied ``image``."""

    active_config = config or load_config()
    resolved_model = model_key or active_config.default_model
    session = _prepare_session(resolved_model, config=active_config, providers=providers)
    options = ProcessingOptions.from_kwargs(feather_radius=feather_radius, **advanced_options)
    try:
        pil_image = load_image_from_array(image)
    except Exception as error:  # pragma: no cover - defensive logging
        raise PipelineError(str(error)) from error
    try:
        LOGGER.info("Starting processing for in-memory image…")
        start = time.perf_counter()
        result = _process_loaded_image(pil_image, session=session, options=options)
        elapsed_ms = (time.perf_counter() - start) * 1000
        LOGGER.info("In-memory image processed successfully ✓ (%.2f ms)", elapsed_ms)
        return result
    except Exception as error:  # pragma: no cover - defensive logging
        message = str(error)
        LOGGER.error("In-memory image failed ✗ — Reason: %s", message)
        raise PipelineError(message) from error


def _process_single_path(
    path: Path,
    *,
    session: BackgroundRemovalSession,
    destination_dir: Path,
    options: ProcessingOptions,
    progress_callback: Callable[[ReportEntry], None] | None = None,
) -> ReportEntry:
    start = time.perf_counter()
    try:
        with Image.open(path) as source:
            pil_image = source.convert("RGBA")
        processing_result = _process_loaded_image(pil_image, session=session, options=options)
        result_image = processing_result.image
        pillow_format, suffix = _infer_output_suffix(options.output_format)
        destination_name = (
            f"{path.stem}.{suffix}"
            if options.preserve_names
            else f"{path.stem}_no_bg.{suffix}"
        )
        destination = destination_dir / destination_name
        if pillow_format != "PNG" and result_image.mode != "RGB":
            result_to_save = result_image.convert("RGB")
        else:
            result_to_save = result_image
        save_image_to_path(result_to_save, destination, format_hint=pillow_format)
        elapsed_ms = (time.perf_counter() - start) * 1000
        LOGGER.info("%s processed successfully ✓", path.name)
        entry = ReportEntry(path_in=path, path_out=destination, success=True, elapsed_ms=elapsed_ms)
        if progress_callback:
            progress_callback(entry)
        return entry
    except UnidentifiedImageError as error:
        elapsed_ms = (time.perf_counter() - start) * 1000
        message = f"Unsupported image format: {error}"
        LOGGER.error("%s failed ✗ — Reason: %s", path.name, message)
        entry = ReportEntry(path_in=path, path_out=None, success=False, elapsed_ms=elapsed_ms, error=message)
        if progress_callback:
            progress_callback(entry)
        return entry
    except Exception as error:  # pragma: no cover - defensive logging
        elapsed_ms = (time.perf_counter() - start) * 1000
        message = str(error)
        LOGGER.exception("Background removal failed for %s", path)
        LOGGER.error("%s failed ✗ — Reason: %s", path.name, message)
        entry = ReportEntry(path_in=path, path_out=None, success=False, elapsed_ms=elapsed_ms, error=message)
        if progress_callback:
            progress_callback(entry)
        return entry


def process_folder(
    input_dir: Path | str,
    output_dir: Path | str | None = None,
    pattern: str = "*",
    *,
    model_key: str | None = None,
    config: Config | None = None,
    recursive: bool = False,
    feather_radius: int = 3,
    progress_callback: Callable[[ReportEntry], None] | None = None,
    **advanced_options: object,
) -> Report:
    """Process images found under ``input_dir`` and return a :class:`Report`."""

    source_dir = Path(input_dir)
    if not source_dir.exists() or not source_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {source_dir}")

    active_config = config or load_config()
    resolved_model = model_key or active_config.default_model
    output_root = resolve_batch_output_dir(source_dir, output_dir)
    session = _prepare_session(resolved_model, config=active_config)
    options = ProcessingOptions.from_kwargs(feather_radius=feather_radius, **advanced_options)
    entries: list[ReportEntry] = []
    candidates = [
        path
        for path in iter_image_files(source_dir, recursive)
        if fnmatch.fnmatch(path.name, pattern)
    ]
    if options.max_workers <= 1:
        for path in candidates:
            LOGGER.info("Starting processing for %s…", path.name)
            entry = _process_single_path(
                path,
                session=session,
                destination_dir=output_root,
                options=options,
                progress_callback=progress_callback,
            )
            entries.append(entry)
    else:
        with ThreadPoolExecutor(max_workers=options.max_workers) as executor:
            future_map = {
                executor.submit(
                    _process_single_path,
                    path,
                    session=session,
                    destination_dir=output_root,
                    options=options,
                    progress_callback=progress_callback,
                ): path
                for path in candidates
            }
            for future in as_completed(future_map):
                entry = future.result()
                entries.append(entry)
    return Report(entries)


__all__ = [
    "Report",
    "ReportEntry",
    "PipelineError",
    "ProcessingDebugInfo",
    "ProcessingResult",
    "process_folder",
    "process_image",
    "remove_background",
    "preprocess",
]
