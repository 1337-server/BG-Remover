"""High level background removal pipelines shared across runtimes."""
from __future__ import annotations

import fnmatch
import logging
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from bgremover_core.config import Config, load_config
from bgremover_core.io.image_io import load_image_from_array, save_image_to_path
from bgremover_core.io.paths import resolve_batch_output_dir
from bgremover_core.models.loader import BackgroundRemovalSession, detect_providers, get_session
from bgremover_core.processing.utils import (
    apply_mask_to_image,
    compute_mask_image,
    iter_image_files,
    normalise_image,
    refine_mask,
)

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
    """Container describing advanced image processing preferences."""

    resize_mode: str = "stretch"
    feather_radius: int = 3
    alpha_matting: bool = False
    foreground_threshold: int = 240
    background_threshold: int = 10
    erode_size: int = 10
    smoothing: float = 0.0
    edge_refinement: bool = False
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
        resolved_feather = _coerce_int(feather_radius, default=3, minimum=0, maximum=50)
        return cls(
            resize_mode=resize_mode,
            feather_radius=resolved_feather,
            alpha_matting=alpha_matting,
            foreground_threshold=foreground_threshold,
            background_threshold=background_threshold,
            erode_size=erode_size,
            smoothing=smoothing,
            edge_refinement=edge_refinement,
            background_color=background_color,
            output_format=output_format,
            preserve_names=preserve_names,
            max_workers=max_workers,
        )


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


def _apply_background(image: Image.Image, color: tuple[int, int, int] | None) -> Image.Image:
    """Return ``image`` composited onto ``color`` when provided."""

    if not color:
        return image
    background = Image.new("RGBA", image.size, (*color, 255))
    return Image.alpha_composite(background, image)


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


def remove_background(
    image: np.ndarray,
    model_key: str,
    *,
    config: Config | None = None,
    feather_radius: int = 3,
    **_advanced_options: object,
) -> np.ndarray:
    """Return ``image`` with its background removed using ``model_key``.

    The ``_advanced_options`` parameter accepts keyword arguments controlling
    resize behaviour, alpha matting, mask smoothing, edge refinement, background
    fills, and other advanced tuning exposed by the GUI and Flask runtimes.  All
    parameters are optional and default to backwards compatible behaviour when
    omitted.
    """

    active_config = config or load_config()
    session = _prepare_session(model_key or active_config.default_model, config=active_config)
    options = ProcessingOptions.from_kwargs(feather_radius=feather_radius, **_advanced_options)
    try:
        pil_image = load_image_from_array(image)
        tensor = normalise_image(pil_image, session.spec, resize_mode=options.resize_mode)
        LOGGER.info("Starting processing for in-memory image…")
        start = time.perf_counter()
        outputs = session.run(tensor)
        mask = compute_mask_image(outputs, pil_image.size)
        mask = refine_mask(
            mask,
            pil_image,
            alpha_matting=options.alpha_matting,
            foreground_threshold=options.foreground_threshold,
            background_threshold=options.background_threshold,
            erode_size=options.erode_size,
            smoothing=options.smoothing,
            edge_refinement=options.edge_refinement,
        )
        result_image = apply_mask_to_image(pil_image, mask, feather_radius=options.feather_radius)
        result_image = _apply_background(result_image, options.background_color)
        elapsed_ms = (time.perf_counter() - start) * 1000
        LOGGER.info("In-memory image processed successfully ✓ (%.2f ms)", elapsed_ms)
        return np.asarray(result_image)
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
            tensor = normalise_image(pil_image, session.spec, resize_mode=options.resize_mode)
        outputs = session.run(tensor)
        mask = compute_mask_image(outputs, pil_image.size)
        mask = refine_mask(
            mask,
            pil_image,
            alpha_matting=options.alpha_matting,
            foreground_threshold=options.foreground_threshold,
            background_threshold=options.background_threshold,
            erode_size=options.erode_size,
            smoothing=options.smoothing,
            edge_refinement=options.edge_refinement,
        )
        result_image = apply_mask_to_image(pil_image, mask, feather_radius=options.feather_radius)
        result_image = _apply_background(result_image, options.background_color)
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


__all__ = ["Report", "ReportEntry", "PipelineError", "process_folder", "remove_background"]
