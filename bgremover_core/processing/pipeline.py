"""High level background removal pipelines shared across runtimes."""
from __future__ import annotations

import fnmatch
import logging
import time
from collections.abc import Sequence
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
) -> np.ndarray:
    """Return ``image`` with its background removed using ``model_key``."""

    active_config = config or load_config()
    session = _prepare_session(model_key or active_config.default_model, config=active_config)
    try:
        pil_image = load_image_from_array(image)
        tensor = normalise_image(pil_image, session.spec)
        LOGGER.info("Starting processing for in-memory image…")
        start = time.perf_counter()
        outputs = session.run(tensor)
        mask = compute_mask_image(outputs, pil_image.size)
        result_image = apply_mask_to_image(pil_image, mask, feather_radius=feather_radius)
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
    feather_radius: int,
) -> ReportEntry:
    start = time.perf_counter()
    try:
        with Image.open(path) as source:
            pil_image = source.convert("RGBA")
            tensor = normalise_image(pil_image, session.spec)
        outputs = session.run(tensor)
        mask = compute_mask_image(outputs, pil_image.size)
        result_image = apply_mask_to_image(pil_image, mask, feather_radius=feather_radius)
        destination = destination_dir / f"{path.stem}.png"
        save_image_to_path(result_image, destination)
        elapsed_ms = (time.perf_counter() - start) * 1000
        LOGGER.info("%s processed successfully ✓", path.name)
        return ReportEntry(path_in=path, path_out=destination, success=True, elapsed_ms=elapsed_ms)
    except UnidentifiedImageError as error:
        elapsed_ms = (time.perf_counter() - start) * 1000
        message = f"Unsupported image format: {error}"
        LOGGER.error("%s failed ✗ — Reason: %s", path.name, message)
        return ReportEntry(path_in=path, path_out=None, success=False, elapsed_ms=elapsed_ms, error=message)
    except Exception as error:  # pragma: no cover - defensive logging
        elapsed_ms = (time.perf_counter() - start) * 1000
        message = str(error)
        LOGGER.exception("Background removal failed for %s", path)
        LOGGER.error("%s failed ✗ — Reason: %s", path.name, message)
        return ReportEntry(path_in=path, path_out=None, success=False, elapsed_ms=elapsed_ms, error=message)


def process_folder(
    input_dir: Path | str,
    output_dir: Path | str | None = None,
    pattern: str = "*",
    *,
    model_key: str | None = None,
    config: Config | None = None,
    recursive: bool = False,
    feather_radius: int = 3,
) -> Report:
    """Process images found under ``input_dir`` and return a :class:`Report`."""

    source_dir = Path(input_dir)
    if not source_dir.exists() or not source_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {source_dir}")

    active_config = config or load_config()
    resolved_model = model_key or active_config.default_model
    output_root = resolve_batch_output_dir(source_dir, output_dir)
    session = _prepare_session(resolved_model, config=active_config)

    entries: list[ReportEntry] = []
    for path in iter_image_files(source_dir, recursive):
        if not fnmatch.fnmatch(path.name, pattern):
            continue
        LOGGER.info("Starting processing for %s…", path.name)
        entry = _process_single_path(
            path,
            session=session,
            destination_dir=output_root,
            feather_radius=feather_radius,
        )
        entries.append(entry)
    return Report(entries)


__all__ = ["Report", "ReportEntry", "PipelineError", "process_folder", "remove_background"]
