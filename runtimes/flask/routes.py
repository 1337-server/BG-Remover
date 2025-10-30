"""HTTP routes backing the interactive Flask UI."""
from __future__ import annotations

import gc
import io
import json
import logging
import shutil
import tempfile
import time
import zipfile
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from PIL import Image, UnidentifiedImageError
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

from bgremover_core import Config, load_config, remove_background
from bgremover_core.config import persist_config
from bgremover_core.io.image_io import image_to_numpy, save_image_to_path
from bgremover_core.models.loader import detect_providers
from bgremover_core.models.specs import MODEL_SPECS, ModelSpec
from bgremover_core.paths import CONFIG_FILE
from bgremover_core.processing import pipeline as pipeline_module
from bgremover_core.processing.pipeline import PipelineError
from flask import (
    Blueprint,
    Flask,
    Response,
    current_app,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    stream_with_context,
    url_for,
)

from .services import BatchJob, BatchJobManager, ResultRecord, ResultStore, ensure_filename

LOGGER = logging.getLogger(__name__)

webui = Blueprint(
    "webui",
    __name__,
    static_folder="static",
    template_folder="templates",
)

__all__ = ["webui"]


ALLOWED_BATCH_EXTENSIONS: frozenset[str] = frozenset({".png", ".jpg", ".jpeg", ".webp"})


@dataclass(slots=True)
class BatchCandidate:
    """Describe a queued file for batch processing."""

    source: Path
    relative_path: Path


@dataclass(slots=True)
class BatchSource:
    """Describe the prepared inputs for a batch job."""

    root: Path
    candidates: list[BatchCandidate]
    label: str
    cleanup: Callable[[], None] | None = None


OPTION_HELP: dict[str, str] = {
    "removal_model": "Neural network architecture used for background removal.",
    "model_precision": "Hint the ONNX runtime precision. Auto selects the best available backend.",
    "provider": "Preferred hardware backend. Auto selects GPU if available, otherwise CPU.",
    "model_dir": "Directory on the server where downloaded model weights are stored.",
    "feather_radius": "Feather the mask edges for smoother blending. Range: 0–50.",
    "alpha_matting": "Enable refined matting for detailed edges such as hair or fur.",
    "alpha_matting_foreground_threshold": (
        "Minimum intensity considered foreground. Range: 0–255. Default: 240."
    ),
    "alpha_matting_background_threshold": (
        "Maximum intensity considered background. Range: 0–255. Default: 10."
    ),
    "alpha_matting_erode_size": "Number of pixels to erode the mask. Range: 0–30. Default: 10.",
    "post_process_mask": "Apply smoothing and refinement heuristics to the raw alpha mask.",
    "mask_blur": "Gaussian blur radius (in pixels) applied to the mask. Set to 0 to disable.",
    "mask_threshold": "Clamp mask values below this ratio to zero. Range: 0.0–1.0.",
    "only_mask": "Export only the alpha mask instead of a composited image.",
    "cut_out_mode": (
        "Control the final crop: keep the full object, output the mask, or crop to the "
        "bounding box."
    ),
    "background_mode": (
        "Choose how the background is composed: keep the original, clear it, or fill "
        "with a colour."
    ),
    "background_color": "Colour used when filling the background. Applies when background mode is Fill.",
    "output_format": "Select the file format for exported results.",
    "output_dir": "Optional subdirectory under the cache where processed files are written.",
    "preserve_names": "Reuse original filenames instead of appending a unique suffix.",
    "remember_preferences": "Persist model directory and provider preferences on the server.",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _get_config() -> Config:
    config = current_app.config.get("BGR_CONFIG")
    if config is None:
        config = load_config(config_path=CONFIG_FILE)
        current_app.config["BGR_CONFIG"] = config
    return config


def _get_executor() -> ThreadPoolExecutor:
    return current_app.extensions["executor"]


def _get_store() -> ResultStore:
    return current_app.extensions["result_store"]


def _get_batch_manager() -> BatchJobManager:
    """Return the shared :class:`BatchJobManager` instance."""

    return current_app.extensions["batch_manager"]


def _provider_badge(providers: list[str]) -> tuple[str, list[str]]:
    if not providers:
        return "CPU", ["CPUExecutionProvider"]
    primary = providers[0]
    if primary.lower().startswith("cuda"):
        return "GPU", providers
    return "CPU", providers


def _hex_to_rgb(value: str | None) -> tuple[int, int, int] | None:
    if not value:
        return None
    value = value.strip().lstrip("#")
    if len(value) != 6:
        return None
    try:
        red = int(value[0:2], 16)
        green = int(value[2:4], 16)
        blue = int(value[4:6], 16)
    except ValueError:
        return None
    return red, green, blue


def _parse_bool(value: str | None, *, default: bool = False) -> bool:
    if value is None:
        return default
    lowered = value.lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    return default


def _prepare_config(base: Config, model_dir: str | None, provider_choice: str | None) -> Config:
    config = base
    if model_dir:
        model_path = Path(model_dir).expanduser()
        config = config.with_updates(model_dir=model_path)
    if provider_choice == "gpu":
        config = config.with_updates(provider_hints=("CUDAExecutionProvider", "CPUExecutionProvider"))
    elif provider_choice == "cpu":
        config = config.with_updates(provider_hints=("CPUExecutionProvider",))
    config.resolved_model_dir()
    return config


def _options_from_request(config: Config) -> dict[str, Any]:
    """Return sanitised processing options parsed from the active request."""

    form = request.form
    removal_model = (form.get("removal_model") or "").strip()
    model_key = removal_model or form.get("model_key") or config.default_model
    try:
        feather_radius = int(form.get("feather_radius", 3))
    except (TypeError, ValueError):
        feather_radius = 3
    feather_radius = max(0, min(50, feather_radius))

    background_color = _hex_to_rgb(form.get("background_color"))
    transparent = _parse_bool(form.get("transparent"), default=True)
    background_mode_raw = (form.get("background_mode") or "").strip().lower()
    if background_mode_raw not in {"none", "fill", "clear"}:
        background_mode = "clear" if transparent else ("fill" if background_color else "clear")
    else:
        background_mode = background_mode_raw
    if background_mode == "fill" and background_color is None:
        background_color = (255, 255, 255)
    if background_mode == "fill":
        transparent = False
    elif background_mode == "clear":
        transparent = True
    output_format = (form.get("output_format") or "PNG").upper()
    provider_choice = form.get("provider")
    preserve_names = _parse_bool(form.get("preserve_names"))
    raw_output_dir = (form.get("output_dir") or form.get("output_directory") or "").strip()
    output_subdir = ensure_filename(raw_output_dir) if raw_output_dir else ""
    model_dir = form.get("model_dir")
    remember = _parse_bool(form.get("remember_preferences"), default=True)
    model_precision_raw = (form.get("model_precision") or "auto").strip().lower()
    model_precision = model_precision_raw if model_precision_raw in {"auto", "fp32", "fp16"} else "auto"

    alpha_matting = _parse_bool(form.get("alpha_matting"))
    try:
        alpha_fg = int(form.get("alpha_matting_foreground_threshold", 240))
    except (TypeError, ValueError):
        alpha_fg = 240
    alpha_fg = max(0, min(255, alpha_fg))
    try:
        alpha_bg = int(form.get("alpha_matting_background_threshold", 10))
    except (TypeError, ValueError):
        alpha_bg = 10
    alpha_bg = max(0, min(255, alpha_bg))
    try:
        alpha_erode = int(form.get("alpha_matting_erode_size", 10))
    except (TypeError, ValueError):
        alpha_erode = 10
    alpha_erode = max(0, min(30, alpha_erode))
    try:
        mask_blur_value = float(form.get("mask_blur", 0))
    except (TypeError, ValueError):
        mask_blur_value = 0.0
    mask_blur = float(max(0.0, min(25.0, mask_blur_value)))
    post_process_mask = _parse_bool(form.get("post_process_mask"), default=True)
    only_mask = _parse_bool(form.get("only_mask"))
    try:
        mask_threshold_value = float(form.get("mask_threshold", 0.0))
    except (TypeError, ValueError):
        mask_threshold_value = 0.0
    mask_threshold = max(0.0, min(1.0, mask_threshold_value))
    cut_out_mode_raw = (form.get("cut_out_mode") or "").strip().lower()
    cut_out_mode = cut_out_mode_raw if cut_out_mode_raw in {"object", "mask", "bbox"} else "object"

    options = {
        "model_key": model_key,
        "removal_model": model_key,
        "feather_radius": feather_radius,
        "background_color": background_color,
        "transparent": transparent,
        "output_format": output_format,
        "preserve_names": preserve_names,
        "output_subdir": output_subdir,
        "output_dir": output_subdir,
        "provider_choice": provider_choice,
        "model_dir": model_dir,
        "remember": remember,
        "alpha_matting": alpha_matting,
        "mask_blur": mask_blur,
        "alpha_matting_foreground_threshold": alpha_fg,
        "alpha_matting_background_threshold": alpha_bg,
        "alpha_matting_erode_size": alpha_erode,
        "post_process_mask": post_process_mask,
        "only_mask": only_mask,
        "mask_threshold": mask_threshold,
        "cut_out_mode": cut_out_mode,
        "background_mode": background_mode,
        "model_precision": model_precision,
    }
    return options


def _pipeline_kwargs(options: dict[str, Any]) -> dict[str, Any]:
    """Return keyword arguments forwarded to the processing pipeline."""

    forwarded: dict[str, Any] = {
        "alpha_matting": bool(options.get("alpha_matting", False)),
        "mask_blur": float(options.get("mask_blur", 0.0)),
        "alpha_matting_foreground_threshold": int(options.get("alpha_matting_foreground_threshold", 240)),
        "alpha_matting_background_threshold": int(options.get("alpha_matting_background_threshold", 10)),
        "alpha_matting_erode_size": int(options.get("alpha_matting_erode_size", 10)),
        "post_process_mask": bool(options.get("post_process_mask", True)),
        "only_mask": bool(options.get("only_mask", False)),
        "mask_threshold": float(options.get("mask_threshold", 0.0)),
        "cut_out_mode": options.get("cut_out_mode", "object"),
        "background_mode": options.get("background_mode", "clear"),
    }
    background_mode = options.get("background_mode")
    if (
        (not options.get("transparent", True) or background_mode == "fill")
        and options.get("background_color")
    ):
        forwarded["background_color"] = options["background_color"]
    return forwarded


def _resolve_output_directory(store: ResultStore, subdir: str) -> Path:
    base_dir = store.base_dir
    if not subdir:
        base_dir.mkdir(parents=True, exist_ok=True)
        return base_dir
    destination = base_dir / subdir
    destination.mkdir(parents=True, exist_ok=True)
    return destination


def _normalise_relative_path(name: str) -> Path:
    """Return a sanitised relative path derived from ``name``."""

    candidate = Path(name)
    safe_parts: list[str] = []
    for part in candidate.parts:
        if part in {"", ".", ".."}:
            continue
        cleaned = secure_filename(part)
        if cleaned:
            safe_parts.append(cleaned)
    if not safe_parts:
        fallback = secure_filename(candidate.name) or "upload"
        return Path(fallback)
    return Path(*safe_parts)


def _ensure_unique_path(destination: Path) -> Path:
    """Return ``destination`` or a unique variant if it already exists."""

    if not destination.exists():
        return destination
    counter = 1
    stem = destination.stem
    suffix = destination.suffix
    parent = destination.parent
    while True:
        candidate = parent / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def _collect_candidates(source_dir: Path, recursive: bool) -> list[BatchCandidate]:
    """Return the list of image files ready for processing."""

    candidates: list[BatchCandidate] = []
    if recursive:
        iterator = source_dir.rglob("*")
    else:
        iterator = source_dir.iterdir()
    for path in iterator:
        if not path.is_file():
            continue
        if path.suffix.lower() not in ALLOWED_BATCH_EXTENSIONS:
            continue
        try:
            relative = path.relative_to(source_dir)
        except ValueError:
            continue
        if not recursive and len(relative.parts) > 1:
            continue
        candidates.append(BatchCandidate(source=path, relative_path=relative))
    candidates.sort(key=lambda item: str(item.relative_path).lower())
    return candidates


def _persist_folder_upload(files: Iterable[FileStorage]) -> Path:
    """Persist uploaded folder contents to a temporary directory."""

    temp_dir = Path(tempfile.mkdtemp(prefix="bgr-folder-"))
    for storage in files:
        if not storage or not storage.filename:
            continue
        relative = _normalise_relative_path(storage.filename)
        if not relative.parts:
            continue
        destination = temp_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        storage.stream.seek(0)
        storage.save(str(destination))
    return temp_dir


def _safe_extract_zip(file_storage: FileStorage) -> Path:
    """Extract ``file_storage`` into a temporary directory with sanitisation."""

    temp_dir = Path(tempfile.mkdtemp(prefix="bgr-zip-"))
    file_storage.stream.seek(0)
    with zipfile.ZipFile(file_storage.stream) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            relative = _normalise_relative_path(info.filename)
            if not relative.suffix:
                continue
            if relative.suffix.lower() not in ALLOWED_BATCH_EXTENSIONS:
                continue
            destination = temp_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info, "r") as source, destination.open("wb") as target:
                shutil.copyfileobj(source, target)
    return temp_dir


def _prepare_batch_source(
    *,
    folder_files: list[FileStorage],
    folder_path_raw: str | None,
    zip_file: FileStorage | None,
    recursive: bool,
) -> BatchSource:
    """Return a :class:`BatchSource` describing the batch inputs."""

    cleanup: Callable[[], None] | None = None
    if folder_files:
        root = _persist_folder_upload(folder_files)

        def _cleanup() -> None:
            """Remove the temporary folder upload directory."""

            shutil.rmtree(root, ignore_errors=True)

        cleanup = _cleanup
        candidates = _collect_candidates(root, recursive)
        label = folder_path_raw or Path(root).name
        return BatchSource(root=root, candidates=candidates, label=label, cleanup=cleanup)
    if folder_path_raw:
        root = Path(folder_path_raw).expanduser()
        if not root.exists() or not root.is_dir():
            raise FileNotFoundError(f"Input directory does not exist: {root}")
        candidates = _collect_candidates(root, recursive)
        return BatchSource(root=root, candidates=candidates, label=root.name)
    if zip_file and zip_file.filename:
        root = _safe_extract_zip(zip_file)

        def _cleanup_zip() -> None:
            """Remove the temporary extracted archive directory."""

            shutil.rmtree(root, ignore_errors=True)

        cleanup = _cleanup_zip
        candidates = _collect_candidates(root, recursive)
        label = Path(zip_file.filename).stem
        return BatchSource(root=root, candidates=candidates, label=label, cleanup=cleanup)
    raise ValueError("No batch inputs provided")


def _execute_batch_job(
    app: Flask,
    job: BatchJob,
    *,
    batch_source: BatchSource,
    output_dir: Path,
    options: dict[str, Any],
    config: Config,
    recursive: bool,
    remember_preferences: bool,
) -> None:
    """Process ``batch_source`` and emit progress events via ``job``."""

    app_logger = LOGGER.getChild(f"batch.{job.identifier}")
    generated_paths: list[Path] = []
    status = "finished"
    message: str | None = None
    processing_kwargs = _pipeline_kwargs(options)
    processing_kwargs.update(
        {
            "output_format": options.get("output_format", "PNG"),
            "preserve_names": options.get("preserve_names", False),
        }
    )
    options_obj = pipeline_module.ProcessingOptions.from_kwargs(
        feather_radius=options.get("feather_radius", 3),
        **processing_kwargs,
    )

    with app.app_context():
        store = _get_store()
        try:
            session = pipeline_module._prepare_session(
                options["model_key"],
                config=config,
                providers=config.provider_hints,
            )
        except Exception as error:  # pragma: no cover - defensive
            message = str(error)
            status = "error"
            job.emit(
                "finished",
                {
                    "status": status,
                    "message": message,
                    "summary": {
                        "total": job.total_items,
                        "success": 0,
                        "failed": job.total_items,
                        "size_bytes": 0,
                    },
                },
            )
            job.mark_finished()
            return

        try:
            for candidate in batch_source.candidates:
                if job.cancelled():
                    status = "cancelled"
                    message = "Batch cancelled by client"
                    break
                start = time.perf_counter()
                relative_text = str(candidate.relative_path)
                try:
                    with Image.open(candidate.source) as source:
                        pil_image = source.convert("RGBA")
                    result = pipeline_module._process_loaded_image(
                        pil_image, session=session, options=options_obj
                    )
                    pillow_format, suffix = pipeline_module._infer_output_suffix(
                        options_obj.output_format
                    )
                    if recursive and len(candidate.relative_path.parts) > 1:
                        destination_parent = output_dir / candidate.relative_path.parent
                    else:
                        destination_parent = output_dir
                    destination_parent.mkdir(parents=True, exist_ok=True)
                    if options_obj.preserve_names:
                        destination_name = f"{candidate.relative_path.stem}.{suffix}"
                    else:
                        destination_name = f"{candidate.relative_path.stem}_no_bg.{suffix}"
                    destination = destination_parent / destination_name
                    destination = _ensure_unique_path(destination)
                    result_image = result.image
                    if pillow_format != "PNG" and result_image.mode != "RGB":
                        result_image = result_image.convert("RGB")
                    save_image_to_path(result_image, destination, format_hint=pillow_format)
                    del result_image
                    del result
                    del pil_image
                    generated_paths.append(destination)
                    job.success_count += 1
                    if destination.exists():
                        job.size_bytes += destination.stat().st_size
                    elapsed_ms = (time.perf_counter() - start) * 1000.0
                    job.emit(
                        "item_success",
                        {
                            "input": relative_text,
                            "output": str(destination.relative_to(output_dir)),
                            "elapsed_ms": round(elapsed_ms, 2),
                        },
                    )
                except UnidentifiedImageError as error:
                    elapsed_ms = (time.perf_counter() - start) * 1000.0
                    job.failure_count += 1
                    error_message = f"Unsupported image format: {error}"
                    app_logger.warning("%s", error_message)
                    job.emit(
                        "item_error",
                        {
                            "input": relative_text,
                            "elapsed_ms": round(elapsed_ms, 2),
                            "error": error_message,
                        },
                    )
                except PermissionError as error:
                    elapsed_ms = (time.perf_counter() - start) * 1000.0
                    job.failure_count += 1
                    error_message = f"Permission error: {error}"
                    app_logger.warning("%s", error_message)
                    job.emit(
                        "item_error",
                        {
                            "input": relative_text,
                            "elapsed_ms": round(elapsed_ms, 2),
                            "error": error_message,
                        },
                    )
                except Exception as error:  # pragma: no cover - defensive
                    elapsed_ms = (time.perf_counter() - start) * 1000.0
                    job.failure_count += 1
                    error_message = str(error)
                    app_logger.exception("Processing failed for %s", candidate.source)
                    job.emit(
                        "item_error",
                        {
                            "input": relative_text,
                            "elapsed_ms": round(elapsed_ms, 2),
                            "error": error_message,
                        },
                    )
                finally:
                    gc.collect()
        finally:
            del session
            gc.collect()

        download_url: str | None = None
        try:
            if generated_paths:
                zip_name = ensure_filename(f"batch_{job.identifier}.zip")
                zip_path = output_dir / zip_name
                with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                    for file_path in generated_paths:
                        arcname = str(file_path.relative_to(output_dir))
                        archive.write(file_path, arcname=arcname)
                job.zip_path = zip_path
                record_options = _serialise_options({"batch": True, **options})
                record_options["output_directory"] = str(output_dir)
                record = ResultRecord(
                    identifier=job.identifier,
                    original_name=batch_source.label,
                    result_name=zip_name,
                    path=zip_path,
                    mime_type="application/zip",
                    created_at=datetime.now(UTC),
                    options=record_options,
                    size_bytes=zip_path.stat().st_size,
                )
                store.add_record(record)
                download_url = url_for("webui.download", identifier=job.identifier)

            if remember_preferences and options.get("model_dir"):
                persist_config(config)

            summary = {
                "total": job.total_items,
                "success": job.success_count,
                "failed": job.failure_count,
                "size_bytes": job.size_bytes,
            }
            payload: dict[str, Any] = {
                "status": status,
                "summary": summary,
                "download_url": download_url,
                "output_dir": str(output_dir),
            }
            if job.zip_path:
                payload["zip_name"] = job.zip_path.name
            if message:
                payload["message"] = message
            job.emit("finished", payload)
        finally:
            job.mark_finished()


def _event_stream(job: BatchJob, manager: BatchJobManager):
    """Yield server-sent events produced by ``job``."""

    try:
        while True:
            event = job.next_event()
            if event is None:
                if job.finished():
                    break
                yield ": keep-alive\n\n"
                continue
            chunk = f"event: {event.name}\ndata: {json.dumps(event.payload)}\n\n"
            yield chunk
            if event.name == "finished":
                break
    except GeneratorExit:  # pragma: no cover - triggered by client disconnect
        job.cancel()
        raise
    finally:
        if job.finished():
            manager.discard(job.identifier)


def _determine_output_meta(format_name: str) -> tuple[str, str]:
    match format_name.upper():
        case "JPEG" | "JPG":
            return "image/jpeg", "jpg"
        case "WEBP":
            return "image/webp", "webp"
        case _:
            return "image/png", "png"


def _prepare_result_name(original: str, suffix: str, preserve: bool, identifier: str) -> str:
    stem = Path(original).stem or "output"
    if preserve:
        candidate = f"{stem}.{suffix}"
    else:
        candidate = f"{stem}_{identifier[:8]}.{suffix}"
    return ensure_filename(candidate)


def _format_float_triplet(values: tuple[float, float, float]) -> str:
    """Return ``values`` formatted for human friendly display."""

    formatted = []
    for value in values:
        text = f"{value:.3f}".rstrip("0").rstrip(".")
        formatted.append(text or "0")
    return ", ".join(formatted)


def _spec_weight_source(spec: ModelSpec) -> str | None:
    """Return a description of where ``spec`` downloads its weights from."""

    if spec.url:
        return spec.url
    if spec.huggingface_repo and spec.huggingface_filename:
        revision = spec.huggingface_revision or "main"
        if revision and revision != "main":
            return f"{spec.huggingface_repo}@{revision}/{spec.huggingface_filename}"
        return f"{spec.huggingface_repo}/{spec.huggingface_filename}"
    if spec.local_filename:
        return spec.local_filename
    return None


def _serialise_model_specs() -> dict[str, dict[str, str]]:
    """Return frontend friendly metadata extracted from :data:`MODEL_SPECS`."""

    serialised: dict[str, dict[str, str]] = {}
    for key in sorted(MODEL_SPECS):
        spec = MODEL_SPECS[key]
        details: dict[str, str] = {
            "Input size": f"{spec.input_size[0]} × {spec.input_size[1]}",
            "Normalisation mean": _format_float_triplet(spec.mean),
            "Normalisation std": _format_float_triplet(spec.std),
            "Scale": f"{spec.normalisation_scale:g}",
        }
        weight_source = _spec_weight_source(spec)
        if weight_source:
            details["Weight source"] = weight_source
        if spec.checksum_md5:
            details["Checksum (MD5)"] = spec.checksum_md5
        serialised[key] = details
    return serialised


def _template_context(active_page: str) -> dict[str, Any]:
    """Return common template context shared across UI pages."""

    config = _get_config()
    providers = detect_providers(config.provider_hints)
    badge_label, provider_list = _provider_badge(providers)
    return {
        "badge_label": badge_label,
        "providers": provider_list,
        "model_options": sorted(MODEL_SPECS.keys()),
        "default_model": config.default_model,
        "model_dir": str(config.model_dir),
        "model_specs": _serialise_model_specs(),
        "current_year": datetime.now(UTC).year,
        "active_page": active_page,
    }


def _serialise_options(options: dict[str, Any]) -> dict[str, Any]:
    serialisable: dict[str, Any] = {}
    for key, value in options.items():
        if key in {"model_dir", "remember", "output_subdir"}:
            continue
        if key == "background_color" and value:
            serialisable[key] = f"#{value[0]:02x}{value[1]:02x}{value[2]:02x}"
        else:
            serialisable[key] = value
    return serialisable


def _process_image(upload, *, config: Config, store: ResultStore, options: dict[str, Any]) -> ResultRecord:
    identifier = uuid4().hex
    upload.stream.seek(0)
    try:
        image = Image.open(upload.stream).convert("RGBA")
    except UnidentifiedImageError as error:
        raise PipelineError(f"Unsupported image format: {error}") from error

    array = image_to_numpy(image)
    processing_kwargs = _pipeline_kwargs(options)
    result_array = remove_background(
        array,
        options["model_key"],
        config=config,
        feather_radius=options["feather_radius"],
        **processing_kwargs,
    )
    result_image = Image.fromarray(result_array)
    image.close()
    del result_array
    del array

    mime_type, suffix = _determine_output_meta(options["output_format"])
    if mime_type != "image/png":
        result_image = result_image.convert("RGB")

    result_name = _prepare_result_name(
        upload.filename or "output.png",
        suffix,
        options["preserve_names"],
        identifier,
    )
    output_dir = _resolve_output_directory(store, options["output_subdir"])
    output_path = output_dir / f"{identifier}_{result_name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    buffer = io.BytesIO()
    result_image.save(buffer, format=options["output_format"].upper())
    buffer.seek(0)
    output_path.write_bytes(buffer.getvalue())
    buffer.close()
    del buffer
    del result_image
    size_bytes = output_path.stat().st_size

    record = ResultRecord(
        identifier=identifier,
        original_name=upload.filename or "upload",
        result_name=result_name,
        path=output_path,
        mime_type=mime_type,
        created_at=datetime.now(UTC),
        options=_serialise_options(options),
        size_bytes=size_bytes,
    )
    stored = store.add_record(record)
    gc.collect()
    return stored


def _process_single_request(
    app: Flask,
    files: Iterable,
    *,
    config: Config,
    options: dict[str, Any],
) -> list[dict[str, Any]]:
    """Process each uploaded file within the provided Flask application context."""

    with app.app_context():
        store = _get_store()
        results: list[dict[str, Any]] = []
        for upload in files:
            record = _process_image(upload, config=config, store=store, options=options)
            results.append(record.as_dict(include_preview=True))
            gc.collect()
        return results


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@webui.route("/api/help/options", methods=["GET"])
def options_help() -> Response:
    """Return helper text describing the available form options."""

    return jsonify({"options": OPTION_HELP})


@webui.route("/", methods=["GET"])
def index() -> Response:
    return redirect(url_for("webui.single"))


@webui.route("/single", methods=["GET"])
def single() -> Response:
    context = _template_context("single")
    return render_template("index.html", **context)


@webui.route("/batch", methods=["GET"])
def batch_page() -> Response:
    context = _template_context("batch")
    return render_template("batch.html", **context)


@webui.route("/process", methods=["POST"])
def process_images() -> Response:
    config = _get_config()
    options = _options_from_request(config)
    active_config = _prepare_config(config, options["model_dir"], options["provider_choice"])

    uploads = request.files.getlist("images")
    if not uploads:
        return jsonify({"error": "No images uploaded"}), 400

    executor = _get_executor()
    app_obj = current_app._get_current_object()
    try:
        future = executor.submit(
            _process_single_request,
            app_obj,
            uploads,
            config=active_config,
            options=options,
        )
        results = future.result()
    except PipelineError as error:
        LOGGER.error("Processing failed: %s", error)
        return jsonify({"error": str(error)}), 422
    except Exception as error:  # noqa: BLE001
        LOGGER.exception("Unexpected processing failure")
        return jsonify({"status": "error", "message": str(error)}), 500

    if options["remember"] and options["model_dir"]:
        persist_config(active_config)

    return jsonify({"results": results})


@webui.route("/api/process/batch", methods=["POST"])
def api_process_batch() -> Response:
    config = _get_config()
    options = _options_from_request(config)
    active_config = _prepare_config(config, options["model_dir"], options["provider_choice"])

    recursive = _parse_bool(request.form.get("recursive"))
    folder_files = [file for file in request.files.getlist("folder_files") if file and file.filename]
    zip_file = request.files.get("zip_file") or request.files.get("archive")
    folder_path_raw = (request.form.get("folder_path") or "").strip() or None

    try:
        batch_source = _prepare_batch_source(
            folder_files=folder_files,
            folder_path_raw=folder_path_raw,
            zip_file=zip_file,
            recursive=recursive,
        )
    except FileNotFoundError as error:
        return jsonify({"error": str(error)}), 404
    except zipfile.BadZipFile:
        return jsonify({"error": "The uploaded file is not a valid ZIP archive."}), 400
    except ValueError:
        return jsonify({"error": "Provide a folder path, folder upload, or ZIP archive."}), 400

    if not batch_source.candidates:
        if batch_source.cleanup:
            batch_source.cleanup()
        return jsonify({"error": "No valid images found in folder."}), 400

    store = _get_store()
    base_output = _resolve_output_directory(store, options["output_subdir"])
    batch_id = uuid4().hex
    batch_output = base_output / f"batch_{batch_id}"
    batch_output.mkdir(parents=True, exist_ok=True)

    job = BatchJob(batch_id, len(batch_source.candidates), output_dir=batch_output)
    manager = _get_batch_manager()
    manager.register(job)
    if batch_source.cleanup:
        job.add_cleanup(batch_source.cleanup)

    job.emit(
        "started",
        {
            "total": job.total_items,
            "label": batch_source.label,
            "recursive": recursive,
        },
    )

    executor = _get_executor()
    app_obj = current_app._get_current_object()
    future = executor.submit(
        _execute_batch_job,
        app_obj,
        job,
        batch_source=batch_source,
        output_dir=batch_output,
        options=options,
        config=active_config,
        recursive=recursive,
        remember_preferences=options.get("remember", True),
    )
    job.attach_future(future)
    return jsonify({"status": "accepted", "job_id": batch_id, "total": job.total_items})


@webui.route("/api/process/batch/<string:job_id>/stream", methods=["GET"])
def api_process_batch_stream(job_id: str) -> Response:
    manager = _get_batch_manager()
    job = manager.get(job_id)
    if job is None:
        return jsonify({"error": "Batch job not found"}), 404

    response = Response(stream_with_context(_event_stream(job, manager)), mimetype="text/event-stream")
    response.headers["Cache-Control"] = "no-cache"
    response.headers["X-Accel-Buffering"] = "no"
    response.headers["Connection"] = "keep-alive"
    return response


@webui.route("/api/process/batch/<string:job_id>/files", methods=["GET"])
def api_process_batch_files(job_id: str) -> Response:
    store = _get_store()
    record = store.get(job_id)
    if record is None:
        return jsonify({"error": "Batch job not found"}), 404
    output_dir_raw = record.options.get("output_directory") if isinstance(record.options, dict) else None
    if not output_dir_raw:
        return jsonify({"error": "Output directory not recorded for this batch."}), 404
    output_dir = Path(output_dir_raw)
    try:
        if not output_dir.exists() or not output_dir.is_dir():
            raise FileNotFoundError
        base_dir = store.base_dir
        if hasattr(output_dir, "is_relative_to"):
            if not output_dir.is_relative_to(base_dir):
                return jsonify({"error": "Access to this directory is not permitted."}), 403
        else:  # pragma: no cover - Python <3.9 fallback not expected
            output_dir.resolve().relative_to(base_dir.resolve())
    except (FileNotFoundError, ValueError):
        return jsonify({"error": "Output directory not found"}), 404

    files: list[str] = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_file():
            relative = path.relative_to(output_dir)
            files.append(str(relative).replace("\\", "/"))
    return jsonify({"output_dir": str(output_dir), "files": files})


@webui.route("/result/<string:identifier>", methods=["GET"])
def result_preview(identifier: str) -> Response:
    store = _get_store()
    record = store.get(identifier)
    if record is None or not record.path.exists():
        return jsonify({"error": "Result not found"}), 404
    return send_file(
        record.path,
        mimetype=record.mime_type,
        download_name=record.result_name,
        as_attachment=False,
    )


@webui.route("/download/<string:identifier>", methods=["GET"])
def download(identifier: str) -> Response:
    store = _get_store()
    record = store.get(identifier)
    if record is None or not record.path.exists():
        return jsonify({"error": "Result not found"}), 404
    return send_file(
        record.path,
        mimetype=record.mime_type,
        download_name=record.result_name,
        as_attachment=True,
    )


@webui.route("/history", methods=["GET"])
def history() -> Response:
    store = _get_store()
    records = [record.as_dict(include_preview=False) for record in store.list_records()]
    return jsonify({"history": records})

