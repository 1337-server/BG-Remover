"""HTTP routes backing the interactive Flask UI."""
from __future__ import annotations

import io
import logging
import tempfile
import zipfile
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

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
    url_for,
)
from PIL import Image, UnidentifiedImageError

from bgremover_core import Config, load_config, remove_background
from bgremover_core.config import persist_config
from bgremover_core.io.image_io import image_to_numpy
from bgremover_core.models.loader import detect_providers
from bgremover_core.models.specs import MODEL_SPECS, ModelSpec
from bgremover_core.processing.pipeline import PipelineError, process_folder

from .services import ResultRecord, ResultStore, ensure_filename, total_size

LOGGER = logging.getLogger(__name__)

webui = Blueprint(
    "webui",
    __name__,
    static_folder="static",
    template_folder="templates",
)

__all__ = ["webui"]


OPTION_HELP: dict[str, str] = {
    "removal_model": "Neural network architecture used for background removal.",
    "model_precision": "Hint the ONNX runtime precision. Auto selects the best available backend.",
    "provider": "Preferred hardware backend. Auto selects GPU if available, otherwise CPU.",
    "model_dir": "Directory on the server where downloaded model weights are stored.",
    "feather_radius": "Feather the mask edges for smoother blending. Range: 0–50.",
    "alpha_matting": "Enable refined matting for detailed edges such as hair or fur.",
    "alpha_matting_foreground_threshold": "Minimum intensity considered foreground. Range: 0–255. Default: 240.",
    "alpha_matting_background_threshold": "Maximum intensity considered background. Range: 0–255. Default: 10.",
    "alpha_matting_erode_size": "Number of pixels to erode the mask. Range: 0–30. Default: 10.",
    "post_process_mask": "Apply smoothing and refinement heuristics to the raw alpha mask.",
    "mask_blur": "Gaussian blur radius (in pixels) applied to the mask. Set to 0 to disable.",
    "mask_threshold": "Clamp mask values below this ratio to zero. Range: 0.0–1.0.",
    "only_mask": "Export only the alpha mask instead of a composited image.",
    "cut_out_mode": "Control the final crop: keep the full object, output the mask, or crop to the bounding box.",
    "background_mode": "Choose how the background is composed: keep the original, clear it, or fill with a colour.",
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
        config = load_config()
        current_app.config["BGR_CONFIG"] = config
    return config


def _get_executor() -> ThreadPoolExecutor:
    return current_app.extensions["executor"]


def _get_store() -> ResultStore:
    return current_app.extensions["result_store"]


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
    return store.add_record(record)


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


@webui.route("/batch", methods=["POST"])
def batch_process() -> Response:
    config = _get_config()
    options = _options_from_request(config)
    active_config = _prepare_config(config, options["model_dir"], options["provider_choice"])

    archive = request.files.get("archive")
    if not archive or not archive.filename:
        return jsonify({"error": "Upload a ZIP archive containing images."}), 400

    try:
        output_dir = _resolve_output_directory(_get_store(), options["output_subdir"])
        batch_id = uuid4().hex
        batch_output = output_dir / f"batch_{batch_id}"
        batch_output.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            archive.stream.seek(0)
            with zipfile.ZipFile(archive.stream) as zip_file:
                zip_file.extractall(temp_path)

            executor = _get_executor()
            future = executor.submit(
                process_folder,
                temp_path,
                output_dir=batch_output,
                model_key=options["model_key"],
                config=active_config,
                feather_radius=options["feather_radius"],
                **_pipeline_kwargs(options),
            )
            report = future.result()

        generated_files = [entry.path_out for entry in report.entries if entry.success and entry.path_out]
        if not generated_files:
            return jsonify({"error": "No images were processed successfully."}), 422

        zip_name = ensure_filename(f"batch_{batch_id}.zip")
        zip_path = batch_output / zip_name
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as result_zip:
            for file_path in generated_files:
                if file_path is None:
                    continue
                result_zip.write(file_path, arcname=file_path.name)

        store = _get_store()
        record = ResultRecord(
            identifier=batch_id,
            original_name=archive.filename,
            result_name=zip_name,
            path=zip_path,
            mime_type="application/zip",
            created_at=datetime.now(UTC),
            options=_serialise_options({"batch": True, **options}),
            size_bytes=zip_path.stat().st_size,
        )
        store.add_record(record)

        response_payload = {
            "batch_id": batch_id,
            "zip_name": zip_name,
            "download_url": f"/download/{batch_id}",
            "entries": report.to_rows(),
            "summary": {
                "total": report.total,
                "success": report.successes,
                "failed": report.failures,
                "size_bytes": total_size(generated_files),
            },
        }
        if options["remember"] and options["model_dir"]:
            persist_config(active_config)
        return jsonify(response_payload)
    except zipfile.BadZipFile:
        return jsonify({"error": "The uploaded file is not a valid ZIP archive."}), 400
    except FileNotFoundError as error:
        return jsonify({"error": str(error)}), 404
    except PipelineError as error:
        LOGGER.error("Batch processing failed: %s", error)
        return jsonify({"error": str(error)}), 422


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

