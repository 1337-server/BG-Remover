"""HTTP routes backing the interactive Flask UI."""
from __future__ import annotations

import hashlib
import json
import logging
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from flask import (
    Blueprint,
    Response,
    current_app,
    jsonify,
    render_template,
    request,
    send_file,
    stream_with_context,
    url_for,
)
from PIL import Image, UnidentifiedImageError

from bgremover_core import Config, load_config, remove_background
from bgremover_core.config import persist_config
from bgremover_core.io.image_io import image_to_numpy, save_image_to_path
from bgremover_core.io.paths import ensure_output_directory, resolve_batch_output_dir
from bgremover_core.models.loader import (
    ModelUnavailableError,
    detect_providers,
    get_session,
)
from bgremover_core.models.specs import MODEL_SPECS, ModelSpec
from bgremover_core.processing.pipeline import PipelineError, process_folder
from bgremover_core.processing.utils import iter_image_files

from .services import BatchEvent, BatchJob, BatchJobManager, ResultRecord, ResultStore

LOGGER = logging.getLogger(__name__)

webui = Blueprint(
    "webui",
    __name__,
    static_folder="static",
    template_folder="templates",
)

__all__ = ["webui"]


# ---------------------------------------------------------------------------
# Constants and metadata
# ---------------------------------------------------------------------------
_ALLOWED_BACKGROUND_MODES = {"none", "fill", "clear"}
_ALLOWED_RESIZE_MODES = {"stretch", "keep-aspect", "crop", "auto"}
_ALLOWED_OUTPUT_FORMATS = {"PNG", "JPEG", "WEBP"}
_MODEL_DISPLAY_NAMES = {
    "isnet-general-use": "IS-Net (General)",
    "u2net": "U2-Net",
    "u2net_human_seg": "U2-Net (Human)",
    "isnet-anime": "IS-Net (Anime)",
    "briaai/RMBG-2.0": "BriaAI RMBG 2.0",
    "matting-by-generation": "Matting by Generation",
    "sam_segmentation_model": "SAM Segmentation",
    "sam_vit_b_01ec64_encoder": "SAM Encoder",
    "sam_vit_b_01ec64_decoder": "SAM Decoder",
}
_ADVANCED_OPTION_HELP = {
    "feather_radius": {
        "name": "Feather radius",
        "description": "Softens mask edges using a Gaussian blur applied to the alpha channel.",
        "min": 0,
        "max": 50,
        "default": 3,
        "recommended": "3",
    },
    "resize_mode": {
        "name": "Resize mode",
        "description": "Controls how images are resized before inference (stretch, keep-aspect, crop, auto).",
        "min": None,
        "max": None,
        "default": "stretch",
        "recommended": "stretch",
    },
    "alpha_matting": {
        "name": "Alpha matting",
        "description": "Refines edges by combining mask predictions with luminance thresholds.",
        "min": 0,
        "max": 1,
        "default": False,
        "recommended": "Enable for hair or fur",
    },
    "foreground_threshold": {
        "name": "Foreground threshold",
        "description": "Pixels brighter than this threshold are forced fully opaque during matting.",
        "min": 0,
        "max": 255,
        "default": 240,
        "recommended": "240",
    },
    "background_threshold": {
        "name": "Background threshold",
        "description": "Pixels darker than this threshold become transparent when matting is active.",
        "min": 0,
        "max": 255,
        "default": 10,
        "recommended": "10",
    },
    "erode_size": {
        "name": "Erode size",
        "description": "Applies morphological erosion/dilation when matting to remove halos.",
        "min": 0,
        "max": 30,
        "default": 10,
        "recommended": "10",
    },
    "smoothing": {
        "name": "Legacy smoothing",
        "description": "0–1 ratio converted into a blur radius when mask blur is zero.",
        "min": 0.0,
        "max": 1.0,
        "default": 0.0,
        "recommended": "0",
    },
    "mask_blur": {
        "name": "Mask blur radius",
        "description": "Direct Gaussian blur radius in pixels applied to the alpha mask.",
        "min": 0.0,
        "max": 25.0,
        "default": 0.0,
        "recommended": "0-5",
    },
    "edge_refinement": {
        "name": "Edge refinement",
        "description": "Applies a sharpening pass to the alpha mask to enhance edges.",
        "min": 0,
        "max": 1,
        "default": False,
        "recommended": "Off",
    },
    "background_color": {
        "name": "Background color",
        "description": "Hex colour used when filling the background instead of keeping transparency.",
        "min": None,
        "max": None,
        "default": "#ffffff",
        "recommended": "Brand colour",
    },
    "transparent": {
        "name": "Keep transparency",
        "description": "Leave background transparent when enabled; otherwise fill using the selected colour.",
        "min": 0,
        "max": 1,
        "default": True,
        "recommended": "On",
    },
    "preserve_names": {
        "name": "Preserve filenames",
        "description": "Write output files using the original stem instead of unique identifiers.",
        "min": 0,
        "max": 1,
        "default": False,
        "recommended": "Off",
    },
    "output_format": {
        "name": "Output format",
        "description": "Image format for saved results. PNG preserves transparency.",
        "min": None,
        "max": None,
        "default": "PNG",
        "recommended": "PNG",
    },
    "max_workers": {
        "name": "Parallel workers",
        "description": "Number of worker threads for batch processing.",
        "min": 1,
        "max": 16,
        "default": 1,
        "recommended": "1-4",
    },
}


class ApiError(RuntimeError):
    """Raised when an API request fails validation or processing."""

    def __init__(self, status_code: int, code: str, message: str, *, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.details = details or {}


# ---------------------------------------------------------------------------
# Dependency accessors
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


def _get_batch_manager() -> BatchJobManager:
    return current_app.extensions["batch_manager"]


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _default_single_output_dir() -> Path:
    """Return the default output directory for single-image processing."""

    return ensure_output_directory(Path.cwd() / "output")


def _parse_bool(value: str | None, *, default: bool = False) -> bool:
    if value is None:
        return default
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    return default


def _parse_int(
    value: Any,
    *,
    field: str,
    minimum: int | None = None,
    maximum: int | None = None,
    default: int,
) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise ApiError(
            400,
            "VALIDATION_ERROR",
            f"{field} must be an integer",
            details={"field": field},
        ) from error
    if minimum is not None and result < minimum:
        raise ApiError(
            400,
            "VALIDATION_ERROR",
            f"{field} must be >= {minimum}",
            details={"field": field, "min": minimum},
        )
    if maximum is not None and result > maximum:
        raise ApiError(
            400,
            "VALIDATION_ERROR",
            f"{field} must be <= {maximum}",
            details={"field": field, "max": maximum},
        )
    return result


def _parse_float(
    value: Any,
    *,
    field: str,
    minimum: float | None = None,
    maximum: float | None = None,
    default: float,
) -> float:
    if value is None:
        return default
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ApiError(
            400,
            "VALIDATION_ERROR",
            f"{field} must be a number",
            details={"field": field},
        ) from error
    if minimum is not None and result < minimum:
        raise ApiError(
            400,
            "VALIDATION_ERROR",
            f"{field} must be >= {minimum}",
            details={"field": field, "min": minimum},
        )
    if maximum is not None and result > maximum:
        raise ApiError(
            400,
            "VALIDATION_ERROR",
            f"{field} must be <= {maximum}",
            details={"field": field, "max": maximum},
        )
    return result


def _parse_color(value: str | None, *, field: str) -> tuple[int, int, int] | None:
    if not value:
        return None
    candidate = value.strip().lstrip("#")
    if len(candidate) != 6:
        raise ApiError(400, "VALIDATION_ERROR", f"{field} must be a hex colour", details={"field": field})
    try:
        red = int(candidate[0:2], 16)
        green = int(candidate[2:4], 16)
        blue = int(candidate[4:6], 16)
    except ValueError as error:
        raise ApiError(
            400,
            "VALIDATION_ERROR",
            f"{field} must be a hex colour",
            details={"field": field},
        ) from error
    return red, green, blue


def _prepare_config(base: Config, model_dir: str | None, provider_choice: str | None) -> Config:
    config = base
    if model_dir:
        config = config.with_updates(model_dir=Path(model_dir).expanduser())
    if provider_choice == "gpu":
        config = config.with_updates(provider_hints=("CUDAExecutionProvider", "CPUExecutionProvider"))
    elif provider_choice == "cpu":
        config = config.with_updates(provider_hints=("CPUExecutionProvider",))
    config.resolved_model_dir()
    return config


def _validate_model_key(model_key: str) -> ModelSpec:
    spec = MODEL_SPECS.get(model_key)
    if spec is None:
        raise ApiError(
            400,
            "VALIDATION_ERROR",
            f"Unknown model '{model_key}'",
            details={"field": "removal_model"},
        )
    return spec


def _decode_advanced_payload(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ApiError(400, "VALIDATION_ERROR", "Advanced options payload is not valid JSON") from error
    if not isinstance(data, dict):
        raise ApiError(400, "VALIDATION_ERROR", "Advanced options must be an object")
    return data


def _collect_advanced_options(payload: dict[str, Any]) -> dict[str, Any]:
    """Return sanitised processing options for downstream consumers."""

    serialised: dict[str, Any] = {}

    feather_radius = _parse_int(
        payload.get("feather_radius", 3),
        field="feather_radius",
        minimum=0,
        maximum=50,
        default=3,
    )
    serialised["feather_radius"] = feather_radius

    resize_mode = str(payload.get("resize_mode", "stretch")).strip().lower() or "stretch"
    if resize_mode not in _ALLOWED_RESIZE_MODES:
        raise ApiError(
            400,
            "VALIDATION_ERROR",
            "resize_mode is invalid",
            details={"field": "resize_mode", "allowed": sorted(_ALLOWED_RESIZE_MODES)},
        )
    serialised["resize_mode"] = resize_mode

    alpha_matting = bool(payload.get("alpha_matting", False))
    serialised["alpha_matting"] = alpha_matting

    foreground_threshold = _parse_int(
        payload.get("foreground_threshold", 240),
        field="foreground_threshold",
        minimum=0,
        maximum=255,
        default=240,
    )
    serialised["foreground_threshold"] = foreground_threshold

    background_threshold = _parse_int(
        payload.get("background_threshold", 10),
        field="background_threshold",
        minimum=0,
        maximum=255,
        default=10,
    )
    serialised["background_threshold"] = background_threshold

    erode_size = _parse_int(
        payload.get("erode_size", 10),
        field="erode_size",
        minimum=0,
        maximum=30,
        default=10,
    )
    serialised["erode_size"] = erode_size

    smoothing = _parse_float(
        payload.get("smoothing", 0.0),
        field="smoothing",
        minimum=0.0,
        maximum=1.0,
        default=0.0,
    )
    serialised["smoothing"] = smoothing

    mask_blur = _parse_float(
        payload.get("mask_blur", 0.0),
        field="mask_blur",
        minimum=0.0,
        maximum=25.0,
        default=0.0,
    )
    serialised["mask_blur"] = mask_blur

    edge_refinement = bool(payload.get("edge_refinement", False))
    serialised["edge_refinement"] = edge_refinement

    transparent = bool(payload.get("transparent", True))
    serialised["transparent"] = transparent

    background_color = _parse_color(payload.get("background_color"), field="background_color")
    if background_color:
        serialised["background_color"] = (
            f"#{background_color[0]:02x}{background_color[1]:02x}{background_color[2]:02x}"
        )
    else:
        serialised["background_color"] = None

    preserve_names = bool(payload.get("preserve_names", False))
    serialised["preserve_names"] = preserve_names

    output_format = str(payload.get("output_format", "PNG")).upper()
    if output_format not in _ALLOWED_OUTPUT_FORMATS:
        raise ApiError(
            400,
            "VALIDATION_ERROR",
            "output_format is invalid",
            details={"field": "output_format", "allowed": sorted(_ALLOWED_OUTPUT_FORMATS)},
        )
    serialised["output_format"] = output_format

    max_workers = _parse_int(
        payload.get("max_workers", 1),
        field="max_workers",
        minimum=1,
        maximum=16,
        default=1,
    )
    serialised["max_workers"] = max_workers

    serialised["remember_preferences"] = bool(payload.get("remember_preferences", True))
    serialised["provider"] = str(payload.get("provider", "auto"))
    serialised["model_dir"] = str(payload.get("model_dir") or "")
    serialised["output_directory"] = str(payload.get("output_directory") or "")

    return serialised


def _infer_output_suffix(format_name: str) -> tuple[str, str]:
    match format_name.upper():
        case "JPEG" | "JPG":
            return "JPEG", "jpg"
        case "WEBP":
            return "WEBP", "webp"
        case _:
            return "PNG", "png"


def _compute_settings_hash(model_key: str, options: dict[str, Any]) -> str:
    sortable = json.dumps({"model_key": model_key, **options}, sort_keys=True, default=str)
    return hashlib.sha1(sortable.encode("utf-8")).hexdigest()


def _serialise_model_specs() -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for _key, spec in sorted(MODEL_SPECS.items()):
        payload.append(
            {
                "key": spec.key,
                "display_name": _MODEL_DISPLAY_NAMES.get(spec.key, spec.key),
                "defaults": {
                    "input_size": list(spec.input_size),
                    "mean": list(spec.mean),
                    "std": list(spec.std),
                    "normalisation_scale": spec.normalisation_scale,
                },
                "constraints": {
                    "checksum_md5": spec.checksum_md5,
                    "download": (
                        spec.url
                        or (
                            f"{spec.huggingface_repo}/{spec.huggingface_filename}"
                            if spec.huggingface_repo
                            else None
                        )
                    ),
                },
            }
        )
    return payload


def _safe_extract(zip_file: zipfile.ZipFile, destination: Path) -> None:
    destination = destination.resolve()
    for member in zip_file.infolist():
        member_path = destination / member.filename
        resolved = member_path.resolve()
        if not str(resolved).startswith(str(destination)):
            raise ApiError(
                400,
                "VALIDATION_ERROR",
                "Archive contains unsafe paths",
                details={"member": member.filename},
            )
    zip_file.extractall(destination)


def _format_pipeline_kwargs(serialised: dict[str, Any]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "resize_mode": serialised["resize_mode"],
        "alpha_matting": serialised["alpha_matting"],
        "foreground_threshold": serialised["foreground_threshold"],
        "background_threshold": serialised["background_threshold"],
        "erode_size": serialised["erode_size"],
        "smoothing": serialised["smoothing"],
        "mask_blur": serialised["mask_blur"],
        "edge_refinement": serialised["edge_refinement"],
        "max_workers": serialised["max_workers"],
    }
    if not serialised["transparent"] and serialised.get("background_color"):
        color = serialised["background_color"]
        kwargs["background_color"] = (
            int(color[1:3], 16),
            int(color[3:5], 16),
            int(color[5:7], 16),
        )
    return kwargs


def _normalise_background(serialised: dict[str, Any], mode: str) -> None:
    mode = mode.lower()
    if mode not in _ALLOWED_BACKGROUND_MODES:
        raise ApiError(400, "VALIDATION_ERROR", f"Invalid background mode: {mode}")
    if mode == "clear":
        serialised["transparent"] = True
        serialised["background_color"] = None
    elif mode == "fill":
        serialised["transparent"] = False
        serialised.setdefault("background_color", "#ffffff")


def _persist_if_requested(active_config: Config, serialised: dict[str, Any]) -> None:
    if serialised.get("remember_preferences") and serialised.get("model_dir"):
        persist_config(active_config)


def _create_record(
    upload_name: str,
    result_image: Image.Image,
    *,
    output_dir: Path,
    serialised: dict[str, Any],
    model_key: str,
) -> tuple[ResultRecord, Path]:
    identifier = uuid4().hex
    stem = Path(upload_name or "image").stem or "image"
    pillow_format, suffix = _infer_output_suffix(serialised["output_format"])
    if serialised.get("preserve_names"):
        filename = f"{stem}.{suffix}"
    else:
        filename = f"{stem}_no_bg.{suffix}"
    output_path = output_dir / filename
    if pillow_format != "PNG" and result_image.mode != "RGB":
        save_image = result_image.convert("RGB")
    else:
        save_image = result_image
    saved_path = save_image_to_path(save_image, output_path, format_hint=pillow_format)
    record = ResultRecord(
        identifier=identifier,
        original_name=upload_name or "upload",
        result_name=filename,
        path=saved_path,
        mime_type=f"image/{suffix}",
        created_at=datetime.now(UTC),
        options={"model_key": model_key, **serialised},
        size_bytes=saved_path.stat().st_size,
    )
    return record, saved_path


def _error_response(error: ApiError) -> tuple[Response, int]:
    payload = {"status": "error", "code": error.code, "message": str(error), "details": error.details}
    LOGGER.error("API error %s: %s", error.code, error.details or error)
    return jsonify(payload), error.status_code


def _model_error(code: str, message: str, *, model_dir: Path, provider: str | None) -> ApiError:
    details = {
        "model_dir": str(model_dir),
        "provider": provider,
        "search_paths": [str(model_dir)],
    }
    return ApiError(400, code, message, details=details)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@webui.route("/", methods=["GET"])
def index() -> Response:
    config = _get_config()
    providers = detect_providers(config.provider_hints)
    badge_label = "GPU" if providers and providers[0].lower().startswith("cuda") else "CPU"
    context = {
        "badge_label": badge_label,
        "providers": providers,
        "default_model": config.default_model,
        "default_output_dir": str(_default_single_output_dir()),
        "current_year": datetime.now(UTC).year,
    }
    return render_template("index.html", **context)


@webui.route("/api/models", methods=["GET"])
def api_models() -> Response:
    config = _get_config()
    payload = {
        "models": _serialise_model_specs(),
        "default_model": config.default_model,
    }
    return jsonify(payload)


@webui.route("/api/help/options", methods=["GET"])
def api_help_options() -> Response:
    return jsonify({"options": _ADVANCED_OPTION_HELP})


@webui.route("/api/process/single", methods=["POST"])
def api_process_single() -> tuple[Response, int]:
    try:
        form = request.form
        upload = request.files.get("file")
        if upload is None or not upload.filename:
            raise ApiError(400, "VALIDATION_ERROR", "No file uploaded", details={"field": "file"})

        model_key = form.get("removal_model") or ""
        spec = _validate_model_key(model_key)

        config = _get_config()
        advanced_payload = _decode_advanced_payload(form.get("advanced"))
        serialised = _collect_advanced_options(advanced_payload)

        provider_choice = serialised.get("provider") or form.get("provider")
        active_config = _prepare_config(config, serialised.get("model_dir"), provider_choice)

        background_mode = form.get("background_mode", "none")
        _normalise_background(serialised, background_mode)

        preview = _parse_bool(form.get("preview"), default=False)

        output_dir_raw = form.get("output_dir") or serialised.get("output_directory") or ""
        if output_dir_raw:
            output_dir = ensure_output_directory(Path(output_dir_raw).expanduser())
        else:
            output_dir = _default_single_output_dir()

        try:
            get_session(spec.key, model_dir=active_config.model_dir, providers=active_config.provider_hints)
        except ModelUnavailableError as error:
            message = str(error)
            if "Failed to load" in message or "Failed to download" in message:
                raise _model_error(
                    "MODEL_NOT_FOUND",
                    message,
                    model_dir=active_config.model_dir,
                    provider=provider_choice,
                ) from error
            raise _model_error(
                "BACKEND_INIT_FAIL",
                message,
                model_dir=active_config.model_dir,
                provider=provider_choice,
            ) from error

        upload.stream.seek(0)
        try:
            image = Image.open(upload.stream).convert("RGBA")
        except UnidentifiedImageError as error:
            raise ApiError(
                400,
                "VALIDATION_ERROR",
                "Unsupported image format",
                details={"field": "file"},
            ) from error

        image_array = image_to_numpy(image)
        pipeline_parameters = _format_pipeline_kwargs(serialised)

        start = time.perf_counter()
        result_array = remove_background(
            image_array,
            spec.key,
            config=active_config,
            feather_radius=serialised["feather_radius"],
            **pipeline_parameters,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        result_image = Image.fromarray(result_array)

        record, saved_path = _create_record(
            upload.filename or "upload.png",
            result_image,
            output_dir=output_dir,
            serialised=serialised,
            model_key=spec.key,
        )
        _get_store().add_record(record)
        _persist_if_requested(active_config, serialised)

        payload = {
            "status": "ok",
            "output_url": url_for("webui.result_preview", identifier=record.identifier),
            "download_url": url_for("webui.download", identifier=record.identifier),
            "output_path": str(saved_path),
            "timings": {"total_ms": elapsed_ms},
            "model_used": spec.key,
            "settings_hash": _compute_settings_hash(spec.key, serialised),
            "preview": preview,
        }
        return jsonify(payload), 200
    except ApiError as error:
        return _error_response(error)
    except PipelineError as error:
        api_error = ApiError(422, "PIPELINE_ERROR", str(error))
        return _error_response(api_error)
    except Exception as error:  # noqa: BLE001
        LOGGER.exception("Unexpected failure during single-image processing")
        api_error = ApiError(500, "INTERNAL_ERROR", str(error))
        return _error_response(api_error)


def _enumerate_inputs(input_dir: Path) -> list[Path]:
    return list(iter_image_files(input_dir, recursive=True))


def _run_batch_job(
    job: BatchJob,
    *,
    input_dir: Path,
    output_dir: Path,
    model_key: str,
    config: Config,
    serialised: dict[str, Any],
) -> None:
    try:
        candidates = _enumerate_inputs(input_dir)
        job.mark_started(total=len(candidates))
        job.publish(BatchEvent("started", {"total": len(candidates), "output_dir": str(output_dir)}))

        pipeline_kwargs = _format_pipeline_kwargs(serialised)

        def progress(entry):
            data = {
                "filename": entry.path_in.name,
                "elapsed_ms": entry.elapsed_ms,
            }
            if entry.success and entry.path_out:
                data["out_path"] = str(entry.path_out)
                job.record_success(data)
                job.publish(BatchEvent("item_success", data))
            else:
                data["error"] = entry.error or "Unknown error"
                job.record_failure(data)
                job.publish(BatchEvent("item_error", data))

        process_folder(
            input_dir,
            output_dir,
            "*",
            model_key=model_key,
            config=config,
            feather_radius=serialised["feather_radius"],
            progress_callback=progress,
            **pipeline_kwargs,
        )

        summary = job.build_summary()
        job.mark_finished()
        job.publish(
            BatchEvent(
                "finished",
                {
                    "status": job.status,
                    "success": summary.successes,
                    "failed": summary.failures,
                    "output_dir": str(output_dir),
                },
            )
        )
    except Exception as error:  # noqa: BLE001
        LOGGER.exception("Batch job %s failed", job.job_id)
        job.mark_finished(error=str(error))
        job.publish(
            BatchEvent(
                "finished",
                {
                    "status": "failed",
                    "error": str(error),
                    "output_dir": str(output_dir),
                },
            )
        )
    finally:
        job.close_stream()


@webui.route("/api/process/batch", methods=["POST"])
def api_process_batch() -> tuple[Response, int]:
    try:
        form = request.form
        archive = request.files.get("file")
        if archive is None or not archive.filename:
            raise ApiError(400, "VALIDATION_ERROR", "Upload a ZIP archive", details={"field": "file"})

        model_key = form.get("removal_model") or ""
        spec = _validate_model_key(model_key)

        advanced_payload = _decode_advanced_payload(form.get("advanced"))
        serialised = _collect_advanced_options(advanced_payload)
        background_mode = form.get("background_mode", "none")
        _normalise_background(serialised, background_mode)
        config = _get_config()
        provider_choice = serialised.get("provider") or form.get("provider")
        active_config = _prepare_config(config, serialised.get("model_dir"), provider_choice)

        try:
            get_session(spec.key, model_dir=active_config.model_dir, providers=active_config.provider_hints)
        except ModelUnavailableError as error:
            message = str(error)
            if "Failed to load" in message or "Failed to download" in message:
                raise _model_error(
                    "MODEL_NOT_FOUND",
                    message,
                    model_dir=active_config.model_dir,
                    provider=provider_choice,
                ) from error
            raise _model_error(
                "BACKEND_INIT_FAIL",
                message,
                model_dir=active_config.model_dir,
                provider=provider_choice,
            ) from error

        store = _get_store()
        job_id = uuid4().hex
        job_root = store.base_dir / "batches" / job_id
        input_root = job_root / "input"
        output_root_default = job_root / "output"
        input_root.mkdir(parents=True, exist_ok=True)

        archive.stream.seek(0)
        try:
            with zipfile.ZipFile(archive.stream) as zip_file:
                _safe_extract(zip_file, input_root)
        except zipfile.BadZipFile as error:
            raise ApiError(400, "VALIDATION_ERROR", "Uploaded file is not a valid ZIP archive") from error

        output_dir_raw = form.get("output_dir") or serialised.get("output_directory") or ""
        if output_dir_raw:
            output_dir = ensure_output_directory(Path(output_dir_raw).expanduser())
        else:
            output_dir = resolve_batch_output_dir(input_root, output_root_default)

        job = BatchJob(job_id, output_dir)
        manager = _get_batch_manager()
        manager.register(job)

        executor = _get_executor()
        executor.submit(
            _run_batch_job,
            job,
            input_dir=input_root,
            output_dir=output_dir,
            model_key=spec.key,
            config=active_config,
            serialised=serialised,
        )

        _persist_if_requested(active_config, serialised)

        return jsonify({"status": "accepted", "job_id": job_id}), 202
    except ApiError as error:
        return _error_response(error)
    except Exception as error:  # noqa: BLE001
        LOGGER.exception("Unexpected error while scheduling batch job")
        api_error = ApiError(500, "INTERNAL_ERROR", str(error))
        return _error_response(api_error)


@webui.route("/api/stream/batch/<string:job_id>", methods=["GET"])
def api_stream_batch(job_id: str) -> Response:
    manager = _get_batch_manager()
    job = manager.get(job_id)
    if job is None:
        return jsonify({"status": "error", "code": "UNKNOWN_JOB", "message": "Job not found"}), 404

    def event_generator():
        for event in job.event_stream():
            yield f"event: {event.event}\ndata: {json.dumps(event.data)}\n\n"

    response = Response(stream_with_context(event_generator()), mimetype="text/event-stream")
    response.headers["Cache-Control"] = "no-cache"
    response.headers["X-Accel-Buffering"] = "no"
    return response


@webui.route("/api/jobs/<string:job_id>/summary", methods=["GET"])
def api_job_summary(job_id: str) -> tuple[Response, int]:
    manager = _get_batch_manager()
    summary = manager.summary(job_id)
    if summary is None:
        return jsonify({"status": "error", "code": "UNKNOWN_JOB", "message": "Job not found"}), 404
    payload = {"status": "ok", "job": summary.as_dict()}
    return jsonify(payload), 200


@webui.route("/result/<string:identifier>", methods=["GET"])
def result_preview(identifier: str) -> Response:
    store = _get_store()
    record = store.get(identifier)
    if record is None or not record.path.exists():
        return jsonify({"status": "error", "code": "NOT_FOUND", "message": "Result not found"}), 404
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
        return jsonify({"status": "error", "code": "NOT_FOUND", "message": "Result not found"}), 404
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
