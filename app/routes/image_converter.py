"""Flask routes for interactive background removal."""
from __future__ import annotations

import json
import threading
import shutil
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Tuple

from app.services import runtime_compat

if TYPE_CHECKING:  # pragma: no cover - hints only
    from flask import Flask

from flask import (
    Blueprint,
    Response,
    abort,
    after_this_request,
    current_app,
    jsonify,
    render_template,
    request,
    send_file,
    url_for,
)
from werkzeug.utils import secure_filename

from app.services.bg_remove import (
    OUTPUT_FORMATS,
    RemovalResult,
    create_session,
    encode_result_image,
    ensure_global_session,
    get_accelerator_status,
    get_mime_type_for_path,
    get_output_format_spec,
    get_runtime_payload,
    remove_bg_file,
    remove_bg_folder,
    OUTPUT_FORMATS,
)

image_converter_bp = Blueprint("image_converter", __name__)

_ZIP_REGISTRY: Dict[str, Path] = {}


@dataclass
class RegistryItem:
    """Metadata describing an entry stored in a download registry."""

    path: Path
    mimetype: str | None = None
    delete_after_read: bool = False
    download_name: str | None = None


_FILE_REGISTRY: Dict[str, RegistryItem] = {}
_PREVIEW_REGISTRY: Dict[str, RegistryItem] = {}
_SESSION_CACHE: Dict[Tuple[str, str, int], "SessionContext"] = {}
_SESSION_CACHE_LOCK = threading.Lock()
FORMAT_OPTIONS = [
    {"key": spec.key, "label": spec.label, "extension": spec.extension}
    for spec in OUTPUT_FORMATS
]
DEFAULT_OUTPUT_FORMAT_KEY = get_output_format_spec(None).key
REMOVAL_MODEL_OPTIONS = [
    {"key": "general", "label": "General Model", "model_name": "isnet-general-use"},
    {"key": "human", "label": "Human Model", "model_name": "u2net_human_seg"},
    {"key": "object", "label": "Object Model", "model_name": "u2net"},
    {"key": "anime", "label": "Anime / Illustration Model", "model_name": "isnet-anime"},
]
_REMOVAL_MODEL_LOOKUP: Dict[str, str] = {
    option["key"]: option["model_name"] for option in REMOVAL_MODEL_OPTIONS
}
DEFAULT_REMOVAL_MODEL_KEY = REMOVAL_MODEL_OPTIONS[0]["key"]
HARDWARE_ACCELERATOR_OPTIONS = [
    {"key": "auto", "label": "Auto (recommended)"},
    {"key": "gpu", "label": "GPU"},
    {"key": "cpu", "label": "CPU"},
]
DEFAULT_HARDWARE_ACCELERATOR_KEY = HARDWARE_ACCELERATOR_OPTIONS[0]["key"]
DEFAULT_SINGLE_OPTIONS: Dict[str, Any] = {
    "am_foreground": 240,
    "am_background": 10,
    "am_erode": 10,
    "colorkey_tolerance": 14,
    "feather_radius": 3,
    "removal_model": DEFAULT_REMOVAL_MODEL_KEY,
    "hardware_accelerator": DEFAULT_HARDWARE_ACCELERATOR_KEY,
}
DEFAULT_CHECKBOX_OPTIONS: Dict[str, bool] = {
    # UI toggles that have sensible disabled defaults.
    "alpha_matting": False,
    "recursive": False,
    "zip": False,
}


def _parse_int(value: str | None, default: int) -> int:
    """Safely parse integers from incoming form values."""

    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _is_truthy(value: str | None) -> bool:
    """Return ``True`` for common representations of truthy checkbox values."""

    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "on", "yes"}


def _collect_single_options(form: Mapping[str, str], defaults: Dict[str, int]) -> Dict[str, Any]:
    """Extract reusable single-image processing options from the request."""

    return {
        "alpha_matting": _is_truthy(form.get("alpha_matting")),
        "am_foreground": _parse_int(form.get("am_foreground"), defaults["am_foreground"]),
        "am_background": _parse_int(form.get("am_background"), defaults["am_background"]),
        "am_erode": _parse_int(form.get("am_erode"), defaults["am_erode"]),
        "colorkey_tolerance": _parse_int(
            form.get("colorkey_tolerance"), defaults["colorkey_tolerance"]
        ),
        "feather_radius": _parse_int(form.get("feather_radius"), defaults["feather_radius"]),
    }


def _get_session_context(model_name: str, config: Mapping[str, Any]) -> "SessionContext":
    """Return a cached background removal session for ``model_name``."""

    accelerator_mode = str(config.get("BG_ACCELERATOR", "auto")).strip().lower()
    device_id = int(config.get("BG_CUDA_DEVICE_ID", 0) or 0)
    cache_key = (model_name, accelerator_mode, device_id)

    with _SESSION_CACHE_LOCK:
        context = _SESSION_CACHE.get(cache_key)
        if context is None:
            context = create_session(model_name=model_name, config=config)
            _SESSION_CACHE[cache_key] = context
        return context


@image_converter_bp.route("/", methods=["GET", "POST"])
@image_converter_bp.route("/image/remove-bg", methods=["GET", "POST"])
def remove_bg_view() -> Response:
    """Render the UI or process incoming form submissions."""

    defaults = DEFAULT_SINGLE_OPTIONS.copy()
    defaults["output_format"] = DEFAULT_OUTPUT_FORMAT_KEY
    defaults.update(DEFAULT_CHECKBOX_OPTIONS)

    ensure_global_session(config=current_app.config if current_app else None)
    runtime_info = get_runtime_payload()

    if request.method == "GET":
        return render_template(
            "image_remove_bg.html",
            defaults=defaults,
            format_options=FORMAT_OPTIONS,
            accelerator_runtime=runtime_info,
            removal_model_options=REMOVAL_MODEL_OPTIONS,
            hardware_accelerator_options=HARDWARE_ACCELERATOR_OPTIONS,
        )

    form = request.form
    process_folder = _is_truthy(form.get("process_folder"))
    recursive = _is_truthy(form.get("recursive")) or request.args.get("recursive") == "1"
    options = _collect_single_options(form, defaults)
    try:
        format_spec = get_output_format_spec(form.get("output_format"))
    except ValueError as exc:
        return _bad_request(str(exc))

    removal_model_key = (form.get("removal_model") or DEFAULT_REMOVAL_MODEL_KEY).strip().lower()
    model_name = _REMOVAL_MODEL_LOOKUP.get(removal_model_key, _REMOVAL_MODEL_LOOKUP[DEFAULT_REMOVAL_MODEL_KEY])
    hardware_accelerator = (form.get("hardware_accelerator") or DEFAULT_HARDWARE_ACCELERATOR_KEY).strip().lower()
    if hardware_accelerator not in {option["key"] for option in HARDWARE_ACCELERATOR_OPTIONS}:
        hardware_accelerator = DEFAULT_HARDWARE_ACCELERATOR_KEY

    preview_size = None
    preview_size_raw = form.get("preview_size")
    if preview_size_raw:
        try:
            preview_size = int(preview_size_raw)
        except (TypeError, ValueError):
            preview_size = None

    session_config: Dict[str, Any] = {}
    if current_app:
        session_config.update(current_app.config)

    accelerator_mode = hardware_accelerator
    if accelerator_mode == "gpu":
        accelerator_mode = "cuda"
    session_config["BG_ACCELERATOR"] = accelerator_mode

    session_context = None
    if current_app and current_app.config.get("TESTING"):
        session = None
        runtime_info = get_runtime_payload()
        gpu_available = bool(runtime_info.get("gpu_available"))
    else:
        session_context = _get_session_context(model_name, session_config)
        session = session_context.session
        runtime_info = session_context.runtime_payload()
        gpu_available = bool(runtime_info.get("gpu_available"))

    json_requested = request.args.get("json") == "1"

    if process_folder or form.get("folder_path"):
        folder_path = (form.get("folder_path") or "").strip()
        if not folder_path:
            return _bad_request("A folder path is required when processing folders.")

        output_dir = (form.get("output_dir") or "").strip() or None

        try:
            results = remove_bg_folder(
                folder_path,
                output_dir,
                output_format=format_spec.key,
                session=session,
                recursive=recursive,
                alpha_matting=options["alpha_matting"],
                am_foreground=options["am_foreground"],
                am_background=options["am_background"],
                am_erode=options["am_erode"],
                colorkey_tolerance=options["colorkey_tolerance"],
                feather_radius=options["feather_radius"],
            )
        except Exception as exc:  # pragma: no cover - depends on runtime environment
            return _bad_request(str(exc))

        payload = _serialise_results(results)
        payload["selected_format"] = format_spec.key
        payload.update(runtime_info)
        payload["selection"] = {
            "removal_model": removal_model_key,
            "model_name": model_name,
            "hardware_accelerator": hardware_accelerator,
            "gpu_available": gpu_available,
            "output_directory": output_dir,
            "preview_size": preview_size,
        }

        if request.args.get("zip") == "1":
            try:
                token, download_url = _create_zip(results)
                payload["zip_download_url"] = download_url
                payload["zip_token"] = token
            except Exception as exc:  # pragma: no cover - filesystem edge cases
                payload["zip_error"] = str(exc)
        return jsonify(payload)

    file_storage = request.files.get("image_file")
    if not file_storage or file_storage.filename == "":
        return _bad_request("Please upload an image or provide a folder path.")

    filename = secure_filename(file_storage.filename or "image.png")
    temp_dir = Path(tempfile.mkdtemp(prefix="bgremove_"))
    input_path = temp_dir / filename
    file_storage.save(input_path)
    persistent_output_dir: Path | None = None
    single_output_dir_text = (form.get("single_output_dir") or "").strip()
    output_base = temp_dir / input_path.stem
    if single_output_dir_text:
        candidate = Path(single_output_dir_text).expanduser()
        if not candidate.is_absolute():
            candidate = Path.cwd() / candidate
        persistent_output_dir = candidate
        output_base = candidate / input_path.stem

    output_path = format_spec.normalise_filename(output_base)
    download_name = f"{input_path.stem}_no_bg{format_spec.extension}"

    result = remove_bg_file(
        input_path,
        output_path,
        session=session,
        alpha_matting=options["alpha_matting"],
        am_foreground=options["am_foreground"],
        am_background=options["am_background"],
        am_erode=options["am_erode"],
        colorkey_tolerance=options["colorkey_tolerance"],
        feather_radius=options["feather_radius"],
        output_format=format_spec.key,
    )

    if not result.success:
        shutil.rmtree(temp_dir, ignore_errors=True)
        return _bad_request(result.error or "Background removal failed.")

    if json_requested:
        final_path = result.path_out or output_path
        encoded = encode_result_image(final_path)
        response_data = {
            "result": result.to_dict(),
            "image_base64": encoded,
            "mime_type": format_spec.mime_type,
            "download_name": download_name,
            "format": format_spec.key,
            "selection": {
                "removal_model": removal_model_key,
                "model_name": model_name,
                "hardware_accelerator": hardware_accelerator,
                "gpu_available": gpu_available,
                "preview_size": preview_size,
                "output_directory": str(persistent_output_dir) if persistent_output_dir else None,
            },
        }
        response_data.update(runtime_info)
        shutil.rmtree(temp_dir, ignore_errors=True)
        return jsonify(response_data)

    @after_this_request
    def cleanup(_: Response) -> Response:
        shutil.rmtree(temp_dir, ignore_errors=True)
        return _

    response = send_file(
        result.path_out or output_path,
        mimetype=format_spec.mime_type,
        download_name=download_name,
    )
    response.headers["X-Removal-Result"] = json.dumps(result.to_dict())
    return response

def _register_registry_item(
    registry: Dict[str, RegistryItem],
    *,
    path: Path,
    mimetype: str | None = None,
    delete_after_read: bool = False,
    download_name: str | None = None,
) -> tuple[str, RegistryItem]:
    """Store ``path`` in the ``registry`` and return the associated token."""

    token = uuid.uuid4().hex
    entry = RegistryItem(
        path=path,
        mimetype=mimetype,
        delete_after_read=delete_after_read,
        download_name=download_name,
    )
    registry[token] = entry
    return token, entry


def _serve_registry_item(
    registry: Dict[str, RegistryItem], token: str, *, as_attachment: bool
) -> Response:
    """Return the file referenced by ``token`` from ``registry``."""

    entry = registry.pop(token, None)
    if entry is None or not entry.path.exists():
        abort(404)

    if entry.delete_after_read:
        @after_this_request
        def cleanup(response: Response) -> Response:
            try:
                entry.path.unlink(missing_ok=True)  # type: ignore[attr-defined]
            except Exception:
                pass
            return response

    mimetype = entry.mimetype or get_mime_type_for_path(entry.path)
    download_name = entry.download_name or entry.path.name if as_attachment else None
    return send_file(
        entry.path,
        mimetype=mimetype,
        as_attachment=as_attachment,
        download_name=download_name,
    )


@image_converter_bp.route("/image/remove-bg/download/<token>")
def download_zip(token: str) -> Response:
    """Serve a generated ZIP archive and clean it up afterwards."""

    path = _ZIP_REGISTRY.pop(token, None)
    if path is None or not path.exists():
        abort(404)

    @after_this_request
    def cleanup(response: Response) -> Response:
        try:
            path.unlink(missing_ok=True)  # type: ignore[attr-defined]
            if path.parent.exists():
                shutil.rmtree(path.parent, ignore_errors=True)
        except Exception:
            pass
        return response

    return send_file(path, mimetype="application/zip", as_attachment=True, download_name=path.name)


@image_converter_bp.route("/image/remove-bg/file/<token>")
def download_file(token: str) -> Response:
    """Serve an exported image referenced by a temporary token."""

    return _serve_registry_item(_FILE_REGISTRY, token, as_attachment=True)


@image_converter_bp.route("/image/remove-bg/preview/<token>")
def preview_file(token: str) -> Response:
    """Serve an inline preview for a processed image."""

    return _serve_registry_item(_PREVIEW_REGISTRY, token, as_attachment=False)


@image_converter_bp.route("/health/accelerator", methods=["GET"])
def accelerator_health() -> Response:
    """Return diagnostic accelerator information for health checks."""

    ensure_global_session(config=current_app.config if current_app else None)
    return jsonify(get_accelerator_status())


def _serialise_results(results: List[RemovalResult]) -> Dict[str, Any]:
    """Convert ``RemovalResult`` objects to JSON-compatible data."""

    serialised: List[Dict[str, Any]] = []
    successes: List[RemovalResult] = []
    failures: List[RemovalResult] = []

    for result in results:
        data = result.to_dict()
        if result.success and result.path_out is not None:
            format_spec = get_output_format_spec(result.path_out.suffix)
            download_token, _ = _register_registry_item(
                _FILE_REGISTRY,
                path=result.path_out,
                mimetype=format_spec.mime_type,
                delete_after_read=False,
                download_name=result.path_out.name,
            )
            preview_token, _ = _register_registry_item(
                _PREVIEW_REGISTRY,
                path=result.path_out,
                mimetype=format_spec.mime_type,
                delete_after_read=False,
            )
            data["download_url"] = url_for("image_converter.download_file", token=download_token)
            data["preview_url"] = url_for("image_converter.preview_file", token=preview_token)
            data["mime_type"] = format_spec.mime_type
            data["format"] = format_spec.key
            successes.append(result)
        else:
            failures.append(result)
        serialised.append(data)

    return {
        "results": serialised,
        "summary": {
            "total": len(results),
            "success": len(successes),
            "failed": len(failures),
        },
    }


def _create_zip(results: List[RemovalResult]) -> tuple[str, str]:
    """Bundle successful outputs into a temporary ZIP archive."""

    successful = [result for result in results if result.success and result.path_out]
    if not successful:
        raise ValueError("No successful results available to bundle.")

    temp_dir = Path(tempfile.mkdtemp(prefix="bgremove_zip_"))
    zip_path = temp_dir / "results.zip"
    import zipfile

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for item in successful:
            if item.path_out is None:
                continue
            arcname = Path(item.path_out).name
            archive.write(item.path_out, arcname=arcname)

    token = uuid.uuid4().hex
    _ZIP_REGISTRY[token] = zip_path
    download_url = url_for("image_converter.download_zip", token=token)
    return token, download_url


def _bad_request(message: str) -> Response:
    """Return a consistent JSON error payload."""

    payload = {"error": message}
    payload.update(get_runtime_payload())
    return jsonify(payload), 400
