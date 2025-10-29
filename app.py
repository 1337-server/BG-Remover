"""Flask application, routes, and dev-server helpers for the background removal UI."""
from __future__ import annotations

import argparse
import atexit
import base64
import binascii
import json
import logging
import shutil
import tempfile
import threading
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from flask import (
    Blueprint,
    Flask,
    Response,
    abort,
    after_this_request,
    jsonify,
    render_template,
    request,
    send_file,
    url_for,
)
from flask.typing import ResponseReturnValue
from werkzeug.utils import secure_filename

from bg_removal import (
    DEFAULT_OUTPUT_FORMAT,
    OUTPUT_FORMATS,
    RemovalResult,
    encode_result_image,
    ensure_global_session,
    ensure_models_downloaded,
    get_accelerator_status,
    get_mime_type_for_path,
    get_output_format_spec,
    get_runtime_payload,
    remove_bg_file,
    remove_bg_folder,
)

LOGGER = logging.getLogger(__name__)

image_converter_bp = Blueprint("image_converter", __name__)


@dataclass(slots=True)
class RegistryItem:
    """Metadata describing an entry stored in a download registry."""

    path: Path
    mimetype: str | None = None
    delete_after_read: bool = False
    download_name: str | None = None


@dataclass(slots=True)
class ZipRegistryItem:
    """Metadata describing scheduled ZIP downloads stored in the registry."""

    path: Path
    temp_dir: Path
    created_at: float
    expiry_timer: threading.Timer | None = None


ZIP_REGISTRY_TTL_SECONDS: float = 600.0
"""Time-to-live for generated ZIP downloads before automatic eviction."""

_ZIP_REGISTRY: dict[str, ZipRegistryItem] = {}
_ZIP_REGISTRY_LOCK = threading.Lock()
_FILE_REGISTRY: dict[str, RegistryItem] = {}
_PREVIEW_REGISTRY: dict[str, RegistryItem] = {}

FORMAT_OPTIONS = [
    {"key": spec.key, "label": spec.label, "extension": spec.extension}
    for spec in OUTPUT_FORMATS
]
DEFAULT_OUTPUT_FORMAT_KEY = DEFAULT_OUTPUT_FORMAT
REMOVAL_MODEL_OPTIONS = [
    {"key": "general", "label": "General Model (isnet-general-use)", "model_name": "isnet-general-use"},
    {
        "key": "general_high_quality",
        "label": "BRIA RMBG v2.0 (High-Quality General)",
        "model_name": "briaai/RMBG-2.0",
    },
    {"key": "human", "label": "Human Model (u2net_human_seg)", "model_name": "u2net_human_seg"},
    {
        "key": "human_matting",
        "label": "BRIA RMBG v1.4 (Portrait Matting)",
        "model_name": "matting-by-generation",
    },
    {"key": "object", "label": "Object Model (u2net)", "model_name": "u2net"},
    {
        "key": "complex_scene",
        "label": "BRIA RMBG v2.0 (Complex Scenes)",
        "model_name": "sam_segmentation_model",
    },
    {"key": "anime", "label": "Anime / Illustration Model (isnet-anime)", "model_name": "isnet-anime"},
]
_REMOVAL_MODEL_LOOKUP: dict[str, str] = {
    option["key"]: option["model_name"] for option in REMOVAL_MODEL_OPTIONS
}
_REMOVAL_MODEL_LABEL_LOOKUP: dict[str, str] = {
    option["key"]: option["label"] for option in REMOVAL_MODEL_OPTIONS
}
DEFAULT_REMOVAL_MODEL_KEY = REMOVAL_MODEL_OPTIONS[0]["key"]
DEFAULT_SINGLE_OPTIONS: dict[str, Any] = {
    "am_foreground": 240,
    "am_background": 10,
    "am_erode": 10,
    "colorkey_tolerance": 14,
    "feather_radius": 3,
    "removal_model": DEFAULT_REMOVAL_MODEL_KEY,
}
DEFAULT_CHECKBOX_OPTIONS: dict[str, bool] = {
    "alpha_matting": False,
    "recursive": False,
    "zip": False,
}


def create_app(
        config_overrides: Mapping[str, object] | None = None,
        *,
        run_startup_tasks: bool = True,
) -> Flask:
    """Create and configure the Flask application instance."""

    app = Flask(__name__, template_folder="templates", static_folder="static")
    if config_overrides:
        app.config.update(config_overrides)

    if run_startup_tasks:
        ensure_models_downloaded()
        ensure_global_session()

    register_routes(app)

    @app.after_request
    def add_static_cache_headers(response: Response) -> Response:
        """Add caching headers for static assets to improve load performance."""

        if request.path.startswith("/static/"):
            response.headers.setdefault("Cache-Control", "public, max-age=31536000, immutable")
        return response

    LOGGER.info("Background removal application initialised.")
    return app


def register_routes(app: Flask) -> None:
    """Attach UI routes to ``app``."""

    app.register_blueprint(image_converter_bp)


def _parse_int(value: Any, default: int) -> int:
    """Safely parse integers from incoming form values."""

    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _is_truthy(value: Any) -> bool:
    """Return ``True`` for common representations of truthy checkbox values."""

    if value is None:
        return False
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "on", "yes"}


def _collect_single_options(form: Mapping[str, Any], defaults: Mapping[str, Any]) -> dict[str, Any]:
    """Extract reusable single-image processing options from the request."""

    return {
        "alpha_matting": _is_truthy(form.get("alpha_matting")),
        "am_foreground": _parse_int(form.get("am_foreground"), int(defaults["am_foreground"])),
        "am_background": _parse_int(form.get("am_background"), int(defaults["am_background"])),
        "am_erode": _parse_int(form.get("am_erode"), int(defaults["am_erode"])),
        "colorkey_tolerance": _parse_int(form.get("colorkey_tolerance"), int(defaults["colorkey_tolerance"])),
        "feather_radius": _parse_int(form.get("feather_radius"), int(defaults["feather_radius"])),
    }


def _load_json_payload() -> Mapping[str, Any] | None:
    """Return the parsed JSON payload when available and valid."""

    if not request.is_json:
        return None
    payload = request.get_json(silent=True)
    if isinstance(payload, Mapping):
        return payload
    LOGGER.error(
        "Invalid JSON payload supplied", extra={"content_type": request.content_type}
    )
    return None


def _decode_base64_image(value: str) -> bytes:
    """Decode a base64-encoded image payload, handling data URLs when present."""

    text = value.strip()
    if not text:
        raise ValueError("Empty base64 payload received.")
    if text.startswith("data:"):
        try:
            _, text = text.split(",", 1)
        except ValueError as exc:  # pragma: no cover - malformed data URL
            raise ValueError("Malformed data URL supplied.") from exc
    try:
        return base64.b64decode(text, validate=True)
    except binascii.Error as exc:
        raise ValueError("Base64 decoding failed for the supplied image payload.") from exc


def _extract_json_image(payload: Mapping[str, Any]) -> tuple[bytes, str | None]:
    """Return raw image bytes and an optional filename from ``payload``."""

    for key in ("image_base64", "image"):
        candidate = payload.get(key)
        if isinstance(candidate, str) and candidate.strip():
            return _decode_base64_image(candidate), payload.get("filename")

    raw_bytes = payload.get("image_bytes")
    if isinstance(raw_bytes, bytes | bytearray):
        return bytes(raw_bytes), payload.get("filename")
    if isinstance(raw_bytes, list) and all(isinstance(item, int) for item in raw_bytes):
        return bytes(raw_bytes), payload.get("filename")

    raise ValueError(
        "JSON payload must include base64 data under 'image_base64' or 'image'."
    )


def _get_stripped(value: Any) -> str:
    """Return ``value`` coerced to ``str`` with surrounding whitespace removed."""

    if value is None:
        return ""
    return str(value).strip()


@image_converter_bp.route("/", methods=["GET", "POST"])
@image_converter_bp.route("/image/remove-bg", methods=["GET", "POST"])
def remove_background_view() -> ResponseReturnValue:
    """Render the UI or process incoming form submissions."""

    defaults = dict(DEFAULT_SINGLE_OPTIONS)
    defaults["output_format"] = DEFAULT_OUTPUT_FORMAT_KEY
    defaults.update(DEFAULT_CHECKBOX_OPTIONS)

    if request.method == "GET":
        ensure_global_session()
        runtime_info = get_runtime_payload()
        return render_template(
            "image_remove_bg.html",
            defaults=defaults,
            format_options=FORMAT_OPTIONS,
            accelerator_runtime=runtime_info,
            removal_model_options=REMOVAL_MODEL_OPTIONS,
        )

    json_payload = _load_json_payload()
    form_data: Mapping[str, Any] = request.form if json_payload is None else json_payload

    process_folder = _is_truthy(form_data.get("process_folder"))
    recursive = _is_truthy(form_data.get("recursive")) or request.args.get("recursive") == "1"
    options = _collect_single_options(form_data, defaults)
    try:
        format_spec = get_output_format_spec(form_data.get("output_format"))
    except ValueError as exc:
        LOGGER.error(
            "Unsupported output format requested", extra={"output_format": form_data.get("output_format")}
        )
        return _bad_request(str(exc))

    removal_model_key = _get_stripped(form_data.get("removal_model") or DEFAULT_REMOVAL_MODEL_KEY)
    if not removal_model_key:
        removal_model_key = DEFAULT_REMOVAL_MODEL_KEY
    removal_model_key = removal_model_key.lower()
    model_name = _REMOVAL_MODEL_LOOKUP.get(
        removal_model_key, _REMOVAL_MODEL_LOOKUP[DEFAULT_REMOVAL_MODEL_KEY]
    )
    preview_size: int | None = None
    preview_size_raw = form_data.get("preview_size")
    if preview_size_raw:
        try:
            preview_size = int(preview_size_raw)
        except (TypeError, ValueError):
            preview_size = None

    ensure_global_session(model_name)
    runtime_info = get_runtime_payload()

    json_requested = request.args.get("json") == "1"

    if process_folder or _get_stripped(form_data.get("folder_path")):
        folder_path = _get_stripped(form_data.get("folder_path"))
        if not folder_path:
            LOGGER.error(
                "Folder processing requested without a folder path", extra={"path": request.path}
            )
            return _bad_request(
                "A folder path is required when processing folders.",
                runtime_info=runtime_info,
            )

        output_dir = _get_stripped(form_data.get("output_dir")) or None

        try:
            results = remove_bg_folder(
                folder_path,
                output_dir,
                output_format=format_spec.key,
                model_name=model_name,
                recursive=recursive,
                alpha_matting=options["alpha_matting"],
                am_foreground=options["am_foreground"],
                am_background=options["am_background"],
                am_erode=options["am_erode"],
                colorkey_tolerance=options["colorkey_tolerance"],
                feather_radius=options["feather_radius"],
            )
        except Exception as exc:  # pragma: no cover - depends on runtime environment
            return _bad_request(str(exc), runtime_info=runtime_info)

        payload = _serialise_results(results)
        payload["selected_format"] = format_spec.key
        payload.update(runtime_info)
        payload["selection"] = {
            "removal_model": removal_model_key,
            "model_name": model_name,
            "removal_model_label": _REMOVAL_MODEL_LABEL_LOOKUP.get(removal_model_key),
            "output_directory": output_dir,
            "preview_size": preview_size,
        }

        if _is_truthy(form_data.get("zip")):
            try:
                token, download_url = _create_zip(results)
                payload["zip_download_url"] = download_url
                payload["zip_token"] = token
            except Exception as exc:  # pragma: no cover - filesystem edge cases
                payload["zip_error"] = str(exc)

        if json_requested:
            return jsonify(payload)

        return render_template(
            "image_remove_bg.html",
            defaults=defaults,
            format_options=FORMAT_OPTIONS,
            accelerator_runtime=runtime_info,
            removal_model_options=REMOVAL_MODEL_OPTIONS,
            results=payload,
        )

    file_storage = request.files.get("image_file")
    image_bytes: bytes | None = None
    upload_name: str | None = None
    if file_storage and file_storage.filename not in {None, ""}:
        upload_name = file_storage.filename
    elif json_payload is not None:
        try:
            image_bytes, upload_name = _extract_json_image(json_payload)
        except ValueError as exc:
            LOGGER.error(
                "JSON payload did not include a valid image", extra={"error": str(exc)}
            )
            return _bad_request(str(exc), runtime_info=runtime_info)
    else:
        LOGGER.error(
            "No image data supplied in request", extra={"path": request.path, "method": request.method}
        )
        return _bad_request(
            "Please choose an image to upload or provide a folder path.",
            runtime_info=runtime_info,
        )

    filename = secure_filename(upload_name or "image.png")
    if not filename:
        filename = "image.png"

    temp_dir = Path(tempfile.mkdtemp(prefix="bgremove_"))
    input_path = temp_dir / filename
    if file_storage and image_bytes is None:
        file_storage.save(input_path)
    else:
        assert image_bytes is not None  # for type-checkers
        input_path.write_bytes(image_bytes)

    persistent_output_dir: Path | None = None
    single_output_dir_text = _get_stripped(form_data.get("single_output_dir"))
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
        output_format=format_spec.key,
        model_name=model_name,
        alpha_matting=options["alpha_matting"],
        am_foreground=options["am_foreground"],
        am_background=options["am_background"],
        am_erode=options["am_erode"],
        colorkey_tolerance=options["colorkey_tolerance"],
        feather_radius=options["feather_radius"],
        retain_image=True,
    )

    if not result.success:
        LOGGER.error(
            "Background removal failed for uploaded image",
            extra={
                "error": result.error,
                "model_name": model_name,
                "input_path": str(result.path_in) if result.path_in else None,
            },
        )
        shutil.rmtree(temp_dir, ignore_errors=True)
        return _bad_request(
            result.error or "Background removal failed.", runtime_info=runtime_info
        )

    if json_requested:
        encoded = encode_result_image(result)
        response_data = {
            "result": result.to_dict(),
            "image_base64": encoded,
            "mime_type": format_spec.mime_type,
            "download_name": download_name,
            "format": format_spec.key,
            "selection": {
                "removal_model": removal_model_key,
                "model_name": model_name,
                "removal_model_label": _REMOVAL_MODEL_LABEL_LOOKUP.get(removal_model_key),
                "preview_size": preview_size,
                "output_directory": str(persistent_output_dir) if persistent_output_dir else None,
            },
        }
        response_data.update(runtime_info)
        if result.image is not None:
            result.image.close()
        shutil.rmtree(temp_dir, ignore_errors=True)
        return jsonify(response_data)

    @after_this_request
    def cleanup(_: Response) -> Response:
        shutil.rmtree(temp_dir, ignore_errors=True)
        if result.image is not None:
            result.image.close()
        return _

    response = send_file(
        result.path_out or output_path,
        mimetype=format_spec.mime_type,
        download_name=download_name,
    )
    response.headers["X-Removal-Result"] = json.dumps(result.to_dict())
    return response


def _register_registry_item(
        registry: dict[str, RegistryItem],
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


def _delete_zip_artifacts(item: ZipRegistryItem) -> None:
    """Remove filesystem artefacts for ``item`` without raising exceptions."""

    try:
        item.path.unlink(missing_ok=True)  # type: ignore[attr-defined]
    except Exception:
        LOGGER.debug("Failed to remove ZIP file %s during cleanup", item.path, exc_info=True)

    try:
        if item.temp_dir.exists():
            shutil.rmtree(item.temp_dir, ignore_errors=True)
    except Exception:
        LOGGER.debug(
            "Failed to remove ZIP temporary directory %s during cleanup", item.temp_dir, exc_info=True
        )


def _cancel_zip_timer(item: ZipRegistryItem) -> None:
    """Stop the expiry timer attached to ``item`` when the download is served."""

    timer = item.expiry_timer
    if timer is not None:
        item.expiry_timer = None
        try:
            timer.cancel()
        except Exception:
            LOGGER.debug("Failed to cancel ZIP expiry timer", exc_info=True)


def _expire_zip_entry(token: str) -> None:
    """Remove the ZIP registry entry ``token`` if it is still present."""

    with _ZIP_REGISTRY_LOCK:
        item = _ZIP_REGISTRY.pop(token, None)
    if item is None:
        return
    _delete_zip_artifacts(item)


def _register_zip_entry(zip_path: Path, temp_dir: Path) -> tuple[str, ZipRegistryItem]:
    """Store a generated ZIP file in the registry and start its expiry timer."""

    token = uuid.uuid4().hex
    item = ZipRegistryItem(path=zip_path, temp_dir=temp_dir, created_at=time.time())
    ttl = ZIP_REGISTRY_TTL_SECONDS
    if ttl <= 0:
        _delete_zip_artifacts(item)
        return token, item

    with _ZIP_REGISTRY_LOCK:
        _ZIP_REGISTRY[token] = item
    timer = threading.Timer(ttl, _expire_zip_entry, args=(token,))
    timer.daemon = True
    item.expiry_timer = timer
    timer.start()
    return token, item


def _drain_zip_registry() -> None:
    """Remove all pending ZIP downloads and their backing timers."""

    with _ZIP_REGISTRY_LOCK:
        pending = list(_ZIP_REGISTRY.values())
        _ZIP_REGISTRY.clear()
    for item in pending:
        _cancel_zip_timer(item)
        _delete_zip_artifacts(item)


atexit.register(_drain_zip_registry)


def _serve_registry_item(registry: dict[str, RegistryItem], token: str, *, as_attachment: bool) -> Response:
    """Return the file referenced by ``token`` from ``registry``."""

    entry = registry.pop(token, None)
    if entry is None:
        abort(404)
        raise RuntimeError("Registry entry missing")  # pragma: no cover - satisfies type checkers
    if not entry.path.exists():
        abort(404)
        raise RuntimeError("Registry entry missing")  # pragma: no cover - satisfies type checkers

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

    with _ZIP_REGISTRY_LOCK:
        item = _ZIP_REGISTRY.pop(token, None)
    if item is None:
        abort(404)
        raise RuntimeError("ZIP entry missing")  # pragma: no cover - satisfies type checkers
    if not item.path.exists():
        abort(404)
        raise RuntimeError("ZIP entry missing")  # pragma: no cover - satisfies type checkers

    _cancel_zip_timer(item)

    @after_this_request
    def cleanup(response: Response) -> Response:
        _delete_zip_artifacts(item)
        return response

    return send_file(
        item.path,
        mimetype="application/zip",
        as_attachment=True,
        download_name=item.path.name,
    )


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

    ensure_global_session()
    return jsonify(get_accelerator_status())


def _serialise_results(results: list[RemovalResult]) -> dict[str, Any]:
    """Convert ``RemovalResult`` objects to JSON-compatible data."""

    serialised: list[dict[str, Any]] = []
    successes: list[RemovalResult] = []
    failures: list[RemovalResult] = []

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


def _create_zip(results: list[RemovalResult]) -> tuple[str, str]:
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

    token, _ = _register_zip_entry(zip_path, temp_dir)
    download_url = url_for("image_converter.download_zip", token=token)
    return token, download_url


def _bad_request(
        message: str, *, runtime_info: Mapping[str, Any] | None = None
) -> ResponseReturnValue:
    """Return a consistent JSON error payload with optional runtime metadata."""

    payload = {"error": message}
    if runtime_info is None:
        runtime_info = get_runtime_payload()
    payload.update(runtime_info)
    response = jsonify(payload)
    response.status_code = 400
    return response


def parse_server_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Return parsed command line arguments for the dev server."""

    parser = argparse.ArgumentParser(description="Run the background removal web UI.")
    parser.add_argument("--host", default="0.0.0.0", help="Interface to bind")
    parser.add_argument("--port", type=int, default=5000, help="Port to bind")
    parser.add_argument("--debug", action="store_true", help="Enable Flask debug mode")
    return parser.parse_args(argv)


def run_dev_server(argv: list[str] | None = None) -> None:
    """Start the Flask development server using parsed CLI arguments."""

    args = parse_server_args(argv)
    app = create_app()
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":  # pragma: no cover - convenience entry point
    run_dev_server()
