"""Flask routes for interactive background removal."""
from __future__ import annotations

import base64
import io
import json
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, List, Mapping

from flask import (
    Blueprint,
    Response,
    abort,
    after_this_request,
    jsonify,
    render_template,
    request,
    send_file,
    url_for,
)
from werkzeug.utils import secure_filename

from PIL import Image

from app.extensions import socketio
from app.services.bg_remove import (
    RemovalResult,
    encode_result_image,
    remove_bg_file,
    remove_bg_folder,
)

image_converter_bp = Blueprint("image_converter", __name__)

_ZIP_REGISTRY: Dict[str, Path] = {}
_FILE_REGISTRY: Dict[str, Path] = {}
_PREVIEW_REGISTRY: Dict[str, Path] = {}

BACKGROUND_PREVIEW_NAMESPACE = "/ws/background-preview"
DEFAULT_SINGLE_OPTIONS: Dict[str, int] = {
    "am_foreground": 240,
    "am_background": 10,
    "am_erode": 10,
    "colorkey_tolerance": 14,
    "feather_radius": 3,
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


@image_converter_bp.route("/image/remove-bg", methods=["GET", "POST"])
def remove_bg_view() -> Response:
    """Render the UI or process incoming form submissions."""

    defaults = DEFAULT_SINGLE_OPTIONS.copy()

    if request.method == "GET":
        return render_template("image_remove_bg.html", defaults=defaults)

    form = request.form
    process_folder = _is_truthy(form.get("process_folder"))
    recursive = _is_truthy(form.get("recursive")) or request.args.get("recursive") == "1"
    options = _collect_single_options(form, defaults)

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
    output_path = temp_dir / f"{input_path.stem}.png"

    result = remove_bg_file(
        input_path,
        output_path,
        alpha_matting=options["alpha_matting"],
        am_foreground=options["am_foreground"],
        am_background=options["am_background"],
        am_erode=options["am_erode"],
        colorkey_tolerance=options["colorkey_tolerance"],
        feather_radius=options["feather_radius"],
    )

    if not result.success:
        shutil.rmtree(temp_dir, ignore_errors=True)
        return _bad_request(result.error or "Background removal failed.")

    if json_requested:
        encoded = encode_result_image(result.path_out or output_path)
        response_data = {
            "result": result.to_dict(),
            "image_base64": encoded,
            "mime_type": "image/png",
        }
        shutil.rmtree(temp_dir, ignore_errors=True)
        return jsonify(response_data)

    @after_this_request
    def cleanup(_: Response) -> Response:
        shutil.rmtree(temp_dir, ignore_errors=True)
        return _

    response = send_file(
        result.path_out or output_path,
        mimetype="image/png",
        download_name=f"{input_path.stem}_no_bg.png",
    )
    response.headers["X-Removal-Result"] = json.dumps(result.to_dict())
    return response


def _encode_image_to_base64(image: Image.Image) -> str:
    """Return a base64-encoded PNG representation of ``image``."""

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _emit_progress(socket_id: str, job_id: str, stage: str, percent: float) -> None:
    """Send a progress update to the connected websocket client."""

    socketio.emit(
        "progress",
        {
            "job_id": job_id,
            "stage": stage,
            "percent": max(0.0, min(100.0, round(percent, 1))),
        },
        namespace=BACKGROUND_PREVIEW_NAMESPACE,
        to=socket_id,
    )


def _emit_preview(socket_id: str, job_id: str, stage: str, image: Image.Image) -> None:
    """Send a preview frame to the websocket client."""

    socketio.emit(
        "preview",
        {
            "job_id": job_id,
            "stage": stage,
            "image": _encode_image_to_base64(image),
        },
        namespace=BACKGROUND_PREVIEW_NAMESPACE,
        to=socket_id,
    )


def _emit_error(socket_id: str, job_id: str, message: str) -> None:
    """Send an error payload to the websocket client."""

    socketio.emit(
        "error",
        {"job_id": job_id, "message": message},
        namespace=BACKGROUND_PREVIEW_NAMESPACE,
        to=socket_id,
    )


def _process_live_job(
    job_id: str,
    socket_id: str,
    temp_dir: Path,
    input_path: Path,
    output_path: Path,
    options: Dict[str, Any],
) -> None:
    """Background task that performs removal and streams websocket updates."""

    def progress_callback(stage: str, percent: float) -> None:
        _emit_progress(socket_id, job_id, stage, percent)

    def preview_callback(image: Image.Image, stage: str) -> None:
        _emit_preview(socket_id, job_id, stage, image)

    try:
        _emit_progress(socket_id, job_id, "queued", 0.0)
        result = remove_bg_file(
            input_path,
            output_path,
            alpha_matting=options["alpha_matting"],
            am_foreground=options["am_foreground"],
            am_background=options["am_background"],
            am_erode=options["am_erode"],
            colorkey_tolerance=options["colorkey_tolerance"],
            feather_radius=options["feather_radius"],
            progress_callback=progress_callback,
            preview_callback=preview_callback,
        )
        if not result.success or result.path_out is None:
            raise RuntimeError(result.error or "Background removal failed.")

        with result.path_out.open("rb") as file_obj:
            encoded = base64.b64encode(file_obj.read()).decode("ascii")

        _emit_progress(socket_id, job_id, "complete", 100.0)
        socketio.emit(
            "completed",
            {
                "job_id": job_id,
                "result": result.to_dict(),
                "image": encoded,
                "mime_type": "image/png",
            },
            namespace=BACKGROUND_PREVIEW_NAMESPACE,
            to=socket_id,
        )
    except Exception as exc:  # pragma: no cover - depends on runtime environment
        _emit_error(socket_id, job_id, str(exc))
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@image_converter_bp.route("/image/remove-bg/live", methods=["POST"])
def remove_bg_live() -> Response:
    """Handle uploads for the live preview background removal pipeline."""

    defaults = DEFAULT_SINGLE_OPTIONS.copy()

    form = request.form
    socket_id = (form.get("socket_id") or "").strip()
    if not socket_id:
        return _bad_request("Missing Socket.IO connection identifier.")

    file_storage = request.files.get("image_file")
    if not file_storage or file_storage.filename == "":
        return _bad_request("Please upload an image to process.")

    options = _collect_single_options(form, defaults)

    temp_dir = Path(tempfile.mkdtemp(prefix="bgremove_live_"))
    filename = secure_filename(file_storage.filename or "image.png")
    input_path = temp_dir / filename
    file_storage.save(input_path)
    output_path = temp_dir / f"{input_path.stem}_no_bg.png"

    job_id = uuid.uuid4().hex

    socketio.start_background_task(
        _process_live_job,
        job_id,
        socket_id,
        temp_dir,
        input_path,
        output_path,
        options,
    )

    return jsonify({"status": "processing", "job_id": job_id})


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
    """Serve an exported PNG referenced by a temporary token."""

    path = _FILE_REGISTRY.pop(token, None)
    if path is None or not path.exists():
        abort(404)
    return send_file(path, mimetype="image/png", as_attachment=True, download_name=path.name)


@image_converter_bp.route("/image/remove-bg/preview/<token>")
def preview_file(token: str) -> Response:
    """Serve an inline preview PNG for a processed image."""

    path = _PREVIEW_REGISTRY.pop(token, None)
    if path is None or not path.exists():
        abort(404)
    return send_file(path, mimetype="image/png", as_attachment=False)


def _serialise_results(results: List[RemovalResult]) -> Dict[str, Any]:
    """Convert ``RemovalResult`` objects to JSON-compatible data."""

    serialised: List[Dict[str, Any]] = []
    successes: List[RemovalResult] = []
    failures: List[RemovalResult] = []

    for result in results:
        data = result.to_dict()
        if result.success and result.path_out is not None:
            download_token = uuid.uuid4().hex
            preview_token = uuid.uuid4().hex
            _FILE_REGISTRY[download_token] = result.path_out
            _PREVIEW_REGISTRY[preview_token] = result.path_out
            data["download_url"] = url_for("image_converter.download_file", token=download_token)
            data["preview_url"] = url_for("image_converter.preview_file", token=preview_token)
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
    return jsonify(payload), 400
