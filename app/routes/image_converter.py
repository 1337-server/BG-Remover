"""Flask routes for interactive background removal."""
from __future__ import annotations

import json
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, List

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

from app.services.bg_remove import (
    RemovalResult,
    encode_result_image,
    remove_bg_file,
    remove_bg_folder,
)

image_converter_bp = Blueprint("image_converter", __name__)

_ZIP_REGISTRY: Dict[str, Path] = {}
_FILE_REGISTRY: Dict[str, Path] = {}


def _parse_int(value: str | None, default: int) -> int:
    """Safely parse integers from incoming form values."""

    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


@image_converter_bp.route("/image/remove-bg", methods=["GET", "POST"])
def remove_bg_view() -> Response:
    """Render the UI or process incoming form submissions."""

    defaults = {
        "am_foreground": 240,
        "am_background": 10,
        "am_erode": 10,
        "colorkey_tolerance": 14,
        "feather_radius": 3,
    }

    if request.method == "GET":
        return render_template("image_remove_bg.html", defaults=defaults)

    form = request.form
    process_folder = form.get("process_folder") in {"on", "true", "1"}
    recursive = form.get("recursive") in {"on", "true", "1"} or request.args.get("recursive") == "1"
    alpha_matting = form.get("alpha_matting") in {"on", "true", "1"}

    am_foreground = _parse_int(form.get("am_foreground"), defaults["am_foreground"])
    am_background = _parse_int(form.get("am_background"), defaults["am_background"])
    am_erode = _parse_int(form.get("am_erode"), defaults["am_erode"])
    colorkey_tolerance = _parse_int(form.get("colorkey_tolerance"), defaults["colorkey_tolerance"])
    feather_radius = _parse_int(form.get("feather_radius"), defaults["feather_radius"])

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
                alpha_matting=alpha_matting,
                am_foreground=am_foreground,
                am_background=am_background,
                am_erode=am_erode,
                colorkey_tolerance=colorkey_tolerance,
                feather_radius=feather_radius,
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
        alpha_matting=alpha_matting,
        am_foreground=am_foreground,
        am_background=am_background,
        am_erode=am_erode,
        colorkey_tolerance=colorkey_tolerance,
        feather_radius=feather_radius,
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


def _serialise_results(results: List[RemovalResult]) -> Dict[str, Any]:
    """Convert ``RemovalResult`` objects to JSON-compatible data."""

    serialised: List[Dict[str, Any]] = []
    successes: List[RemovalResult] = []
    failures: List[RemovalResult] = []

    for result in results:
        data = result.to_dict()
        if result.success and result.path_out is not None:
            token = uuid.uuid4().hex
            _FILE_REGISTRY[token] = result.path_out
            data["download_url"] = url_for("image_converter.download_file", token=token)
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
