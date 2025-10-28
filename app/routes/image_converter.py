"""Flask routes exposing a minimal background removal interface."""
from __future__ import annotations

from http import HTTPStatus

from flask import Blueprint, flash, render_template, request
from flask.typing import ResponseReturnValue

from app.services.bg_remove import (
    DEFAULT_OUTPUT_FORMAT,
    OUTPUT_FORMATS,
    encode_result_image,
    ensure_global_session,
    get_runtime_payload,
    remove_background_stream,
)

image_converter_bp = Blueprint("image_converter", __name__)


@image_converter_bp.route("/", methods=["GET", "POST"])
@image_converter_bp.route("/image/remove-bg", methods=["GET", "POST"])
def remove_background_view() -> ResponseReturnValue:
    """Render the upload form and process submitted images."""

    ensure_global_session()
    runtime_info = get_runtime_payload()
    selected_format = request.form.get("output_format", DEFAULT_OUTPUT_FORMAT)
    result_data: str | None = None
    error_message: str | None = None
    status = HTTPStatus.OK

    if request.method == "POST":
        upload = request.files.get("image")
        if upload is None or not upload.filename:
            error_message = "Please choose an image to process."
            status = HTTPStatus.BAD_REQUEST
        else:
            upload.stream.seek(0)
            try:
                result = remove_background_stream(
                    upload.stream,
                    output_format=selected_format,
                )
            except Exception as exc:  # pragma: no cover - surfaced as user feedback
                error_message = str(exc)
                status = HTTPStatus.INTERNAL_SERVER_ERROR
            else:
                result_data = encode_result_image(result)
                selected_format = result.format_spec.key
                flash("Background removed successfully.", "success")

    if error_message:
        flash(error_message, "error")

    return (
        render_template(
            "image_remove_bg.html",
            runtime_info=runtime_info,
            format_options=OUTPUT_FORMATS,
            selected_format=selected_format,
            result_data=result_data,
        ),
        status,
    )
