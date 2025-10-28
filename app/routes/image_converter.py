"""Flask routes exposing the interactive background removal interface."""
from __future__ import annotations

from http import HTTPStatus

from flask import Blueprint, flash, render_template, request
from flask.typing import ResponseReturnValue

from app.services.bg_remove import (
    DEFAULT_OUTPUT_FORMAT,
    OUTPUT_FORMATS,
    encode_result_image,
    ensure_global_session,
    get_output_format_spec,
    get_runtime_payload,
    remove_background_stream,
)

DEFAULT_PREVIEW_SIZE = 1024
REMOVAL_MODEL_OPTIONS: tuple[dict[str, str], ...] = (
    {
        "key": "general",
        "label": "General Model",
        "model_name": "isnet-general-use",
    },
    {
        "key": "human",
        "label": "Human Model",
        "model_name": "u2net_human_seg",
    },
    {
        "key": "object",
        "label": "Object Model",
        "model_name": "u2net",
    },
    {
        "key": "anime",
        "label": "Anime / Illustration Model",
        "model_name": "isnet-anime",
    },
)
_REMOVAL_MODEL_LOOKUP: dict[str, str] = {
    option["key"]: option["model_name"] for option in REMOVAL_MODEL_OPTIONS
}
DEFAULT_REMOVAL_MODEL_KEY = REMOVAL_MODEL_OPTIONS[0]["key"]

image_converter_bp = Blueprint("image_converter", __name__)


def _resolve_removal_model(raw_key: str | None) -> str:
    """Return a supported model key for ``raw_key`` falling back to the default."""

    if not raw_key:
        return DEFAULT_REMOVAL_MODEL_KEY
    key = raw_key.strip().lower()
    if key in _REMOVAL_MODEL_LOOKUP:
        return key
    return DEFAULT_REMOVAL_MODEL_KEY


def _parse_preview_size(raw_value: str | None) -> int:
    """Parse ``raw_value`` into an integer preview size within expected bounds."""

    if not raw_value:
        return DEFAULT_PREVIEW_SIZE
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        return DEFAULT_PREVIEW_SIZE
    return max(320, min(1280, value))


@image_converter_bp.route("/", methods=["GET", "POST"])
@image_converter_bp.route("/image/remove-bg", methods=["GET", "POST"])
def remove_background_view() -> ResponseReturnValue:
    """Render the upload form and process submitted images."""

    ensure_global_session()
    runtime_info = get_runtime_payload()
    selected_format = request.form.get("output_format", DEFAULT_OUTPUT_FORMAT)
    selected_model = _resolve_removal_model(request.form.get("removal_model"))
    preview_size = _parse_preview_size(request.form.get("preview_size"))
    result_data: str | None = None
    result_summary: dict[str, str | float] | None = None
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
                format_spec = get_output_format_spec(selected_format)
            except ValueError as exc:
                error_message = str(exc)
                status = HTTPStatus.BAD_REQUEST
            else:
                selected_format = format_spec.key
                model_name = _REMOVAL_MODEL_LOOKUP[selected_model]
                try:
                    result = remove_background_stream(
                        upload.stream,
                        output_format=selected_format,
                        model_name=model_name,
                    )
                except Exception as exc:  # pragma: no cover - surfaced as user feedback
                    error_message = str(exc)
                    status = HTTPStatus.INTERNAL_SERVER_ERROR
                else:
                    result_data = encode_result_image(result)
                    selected_format = result.format_spec.key
                    runtime_info = get_runtime_payload()
                    flash("Background removed successfully.", "success")
                    model_label = next(
                        (
                            option["label"]
                            for option in REMOVAL_MODEL_OPTIONS
                            if option["key"] == selected_model
                        ),
                        REMOVAL_MODEL_OPTIONS[0]["label"],
                    )
                    result_summary = {
                        "elapsed_ms": round(result.elapsed_ms, 2),
                        "format_label": result.format_spec.label,
                        "model_label": model_label,
                    }

    if error_message:
        flash(error_message, "error")

    return (
        render_template(
            "image_remove_bg.html",
            runtime_info=runtime_info,
            format_options=OUTPUT_FORMATS,
            selected_format=selected_format,
            result_data=result_data,
            removal_model_options=REMOVAL_MODEL_OPTIONS,
            selected_model=selected_model,
            preview_size=preview_size,
            result_summary=result_summary,
        ),
        status,
    )
