"""Flask blueprint exposing the background removal web UI."""
from __future__ import annotations

import io

import numpy as np
from flask import (
    Blueprint,
    Response,
    current_app,
    flash,
    redirect,
    render_template,
    request,
    send_file,
    url_for,
)
from PIL import Image, UnidentifiedImageError

from bgremover_core import Config, load_config, remove_background
from bgremover_core.models.loader import detect_providers
from bgremover_core.models.specs import MODEL_SPECS
from bgremover_core.processing.pipeline import PipelineError

from .forms import RemovalFormData

bp = Blueprint("bgremover", __name__)


def _get_config() -> Config:
    config = current_app.config.get("BGR_CONFIG")
    if config is None:
        config = load_config()
        current_app.config["BGR_CONFIG"] = config
    return config


def _provider_badge(providers: list[str]) -> tuple[str, list[str]]:
    if not providers:
        return "CPU", ["CPUExecutionProvider"]
    primary = providers[0]
    if primary.lower().startswith("cuda"):
        return "GPU", providers
    return "CPU", providers


@bp.route("/", methods=["GET", "POST"])
def index() -> Response:
    config = _get_config()
    providers = detect_providers(config.provider_hints)
    badge_label, provider_list = _provider_badge(providers)

    if request.method == "POST":
        form = RemovalFormData.from_request(request, config)
        upload = request.files.get("image")
        if not upload or not upload.filename:
            flash("Please choose an image to process.", "error")
            return redirect(url_for("bgremover.index"))
        try:
            image = Image.open(upload.stream).convert("RGBA")
        except UnidentifiedImageError:
            flash("The uploaded file is not a supported image format.", "error")
            return redirect(url_for("bgremover.index"))

        array = np.asarray(image)
        try:
            result = remove_background(
                array,
                form.model_key,
                config=form.config,
                feather_radius=form.feather_radius,
            )
        except PipelineError as error:
            flash(f"Processing failed: {error}", "error")
            return redirect(url_for("bgremover.index"))
        output_image = Image.fromarray(result)
        buffer = io.BytesIO()
        output_image.save(buffer, "PNG")
        buffer.seek(0)
        download_name = form.output_name(upload.filename)
        return send_file(
            buffer,
            mimetype="image/png",
            as_attachment=True,
            download_name=download_name,
        )

    form = RemovalFormData.from_defaults(config)
    context = {
        "form": form,
        "badge_label": badge_label,
        "providers": provider_list,
        "model_options": sorted(MODEL_SPECS.keys()),
        "model_dir": str(config.model_dir),
    }
    return render_template("index.html", **context)
