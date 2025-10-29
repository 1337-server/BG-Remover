"""Form helpers for the Flask runtime."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from flask import Request

from bgremover_core import Config


@dataclass(slots=True)
class RemovalFormData:
    """Parsed form data for a background removal request."""

    model_key: str
    feather_radius: int
    model_dir: Path | None
    config: Config

    @classmethod
    def from_request(cls, request: Request, base_config: Config) -> RemovalFormData:
        model_key = request.form.get("model_key") or base_config.default_model
        feather_raw = request.form.get("feather_radius", 3)
        try:
            feather_radius = int(feather_raw)
        except (TypeError, ValueError):
            feather_radius = 3
        feather_radius = max(0, min(50, feather_radius))
        model_dir_raw = request.form.get("model_dir")
        if model_dir_raw:
            model_dir = Path(model_dir_raw).expanduser()
            config = base_config.with_updates(model_dir=model_dir)
        else:
            model_dir = None
            config = base_config
        config.resolved_model_dir()
        return cls(
            model_key=model_key,
            feather_radius=feather_radius,
            model_dir=model_dir,
            config=config,
        )

    @classmethod
    def from_defaults(cls, config: Config) -> RemovalFormData:
        return cls(
            model_key=config.default_model,
            feather_radius=3,
            model_dir=config.model_dir,
            config=config,
        )

    def output_name(self, original_filename: str) -> str:
        stem = Path(original_filename).stem or "output"
        return f"{stem}_no_bg.png"
