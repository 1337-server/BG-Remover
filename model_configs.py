"""Model configuration registry and validation helpers for advanced options."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class ParameterOption:
    """Represents a selectable option for a model configuration parameter."""

    value: Any
    label: str | None = None

    def to_frontend(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation for UI rendering."""

        return {"value": self.value, "label": self.label or str(self.value)}


@dataclass(frozen=True)
class ParameterSchema:
    """Schema describing metadata and validation rules for a model parameter."""

    value_type: type[Any]
    default: Any
    min_value: float | None = None
    max_value: float | None = None
    options: tuple[ParameterOption, ...] | None = None
    step: float | None = None
    label: str | None = None
    help_text: str | None = None

    def type_name(self) -> str:
        """Return a string identifier describing ``value_type``."""

        if self.value_type is bool:
            return "bool"
        if self.value_type is int:
            return "int"
        if self.value_type is float:
            return "float"
        return "str"

    def coerce(self, raw: Any) -> Any:
        """Validate and coerce ``raw`` into the configured ``value_type``."""

        if raw is None or (isinstance(raw, str) and raw.strip() == ""):
            value = self.default
        elif self.value_type is bool:
            if isinstance(raw, str):
                text = raw.strip().lower()
                if text in {"1", "true", "on", "yes"}:
                    value = True
                elif text in {"0", "false", "off", "no"}:
                    value = False
                else:
                    raise ValueError("must be true or false")
            elif isinstance(raw, bool):
                value = raw
            elif isinstance(raw, (int, float)):
                value = bool(raw)
            else:
                raise ValueError("must be a boolean value")
        elif self.value_type is int:
            try:
                if isinstance(raw, bool):
                    raise TypeError
                value = int(raw)
            except (TypeError, ValueError) as exc:  # pragma: no cover - defensive guard
                raise ValueError("must be an integer") from exc
        elif self.value_type is float:
            try:
                if isinstance(raw, bool):
                    raise TypeError
                value = float(raw)
            except (TypeError, ValueError) as exc:  # pragma: no cover - defensive guard
                raise ValueError("must be a number") from exc
        else:
            value = str(raw)

        if isinstance(value, (int, float)):
            if self.min_value is not None and value < self.min_value:
                raise ValueError(f"must be >= {self.min_value}")
            if self.max_value is not None and value > self.max_value:
                raise ValueError(f"must be <= {self.max_value}")

        if self.options:
            allowed = [option.value for option in self.options]
            if value not in allowed:
                allowed_text = ", ".join(str(item) for item in allowed)
                raise ValueError(f"must be one of: {allowed_text}")

        return value

    def to_frontend(self) -> dict[str, Any]:
        """Return a JSON-serialisable payload for the frontend templates."""

        return {
            "type": self.type_name(),
            "default": self.default,
            "min": self.min_value,
            "max": self.max_value,
            "step": self.step,
            "label": self.label,
            "help_text": self.help_text,
            "options": [option.to_frontend() for option in self.options or ()],
        }


MODEL_CONFIGS: dict[str, dict[str, ParameterSchema]] = {
    "briaai/RMBG-2.0": {
        "input_size": ParameterSchema(
            value_type=int,
            default=1024,
            min_value=256,
            max_value=2048,
            step=64,
            label="Input size",
            help_text="Resize the longest edge before inference. Larger values improve detail but increase processing time.",
        ),
        "output_format": ParameterSchema(
            value_type=str,
            default="rgba",
            options=(
                ParameterOption("mask", "Mask (alpha only)"),
                ParameterOption("rgba", "RGBA (transparent)"),
            ),
            label="Output format",
            help_text="Choose between a binary mask or a transparent RGBA composite.",
        ),
        "matte_strength": ParameterSchema(
            value_type=float,
            default=1.0,
            min_value=0.1,
            max_value=2.0,
            step=0.1,
            label="Matte strength",
            help_text="Fine-tune the softness of the predicted matte. Higher values yield softer edges.",
        ),
        "device": ParameterSchema(
            value_type=str,
            default="auto",
            options=(
                ParameterOption("auto", "Auto"),
                ParameterOption("cpu", "CPU"),
                ParameterOption("cuda", "CUDA"),
            ),
            label="Inference device",
            help_text="Select the preferred execution provider when multiple accelerators are available.",
        ),
    },
    "matting-by-generation": {
        "detail_level": ParameterSchema(
            value_type=str,
            default="high",
            options=(
                ParameterOption("standard", "Standard"),
                ParameterOption("high", "High"),
                ParameterOption("ultra", "Ultra"),
            ),
            label="Detail level",
            help_text="Adjust the intensity of detail preservation for hair and semi-transparent regions.",
        ),
        "mask_blur": ParameterSchema(
            value_type=float,
            default=1.0,
            min_value=0.0,
            max_value=10.0,
            step=0.5,
            label="Mask blur",
            help_text="Apply additional blur to the predicted matte. Helpful when blending subjects into new backgrounds.",
        ),
        "threshold": ParameterSchema(
            value_type=float,
            default=0.5,
            min_value=0.0,
            max_value=1.0,
            step=0.05,
            label="Confidence threshold",
            help_text="Pixels below this confidence are treated as background for a crisper composite.",
        ),
    },
    "sam_segmentation_model": {
        "precision": ParameterSchema(
            value_type=str,
            default="auto",
            options=(
                ParameterOption("auto", "Auto"),
                ParameterOption("fp16", "FP16"),
                ParameterOption("fp32", "FP32"),
            ),
            label="Precision mode",
            help_text="Lower precision can improve performance on GPUs with limited memory.",
        ),
        "segmentation_points": ParameterSchema(
            value_type=int,
            default=8,
            min_value=1,
            max_value=24,
            step=1,
            label="Segmentation points",
            help_text="Controls how many prompt points are sampled when generating masks.",
        ),
        "prompt_mode": ParameterSchema(
            value_type=str,
            default="automatic",
            options=(
                ParameterOption("automatic", "Automatic"),
                ParameterOption("foreground", "Foreground-focused"),
                ParameterOption("background", "Background-focused"),
            ),
            label="Prompt mode",
            help_text="Select the prompting strategy used when generating segments for complex scenes.",
        ),
    },
    "u2net": {},
    "u2net_human_seg": {},
    "isnet-general-use": {},
    "isnet-anime": {},
}
"""Mapping of model identifiers to their optional configuration schemas."""


def serialise_model_configs() -> dict[str, dict[str, dict[str, Any]]]:
    """Return a JSON-friendly representation of :data:`MODEL_CONFIGS`."""

    return {
        model_name: {param: schema.to_frontend() for param, schema in params.items()}
        for model_name, params in MODEL_CONFIGS.items()
    }


def parse_model_options(
    model_name: str, raw_values: Mapping[str, Any] | None
) -> dict[str, Any]:
    """Validate and normalise ``raw_values`` for ``model_name``.

    Raises:
        ValueError: If any supplied parameter fails validation or an unknown
            parameter is provided.
    """

    schema = MODEL_CONFIGS.get(model_name)
    if not schema:
        return {}

    raw_values = dict(raw_values or {})
    unexpected = set(raw_values) - set(schema)
    if unexpected:
        names = ", ".join(sorted(unexpected))
        raise ValueError(f"Unsupported parameter(s) for {model_name}: {names}")

    parsed: dict[str, Any] = {}
    for key, parameter in schema.items():
        raw = raw_values.get(key)
        try:
            parsed[key] = parameter.coerce(raw)
        except ValueError as exc:  # pragma: no cover - defensive conversion guard
            raise ValueError(f"Invalid value for {key}: {exc}") from exc

    return parsed


# Adding a new configuration entry is as simple as defining a new key in
# ``MODEL_CONFIGS``. Provide a :class:`ParameterSchema` for each model-specific
# parameter you want to expose, then reference its ``default`` value in
# inference code as needed. The UI automatically renders the declared fields.
