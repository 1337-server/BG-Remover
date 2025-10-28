"""Utilities for removing image backgrounds using ONNX Runtime sessions."""
from __future__ import annotations

import base64
import logging
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image, ImageFilter, ImageOps

try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    cv2 = None  # type: ignore

try:  # pragma: no cover - optional dependency when Eventlet is unavailable
    from eventlet.green import threading as cooperative_threading  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - Eventlet not installed in some environments
    import threading as cooperative_threading  # type: ignore

try:  # pragma: no cover - optional dependency during tests
    import onnxruntime as ort
except ModuleNotFoundError:  # pragma: no cover - resolved by runtime checks
    ort = None  # type: ignore[assignment]

if TYPE_CHECKING:  # pragma: no cover - typing assistance only
    from onnxruntime import InferenceSession as Session
else:
    Session = Any  # type: ignore[assignment]

from app.services import accelerator, model_registry, runtime_compat

LOGGER = logging.getLogger(__name__)

ProgressCallback = Callable[[str, float], None]
PreviewCallback = Callable[[Image.Image, str], None]

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
MAX_WORK_DIMENSION = 8000

# The background-removal service supports a curated list of ONNX models. These
# names mirror the defaults exposed by the rembg command-line interface and
# cover common use-cases (general photography, portraits, and anime artwork).
_PRELOAD_MODEL_NAMES: tuple[str, ...] = (
    "u2net",
    "u2netp",
    "isnet-general-use",
    "isnet-anime",
)


@dataclass(frozen=True)
class ModelSpec:
    """Describes normalisation parameters for a supported ONNX model."""

    input_size: tuple[int, int]
    mean: tuple[float, float, float]
    std: tuple[float, float, float]


_MODEL_SPECS: dict[str, ModelSpec] = {
    "u2net": ModelSpec((320, 320), (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    "u2netp": ModelSpec((320, 320), (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    "u2net_human_seg": ModelSpec((320, 320), (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    "isnet-general-use": ModelSpec((1024, 1024), (0.5, 0.5, 0.5), (1.0, 1.0, 1.0)),
    "isnet-anime": ModelSpec((1024, 1024), (0.485, 0.456, 0.406), (1.0, 1.0, 1.0)),
}

_DEFAULT_MODEL_SPEC = ModelSpec((320, 320), (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


def _get_model_spec(model_name: str) -> ModelSpec:
    """Return the normalisation parameters for ``model_name``."""

    return _MODEL_SPECS.get(model_name, _DEFAULT_MODEL_SPEC)


def sanitize_mask(values: np.ndarray) -> np.ndarray:
    """Clamp the network output to the valid probability range."""

    sanitised = np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(sanitised, 0.0, 1.0)


@dataclass(frozen=True)
class OutputFormat:
    """Descriptor for a supported export format."""

    key: str
    extension: str
    label: str
    pil_format: str
    mime_type: str
    supports_alpha: bool
    aliases: tuple[str, ...] = ()
    save_kwargs: Mapping[str, Any] = field(default_factory=dict)

    def normalise_filename(self, path: Path) -> Path:
        """Return ``path`` with the correct extension applied."""

        return path.with_suffix(self.extension)


DEFAULT_OUTPUT_FORMAT = "png"

OUTPUT_FORMATS: tuple[OutputFormat, ...] = (
    OutputFormat(
        key="png",
        extension=".png",
        label="PNG (lossless, supports transparency)",
        pil_format="PNG",
        mime_type="image/png",
        supports_alpha=True,
        save_kwargs={"optimize": True},
    ),
    OutputFormat(
        key="webp",
        extension=".webp",
        label="WebP (modern, supports transparency)",
        pil_format="WEBP",
        mime_type="image/webp",
        supports_alpha=True,
        save_kwargs={"lossless": True},
    ),
    OutputFormat(
        key="jpg",
        extension=".jpg",
        label="JPG (lossy, no transparency)",
        pil_format="JPEG",
        mime_type="image/jpeg",
        supports_alpha=False,
        aliases=("jpeg",),
        save_kwargs={"quality": 95},
    ),
    OutputFormat(
        key="bmp",
        extension=".bmp",
        label="BMP (uncompressed, no transparency)",
        pil_format="BMP",
        mime_type="image/bmp",
        supports_alpha=False,
    ),
    OutputFormat(
        key="tiff",
        extension=".tiff",
        label="TIFF (lossless, supports transparency)",
        pil_format="TIFF",
        mime_type="image/tiff",
        supports_alpha=True,
        aliases=("tif",),
    ),
)

_OUTPUT_FORMAT_LOOKUP: dict[str, OutputFormat] = {}
for _format in OUTPUT_FORMATS:
    _OUTPUT_FORMAT_LOOKUP[_format.key] = _format
    _OUTPUT_FORMAT_LOOKUP[_format.key.lower()] = _format
    _OUTPUT_FORMAT_LOOKUP[_format.extension.lstrip(".").lower()] = _format
    for _alias in _format.aliases:
        alias_key = _alias.lower().lstrip(".")
        _OUTPUT_FORMAT_LOOKUP[alias_key] = _format


def get_output_format_spec(value: str | None) -> OutputFormat:
    """Return the :class:`OutputFormat` matching ``value``.

    ``value`` may be a canonical key such as ``"png"`` or an extension such as
    ``".png"``. When ``value`` is ``None`` or empty the default format is
    returned. A :class:`ValueError` is raised for unsupported formats.
    """

    if value is None:
        key = DEFAULT_OUTPUT_FORMAT
    else:
        key = str(value).strip().lower()
        key = key.lstrip(".")
        if not key:
            key = DEFAULT_OUTPUT_FORMAT
    spec = _OUTPUT_FORMAT_LOOKUP.get(key)
    if spec is None:
        raise ValueError(f"Unsupported output format: {value}")
    return spec


def list_output_format_choices() -> list[str]:
    """Return the canonical keys for supported export formats."""

    return sorted({format_spec.key for format_spec in OUTPUT_FORMATS})


def get_mime_type_for_path(path: Path) -> str:
    """Return the MIME type for ``path`` based on its suffix."""

    try:
        spec = get_output_format_spec(path.suffix)
    except ValueError:
        return "application/octet-stream"
    return spec.mime_type


def _resolve_output_directory(output_dir: str | Path | None) -> Path:
    """Return an absolute output directory, defaulting to ``cwd / 'output'``."""

    if output_dir is None:
        base = Path.cwd() / "output"
    else:
        base = Path(output_dir).expanduser()
        if not base.is_absolute():
            base = Path.cwd() / base
    return base


def _resolve_output_path(path: str | Path) -> Path:
    """Return an absolute path for output files."""

    destination = Path(path).expanduser()
    if not destination.is_absolute():
        destination = Path.cwd() / destination
    return destination


def _looks_like_directory(original: str | Path, resolved: Path) -> bool:
    """Return ``True`` when ``original`` should be treated as a directory."""

    if resolved.exists():
        return resolved.is_dir()
    text = str(original)
    return text.endswith(("/", "\\"))

@dataclass
class SessionContext:
    """Container describing the active ONNX Runtime session and accelerator state."""

    model_name: str
    session: Session
    runtime: str
    provider: str
    provider_options: Mapping[str, Any]
    providers_available: list[str]
    gpu_name: str | None
    rtx_50_series: bool
    warning: str | None
    device_id: int
    gpu_available: bool
    accelerator_message: str | None

    def runtime_payload(self) -> dict[str, Any]:
        """Return a serialisable snapshot of the accelerator runtime."""

        return {
            "runtime": self.runtime,
            "provider": self.provider,
            "gpu_name": self.gpu_name,
            "warning": self.warning,
            "providers_available": self.providers_available,
            "gpu_available": self.gpu_available,
            "accelerator_message": self.accelerator_message,
        }


_SESSION_CONTEXT: SessionContext | None = None
_SESSION_CONFIG_SIGNATURE: tuple[str, int, bool] | None = None
_SESSION_LOCK = cooperative_threading.Lock()
# Guard access to the preloaded session pools to ensure thread-safety when the
# Flask application serves concurrent requests.
_POOL_LOCK = cooperative_threading.Lock()
# Cache of ``SessionContext`` objects grouped by accelerator configuration
# signature. Each entry stores model-name keys mapped to active ONNX sessions.
_SESSION_POOLS: dict[tuple[str, int, bool], dict[str, SessionContext]] = {}
# Track which accelerator signatures have already been preloaded to avoid
# re-running the expensive warm-up pipeline.
_PRELOADED_SIGNATURES: set[tuple[str, int, bool]] = set()
# Remember whether diagnostic provider logs have been emitted for each
# accelerator signature to prevent noisy, repeated log messages when multiple
# models are initialised.
_STARTUP_LOGGED_SIGNATURES: set[tuple[str, int, bool]] = set()
# Map ``id(session)`` to lightweight metadata so runtime logs can reference the
# active model and provider when reporting inference durations.
_SESSION_METADATA: dict[int, dict[str, str]] = {}
_CPU_WARNING_LOGGED = False
_CUDA_HINT_LOGGED = False


@dataclass
class RemovalResult:
    """Represents the outcome of processing a single file."""

    path_in: Path
    path_out: Path | None
    success: bool
    error: str | None
    timing_ms: float

    def to_dict(self) -> dict:
        """Return a serialisable representation."""

        return {
            "path_in": str(self.path_in),
            "path_out": str(self.path_out) if self.path_out else None,
            "success": self.success,
            "error": self.error,
            "timing_ms": self.timing_ms,
        }


def _normalise_mode(value: Any) -> str:
    """Return a valid accelerator mode string."""

    if value is None:
        return "auto"
    text = str(value).strip().lower()
    if text not in {"auto", "cuda", "cpu"}:
        return "auto"
    return text


def _coerce_int(value: Any, default: int) -> int:
    """Return ``value`` as an integer or ``default`` on failure."""

    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _coerce_bool(value: Any, default: bool) -> bool:
    """Return ``value`` as a boolean with sensible string handling."""

    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    try:
        return bool(int(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default

def _extract_session_config(config: Mapping[str, Any] | None) -> tuple[str, int, bool]:
    """Return normalised accelerator configuration values."""

    source = config or {}
    mode = _normalise_mode(source.get("BG_ACCELERATOR"))
    device_id = _coerce_int(source.get("BG_CUDA_DEVICE_ID", 0), 0)
    warn_on_cpu = _coerce_bool(source.get("BG_WARN_ON_CPU", True), True)
    if runtime_compat.is_force_cpu_enabled():
        mode = "cpu"
    return mode, device_id, warn_on_cpu


def _signature_for_config(config: Mapping[str, Any] | None) -> tuple[str, int, bool]:
    """Return a cache signature for accelerator-related configuration."""

    mode, device_id, warn_on_cpu = _extract_session_config(config)
    return mode, device_id, warn_on_cpu


def _register_session_context(context: SessionContext) -> None:
    """Store metadata describing ``context`` for diagnostic logging."""

    _SESSION_METADATA[id(context.session)] = {
        "model": context.model_name,
        "provider": context.provider,
        "runtime": context.runtime,
    }


def _describe_session(session: Session) -> str:
    """Return a short description of the ONNX session for logging."""

    metadata = _SESSION_METADATA.get(id(session))
    if not metadata:
        return "background removal session"
    model = metadata.get("model", "unknown model")
    provider = metadata.get("provider", "unknown provider")
    runtime = metadata.get("runtime")
    if runtime:
        return f"{model} via {provider} ({runtime})"
    return f"{model} via {provider}"


def _session_model_name(session: Session) -> str:
    """Return the model name associated with ``session``."""

    metadata = _SESSION_METADATA.get(id(session))
    return metadata.get("model", "u2net") if metadata else "u2net"


def _warm_up_session(context: SessionContext) -> None:
    """Execute a one-time warm-up inference for ``context`` to prime CUDA kernels."""

    if ort is None:
        LOGGER.debug("Skipping warm-up because onnxruntime is unavailable")
        return

    dummy_size = _get_model_spec(context.model_name).input_size
    dummy_image = Image.new("RGB", dummy_size, color=(0, 0, 0))

    try:
        start = time.perf_counter()
        _run_inference(dummy_image, context.session, context.model_name, log_timing=False)
        elapsed_ms = (time.perf_counter() - start) * 1000
        LOGGER.info(
            "Warm-up inference for %s completed in %.2f ms (%s)",
            context.model_name,
            elapsed_ms,
            context.provider,
        )
    except Exception:
        LOGGER.warning(
            "Warm-up inference for %s failed; continuing without GPU priming",
            context.model_name,
            exc_info=True,
        )


def _initialise_session_context(
    model_name: str,
    *,
    requested_mode: str,
    device_id: int,
    warn_on_cpu: bool,
    log_diagnostics: bool,
) -> SessionContext:
    """Return a ready-to-use :class:`SessionContext` for ``model_name``."""

    model_registry.preload_models(config={"BG_ACCELERATOR": requested_mode, "BG_CUDA_DEVICE_ID": device_id})
    session_obj = model_registry.get_session(model_name)
    if session_obj is None:
        raise FileNotFoundError(
            f"Model '{model_name}' is not available. Download weights into {Path.home() / '.u2net'}."
        )

    providers_available = model_registry.get_available_providers() or accelerator.onnx_providers_available()
    resolved_providers = list(getattr(session_obj, "get_providers", lambda: [])())  # type: ignore[call-arg]
    provider_name = resolved_providers[0] if resolved_providers else "CPUExecutionProvider"

    gpu_name = accelerator.detect_gpu_name()
    rtx_50_series = bool(gpu_name and accelerator.is_rtx_50xx(gpu_name))
    runtime = "cuda" if provider_name in {"CUDAExecutionProvider", "TensorrtExecutionProvider"} else "cpu"
    gpu_available = runtime == "cuda"

    if log_diagnostics:
        LOGGER.info("Available ONNX Runtime providers: %s", ", ".join(providers_available) or "none")
        LOGGER.info("Selected execution provider: %s", provider_name)
        if gpu_name:
            suffix = " (RTX 50-series detected)" if rtx_50_series else ""
            LOGGER.info("Detected GPU: %s%s", gpu_name, suffix)
        else:
            LOGGER.info("No NVIDIA GPU detected")
        LOGGER.info("Accelerator preference: %s", requested_mode)

    global _CUDA_HINT_LOGGED
    if gpu_name and "CUDAExecutionProvider" not in providers_available and not _CUDA_HINT_LOGGED:
        LOGGER.warning(
            "Detected GPU %s but CUDAExecutionProvider is unavailable. Install a matching "
            "onnxruntime-gpu wheel.",
            gpu_name,
        )
        _CUDA_HINT_LOGGED = True

    accelerator_message: str | None
    warning_message: str | None
    if requested_mode in {"cuda", "auto"} and gpu_available:
        accelerator_message = f"Using GPU ({provider_name})"
        warning_message = None
    elif requested_mode in {"cuda", "auto"} and warn_on_cpu:
        accelerator_message = "GPU requested but unavailable — falling back to CPU"
        warning_message = accelerator_message
    else:
        accelerator_message = f"Using CPU ({provider_name})"
        warning_message = None

    global _CPU_WARNING_LOGGED
    if warning_message and not _CPU_WARNING_LOGGED:
        LOGGER.warning(
            "GPU acceleration requested but unavailable; verify that onnxruntime-gpu >= 1.20.1 is installed",
        )
        _CPU_WARNING_LOGGED = True

    provider_options: dict[str, Any] = {"device_id": int(device_id)} if gpu_available else {}

    context = SessionContext(
        model_name=model_name,
        session=session_obj,
        runtime=runtime,
        provider=provider_name,
        provider_options=provider_options,
        providers_available=providers_available,
        gpu_name=gpu_name,
        rtx_50_series=rtx_50_series,
        warning=warning_message,
        device_id=device_id,
        gpu_available=gpu_available,
        accelerator_message=accelerator_message,
    )
    _register_session_context(context)
    _warm_up_session(context)
    LOGGER.info(
        "Model '%s' initialised using provider %s (runtime=%s)",
        model_name,
        provider_name,
        runtime,
    )
    return context


def _preload_default_models(
    signature: tuple[str, int, bool],
    *,
    requested_mode: str,
    device_id: int,
    warn_on_cpu: bool,
) -> None:
    """Preload the curated list of models for the provided accelerator signature."""

    model_registry.preload_models(
        config={"BG_ACCELERATOR": requested_mode, "BG_CUDA_DEVICE_ID": device_id},
        model_names=_PRELOAD_MODEL_NAMES,
    )
    pool = _SESSION_POOLS.setdefault(signature, {})
    logged = signature in _STARTUP_LOGGED_SIGNATURES
    for model_name in _PRELOAD_MODEL_NAMES:
        if model_name in pool:
            continue
        context = _initialise_session_context(
            model_name,
            requested_mode=requested_mode,
            device_id=device_id,
            warn_on_cpu=warn_on_cpu,
            log_diagnostics=not logged,
        )
        pool[model_name] = context
        if not logged:
            _STARTUP_LOGGED_SIGNATURES.add(signature)
            logged = True

def create_session(
    model_name: str = "u2net", config: Mapping[str, Any] | None = None
) -> SessionContext:
    """Create a new background removal session with optional GPU acceleration."""

    requested_mode, device_id, warn_on_cpu = _extract_session_config(config)
    runtime_compat.ensure_runtime_ready()

    signature = (requested_mode, device_id, warn_on_cpu)
    with _POOL_LOCK:
        pool = _SESSION_POOLS.setdefault(signature, {})
        if signature not in _PRELOADED_SIGNATURES:
            _preload_default_models(
                signature,
                requested_mode=requested_mode,
                device_id=device_id,
                warn_on_cpu=warn_on_cpu,
            )
            _PRELOADED_SIGNATURES.add(signature)

        context = pool.get(model_name)
        log_required = signature not in _STARTUP_LOGGED_SIGNATURES
        if context is None:
            context = _initialise_session_context(
                model_name,
                requested_mode=requested_mode,
                device_id=device_id,
                warn_on_cpu=warn_on_cpu,
                log_diagnostics=log_required,
            )
            pool[model_name] = context
        if log_required:
            _STARTUP_LOGGED_SIGNATURES.add(signature)

    return context


def ensure_global_session(
    model_name: str = "u2net", config: Mapping[str, Any] | None = None
) -> Session:
    """Initialise and cache a global background removal session."""

    global _SESSION_CONTEXT, _SESSION_CONFIG_SIGNATURE
    signature = _signature_for_config(config)
    with _SESSION_LOCK:
        if _SESSION_CONTEXT is None or _SESSION_CONFIG_SIGNATURE != signature:
            _SESSION_CONTEXT = create_session(model_name=model_name, config=config)
            _SESSION_CONFIG_SIGNATURE = signature
    assert _SESSION_CONTEXT is not None
    return _SESSION_CONTEXT.session


def _get_session(session: Session | None = None) -> Session:
    """Return the provided session or the cached singleton."""

    if session is not None:
        return session
    context = _SESSION_CONTEXT
    if context is None:
        return ensure_global_session()
    return context.session


def get_session_context() -> SessionContext | None:
    """Return the cached :class:`SessionContext`, if initialised."""

    return _SESSION_CONTEXT


def get_runtime_payload() -> dict[str, Any]:
    """Return runtime metadata for embedding in API responses."""

    context = _SESSION_CONTEXT
    if context is None:
        providers_available = accelerator.onnx_providers_available()
        return {
            "runtime": "cpu",
            "provider": "CPUExecutionProvider",
            "gpu_name": None,
            "warning": None,
            "providers_available": providers_available,
            "gpu_available": False,
            "accelerator_message": None,
        }
    return context.runtime_payload()


def get_accelerator_status() -> dict[str, Any]:
    """Return diagnostic accelerator details for health checks."""

    context = _SESSION_CONTEXT
    if context is None:
        providers = accelerator.onnx_providers_available()
        gpu_name = accelerator.detect_gpu_name()
        runtime = "cpu"
        rtx = bool(gpu_name and accelerator.is_rtx_50xx(gpu_name)) if gpu_name else False
        provider_name = "CPUExecutionProvider"
    else:
        providers = context.providers_available
        gpu_name = context.gpu_name
        runtime = context.runtime
        rtx = context.rtx_50_series
        provider_name = context.provider
    return {
        "providers": providers,
        "selected": "cuda" if runtime == "cuda" else "cpu",
        "provider": provider_name,
        "gpu_name": gpu_name,
        "rtx_50_series": rtx,
    }


def build_colorkey_mask(image: Image.Image, tolerance: int = 14) -> np.ndarray | None:
    """Return a colour-key alpha mask when a near-solid background is detected."""

    if tolerance <= 0:
        return None

    rgb_image = image.convert("RGB")
    np_image = np.asarray(rgb_image, dtype=np.uint8)
    height, width, _ = np_image.shape
    patch = max(5, min(height, width) // 10)
    patch = min(patch, height, width)
    if patch <= 0:
        return None

    corners = [
        np_image[0:patch, 0:patch],
        np_image[0:patch, width - patch : width],
        np_image[height - patch : height, 0:patch],
        np_image[height - patch : height, width - patch : width],
    ]
    corner_means = np.array([corner.reshape(-1, 3).mean(axis=0) for corner in corners])
    background_colour = corner_means.mean(axis=0)
    deviations = [np.abs(corner - background_colour).max() for corner in corners]
    if max(deviations) > tolerance * 1.5:
        return None

    delta = np.max(np.abs(np_image.astype(np.int16) - background_colour.astype(np.int16)), axis=2)
    mask = np.where(delta <= tolerance, 0, 255).astype(np.uint8)
    return mask


def _feather_alpha(alpha: np.ndarray, radius: int) -> np.ndarray:
    """Apply Gaussian blur to soften mask edges when OpenCV is available."""

    if radius <= 0 or cv2 is None:
        return alpha
    kernel = radius * 2 + 1
    try:
        blurred = cv2.GaussianBlur(alpha, (kernel, kernel), 0)  # type: ignore[arg-type]
    except Exception:
        return alpha
    return blurred.astype(np.uint8)


def _run_inference(
    image: Image.Image,
    session: Session,
    model_name: str,
    *,
    log_timing: bool = True,
) -> Image.Image:
    """Execute ONNX Runtime inference and return an alpha mask image."""

    spec = _get_model_spec(model_name)
    resized = image.convert("RGB").resize(spec.input_size, Image.Resampling.LANCZOS)
    np_image = np.asarray(resized, dtype=np.float32)
    max_value = float(np.max(np_image)) or 1.0
    np_image /= max_value

    normalised = np.empty_like(np_image, dtype=np.float32)
    for index in range(3):
        normalised[:, :, index] = (np_image[:, :, index] - spec.mean[index]) / spec.std[index]

    tensor = normalised.transpose((2, 0, 1))[np.newaxis, ...].astype(np.float32)
    input_name = session.get_inputs()[0].name
    feed = {input_name: tensor}

    start = time.perf_counter()
    outputs = session.run(None, feed)
    elapsed_ms = (time.perf_counter() - start) * 1000
    if log_timing:
        LOGGER.info("%s inference completed in %.2f ms", _describe_session(session), elapsed_ms)

    pred = sanitize_mask(outputs[0][:, 0, :, :])
    pred = np.squeeze(pred)
    mask = Image.fromarray((pred * 255).astype(np.uint8), mode="L")
    mask = mask.resize(image.size, Image.Resampling.LANCZOS)
    return mask


def _apply_alpha_matting(
    alpha: np.ndarray,
    *,
    foreground_threshold: int,
    background_threshold: int,
    erode_size: int,
) -> np.ndarray:
    """Apply basic alpha refinement inspired by legacy rembg settings."""

    fg = int(np.clip(foreground_threshold, 0, 255))
    bg = int(np.clip(background_threshold, 0, 254))
    if fg <= bg:
        fg = min(255, bg + 1)

    normalized = alpha.astype(np.float32)
    normalized = np.clip(normalized - bg, 0, None)
    scale = max(fg - bg, 1)
    normalized = np.clip(normalized * (255.0 / scale), 0, 255)
    refined = normalized.astype(np.uint8)

    if erode_size > 0:
        if cv2 is not None:
            kernel = np.ones((erode_size, erode_size), dtype=np.uint8)
            refined = np.asarray(cv2.erode(refined, kernel, iterations=1), dtype=np.uint8)
        else:
            filter_size = max(3, (erode_size // 2) * 2 + 1)
            mask_image = Image.fromarray(refined)
            for _ in range(max(1, erode_size // 3 + 1)):
                mask_image = mask_image.filter(ImageFilter.MinFilter(filter_size))
            refined = np.asarray(mask_image, dtype=np.uint8)

    return refined


def remove_bg_file(
    input_path: str | Path,
    output_path: str | Path | None = None,
    *,
    session: Session | None = None,
    alpha_matting: bool = False,
    am_foreground: int = 240,
    am_background: int = 10,
    am_erode: int = 10,
    use_colorkey_fallback: bool = True,
    colorkey_tolerance: int = 14,
    feather_radius: int = 3,
    progress_callback: ProgressCallback | None = None,
    preview_callback: PreviewCallback | None = None,
    output_format: str | None = None,
) -> RemovalResult:
    """Remove the background from ``input_path`` and export the chosen format."""

    session = _get_session(session)
    model_name = _session_model_name(session)
    source = Path(input_path)
    format_spec = get_output_format_spec(output_format)
    if output_path is None:
        output_dir = _resolve_output_directory(None)
        destination = format_spec.normalise_filename(output_dir / source.name)
    else:
        resolved_output = _resolve_output_path(output_path)
        if _looks_like_directory(output_path, resolved_output):
            destination = format_spec.normalise_filename(resolved_output / source.name)
        else:
            destination = format_spec.normalise_filename(resolved_output)
    destination.parent.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    error: str | None = None

    def emit_progress(stage: str, percent: float) -> None:
        """Forward progress updates to the optional callback."""

        if progress_callback is None:
            return
        clamped = max(0.0, min(100.0, float(percent)))
        progress_callback(stage, clamped)

    try:
        emit_progress("load", 5.0)
        with Image.open(source) as raw_image:
            oriented = ImageOps.exif_transpose(raw_image)
            if oriented is None:
                oriented = raw_image.copy()
            rgba_source = oriented.convert("RGBA")
            work_image = rgba_source.convert("RGB")

            def emit_preview_from_alpha(alpha_data: np.ndarray, stage: str) -> None:
                """Send a downscaled preview constructed from the alpha mask."""

                if preview_callback is None:
                    return

                preview_image = rgba_source.copy()
                alpha_image = Image.fromarray(alpha_data, mode="L")
                preview_image.putalpha(alpha_image)
                if max(preview_image.size) > 1024:
                    preview_image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
                preview_callback(preview_image, stage)

            if max(work_image.size) > MAX_WORK_DIMENSION:
                resized = work_image.copy()
                resized.thumbnail((MAX_WORK_DIMENSION, MAX_WORK_DIMENSION), Image.Resampling.LANCZOS)
                emit_progress("resize", 15.0)
                mask_image = _run_inference(resized, session, model_name)
                alpha_channel = mask_image.resize(rgba_source.size, Image.Resampling.LANCZOS)
                emit_progress("remove_background", 40.0)
            else:
                mask_image = _run_inference(work_image, session, model_name)
                alpha_channel = mask_image
                emit_progress("remove_background", 40.0)

            alpha_np = np.asarray(alpha_channel, dtype=np.uint8)
            emit_progress("mask", 55.0)
            emit_preview_from_alpha(alpha_np, "initial")

            if alpha_matting:
                emit_progress("alpha_matting", 65.0)
                alpha_np = _apply_alpha_matting(
                    alpha_np,
                    foreground_threshold=am_foreground,
                    background_threshold=am_background,
                    erode_size=am_erode,
                )

            if use_colorkey_fallback:
                emit_progress("colorkey", 70.0)
                fallback_mask = build_colorkey_mask(rgba_source, tolerance=colorkey_tolerance)
                if fallback_mask is not None:
                    alpha_np = np.maximum(alpha_np, fallback_mask)

            emit_progress("feather", 75.0)
            alpha_np = _feather_alpha(alpha_np, feather_radius)
            emit_preview_from_alpha(alpha_np, "refined")
            final_alpha = Image.fromarray(alpha_np, mode="L")
            output_image = rgba_source.copy()
            output_image.putalpha(final_alpha)
            emit_progress("save", 90.0)
            save_kwargs = dict(format_spec.save_kwargs)
            if format_spec.supports_alpha:
                output_image.save(destination, format=format_spec.pil_format, **save_kwargs)
            else:
                background = Image.new("RGB", output_image.size, color=(255, 255, 255))
                background.paste(output_image, mask=final_alpha)
                background.save(destination, format=format_spec.pil_format, **save_kwargs)
                del background

            # Explicitly release large arrays to limit memory pressure.
            del alpha_np
            del final_alpha
            del output_image
    except Exception as exc:  # pragma: no cover - depends on external files
        error = str(exc)

    elapsed_ms = (time.perf_counter() - start) * 1000
    success = error is None
    if success:
        emit_progress("complete", 100.0)
    return RemovalResult(source, destination if success else None, success, error, elapsed_ms)


def _iter_input_files(input_dir: Path, recursive: bool) -> Iterable[Path]:
    """Yield input files matching supported extensions."""

    if recursive:
        iterator: Iterable[Path] = input_dir.rglob("*")
    else:
        iterator = input_dir.iterdir()
    for candidate in iterator:
        if candidate.is_file() and candidate.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield candidate


def remove_bg_folder(
    input_dir: str | Path,
    output_dir: str | Path | None = None,
    output_format: str | None = None,
    *,
    session: Session | None = None,
    recursive: bool = False,
    alpha_matting: bool = False,
    am_foreground: int = 240,
    am_background: int = 10,
    am_erode: int = 10,
    use_colorkey_fallback: bool = True,
    colorkey_tolerance: int = 14,
    feather_radius: int = 3,
) -> list[RemovalResult]:
    """Process every supported image in ``input_dir`` sequentially."""

    session = _get_session(session)
    input_path = Path(input_dir)
    if not input_path.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {input_path}")

    output_path = _resolve_output_directory(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    format_spec = get_output_format_spec(output_format)

    results: list[RemovalResult] = []
    for source in _iter_input_files(input_path, recursive):
        relative = source.relative_to(input_path) if recursive else Path(source.name)
        destination = format_spec.normalise_filename(output_path / relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        result = remove_bg_file(
            source,
            destination,
            session=session,
            alpha_matting=alpha_matting,
            am_foreground=am_foreground,
            am_background=am_background,
            am_erode=am_erode,
            use_colorkey_fallback=use_colorkey_fallback,
            colorkey_tolerance=colorkey_tolerance,
            feather_radius=feather_radius,
            output_format=format_spec.key,
        )
        results.append(result)
    return results


def encode_result_image(path: Path) -> str:
    """Return a base64-encoded representation of an output image."""

    with path.open("rb") as file_obj:
        data = file_obj.read()
    return base64.b64encode(data).decode("ascii")
