"""Utilities for removing image backgrounds using ``rembg`` sessions."""
from __future__ import annotations

import base64
import io
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional

import numpy as np
from PIL import Image, ImageFilter, ImageOps

try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    cv2 = None  # type: ignore

try:  # pragma: no cover - optional dependency during testing
    from rembg import new_session as _rembg_new_session
    from rembg import remove as _rembg_remove
except ModuleNotFoundError as exc:  # pragma: no cover - propagated at runtime
    _rembg_remove = None
    _rembg_new_session = None
    _REMBG_IMPORT_ERROR = exc
else:
    _REMBG_IMPORT_ERROR = None

try:  # pragma: no cover - optional dependency when Eventlet is unavailable
    from eventlet.green import threading as cooperative_threading  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - Eventlet not installed in some environments
    import threading as cooperative_threading  # type: ignore

from app.services import runtime_compat
from app.services.accelerator import (
    detect_gpu_name,
    describe_selected_provider,
    is_rtx_50xx,
    onnx_providers_available,
    pick_execution_provider,
)


LOGGER = logging.getLogger(__name__)

Session = Any

ProgressCallback = Callable[[str, float], None]
PreviewCallback = Callable[[Image.Image, str], None]

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
MAX_WORK_DIMENSION = 8000


@dataclass(frozen=True)
class AcceleratorStatus:
    """Describe the currently selected execution provider and GPU metadata."""

    provider: str
    provider_options: Mapping[str, Any]
    available_providers: tuple[str, ...]
    gpu_name: Optional[str]
    rtx_50_series: bool
    warning: Optional[str]
    requested_mode: str
    provider_description: str

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable representation of the accelerator status."""

        return {
            "providers": list(self.available_providers),
            "selected": self.provider,
            "gpu_name": self.gpu_name,
            "rtx_50_series": self.rtx_50_series,
            "warning": self.warning,
            "mode": self.requested_mode,
            "provider_description": self.provider_description,
        }


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

_OUTPUT_FORMAT_LOOKUP: Dict[str, OutputFormat] = {}
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


def list_output_format_choices() -> List[str]:
    """Return the canonical keys for supported export formats."""

    return sorted({format_spec.key for format_spec in OUTPUT_FORMATS})


def get_mime_type_for_path(path: Path) -> str:
    """Return the MIME type for ``path`` based on its suffix."""

    try:
        spec = get_output_format_spec(path.suffix)
    except ValueError:
        return "application/octet-stream"
    return spec.mime_type


def _resolve_output_directory(output_dir: Optional[str | Path]) -> Path:
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

_SESSION_SINGLETON: Optional[Session] = None
_SESSION_LOCK = cooperative_threading.Lock()
_SESSION_CONFIG: Dict[str, Any] = {}

_DEFAULT_ACCELERATOR_STATUS = AcceleratorStatus(
    provider="cpu",
    provider_options={},
    available_providers=("CPUExecutionProvider",),
    gpu_name=None,
    rtx_50_series=False,
    warning=None,
    requested_mode="auto",
    provider_description="CPUExecutionProvider",
)
_ACCELERATOR_STATUS = _DEFAULT_ACCELERATOR_STATUS
_CPU_WARNING_EMITTED = False


@dataclass
class RemovalResult:
    """Represents the outcome of processing a single file."""

    path_in: Path
    path_out: Optional[Path]
    success: bool
    error: Optional[str]
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


def _normalise_accelerator_config(config: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Return a normalised accelerator configuration dictionary."""

    config = dict(config or {})
    mode = str(config.get("BG_ACCELERATOR", "auto") or "auto").strip().lower()
    try:
        device_id = int(config.get("BG_CUDA_DEVICE_ID", 0))
    except (TypeError, ValueError):
        device_id = 0

    raw_warn = config.get("BG_WARN_ON_CPU", True)
    if isinstance(raw_warn, str):
        warn_on_cpu = raw_warn.strip().lower() in {"1", "true", "yes", "on"}
    else:
        warn_on_cpu = bool(raw_warn)

    return {
        "BG_ACCELERATOR": mode or "auto",
        "BG_CUDA_DEVICE_ID": device_id,
        "BG_WARN_ON_CPU": warn_on_cpu,
    }


def _should_warn_on_cpu(mode: str, warn_on_cpu: bool) -> bool:
    """Return ``True`` when a CPU warning should be emitted."""

    return warn_on_cpu and mode in {"auto", "cuda"}


def _emit_cpu_warning_once(message: Optional[str], providers: tuple[str, ...], gpu_name: Optional[str]) -> None:
    """Log ``message`` only once per process to avoid noisy warnings."""

    global _CPU_WARNING_EMITTED
    if not message or _CPU_WARNING_EMITTED:
        return

    LOGGER.warning(message)

    if gpu_name and "CUDAExecutionProvider" not in providers:
        LOGGER.warning(
            "Detected %s but CUDAExecutionProvider is unavailable. Install onnxruntime-gpu>=1.18.1, "
            "enable the NVIDIA Container Toolkit, and launch with --gpus all when using Docker.",
            gpu_name,
        )
    elif gpu_name:
        LOGGER.warning("Detected GPU %s but running on CPU; performance will be slower.", gpu_name)
    else:
        LOGGER.warning("No compatible NVIDIA GPU detected; continuing with CPU execution.")

    _CPU_WARNING_EMITTED = True


def _update_accelerator_status(status: AcceleratorStatus) -> None:
    """Store ``status`` as the global accelerator status snapshot."""

    global _ACCELERATOR_STATUS
    _ACCELERATOR_STATUS = status


def get_accelerator_status() -> AcceleratorStatus:
    """Return the most recently observed accelerator status."""

    return _ACCELERATOR_STATUS


def create_session(model_name: str = "u2net", config: Optional[Mapping[str, Any]] = None) -> Session:
    """Create a new ``rembg`` session with safe GPU/CPU configuration."""

    if _rembg_new_session is None:
        raise RuntimeError("rembg is required to create a background removal session.") from _REMBG_IMPORT_ERROR

    runtime_compat.ensure_runtime_ready()

    normalised_config = _normalise_accelerator_config(config)
    mode = normalised_config["BG_ACCELERATOR"]
    device_id = normalised_config["BG_CUDA_DEVICE_ID"]
    warn_on_cpu = bool(normalised_config["BG_WARN_ON_CPU"])

    available_providers = tuple(onnx_providers_available())
    providers_label = ", ".join(available_providers) if available_providers else "<none>"
    LOGGER.info("Available ONNXRuntime providers: %s", providers_label)

    provider, provider_options = pick_execution_provider(mode, device_id)
    provider_description = describe_selected_provider(provider, provider_options)

    gpu_name = detect_gpu_name()
    if gpu_name:
        flair = " (RTX 50-series)" if is_rtx_50xx(gpu_name) else ""
        LOGGER.info("Detected NVIDIA GPU: %s%s", gpu_name, flair)
    else:
        LOGGER.info("No NVIDIA GPU detected via NVML/torch probes")

    warning_message: Optional[str] = None
    providers_config: List[Any]

    if provider == "cuda":
        providers_config = [
            ("CUDAExecutionProvider", dict(provider_options)),
            "CPUExecutionProvider",
        ]
    else:
        providers_config = ["CPUExecutionProvider"]
        if gpu_name and "CUDAExecutionProvider" not in available_providers:
            log_fn = LOGGER.warning if warn_on_cpu else LOGGER.info
            log_fn(
                "Detected %s but CUDAExecutionProvider was not reported by ONNXRuntime.",
                gpu_name,
            )
        if _should_warn_on_cpu(mode, warn_on_cpu):
            if gpu_name:
                warning_message = "Running on CPU because CUDAExecutionProvider is unavailable. Performance will be slower."
            else:
                warning_message = "Running on CPU because no compatible GPU was detected. Performance will be slower."

    LOGGER.info("Selected execution provider: %s", provider_description)

    try:
        session = _rembg_new_session(model_name, providers=providers_config)
    except Exception as exc:
        LOGGER.exception("Failed to initialise %s: %s", provider_description, exc)
        providers_config = ["CPUExecutionProvider"]
        provider = "cpu"
        provider_options = {}
        provider_description = "CPUExecutionProvider"
        if _should_warn_on_cpu(mode, warn_on_cpu):
            warning_message = (
                "Falling back to CPU because GPU session initialisation failed. Performance will be slower."
            )
        session = _rembg_new_session(model_name, providers=providers_config)

    rtx_flair = bool(gpu_name and is_rtx_50xx(gpu_name))
    status = AcceleratorStatus(
        provider=provider,
        provider_options=dict(provider_options),
        available_providers=available_providers,
        gpu_name=gpu_name,
        rtx_50_series=rtx_flair,
        warning=warning_message,
        requested_mode=mode,
        provider_description=provider_description,
    )
    _update_accelerator_status(status)
    _emit_cpu_warning_once(warning_message, available_providers, gpu_name)

    return session


def ensure_global_session(
    model_name: str = "u2net", config: Optional[Mapping[str, Any]] = None
) -> Session:
    """Initialise and cache a global ``rembg`` session."""

    global _SESSION_SINGLETON, _SESSION_CONFIG
    normalised_config = _normalise_accelerator_config(config)
    with _SESSION_LOCK:
        if _SESSION_SINGLETON is None or _SESSION_CONFIG != normalised_config:
            _SESSION_SINGLETON = create_session(model_name, normalised_config)
            _SESSION_CONFIG = normalised_config
    assert _SESSION_SINGLETON is not None
    return _SESSION_SINGLETON


def _get_session(session: Optional[Session] = None) -> Session:
    """Return the provided session or the cached singleton."""

    if session is not None:
        return session
    return ensure_global_session()


def build_colorkey_mask(image: Image.Image, tolerance: int = 14) -> Optional[np.ndarray]:
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


def _run_rembg(image: Image.Image, session: Session) -> Image.Image:
    """Execute ``rembg.remove`` and return an RGBA mask image."""

    if _rembg_remove is None:
        raise RuntimeError("rembg is required to process images.") from _REMBG_IMPORT_ERROR

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    result_bytes = _rembg_remove(buffer.read(), session=session)
    result_stream = io.BytesIO(result_bytes)
    result_image = Image.open(result_stream).convert("RGBA")
    result_image.load()
    return result_image


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
            refined = cv2.erode(refined, kernel, iterations=1)
        else:
            filter_size = max(3, (erode_size // 2) * 2 + 1)
            mask_image = Image.fromarray(refined)
            for _ in range(max(1, erode_size // 3 + 1)):
                mask_image = mask_image.filter(ImageFilter.MinFilter(filter_size))
            refined = np.asarray(mask_image, dtype=np.uint8)

    return refined


def remove_bg_file(
    input_path: str | Path,
    output_path: Optional[str | Path] = None,
    *,
    session: Optional[Session] = None,
    alpha_matting: bool = False,
    am_foreground: int = 240,
    am_background: int = 10,
    am_erode: int = 10,
    use_colorkey_fallback: bool = True,
    colorkey_tolerance: int = 14,
    feather_radius: int = 3,
    progress_callback: Optional[ProgressCallback] = None,
    preview_callback: Optional[PreviewCallback] = None,
    output_format: Optional[str] = None,
) -> RemovalResult:
    """Remove the background from ``input_path`` and export the chosen format."""

    session = _get_session(session)
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
    error: Optional[str] = None

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
                mask_image = _run_rembg(resized, session)
                alpha_channel = mask_image.split()[-1]
                alpha_channel = alpha_channel.resize(rgba_source.size, Image.Resampling.LANCZOS)
                emit_progress("remove_background", 40.0)
            else:
                mask_image = _run_rembg(work_image, session)
                alpha_channel = mask_image.split()[-1]
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
    output_dir: Optional[str | Path] = None,
    output_format: Optional[str] = None,
    *,
    session: Optional[Session] = None,
    recursive: bool = False,
    alpha_matting: bool = False,
    am_foreground: int = 240,
    am_background: int = 10,
    am_erode: int = 10,
    use_colorkey_fallback: bool = True,
    colorkey_tolerance: int = 14,
    feather_radius: int = 3,
) -> List[RemovalResult]:
    """Process every supported image in ``input_dir`` sequentially."""

    session = _get_session(session)
    input_path = Path(input_dir)
    if not input_path.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {input_path}")

    output_path = _resolve_output_directory(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    format_spec = get_output_format_spec(output_format)

    results: List[RemovalResult] = []
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
