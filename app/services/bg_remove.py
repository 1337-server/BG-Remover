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

from app.services import accelerator, runtime_compat


LOGGER = logging.getLogger(__name__)

Session = Any

ProgressCallback = Callable[[str, float], None]
PreviewCallback = Callable[[Image.Image, str], None]

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
MAX_WORK_DIMENSION = 8000


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

@dataclass
class SessionContext:
    """Container describing the active rembg session and accelerator state."""

    session: Session
    runtime: str
    provider: str
    provider_options: Mapping[str, Any]
    providers_available: List[str]
    gpu_name: str | None
    rtx_50_series: bool
    warning: str | None
    device_id: int

    def runtime_payload(self) -> Dict[str, Any]:
        """Return a serialisable snapshot of the accelerator runtime."""

        return {
            "runtime": self.runtime,
            "gpu_name": self.gpu_name,
            "warning": self.warning,
        }


_SESSION_CONTEXT: Optional[SessionContext] = None
_SESSION_CONFIG_SIGNATURE: Optional[tuple[str, int, bool]] = None
_SESSION_LOCK = cooperative_threading.Lock()
_CPU_WARNING_LOGGED = False
_CUDA_HINT_LOGGED = False


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


def create_session(
    model_name: str = "u2net", config: Mapping[str, Any] | None = None
) -> SessionContext:
    """Create a new ``rembg`` session with optional GPU acceleration."""

    if _rembg_new_session is None:
        raise RuntimeError("rembg is required to create a background removal session.") from _REMBG_IMPORT_ERROR

    requested_mode, device_id, warn_on_cpu = _extract_session_config(config)
    runtime_compat.ensure_runtime_ready()

    providers_available = accelerator.onnx_providers_available()
    provider_key, provider_options = accelerator.pick_execution_provider(requested_mode, device_id)
    runtime = "cuda" if provider_key == "cuda" else "cpu"
    provider_name = "CUDAExecutionProvider" if runtime == "cuda" else "CPUExecutionProvider"
    providers_argument: List[Any]
    if runtime == "cuda":
        providers_argument = [("CUDAExecutionProvider", dict(provider_options or {})), "CPUExecutionProvider"]
    else:
        providers_argument = ["CPUExecutionProvider"]

    gpu_name = accelerator.detect_gpu_name()
    rtx_50_series = bool(gpu_name and accelerator.is_rtx_50xx(gpu_name))

    LOGGER.info("Available ONNX Runtime providers: %s", ", ".join(providers_available) or "none")
    if gpu_name:
        suffix = " (RTX 50-series detected)" if rtx_50_series else ""
        LOGGER.info("Detected GPU: %s%s", gpu_name, suffix)
    else:
        LOGGER.info("No NVIDIA GPU detected")
    LOGGER.info("Accelerator preference: %s", requested_mode)

    warning_message: Optional[str] = None
    session_obj: Optional[Session] = None

    try:
        session_obj = _rembg_new_session(model_name, providers=providers_argument)
    except Exception as exc:
        if runtime == "cuda":
            LOGGER.exception(
                "Falling back to CPU background removal after CUDA initialisation failure: %s",
                exc,
            )
            runtime = "cpu"
            provider_name = "CPUExecutionProvider"
            providers_argument = ["CPUExecutionProvider"]
            warning_message = "Running on CPU, performance will be slower."
            session_obj = _rembg_new_session(model_name, providers=providers_argument)
        else:
            raise

    if session_obj is None:
        session_obj = _rembg_new_session(model_name, providers=providers_argument)

    LOGGER.info("Selected execution provider: %s", provider_name)

    if gpu_name and "CUDAExecutionProvider" not in providers_available:
        global _CUDA_HINT_LOGGED
        if not _CUDA_HINT_LOGGED:
            LOGGER.warning(
                "Detected GPU %s but CUDAExecutionProvider is unavailable. Install "
                "onnxruntime-gpu that matches your CUDA runtime, configure nvidia-container-toolkit "
                "for Docker, and run containers with --gpus all.",
                gpu_name,
            )
            _CUDA_HINT_LOGGED = True

    should_warn = warn_on_cpu and runtime == "cpu" and requested_mode != "cpu"
    if should_warn:
        warning_message = warning_message or "Running on CPU, performance will be slower."
        global _CPU_WARNING_LOGGED
        if not _CPU_WARNING_LOGGED:
            LOGGER.warning(warning_message)
            _CPU_WARNING_LOGGED = True

    return SessionContext(
        session=session_obj,
        runtime=runtime,
        provider=provider_name,
        provider_options=dict(provider_options or {}),
        providers_available=providers_available,
        gpu_name=gpu_name,
        rtx_50_series=rtx_50_series,
        warning=warning_message if should_warn else None,
        device_id=device_id,
    )


def ensure_global_session(
    model_name: str = "u2net", config: Mapping[str, Any] | None = None
) -> Session:
    """Initialise and cache a global ``rembg`` session."""

    global _SESSION_CONTEXT, _SESSION_CONFIG_SIGNATURE
    signature = _signature_for_config(config)
    with _SESSION_LOCK:
        if _SESSION_CONTEXT is None or _SESSION_CONFIG_SIGNATURE != signature:
            _SESSION_CONTEXT = create_session(model_name=model_name, config=config)
            _SESSION_CONFIG_SIGNATURE = signature
    assert _SESSION_CONTEXT is not None
    return _SESSION_CONTEXT.session


def _get_session(session: Optional[Session] = None) -> Session:
    """Return the provided session or the cached singleton."""

    if session is not None:
        return session
    context = _SESSION_CONTEXT
    if context is None:
        return ensure_global_session()
    return context.session


def get_session_context() -> Optional[SessionContext]:
    """Return the cached :class:`SessionContext`, if initialised."""

    return _SESSION_CONTEXT


def get_runtime_payload() -> Dict[str, Any]:
    """Return runtime metadata for embedding in API responses."""

    context = _SESSION_CONTEXT
    if context is None:
        return {"runtime": "cpu", "gpu_name": None, "warning": None}
    return context.runtime_payload()


def get_accelerator_status() -> Dict[str, Any]:
    """Return diagnostic accelerator details for health checks."""

    context = _SESSION_CONTEXT
    if context is None:
        providers = accelerator.onnx_providers_available()
        gpu_name = accelerator.detect_gpu_name()
        runtime = "cpu"
        rtx = bool(gpu_name and accelerator.is_rtx_50xx(gpu_name)) if gpu_name else False
    else:
        providers = context.providers_available
        gpu_name = context.gpu_name
        runtime = context.runtime
        rtx = context.rtx_50_series
    return {
        "providers": providers,
        "selected": "cuda" if runtime == "cuda" else "cpu",
        "gpu_name": gpu_name,
        "rtx_50_series": rtx,
    }


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
