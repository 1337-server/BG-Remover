"""Simplified background removal helpers using rembg sessions."""
from __future__ import annotations

import base64
import io
import logging
import threading
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from typing import IO, Any, cast

import numpy as np

try:  # pragma: no cover - optional dependency during certain deployments
    import cv2  # type: ignore
except Exception:  # pragma: no cover - gracefully handle missing OpenCV
    cv2 = None  # type: ignore

from PIL import Image, ImageFilter, ImageOps

LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "isnet-general-use"


@dataclass(frozen=True)
class OutputFormat:
    """Descriptor describing a supported export format."""

    key: str
    extension: str
    label: str
    pil_format: str
    mime_type: str
    supports_alpha: bool
    save_kwargs: Mapping[str, Any] = field(default_factory=dict)

    def normalise_filename(self, path: Path) -> Path:
        """Return ``path`` with the configured extension."""

        return path.with_suffix(self.extension)


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
        label="WebP (lossless, supports transparency)",
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
        save_kwargs={"quality": 95},
    ),
)


_OUTPUT_FORMAT_LOOKUP: dict[str, OutputFormat] = {}
for _format in OUTPUT_FORMATS:
    _OUTPUT_FORMAT_LOOKUP[_format.key] = _format
    _OUTPUT_FORMAT_LOOKUP[_format.key.lower()] = _format
    _OUTPUT_FORMAT_LOOKUP[_format.extension.lstrip(".").lower()] = _format


DEFAULT_OUTPUT_FORMAT = OUTPUT_FORMATS[0].key


@dataclass(frozen=True)
class SessionContext:
    """Container representing a rembg session and provider metadata."""

    model_name: str
    session: Any
    provider: str
    providers_available: tuple[str, ...]

    def runtime_payload(self) -> dict[str, Any]:
        """Return diagnostics describing the active runtime."""

        return {
            "runtime": "cpu",
            "provider": self.provider,
            "providers_available": list(self.providers_available) or [self.provider],
            "gpu_name": None,
            "gpu_available": False,
            "warning": None,
            "accelerator_message": f"Using CPU ({self.provider})",
        }


SUPPORTED_EXTENSIONS: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff")
MAX_FEATHER_RADIUS = 50


@dataclass(frozen=True)
class RemovalResult:
    """Represents the processed output from a background removal request."""

    image: Image.Image | None
    format_spec: OutputFormat
    elapsed_ms: float
    path_in: Path | None = None
    path_out: Path | None = None
    error: str | None = None

    @property
    def success(self) -> bool:
        """Return ``True`` when the removal completed successfully."""

        return self.error is None

    @property
    def timing_ms(self) -> float:
        """Return the recorded processing time in milliseconds."""

        return self.elapsed_ms

    def to_dict(self) -> dict[str, Any]:
        """Return a serialisable representation of the result."""

        return {
            "path_in": str(self.path_in) if self.path_in else None,
            "path_out": str(self.path_out) if self.path_out else None,
            "success": self.success,
            "error": self.error,
            "timing_ms": self.elapsed_ms,
            "format": self.format_spec.key,
        }

    def save(self, path: Path) -> Path:
        """Persist the processed image to ``path`` using the target format."""

        if self.image is None:
            raise ValueError("No processed image available to save.")
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as buffer:
            buffer.write(self.as_bytes())
        return destination

    def as_bytes(self) -> bytes:
        """Return the processed image encoded in the configured format."""

        if self.image is None:
            raise ValueError("No processed image available to encode.")
        buffer = io.BytesIO()
        prepared = _prepare_for_format(self.image, self.format_spec)
        prepared.save(buffer, self.format_spec.pil_format, **self.format_spec.save_kwargs)
        return buffer.getvalue()


_SESSION_CACHE: dict[str, SessionContext] = {}
_SESSION_CACHE_LOCK = threading.Lock()
_GLOBAL_SESSION: SessionContext | None = None
_GLOBAL_SESSION_LOCK = threading.Lock()


@lru_cache(maxsize=4)
def _load_session(model_name: str) -> Any:
    """Return a cached rembg session for ``model_name``."""

    from rembg import new_session

    LOGGER.info("Creating rembg session for model: %s", model_name)
    return new_session(model_name=model_name)


def _resolve_providers(session: Any) -> tuple[str, ...]:
    """Extract provider information from ``session`` where possible."""

    candidates: list[str] = []
    for attr in ("providers", "get_providers"):
        provider_info = getattr(session, attr, None)
        if callable(provider_info):
            try:
                resolved = list(provider_info())  # type: ignore[call-arg]
            except Exception:  # pragma: no cover - best effort diagnostics
                LOGGER.debug("Failed to query providers via %s", attr, exc_info=True)
                continue
        elif provider_info:
            resolved = list(provider_info)
        else:
            continue
        candidates.extend(str(item) for item in resolved if item)
    if not candidates:
        candidates.append("CPUExecutionProvider")
    return tuple(dict.fromkeys(candidates))


def create_session(model_name: str = DEFAULT_MODEL_NAME) -> SessionContext:
    """Return a cached session context for ``model_name``."""

    with _SESSION_CACHE_LOCK:
        cached = _SESSION_CACHE.get(model_name)
    if cached is not None:
        return cached

    session = _load_session(model_name)
    providers = _resolve_providers(session)
    context = SessionContext(
        model_name=model_name,
        session=session,
        provider=providers[0],
        providers_available=providers,
    )
    with _SESSION_CACHE_LOCK:
        _SESSION_CACHE[model_name] = context
    return context


def ensure_global_session(model_name: str = DEFAULT_MODEL_NAME) -> Any:
    """Ensure a global rembg session exists and return it."""

    global _GLOBAL_SESSION
    context = _GLOBAL_SESSION
    if context is not None and context.model_name == model_name:
        return context.session

    context = create_session(model_name)
    with _GLOBAL_SESSION_LOCK:
        _GLOBAL_SESSION = context
    return context.session


def get_session_context() -> SessionContext | None:
    """Return the cached global session context, if initialised."""

    return _GLOBAL_SESSION


def get_output_format_spec(value: str | None) -> OutputFormat:
    """Return the :class:`OutputFormat` for ``value``.

    The lookup accepts canonical keys such as ``"png"`` or extensions such as
    ``".png"``. When ``value`` is missing the default PNG format is returned.
    A :class:`ValueError` is raised for unsupported formats.
    """

    if value is None:
        key = DEFAULT_OUTPUT_FORMAT
    else:
        key = value.strip().lower().lstrip(".")
        if not key:
            key = DEFAULT_OUTPUT_FORMAT
    spec = _OUTPUT_FORMAT_LOOKUP.get(key)
    if spec is None:
        raise ValueError(f"Unsupported output format: {value}")
    return spec


def list_output_format_choices() -> list[str]:
    """Return the sorted list of available output format keys."""

    return sorted({format_spec.key for format_spec in OUTPUT_FORMATS})


def get_mime_type_for_path(path: Path) -> str:
    """Return the MIME type inferred from ``path``."""

    try:
        spec = get_output_format_spec(path.suffix)
    except ValueError:
        return "application/octet-stream"
    return spec.mime_type


def _prepare_for_format(image: Image.Image, format_spec: OutputFormat) -> Image.Image:
    """Return ``image`` converted for ``format_spec``."""

    if format_spec.supports_alpha:
        return image.convert("RGBA")
    rgba = image.convert("RGBA")
    background = Image.new("RGB", rgba.size, (255, 255, 255))
    background.paste(rgba, mask=rgba.split()[-1])
    return background


def _image_to_bytes(image: Image.Image) -> bytes:
    """Return ``image`` encoded as PNG bytes for rembg input."""

    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    return buffer.getvalue()


def _rembg_remove(data: bytes, session: Any, **options: Any) -> bytes:
    """Execute ``rembg.remove`` with ``data`` and ``session``."""

    from rembg import remove

    return remove(data, session=session, **options)


def build_colorkey_mask(image: Image.Image, tolerance: int = 14) -> np.ndarray | None:
    """Return an alpha mask for near-solid backgrounds when detected."""

    if tolerance <= 0:
        return None

    rgb_image = image.convert("RGB")
    np_image = np.asarray(rgb_image, dtype=np.uint8)
    height, width, _ = np_image.shape
    patch = max(5, min(height, width) // 10)
    patch = min(patch, height, width)
    if patch <= 0:
        return None

    corners = (
        np_image[0:patch, 0:patch],
        np_image[0:patch, width - patch : width],
        np_image[height - patch : height, 0:patch],
        np_image[height - patch : height, width - patch : width],
    )
    corner_means = np.array([corner.reshape(-1, 3).mean(axis=0) for corner in corners])
    background_colour = corner_means.mean(axis=0)
    deviations = [np.abs(corner - background_colour).max() for corner in corners]
    if max(deviations) > tolerance * 1.5:
        return None

    delta = np.max(
        np.abs(np_image.astype(np.int16) - background_colour.astype(np.int16)), axis=2
    )
    mask = np.where(delta <= tolerance, 0, 255).astype(np.uint8)
    return mask


def _feather_alpha(alpha: np.ndarray, radius: int) -> np.ndarray:
    """Return a softened alpha channel using Gaussian blur when possible."""

    if radius <= 0:
        return alpha
    clamped_radius = min(int(radius), MAX_FEATHER_RADIUS)
    if clamped_radius <= 0:
        return alpha
    if cv2 is not None:
        kernel = clamped_radius * 2 + 1
        try:
            blurred = cv2.GaussianBlur(alpha, (kernel, kernel), 0)  # type: ignore[arg-type]
            return blurred.astype(np.uint8)
        except Exception:
            LOGGER.debug("Falling back to Pillow feathering", exc_info=True)
    image = Image.fromarray(alpha, mode="L")
    blurred_image = image.filter(ImageFilter.GaussianBlur(radius=clamped_radius))
    return np.asarray(blurred_image, dtype=np.uint8)


def _read_stream(stream: IO[bytes]) -> bytes:
    """Return the bytes from ``stream`` ensuring a useful error on empties."""

    data = stream.read()
    if not data:
        raise ValueError("No image data supplied for background removal.")
    return data


def remove_background_bytes(
    data: bytes,
    *,
    output_format: str | None = None,
    model_name: str = DEFAULT_MODEL_NAME,
    alpha_matting: bool = False,
    am_foreground: int = 240,
    am_background: int = 10,
    am_erode: int = 10,
    use_colorkey_fallback: bool = True,
    colorkey_tolerance: int = 14,
    feather_radius: int = 3,
) -> RemovalResult:
    """Remove the background from raw ``data`` and return the processed result."""

    format_spec = get_output_format_spec(output_format)
    session = ensure_global_session(model_name)

    original = Image.open(io.BytesIO(data))
    processed_input = cast(Image.Image, ImageOps.exif_transpose(original)).convert("RGBA")
    original.close()

    am_foreground = max(0, min(255, int(am_foreground)))
    am_background = max(0, min(255, int(am_background)))
    am_erode = max(0, min(255, int(am_erode)))
    colorkey_tolerance = max(0, min(255, int(colorkey_tolerance)))
    feather_radius = max(0, min(MAX_FEATHER_RADIUS, int(feather_radius)))

    options = {
        "alpha_matting": bool(alpha_matting),
        "alpha_matting_foreground_threshold": am_foreground,
        "alpha_matting_background_threshold": am_background,
        "alpha_matting_erode_size": am_erode,
        "only_mask": True,
    }

    start_time = time.perf_counter()
    error: str | None = None
    output_image: Image.Image | None = None
    try:
        encoded_input = _image_to_bytes(processed_input)
        mask_bytes = _rembg_remove(encoded_input, session, **options)
        mask_image = Image.open(io.BytesIO(mask_bytes))
        with mask_image:
            mask_l = cast(Image.Image, ImageOps.exif_transpose(mask_image)).convert("L")
            mask_l.load()
        alpha_np = np.asarray(mask_l, dtype=np.uint8)
        mask_l.close()
        if use_colorkey_fallback:
            fallback_mask = build_colorkey_mask(processed_input, tolerance=colorkey_tolerance)
            if fallback_mask is not None:
                alpha_np = np.maximum(alpha_np, fallback_mask)
        alpha_np = _feather_alpha(alpha_np, feather_radius)
        refined_mask = Image.fromarray(alpha_np, mode="L")
        output_image = processed_input.copy()
        output_image.putalpha(refined_mask)
    except Exception as exc:  # pragma: no cover - depends on third-party libraries
        error = str(exc)
        output_image = None
    finally:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        processed_input.close()
    return RemovalResult(
        image=output_image,
        format_spec=format_spec,
        elapsed_ms=elapsed_ms,
        error=error,
    )


def remove_background_stream(
    stream: IO[bytes],
    *,
    output_format: str | None = None,
    model_name: str = DEFAULT_MODEL_NAME,
    alpha_matting: bool = False,
    am_foreground: int = 240,
    am_background: int = 10,
    am_erode: int = 10,
    use_colorkey_fallback: bool = True,
    colorkey_tolerance: int = 14,
    feather_radius: int = 3,
) -> RemovalResult:
    """Process ``stream`` by reading its bytes before delegating to ``remove_background_bytes``."""

    data = _read_stream(stream)
    return remove_background_bytes(
        data,
        output_format=output_format,
        model_name=model_name,
        alpha_matting=alpha_matting,
        am_foreground=am_foreground,
        am_background=am_background,
        am_erode=am_erode,
        use_colorkey_fallback=use_colorkey_fallback,
        colorkey_tolerance=colorkey_tolerance,
        feather_radius=feather_radius,
    )


def _resolve_output_path(
    input_path: Path,
    output: str | Path | None,
    format_spec: OutputFormat,
) -> Path:
    """Return the destination path for ``input_path`` respecting ``output``."""

    if output is None:
        return format_spec.normalise_filename(
            input_path.with_name(f"{input_path.stem}_no_bg")
        )

    candidate = Path(output).expanduser()
    path_text = str(output)
    should_treat_as_directory = candidate.is_dir() or (
        not candidate.exists()
        and (path_text.endswith(("/", "\\")) or candidate.suffix == "")
    )
    if should_treat_as_directory:
        candidate.mkdir(parents=True, exist_ok=True)
        return candidate / f"{input_path.stem}{format_spec.extension}"
    return candidate


def _resolve_output_directory(
    output_dir: str | Path | None, input_dir: Path | None = None
) -> Path:
    """Return a resolved directory for batch exports."""

    if output_dir is None:
        base = Path(input_dir) if input_dir is not None else Path.cwd()
        destination = base / "_no_bg"
    else:
        destination = Path(output_dir).expanduser()
        if not destination.is_absolute():
            destination = Path.cwd() / destination
    destination.mkdir(parents=True, exist_ok=True)
    return destination


def _iter_input_files(input_dir: Path, recursive: bool) -> Iterable[Path]:
    """Yield supported input files from ``input_dir`` respecting ``recursive``."""

    iterator: Iterable[Path]
    if recursive:
        iterator = input_dir.rglob("*")
    else:
        iterator = input_dir.iterdir()
    for candidate in iterator:
        if candidate.is_file() and candidate.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield candidate


def remove_bg_file(
    input_path: Path | str,
    output: Path | str | None,
    *,
    output_format: str | None = None,
    model_name: str = DEFAULT_MODEL_NAME,
    alpha_matting: bool = False,
    am_foreground: int = 240,
    am_background: int = 10,
    am_erode: int = 10,
    use_colorkey_fallback: bool = True,
    colorkey_tolerance: int = 14,
    feather_radius: int = 3,
    retain_image: bool = False,
) -> RemovalResult:
    """Remove the background from ``input_path`` and write the result to ``output``."""

    source_path = Path(input_path)
    with source_path.open("rb") as stream:
        data = stream.read()
    result = remove_background_bytes(
        data,
        output_format=output_format,
        model_name=model_name,
        alpha_matting=alpha_matting,
        am_foreground=am_foreground,
        am_background=am_background,
        am_erode=am_erode,
        use_colorkey_fallback=use_colorkey_fallback,
        colorkey_tolerance=colorkey_tolerance,
        feather_radius=feather_radius,
    )
    destination: Path | None = None
    if result.success:
        destination = _resolve_output_path(source_path, output, result.format_spec)
        result.save(destination)
    image_ref = result.image if retain_image else None
    if result.image is not None and not retain_image:
        result.image.close()
    return replace(result, path_in=source_path, path_out=destination, image=image_ref)


def remove_bg_folder(
    input_dir: Path | str,
    output_dir: Path | str | None = None,
    *,
    output_format: str | None = None,
    model_name: str = DEFAULT_MODEL_NAME,
    recursive: bool = False,
    alpha_matting: bool = False,
    am_foreground: int = 240,
    am_background: int = 10,
    am_erode: int = 10,
    use_colorkey_fallback: bool = True,
    colorkey_tolerance: int = 14,
    feather_radius: int = 3,
) -> list[RemovalResult]:
    """Process every supported image found under ``input_dir``."""

    source_dir = Path(input_dir)
    if not source_dir.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {source_dir}")

    output_root = _resolve_output_directory(output_dir, source_dir)
    format_spec = get_output_format_spec(output_format)

    results: list[RemovalResult] = []
    for source in _iter_input_files(source_dir, recursive):
        relative = source.relative_to(source_dir) if recursive else Path(source.name)
        destination = format_spec.normalise_filename(output_root / relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        result = remove_bg_file(
            source,
            destination,
            output_format=format_spec.key,
            model_name=model_name,
            alpha_matting=alpha_matting,
            am_foreground=am_foreground,
            am_background=am_background,
            am_erode=am_erode,
            use_colorkey_fallback=use_colorkey_fallback,
            colorkey_tolerance=colorkey_tolerance,
            feather_radius=feather_radius,
            retain_image=False,
        )
        results.append(result)
    return results


def get_runtime_payload() -> dict[str, Any]:
    """Return diagnostic metadata for templates and API responses."""

    context = get_session_context()
    if context is None:
        return {
            "runtime": "cpu",
            "provider": "CPUExecutionProvider",
            "providers_available": ["CPUExecutionProvider"],
            "gpu_name": None,
            "gpu_available": False,
            "warning": "Session not initialised",
            "accelerator_message": "Using CPU (CPUExecutionProvider)",
        }
    return context.runtime_payload()


def ensure_runtime_ready() -> None:
    """Compatibility shim retained for callers expecting a guard."""

    ensure_global_session()


def encode_result_image(result: RemovalResult) -> str:
    """Return the processed image encoded as a base64 data URL payload."""

    encoded = base64.b64encode(result.as_bytes()).decode("ascii")
    return f"data:{result.format_spec.mime_type};base64,{encoded}"


def get_accelerator_status() -> dict[str, Any]:
    """Return simplified accelerator information for health checks."""

    context = get_session_context()
    provider = context.provider if context else "CPUExecutionProvider"
    providers_available = list(context.providers_available) if context else [provider]
    return {
        "runtime": "cpu",
        "provider": provider,
        "providers_available": providers_available,
        "has_gpu": False,
        "accelerator_message": f"Using CPU ({provider})",
    }
