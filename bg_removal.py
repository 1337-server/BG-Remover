"""Background removal helpers powered by ONNX Runtime sessions.

The original implementation proxied the :mod:`rembg` package. Recent
versions of :mod:`rembg` pull in :mod:`pymatting`, which depends on
``numba`` features that are not yet available on Python 3.12. Importing
``rembg`` therefore raised ``NotImplementedError`` while attempting to
compile those extensions, leaving the application stuck before it could
run inference. To keep the project lightweight and reliable we now drive
the ONNX models directly through :mod:`onnxruntime`, reproducing the
minimal pre/post-processing steps required by the original sessions.
"""
from __future__ import annotations

import base64
import contextlib
import hashlib
import io
import logging
import os
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from typing import IO, Any, cast

from enum import Enum

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageFilter, ImageOps

try:  # pragma: no cover - optional dependency fallback
    import requests  # type: ignore[import]
except ModuleNotFoundError:  # pragma: no cover - fallback for restricted environments
    from types import SimpleNamespace

    from requests_shim import HTTPError, RequestException, Response, get

    requests = SimpleNamespace(  # type: ignore[assignment]
        get=get,
        HTTPError=HTTPError,
        RequestException=RequestException,
        Response=Response,
    )

try:  # pragma: no cover - optional dependency during certain deployments
    import cv2  # type: ignore
except Exception:  # pragma: no cover - gracefully handle missing OpenCV
    cv2 = None  # type: ignore

LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "isnet-general-use"
MODELS_DIRECTORY = Path(__file__).resolve().parent / "models"
MODEL_DOWNLOAD_ROOT = MODELS_DIRECTORY
MODEL_DOWNLOAD_ROOT.mkdir(parents=True, exist_ok=True)


class DownloadState(Enum):
    """Enumeration describing lifecycle states for model downloads."""

    AVAILABLE = "available"
    PENDING = "pending"
    IN_PROGRESS = "in-progress"
    FAILED = "failed"


@dataclass(frozen=True)
class DownloadStatus:
    """Snapshot describing the current state of a model download."""

    key: str
    state: DownloadState
    path: Path | None = None
    error: str | None = None
    attempts: int = 0
    updated_at: float = field(default_factory=time.time)


_DOWNLOAD_STATUS_LOCK = threading.Lock()
_DOWNLOAD_STATUSES: dict[str, DownloadStatus] = {}
_PREFETCH_THREADS: dict[str, threading.Thread] = {}


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
class ModelSpec:
    """Descriptor describing an ONNX segmentation model."""

    key: str
    input_size: tuple[int, int]
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    normalisation_scale: float = 255.0  #: Divisor applied before mean/std normalisation.
    checksum_md5: str | None = None
    url: str | None = None
    huggingface_repo: str | None = None
    huggingface_filename: str | None = None
    huggingface_revision: str = "main"


MODEL_SPECS: dict[str, ModelSpec] = {
    "isnet-general-use": ModelSpec(
        key="isnet-general-use",
        url="https://github.com/danielgatis/rembg/releases/download/v0.0.0/isnet-general-use.onnx",
        input_size=(1024, 1024),
        mean=(0.5, 0.5, 0.5),
        std=(1.0, 1.0, 1.0),
        checksum_md5="fc16ebd8b0c10d971d3513d564d01e29",
    ),
    "u2net": ModelSpec(
        key="u2net",
        url="https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx",
        input_size=(320, 320),
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        checksum_md5="60024c5c889badc19c04ad937298a77b",
    ),
    "u2net_human_seg": ModelSpec(
        key="u2net_human_seg",
        url="https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net_human_seg.onnx",
        input_size=(320, 320),
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        checksum_md5="c09ddc2e0104f800e3e1bb4652583d1f",
    ),
    "isnet-anime": ModelSpec(
        key="isnet-anime",
        input_size=(1024, 1024),
        mean=(0.485, 0.456, 0.406),
        std=(1.0, 1.0, 1.0),
        checksum_md5="6f184e756bb3bd901c8849220a83e38e",
        url="https://github.com/danielgatis/rembg/releases/download/v0.0.0/isnet-anime.onnx",
    ),
    "briaai/RMBG-2.0": ModelSpec(
        key="briaai/RMBG-2.0",
        input_size=(1024, 1024),
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        huggingface_repo="briaai/RMBG-2.0",
        huggingface_filename="RMBG-2.0.onnx",
    ),
    # The original matting-by-generation weights are no longer published. The
    # BRIA RMBG v1.4 portrait model offers comparable fine-edge performance and
    # remains actively hosted on the Hugging Face Hub.
    "matting-by-generation": ModelSpec(
        key="matting-by-generation",
        input_size=(1024, 1024),
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        huggingface_repo="briaai/RMBG-1.4",
        huggingface_filename="onnx/model.onnx",
    ),

    # Alias of the BRIA RMBG v2.0 weights used for the "complex scene" option in the UI.
    "sam_segmentation_model": ModelSpec(
        key="sam_segmentation_model",
        input_size=(1024, 1024),
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        huggingface_repo="briaai/RMBG-2.0",
        huggingface_filename="RMBG-2.0.onnx",
    ),

    # SAM ViT-B Encoder (Hugging Face)
    "sam_vit_b_01ec64_encoder": ModelSpec(
        key="sam_vit_b_01ec64_encoder",
        input_size=(1024, 1024),
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        huggingface_repo="microsoft/segment-anything-model-webnn",
        huggingface_filename="sam_vit_b_01ec64.encoder-fp16.onnx",
    ),

    # SAM ViT-B Decoder (Hugging Face)
    "sam_vit_b_01ec64_decoder": ModelSpec(
        key="sam_vit_b_01ec64_decoder",
        input_size=(1024, 1024),
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        huggingface_repo="microsoft/segment-anything-model-webnn",
        huggingface_filename="sam_vit_b_01ec64.decoder-fp16.onnx",
    ),
}


@dataclass(frozen=True)
class SessionContext:
    """Container representing an ONNX session and provider metadata."""

    model_name: str
    session: BackgroundRemovalSession
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


def _normalise_image(image: Image.Image, spec: ModelSpec) -> np.ndarray:
    """Return a model-ready tensor for ``image`` according to ``spec``."""

    rgb_image = image.convert("RGB").resize(spec.input_size, Image.Resampling.LANCZOS)
    rgb_array = np.asarray(rgb_image, dtype=np.float32)
    scale = spec.normalisation_scale if spec.normalisation_scale > 0 else 1.0
    # Preserve relative luminance by scaling with the model-defined range rather than
    # the brightest pixel in the current image.
    rgb_array /= scale
    normalised = np.zeros_like(rgb_array, dtype=np.float32)
    for index in range(3):
        normalised[:, :, index] = (rgb_array[:, :, index] - spec.mean[index]) / spec.std[index]
    normalised = normalised.transpose((2, 0, 1))
    return np.expand_dims(normalised, 0).astype(np.float32)


def _compute_mask(array: np.ndarray, original_size: tuple[int, int]) -> Image.Image:
    """Convert an ONNX output ``array`` into a resized mask image."""

    mask = array
    while mask.ndim > 2:
        mask = mask[0]
    max_value = float(mask.max())
    min_value = float(mask.min())
    if max_value - min_value > 1e-5:
        mask = (mask - min_value) / (max_value - min_value)
    else:
        mask = np.zeros_like(mask)
    mask = (mask * 255).clip(0, 255).astype("uint8")
    image = Image.fromarray(mask, mode="L")
    if image.size != original_size:
        image = image.resize(original_size, Image.Resampling.LANCZOS)
    return image


def _build_model_filename(spec: ModelSpec) -> str:
    """Return the expected filename for ``spec`` within the models directory."""

    return f"{spec.key}.onnx"


def _verify_md5(path: Path, expected: str | None) -> bool:
    """Return ``True`` when the file at ``path`` matches ``expected``.

    When ``expected`` is ``None`` or empty the check is reduced to
    verifying the file exists. Some community hosted models do not
    publish checksums, so we prefer opportunistic validation rather than
    blocking the download entirely.
    """

    if not path.exists():
        return False
    if not expected:
        return True
    checksum = hashlib.md5()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest() == expected.lower()


def _update_download_status(
    key: str,
    *,
    state: DownloadState,
    path: Path | None = None,
    error: str | None = None,
    attempts: int | None = None,
) -> DownloadStatus:
    """Update the cached download status for ``key`` and return the snapshot."""

    with _DOWNLOAD_STATUS_LOCK:
        previous = _DOWNLOAD_STATUSES.get(key)
        resolved_path = path if path is not None else (previous.path if previous else None)
        resolved_error = error if error is not None else (previous.error if previous else None)
        resolved_attempts = attempts if attempts is not None else (previous.attempts if previous else 0)
        status = DownloadStatus(
            key=key,
            state=state,
            path=resolved_path,
            error=resolved_error,
            attempts=resolved_attempts,
            updated_at=time.time(),
        )
        _DOWNLOAD_STATUSES[key] = status
        return status


def get_download_statuses() -> dict[str, DownloadStatus]:
    """Return a shallow copy of the current download status mapping."""

    with _DOWNLOAD_STATUS_LOCK:
        return dict(_DOWNLOAD_STATUSES)


def _prefetch_model(spec: ModelSpec) -> None:
    """Background worker that downloads ``spec`` without blocking startup."""

    try:
        _download_model(spec)
    except Exception as error:  # pragma: no cover - network dependent
        LOGGER.warning("Background download for %s failed: %s", spec.key, error)
    finally:
        with _DOWNLOAD_STATUS_LOCK:
            thread = _PREFETCH_THREADS.get(spec.key)
            if thread and thread is threading.current_thread():
                _PREFETCH_THREADS.pop(spec.key, None)


def _schedule_prefetch(spec: ModelSpec) -> None:
    """Start a background thread to download ``spec`` when not already cached."""

    filename = _build_model_filename(spec)
    destination = MODEL_DOWNLOAD_ROOT / filename
    existing = get_download_statuses().get(spec.key)
    if existing and existing.state is DownloadState.AVAILABLE:
        return

    with _DOWNLOAD_STATUS_LOCK:
        thread = _PREFETCH_THREADS.get(spec.key)
        if thread and thread.is_alive():
            return
        thread = threading.Thread(
            target=_prefetch_model,
            name=f"prefetch-{spec.key}",
            args=(spec,),
            daemon=True,
        )
        _PREFETCH_THREADS[spec.key] = thread

    _update_download_status(spec.key, state=DownloadState.IN_PROGRESS, path=destination)
    thread.start()


def ensure_models_downloaded(prefetch: bool | Iterable[str] = True) -> dict[str, DownloadStatus]:
    """Ensure the models directory exists and optionally prefetch weights."""

    MODELS_DIRECTORY.mkdir(parents=True, exist_ok=True)

    if isinstance(prefetch, bool):
        prefetch_keys = {DEFAULT_MODEL_NAME} if prefetch else set()
    else:
        requested = {name for name in prefetch}
        prefetch_keys = {name for name in requested if name in MODEL_SPECS}
        missing = requested - prefetch_keys
        if missing:
            LOGGER.warning("Ignoring unknown models requested for prefetch: %s", ", ".join(sorted(missing)))

    statuses: dict[str, DownloadStatus] = {}
    for spec in MODEL_SPECS.values():
        filename = _build_model_filename(spec)
        destination = MODEL_DOWNLOAD_ROOT / filename
        relative_destination = Path("models") / filename
        if _verify_md5(destination, spec.checksum_md5):
            status = _update_download_status(
                spec.key,
                state=DownloadState.AVAILABLE,
                path=destination,
                error=None,
            )
            print(f"[✓] {spec.key} available at ./{relative_destination.as_posix()}")
        else:
            status = _update_download_status(
                spec.key,
                state=DownloadState.PENDING,
                path=destination if destination.exists() else None,
                error=None,
            )
            print(f"[ ] {spec.key} will download on first use.")
            if spec.key in prefetch_keys:
                print(f"[~] Prefetching {spec.key} in background…")
                _schedule_prefetch(spec)
        statuses[spec.key] = status
    return statuses


def _read_token_file(path: Path) -> str | None:
    """Return the first non-empty line from ``path`` when present."""

    try:
        contents = path.read_text(encoding="utf-8")
    except OSError:
        return None
    lines = [line.strip() for line in contents.splitlines() if line.strip()]
    if not lines:
        return None
    return lines[0]


def _resolve_huggingface_token() -> str | None:
    """Return an authentication token for Hugging Face downloads when available."""

    for variable in ("HUGGINGFACEHUB_API_TOKEN", "HF_API_TOKEN"):
        token = os.environ.get(variable)
        if token:
            cleaned = token.strip()
            if cleaned:
                return cleaned

    home = Path.home()
    candidate_paths = (
        home / ".huggingface" / "token",
        home / ".cache" / "huggingface" / "token",
    )
    for token_path in candidate_paths:
        token = _read_token_file(token_path)
        if token:
            return token
    return None


def _download_model_from_url(spec: ModelSpec, destination: Path) -> Path:
    """Download ``spec`` using a direct HTTP request."""

    if not spec.url:
        raise ValueError(f"No download URL configured for model {spec.key}")
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; br-remover/1.0; +https://github.com/your-org/br-remover)",
        "Accept": "application/octet-stream",
    }
    return _download_model_via_http(spec, spec.url, destination, headers)


def _download_model_from_huggingface(spec: ModelSpec, destination: Path) -> Path:
    """Download ``spec`` from the Hugging Face Hub."""

    if not spec.huggingface_repo or not spec.huggingface_filename:
        raise ValueError(f"Incomplete Hugging Face configuration for model {spec.key}")
    revision = spec.huggingface_revision or "main"
    url = (
        "https://huggingface.co/"
        f"{spec.huggingface_repo}/resolve/{revision}/{spec.huggingface_filename}"
        "?download=1"
    )
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; br-remover/1.0; +https://github.com/your-org/br-remover)",
        "Accept": "application/octet-stream",
    }
    token = _resolve_huggingface_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        return _download_model_via_http(spec, url, destination, headers)
    except RuntimeError as error:
        if "HTTP Error 401" in str(error):
            raise RuntimeError(
                f"Failed to download {spec.key} from Hugging Face: authentication is required. "
                "Provide a token via the HUGGINGFACEHUB_API_TOKEN or HF_API_TOKEN environment variables."
            ) from error
        raise


def _download_model_via_http(
    spec: ModelSpec, url: str, destination: Path, headers: Mapping[str, str]
) -> Path:
    """Download ``spec`` from ``url`` into ``destination`` using ``headers``."""

    tmp_path = destination.with_suffix(".tmp")
    try:
        with contextlib.ExitStack() as stack:
            response = stack.enter_context(
                requests.get(url, headers=dict(headers), stream=True, timeout=60)
            )
            try:
                response.raise_for_status()
            except requests.HTTPError as error:  # pragma: no cover - exercised in ensure helper
                tmp_path.unlink(missing_ok=True)
                raise RuntimeError(f"Failed to download {spec.key} from {url}: {error}") from error
            with stack.enter_context(tmp_path.open("wb")) as buffer:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    buffer.write(chunk)
    except requests.RequestException as error:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to download {spec.key} from {url}: {error}") from error
    if not _verify_md5(tmp_path, spec.checksum_md5):
        tmp_path.unlink(missing_ok=True)
        raise ValueError(f"Checksum mismatch for model {spec.key}")
    tmp_path.replace(destination)
    return destination


def _download_model(
    spec: ModelSpec, *, max_attempts: int = 3, backoff_base: float = 0.5
) -> Path:
    """Download the ONNX model defined by ``spec`` when needed."""

    destination = MODEL_DOWNLOAD_ROOT / f"{spec.key}.onnx"
    if _verify_md5(destination, spec.checksum_md5):
        LOGGER.debug("Model %s already present at %s", spec.key, destination)
        _update_download_status(spec.key, state=DownloadState.AVAILABLE, path=destination, error=None)
        return destination

    if not (spec.huggingface_repo and spec.huggingface_filename) and not spec.url:
        message = f"No download source configured for model {spec.key}"
        _update_download_status(spec.key, state=DownloadState.FAILED, path=destination, error=message)
        raise ValueError(message)

    destination.parent.mkdir(parents=True, exist_ok=True)
    attempts = 0
    while attempts < max_attempts:
        attempts += 1
        _update_download_status(
            spec.key,
            state=DownloadState.IN_PROGRESS,
            path=destination,
            attempts=attempts,
        )
        try:
            if spec.huggingface_repo and spec.huggingface_filename:
                LOGGER.info(
                    "Fetching model %s from Hugging Face repo %s (attempt %s)",
                    spec.key,
                    spec.huggingface_repo,
                    attempts,
                )
                path = _download_model_from_huggingface(spec, destination)
            else:
                LOGGER.info(
                    "Fetching model %s from %s (attempt %s)",
                    spec.key,
                    spec.url,
                    attempts,
                )
                path = _download_model_from_url(spec, destination)
        except Exception as error:
            LOGGER.warning("Attempt %s to download %s failed: %s", attempts, spec.key, error)
            if attempts >= max_attempts:
                _update_download_status(
                    spec.key,
                    state=DownloadState.FAILED,
                    path=destination if destination.exists() else None,
                    error=str(error),
                    attempts=attempts,
                )
                raise
            _update_download_status(
                spec.key,
                state=DownloadState.PENDING,
                path=destination if destination.exists() else None,
                error=str(error),
                attempts=attempts,
            )
            delay = backoff_base * (2 ** (attempts - 1))
            LOGGER.debug(
                "Retrying download for %s in %.2f seconds (attempt %s of %s)",
                spec.key,
                delay,
                attempts + 1,
                max_attempts,
            )
            time.sleep(delay)
            continue

        _update_download_status(
            spec.key,
            state=DownloadState.AVAILABLE,
            path=path,
            error=None,
            attempts=attempts,
        )
        LOGGER.info("Model %s stored at %s", spec.key, path)
        return path

    # The loop should always return or raise beforehand, but the interpreter
    # requires an explicit ``raise`` in case logic changes in the future.
    raise RuntimeError(f"Failed to download {spec.key}")


class BackgroundRemovalSession:
    """Small wrapper around an :class:`onnxruntime.InferenceSession`."""

    def __init__(self, spec: ModelSpec):
        self.spec = spec
        model_path = _download_model(spec)
        session_options = ort.SessionOptions()
        if "OMP_NUM_THREADS" in os.environ:  # type: ignore[name-defined]
            threads = int(os.environ["OMP_NUM_THREADS"])  # type: ignore[name-defined]
            session_options.inter_op_num_threads = threads
            session_options.intra_op_num_threads = threads
        self.inner = ort.InferenceSession(
            str(model_path),
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.inner.get_inputs()[0].name
        providers = self.inner.get_providers()
        self.providers_available = tuple(providers)
        self.primary_provider = providers[0] if providers else "CPUExecutionProvider"

    def predict_mask(self, image: Image.Image) -> Image.Image:
        """Return the segmentation mask predicted for ``image``."""

        tensor = _normalise_image(image, self.spec)
        LOGGER.debug(
            "Running ONNX inference", extra={"input_size": self.spec.input_size, "model": self.spec.key}
        )
        outputs = self.inner.run(None, {self.input_name: tensor})
        return _compute_mask(outputs[0], image.size)


@lru_cache(maxsize=4)
def _load_session(model_name: str) -> Any:
    """Return a cached ONNX runtime session for ``model_name``."""

    spec = MODEL_SPECS.get(model_name)
    if spec is None:
        LOGGER.warning("Unknown model %s requested; using default %s", model_name, DEFAULT_MODEL_NAME)
        spec = MODEL_SPECS[DEFAULT_MODEL_NAME]
    LOGGER.info("Initialising ONNX session for model: %s", spec.key)
    return BackgroundRemovalSession(spec)


def _resolve_providers(session: BackgroundRemovalSession) -> tuple[str, ...]:
    """Extract provider information from ``session`` where possible."""

    providers = session.providers_available
    if not providers:
        return ("CPUExecutionProvider",)
    return providers


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
    """Ensure a global ONNX runtime session exists and return it."""

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


def _predict_mask(image: Image.Image, session: BackgroundRemovalSession) -> Image.Image:
    """Return the segmentation mask predicted for ``image``."""

    return session.predict_mask(image)


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


def _remove_background_from_image_loader(
    loader: Callable[[], Image.Image],
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
    session: BackgroundRemovalSession | None = None,
) -> RemovalResult:
    """Run the shared removal pipeline using ``loader`` to obtain the source image."""

    # Reuse ``session`` when provided so large batch jobs avoid repeated
    # initialisation overhead and share ONNX runtime state across threads.

    format_spec = get_output_format_spec(output_format)
    active_session = session if session is not None else ensure_global_session(model_name)
    model_key = getattr(getattr(active_session, "spec", None), "key", model_name)

    LOGGER.info(
        "Starting background removal request",
        extra={
            "model": model_key,
            "output_format": format_spec.key,
            "alpha_matting": bool(alpha_matting),
            "feather_radius": int(feather_radius),
        },
    )

    am_foreground = max(0, min(255, int(am_foreground)))
    am_background = max(0, min(255, int(am_background)))
    am_erode = max(0, min(255, int(am_erode)))
    colorkey_tolerance = max(0, min(255, int(colorkey_tolerance)))
    feather_radius = max(0, min(MAX_FEATHER_RADIUS, int(feather_radius)))

    start_time = time.perf_counter()
    error: str | None = None
    output_image: Image.Image | None = None
    original: Image.Image | None = None
    processed_input: Image.Image | None = None
    try:
        mask_image = _predict_mask(processed_input, active_session)
        mask_l = cast(Image.Image, ImageOps.exif_transpose(mask_image)).convert("L")
        mask_image.close()
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
        if processed_input is not None:
            processed_input.close()
        if original is not None:
            with contextlib.suppress(Exception):
                original.close()

    if error is None:
        LOGGER.info(
            "Background removal completed",
            extra={
                "model": model_key,
                "output_format": format_spec.key,
                "elapsed_ms": round(elapsed_ms, 2),
            },
        )
    else:
        LOGGER.error(
            "Background removal failed",
            extra={
                "model": model_key,
                "output_format": format_spec.key,
                "elapsed_ms": round(elapsed_ms, 2),
                "error": error,
            },
        )
    return RemovalResult(
        image=output_image,
        format_spec=format_spec,
        elapsed_ms=elapsed_ms,
        error=error,
    )


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
    session: BackgroundRemovalSession | None = None,
) -> RemovalResult:
    """Remove the background from raw ``data`` and return the processed result.

    Args:
        data: Raw image bytes to process. Must not be empty.
        output_format: Optional key describing the desired output format.
        model_name: Identifier used to resolve the ONNX session.
        alpha_matting: Whether alpha matting refinement should be applied.
        am_foreground: Foreground threshold for alpha matting.
        am_background: Background threshold for alpha matting.
        am_erode: Erosion kernel size for alpha matting.
        use_colorkey_fallback: Whether colour-key fallback is enabled.
        colorkey_tolerance: Tolerance value used for colour-key fallback.
        feather_radius: Radius for feathering the alpha channel.
        session: Optional ONNX runtime session to reuse for inference.

    Returns:
        RemovalResult: The processed image data and associated metadata.
    """

    if not data:
        raise ValueError("No image data supplied for background removal.")

    buffer = io.BytesIO(data)
    return _remove_background_from_image_loader(
        lambda: Image.open(buffer),
        output_format=output_format,
        model_name=model_name,
        alpha_matting=alpha_matting,
        am_foreground=am_foreground,
        am_background=am_background,
        am_erode=am_erode,
        use_colorkey_fallback=use_colorkey_fallback,
        colorkey_tolerance=colorkey_tolerance,
        feather_radius=feather_radius,
        session=session,
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
    """Process ``stream`` directly without materialising the entire payload."""

    try:
        stream.seek(0)
    except (AttributeError, OSError):  # pragma: no cover - seek may be unsupported
        pass

    return _remove_background_from_image_loader(
        lambda: Image.open(stream),
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
    session: BackgroundRemovalSession | None = None,
) -> RemovalResult:
    """Remove the background from ``input_path`` and write the result to ``output``."""

    # Allow callers to inject a pre-created session so batch processing can share
    # the same inference instance when running concurrently.

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
        session=session,
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

    # The folder workflow submits each image to a shared executor so that
    # background removal can be parallelised while retaining deterministic
    # ordering of the returned results.

    source_dir = Path(input_dir)
    if not source_dir.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {source_dir}")

    output_root = _resolve_output_directory(output_dir, source_dir)
    format_spec = get_output_format_spec(output_format)

    session = ensure_global_session(model_name)

    submitted: list[tuple[Path, Future[RemovalResult]]] = []
    with ThreadPoolExecutor() as executor:
        for source in _iter_input_files(source_dir, recursive):
            relative = source.relative_to(source_dir) if recursive else Path(source.name)
            destination = format_spec.normalise_filename(output_root / relative)
            destination.parent.mkdir(parents=True, exist_ok=True)
            future = executor.submit(
                remove_bg_file,
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
                session=session,
            )
            submitted.append((relative, future))

    results: list[RemovalResult] = []
    for relative, future in submitted:
        try:
            results.append(future.result())
        except Exception:
            LOGGER.error("Background removal failed for %s", relative, exc_info=True)
            raise
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
