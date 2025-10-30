"""ONNX runtime session management and model download helpers."""

from __future__ import annotations

import contextlib
import hashlib
import logging
import os
import subprocess
import threading
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

try:  # pragma: no cover - optional dependency fallback
    import pynvml  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - optional dependency fallback
    pynvml = None

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

from ..paths import MODELS_DIR
from .specs import MODEL_SPECS, ModelSpec

LOGGER = logging.getLogger(__name__)


ProviderEntry = str | tuple[str, Mapping[str, Any]]


class ModelUnavailableError(RuntimeError):
    """Raised when an ONNX model cannot be prepared for inference."""

    def __init__(self, message: str, *, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class ModelDownloadError(RuntimeError):
    """Raised when a download attempt fails while fetching model weights."""

    def __init__(self, message: str, *, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class DownloadStatus:
    """Snapshot describing the current state of a model download."""

    key: str
    state: str
    path: Path | None = None
    error: str | None = None
    attempts: int = 0
    updated_at: float = field(default_factory=time.time)


DOWNLOAD_AVAILABLE = "available"
DOWNLOAD_PENDING = "pending"
DOWNLOAD_IN_PROGRESS = "in-progress"
DOWNLOAD_FAILED = "failed"


_SESSION_CACHE: dict[tuple[Path, str], BackgroundRemovalSession] = {}
_SESSION_CACHE_LOCK = threading.Lock()
_DOWNLOAD_STATUS_LOCK = threading.Lock()
_DOWNLOAD_STATUSES: dict[tuple[Path, str], DownloadStatus] = {}


def get_gpu_memory(device_index: int = 0) -> tuple[int | None, int | None, int | None]:
    """Return the total, free, and used VRAM for ``device_index`` in bytes."""

    nvml_initialised = False
    try:
        if pynvml is None:
            raise RuntimeError("pynvml is not available")
        pynvml.nvmlInit()
        nvml_initialised = True
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        total = int(info.total)
        free = int(info.free)
        used = int(info.used)
        return total, free, used
    except Exception as error:  # pragma: no cover - depends on NVML availability
        LOGGER.warning("Failed to query GPU memory via NVML: %s", error)
        try:
            output = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.total,memory.free,memory.used",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
            )
        except Exception as fallback_error:  # pragma: no cover - depends on environment
            LOGGER.warning(
                "Fallback nvidia-smi memory query failed: %s",
                fallback_error,
            )
            return None, None, None
        lines = [line.strip() for line in output.splitlines() if line.strip()]
        if not lines:
            return None, None, None
        target = lines[device_index] if device_index < len(lines) else lines[0]
        try:
            total_mib, free_mib, used_mib = (
                int(part.strip()) for part in target.split(",")
            )
        except ValueError:  # pragma: no cover - defensive parsing guard
            return None, None, None
        mib_to_bytes = 1024 * 1024
        return (
            total_mib * mib_to_bytes,
            free_mib * mib_to_bytes,
            used_mib * mib_to_bytes,
        )
    finally:
        if nvml_initialised:
            try:  # pragma: no cover - NVML shutdown failures are benign
                pynvml.nvmlShutdown()
            except Exception:
                pass


def _provider_name(entry: ProviderEntry) -> str:
    """Return the provider name extracted from ``entry``."""

    if isinstance(entry, tuple):
        return entry[0]
    return entry


def detect_providers(provider_hints: Iterable[str] | None = None) -> list[str]:
    """Return the preferred execution providers available for inference."""

    available = list(ort.get_available_providers())
    if not available:
        available = ["CPUExecutionProvider"]

    normalised: list[str] = []
    hints = [hint.strip().lower() for hint in provider_hints or () if hint.strip()]
    for hint in hints:
        if hint.startswith("cuda") and "CUDAExecutionProvider" in available:
            normalised.append("CUDAExecutionProvider")
        if hint.startswith("cpu") and "CPUExecutionProvider" in available:
            normalised.append("CPUExecutionProvider")
        if hint.startswith("directml") and "DmlExecutionProvider" in available:
            normalised.append("DmlExecutionProvider")
    if "CUDAExecutionProvider" in available and "CUDAExecutionProvider" not in normalised:
        normalised.insert(0, "CUDAExecutionProvider")
    for provider in available:
        if provider not in normalised:
            normalised.append(provider)
    if "CPUExecutionProvider" not in normalised:
        normalised.append("CPUExecutionProvider")
    return normalised


def _build_model_filename(spec: ModelSpec) -> str:
    """Return the expected filename for ``spec`` within the models directory."""

    if spec.local_filename:
        return spec.local_filename
    return f"{spec.key}.onnx"


def _verify_md5(path: Path, expected: str | None) -> bool:
    """Return ``True`` when the file at ``path`` matches ``expected``."""

    if not path.exists():
        return False
    if not expected:
        return True
    checksum = hashlib.md5()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest() == expected.lower()


def _default_model_dir() -> Path:
    """Return the default model directory used when none is supplied."""

    env_dir = os.getenv("MODEL_DIR")
    if env_dir:
        return Path(env_dir).expanduser()
    return MODELS_DIR


def _update_download_status(
    model_dir: Path,
    key: str,
    *,
    state: str,
    path: Path | None = None,
    error: str | None = None,
    attempts: int | None = None,
) -> DownloadStatus:
    """Update the cached download status for ``key`` in ``model_dir``."""

    status = DownloadStatus(
        key=key,
        state=state,
        path=path,
        error=error,
        attempts=attempts or 0,
    )
    with _DOWNLOAD_STATUS_LOCK:
        _DOWNLOAD_STATUSES[(model_dir, key)] = status
    return status


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


def _download_model_via_http(
    spec: ModelSpec,
    url: str,
    destination: Path,
    headers: Mapping[str, str],
) -> Path:
    """Download ``spec`` from ``url`` into ``destination`` using ``headers``."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.with_suffix(".tmp")
    try:
        with contextlib.ExitStack() as stack:
            response = stack.enter_context(
                requests.get(url, headers=dict(headers), stream=True, timeout=60)
            )
            try:
                response.raise_for_status()
            except requests.HTTPError as error:  # pragma: no cover - depends on network
                tmp_path.unlink(missing_ok=True)
                status_code = getattr(getattr(error, "response", None), "status_code", None)
                if status_code is None:
                    status_code = getattr(error, "status_code", None)
                message = f"Failed to download {spec.key} from {url}: {error}"
                raise ModelDownloadError(message, status_code=status_code) from error
            with stack.enter_context(tmp_path.open("wb")) as buffer:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    buffer.write(chunk)
    except requests.RequestException as error:  # pragma: no cover - depends on network
        tmp_path.unlink(missing_ok=True)
        status_code = getattr(error, "status_code", None)
        message = f"Failed to download {spec.key} from {url}: {error}"
        raise ModelDownloadError(message, status_code=status_code) from error
    if not _verify_md5(tmp_path, spec.checksum_md5):
        tmp_path.unlink(missing_ok=True)
        raise ModelDownloadError(f"Checksum mismatch for model {spec.key}")
    tmp_path.replace(destination)
    return destination


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
    return _download_model_via_http(spec, url, destination, headers)


def _format_download_failure(spec: ModelSpec, error: ModelDownloadError | Exception) -> str:
    """Return a human-readable message describing a download failure."""

    status_code = getattr(error, "status_code", None)
    base = f"Unable to download model '{spec.key}'."
    if status_code == 403:
        return (
            f"{base} Access was denied (HTTP 403). Supply a Hugging Face token or download the "
            "weights manually into the models directory."
        )
    if status_code == 404:
        return (
            f"{base} The requested file was not found (HTTP 404). Confirm the filename and revision."
        )
    return f"{base} {error}"


def _download_model(
    spec: ModelSpec,
    model_dir: Path,
    *,
    max_attempts: int = 3,
    backoff_base: float = 0.5,
) -> Path:
    """Ensure ``spec`` exists under ``model_dir`` and return the path."""

    destination = model_dir / _build_model_filename(spec)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if _verify_md5(destination, spec.checksum_md5):
        _update_download_status(model_dir, spec.key, state=DOWNLOAD_AVAILABLE, path=destination)
        return destination

    model_dir.mkdir(parents=True, exist_ok=True)

    attempts = 0
    while attempts < max_attempts:
        attempts += 1
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
        except ModelDownloadError as error:
            message = _format_download_failure(spec, error)
            LOGGER.warning("Attempt %s to download %s failed: %s", attempts, spec.key, message)
            _update_download_status(
                model_dir,
                spec.key,
                state=DOWNLOAD_PENDING if attempts < max_attempts else DOWNLOAD_FAILED,
                path=destination if destination.exists() else None,
                error=message,
                attempts=attempts,
            )
            if attempts >= max_attempts:
                raise ModelUnavailableError(message, status_code=error.status_code) from error
            delay = backoff_base * (2 ** (attempts - 1))
            time.sleep(delay)
            continue
        except Exception as error:
            message = str(error)
            LOGGER.warning("Attempt %s to download %s failed: %s", attempts, spec.key, message)
            _update_download_status(
                model_dir,
                spec.key,
                state=DOWNLOAD_PENDING if attempts < max_attempts else DOWNLOAD_FAILED,
                path=destination if destination.exists() else None,
                error=message,
                attempts=attempts,
            )
            if attempts >= max_attempts:
                raise ModelUnavailableError(
                    f"Failed to download model {spec.key}: {message}"
                ) from error
            delay = backoff_base * (2 ** (attempts - 1))
            time.sleep(delay)
            continue

        _update_download_status(
            model_dir,
            spec.key,
            state=DOWNLOAD_AVAILABLE,
            path=path,
            error=None,
            attempts=attempts,
        )
        LOGGER.info("Model %s stored at %s", spec.key, path)
        return path

    raise ModelUnavailableError(f"Failed to download {spec.key}")


class BackgroundRemovalSession:
    """Small wrapper around an :class:`onnxruntime.InferenceSession`."""

    def __init__(
        self,
        spec: ModelSpec,
        model_path: Path,
        *,
        providers: Sequence[ProviderEntry],
    ) -> None:
        self.spec = spec
        self.model_path = Path(model_path)
        session_options = ort.SessionOptions()
        threads = os.getenv("OMP_NUM_THREADS")
        if threads:
            try:
                value = int(threads)
            except ValueError:
                value = None
            else:
                session_options.inter_op_num_threads = value
                session_options.intra_op_num_threads = value

        requested = list(providers or ["CPUExecutionProvider"])
        extras: list[ProviderEntry] = []
        extras_seen: set[str] = set()
        wants_cuda = False
        for entry in requested:
            name = _provider_name(entry)
            if name == "CUDAExecutionProvider":
                wants_cuda = True
                continue
            if name == "CPUExecutionProvider":
                continue
            if name in extras_seen:
                continue
            extras.append(entry)
            extras_seen.add(name)

        provider_entries: list[ProviderEntry] = []
        total_vram: int | None = None
        free_vram: int | None = None
        gpu_mem_limit = 0
        cuda_provider_options: dict[str, Any] | None = None

        if wants_cuda:
            total_vram, free_vram, _ = get_gpu_memory(0)
            if free_vram:
                gpu_mem_limit = int(free_vram * 0.9)
            LOGGER.info(
                "Detected GPU memory total=%s, free=%s, setting limit=%s",
                total_vram,
                free_vram,
                gpu_mem_limit,
            )
            cuda_provider_options = {
                "device_id": 0,
                "arena_extend_strategy": "kSameAsRequested",
                "gpu_mem_limit": gpu_mem_limit,
                "cudnn_conv_algo_search": "EXHAUSTIVE",
            }
            provider_entries.append(("CUDAExecutionProvider", cuda_provider_options))

        provider_entries.extend(extras)
        provider_entries.append(("CPUExecutionProvider", {}))

        try:
            self.inner = ort.InferenceSession(
                str(self.model_path),
                sess_options=session_options,
                providers=provider_entries,
            )
        except Exception as error:  # pragma: no cover - depends on onnxruntime
            if wants_cuda:
                LOGGER.warning(
                    "GPU session failed (%s); retrying with reduced memory limit...",
                    error,
                )
                if gpu_mem_limit > 0 and cuda_provider_options is not None:
                    reduced_limit = int(gpu_mem_limit * 0.8)
                    cuda_provider_options["gpu_mem_limit"] = reduced_limit
                    try:
                        self.inner = ort.InferenceSession(
                            str(self.model_path),
                            sess_options=session_options,
                            providers=provider_entries,
                        )
                    except Exception as retry_error:  # pragma: no cover - depends on onnxruntime
                        LOGGER.error(
                            "GPU retry failed (%s); falling back to CPU provider.",
                            retry_error,
                        )
                        try:
                            self.inner = ort.InferenceSession(
                                str(self.model_path),
                                sess_options=session_options,
                                providers=["CPUExecutionProvider"],
                            )
                        except Exception as cpu_error:  # pragma: no cover - depends on onnxruntime
                            message = (
                                f"Failed to load ONNX model {spec.key} from {self.model_path}: {cpu_error}"
                            )
                            raise ModelUnavailableError(message) from cpu_error
                else:
                    LOGGER.error(
                        "No GPU memory limit available; falling back to CPU provider.",
                    )
                    try:
                        self.inner = ort.InferenceSession(
                            str(self.model_path),
                            sess_options=session_options,
                            providers=["CPUExecutionProvider"],
                        )
                    except Exception as cpu_error:  # pragma: no cover - depends on onnxruntime
                        message = (
                            f"Failed to load ONNX model {spec.key} from {self.model_path}: {cpu_error}"
                        )
                        raise ModelUnavailableError(message) from cpu_error
            else:
                message = f"Failed to load ONNX model {spec.key} from {self.model_path}: {error}"
                raise ModelUnavailableError(message) from error
        LOGGER.info("Active providers: %s", self.inner.get_providers())
        self.input_name = self.inner.get_inputs()[0].name
        providers_available = self.inner.get_providers()
        self.providers_available = tuple(providers_available)
        self.primary_provider = (
            providers_available[0] if providers_available else "CPUExecutionProvider"
        )

    def run(self, tensor: np.ndarray) -> np.ndarray:
        """Execute inference using ``tensor`` and return the first output."""

        outputs = self.inner.run(None, {self.input_name: tensor})
        return np.asarray(outputs[0])


def _resolve_spec(model_key: str) -> ModelSpec:
    """Return the :class:`ModelSpec` for ``model_key`` falling back to default."""

    spec = MODEL_SPECS.get(model_key)
    if spec is None:
        LOGGER.warning(
            "Unknown model %s requested; falling back to %s", model_key, "isnet-general-use"
        )
        spec = MODEL_SPECS["isnet-general-use"]
    return spec


def get_session(
    model_key: str,
    *,
    providers: Sequence[ProviderEntry] | None = None,
    model_dir: Path | None = None,
) -> BackgroundRemovalSession:
    """Return a cached :class:`BackgroundRemovalSession` for ``model_key``."""

    resolved_dir = Path(model_dir or _default_model_dir()).expanduser()
    resolved_dir.mkdir(parents=True, exist_ok=True)
    cache_key = (resolved_dir, model_key)

    with _SESSION_CACHE_LOCK:
        cached = _SESSION_CACHE.get(cache_key)
    if cached is not None:
        return cached

    spec = _resolve_spec(model_key)
    providers = list(providers or detect_providers())
    try:
        model_path = _download_model(spec, resolved_dir)
    except ModelUnavailableError:
        raise
    except Exception as error:  # pragma: no cover - defensive wrapper
        message = f"Failed to prepare model {spec.key}: {error}"
        raise ModelUnavailableError(message) from error

    session = BackgroundRemovalSession(spec, model_path, providers=providers)
    with _SESSION_CACHE_LOCK:
        _SESSION_CACHE[cache_key] = session
    return session


def release_session(session: BackgroundRemovalSession | None) -> None:
    """Remove ``session`` from caches and drop references to release resources."""

    if session is None:
        return

    with _SESSION_CACHE_LOCK:
        keys_to_delete = [
            key for key, cached in _SESSION_CACHE.items() if cached is session
        ]
        for key in keys_to_delete:
            _SESSION_CACHE.pop(key, None)

    if hasattr(session, "inner"):
        try:
            session.inner = None  # type: ignore[assignment]
        except Exception:  # pragma: no cover - defensive fallback
            pass


__all__ = [
    "BackgroundRemovalSession",
    "DownloadStatus",
    "ModelDownloadError",
    "ModelUnavailableError",
    "detect_providers",
    "get_session",
    "release_session",
]
