"""ONNX runtime session management and model download helpers."""
from __future__ import annotations

import contextlib
import hashlib
import logging
import multiprocessing
import os
import queue
import threading
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

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
from ..utils.gpu_memory import query_gpu_memory
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


ProviderOptionsSignature = tuple[tuple[str, tuple[tuple[str, str], ...] | None], ...] | None
CacheKey = tuple[Path, str, int | None, bool, ProviderOptionsSignature]


_SESSION_CACHE: dict[CacheKey, BackgroundRemovalSession] = {}
_SESSION_CACHE_LOCK = threading.Lock()
_DOWNLOAD_STATUS_LOCK = threading.Lock()
_DOWNLOAD_STATUSES: dict[tuple[Path, str], DownloadStatus] = {}

try:  # pragma: no cover - depends on onnxruntime internals
    from onnxruntime.capi.onnxruntime_pybind11_state import Fail as OrtFail
except Exception:  # pragma: no cover - fall back when symbol not exposed
    OrtFail = RuntimeError  # type: ignore[assignment]


def _normalise_cuda_options(
    options: Mapping[str, Any] | None,
    *,
    max_performance: bool = False,
) -> dict[str, Any]:
    """Return CUDA provider options augmented with performance aware defaults.

    The function keeps the defaults permissive so that inference can saturate
    modern GPUs. When ``max_performance`` is ``True`` every soft restriction is
    removed and additional cuDNN/graph optimisations are enabled to keep the
    device fully utilised. In conservative mode a VRAM limit derived from the
    currently available memory is retained to reduce the likelihood of OOM
    errors on shared systems.
    """

    merged: dict[str, Any] = {}
    if options:
        merged.update({str(key): value for key, value in options.items()})

    merged.setdefault("arena_extend_strategy", "kSameAsRequested")
    merged.setdefault("do_copy_in_default_stream", "1")

    if max_performance:
        # Remove every VRAM restriction and lean on the CUDA provider to grab
        # as much memory as it needs. Additional tunable op features give
        # recent GPUs the flexibility to pick the fastest kernels.
        merged.pop("gpu_mem_limit", None)
        merged["cudnn_conv_use_max_workspace"] = "1"
        merged["cudnn_conv_algo_search"] = "EXHAUSTIVE"
        merged["tunable_op_enable"] = "1"
        merged["tunable_op_tuning_enable"] = "1"
        merged["enable_cuda_graph"] = "1"
        LOGGER.info(
            "Max Performance: enabling unrestricted CUDAExecutionProvider with exhaustive "
            "algorithms.",
        )
        return merged

    merged.setdefault("cudnn_conv_use_max_workspace", "1")
    merged.setdefault("cudnn_conv_algo_search", "HEURISTIC")

    snapshot = query_gpu_memory()
    limit_bytes: int | None = None
    if snapshot is not None and snapshot.free > 0:
        limit_bytes = int(snapshot.free * 0.8)
        LOGGER.info(
            "Configuring CUDAExecutionProvider gpu_mem_limit to %s MiB based on NVML free memory.",
            limit_bytes // (1024 * 1024),
        )
    if limit_bytes is not None and limit_bytes > 0:
        merged["gpu_mem_limit"] = str(limit_bytes)
    elif "gpu_mem_limit" in merged:
        merged["gpu_mem_limit"] = str(merged["gpu_mem_limit"])
    else:
        LOGGER.info(
            "CUDAExecutionProvider memory limit could not be determined from NVML; proceeding "
            "without an explicit limit.",
        )

    return merged


def _apply_cuda_provider_defaults(
    entry: ProviderEntry,
    *,
    max_performance: bool = False,
) -> ProviderEntry:
    """Return ``entry`` with adaptive CUDA provider defaults applied."""

    if isinstance(entry, tuple):
        name, options = entry
    else:
        name, options = entry, {}
    if name != "CUDAExecutionProvider":
        return entry

    if isinstance(options, Mapping):
        raw_options: Mapping[str, Any] = options
    else:  # pragma: no cover - defensive fallback for unexpected sequences
        raw_options = dict(options)  # type: ignore[arg-type]
    return (name, _normalise_cuda_options(raw_options, max_performance=max_performance))


def _normalise_providers(
    providers: Sequence[ProviderEntry],
    *,
    max_performance: bool = False,
) -> list[ProviderEntry]:
    """Return ``providers`` augmented with safe defaults for GPU execution."""

    normalised: list[ProviderEntry] = []
    for entry in providers:
        if isinstance(entry, tuple):
            name = entry[0]
        else:
            name = entry
        if name == "CUDAExecutionProvider":
            normalised.append(
                _apply_cuda_provider_defaults(entry, max_performance=max_performance)
            )
        else:
            normalised.append(entry)
    return normalised


def _provider_options_signature(
    providers: Sequence[ProviderEntry] | None,
) -> ProviderOptionsSignature:
    """Return a hashable signature representing ``providers`` for caching."""

    if not providers:
        return None

    signature: list[tuple[str, tuple[tuple[str, str], ...] | None]] = []
    for entry in providers:
        if isinstance(entry, tuple):
            name, options = entry
            if isinstance(options, Mapping):
                raw_items = options.items()
            else:  # pragma: no cover - defensive fallback for unexpected sequences
                raw_items = dict(options).items()  # type: ignore[arg-type]
            normalised_options = tuple(
                sorted((str(key), str(value)) for key, value in raw_items)
            )
            signature.append((name, normalised_options))
        else:
            signature.append((entry, None))
    return tuple(signature)


def _provider_name(entry: ProviderEntry) -> str:
    """Return the provider name extracted from ``entry``."""

    if isinstance(entry, tuple):
        return entry[0]
    return entry


def _match_provider_hint(hint: str, available: Sequence[str]) -> str | None:
    """Return an available provider matching ``hint`` or ``None`` when missing."""

    candidate = hint.strip()
    if not candidate:
        return None

    canonical = candidate.lower()
    alias_map = {
        "cpu": "CPUExecutionProvider",
        "cpuexecutionprovider": "CPUExecutionProvider",
        "cuda": "CUDAExecutionProvider",
        "cudaexecutionprovider": "CUDAExecutionProvider",
        "gpu": "CUDAExecutionProvider",
        "dml": "DmlExecutionProvider",
        "dmlexecutionprovider": "DmlExecutionProvider",
        "directml": "DmlExecutionProvider",
    }

    mapped = alias_map.get(canonical)
    if mapped and mapped in available:
        return mapped

    for provider in available:
        if provider.lower() == canonical:
            return provider

    return None


def detect_providers(provider_hints: Iterable[str] | None = None) -> list[str]:
    """Return the preferred execution providers available for inference."""

    available = list(ort.get_available_providers())
    if not available:
        available = ["CPUExecutionProvider"]

    normalised: list[str] = []
    hints = [hint for hint in provider_hints or () if str(hint).strip()]

    if hints:
        for hint in hints:
            provider = _match_provider_hint(str(hint), available)
            if provider and provider not in normalised:
                normalised.append(provider)
        if normalised:
            return normalised
        LOGGER.warning(
            "Provider hints %s could not be satisfied; defaulting to CPUExecutionProvider.",
            hints,
        )
        if "CPUExecutionProvider" in available:
            return ["CPUExecutionProvider"]
        return [available[0]]

    if "CUDAExecutionProvider" in available and "CUDAExecutionProvider" not in normalised:
        normalised.append("CUDAExecutionProvider")
    for provider in available:
        if provider not in normalised:
            normalised.append(provider)
    if "CPUExecutionProvider" not in normalised and "CPUExecutionProvider" in available:
        normalised.append("CPUExecutionProvider")
    return normalised or ["CPUExecutionProvider"]


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
    """Small wrapper around an :class:`onnxruntime.InferenceSession`.

    The ``disable_cuda_graph`` flag instructs ONNX Runtime to skip CUDA graph
    capture, which improves stability when multiple GPU-bound sessions execute
    concurrently on dedicated worker threads.
    """

    def __init__(
        self,
        spec: ModelSpec,
        model_path: Path,
        *,
        providers: Sequence[str],
        max_performance: bool = False,
        disable_cuda_graph: bool = False,
    ) -> None:
        self.spec = spec
        self.model_path = Path(model_path)
        session_options = ort.SessionOptions()
        max_perf_enabled = bool(max_performance)
        thread_override: int | None = None
        if max_perf_enabled:
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            session_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
            session_options.enable_mem_pattern = True
            session_options.enable_mem_reuse = True
            session_options.add_session_config_entry("session.intra_op_allow_spinning", "1")
            session_options.add_session_config_entry("session.use_env_allocators", "1")
            try:
                thread_override = multiprocessing.cpu_count()
            except NotImplementedError:  # pragma: no cover - platform specific
                thread_override = None
            if thread_override and thread_override > 0:
                session_options.inter_op_num_threads = thread_override
                session_options.intra_op_num_threads = thread_override
        else:
            threads = os.getenv("OMP_NUM_THREADS")
            if threads:
                try:
                    thread_override = int(threads)
                except ValueError:
                    thread_override = None
                else:
                    session_options.inter_op_num_threads = thread_override
                    session_options.intra_op_num_threads = thread_override

        if disable_cuda_graph:
            session_options.add_session_config_entry("session.disable_cuda_graph", "1")
            with contextlib.suppress(Exception):
                session_options.add_session_config_entry("session.use_cuda_graph", "0")

        provider_entries = _normalise_providers(
            list(providers) or ["CPUExecutionProvider"],
            max_performance=max_perf_enabled,
        )
        provider_names = [_provider_name(entry) for entry in provider_entries]
        LOGGER.info("Using providers: %s", provider_names)
        if max_perf_enabled:
            gpu_mem_limit: str | None = None
            for entry in provider_entries:
                if isinstance(entry, tuple) and entry[0] == "CUDAExecutionProvider":
                    options = entry[1]
                    if isinstance(options, Mapping):
                        raw_limit = options.get("gpu_mem_limit")
                        if raw_limit is not None:
                            try:
                                gpu_mem_limit = str(int(raw_limit))
                            except (TypeError, ValueError):
                                gpu_mem_limit = str(raw_limit)
                    break
            LOGGER.info(
                "Max Performance: GPU VRAM limit=%s, threads=%s",
                gpu_mem_limit or "unrestricted",
                thread_override if thread_override else "default",
            )
        try:
            self.inner = ort.InferenceSession(
                str(self.model_path),
                sess_options=session_options,
                providers=provider_entries,
            )
        except Exception as error:  # pragma: no cover - depends on onnxruntime
            if any(_provider_name(entry) == "CUDAExecutionProvider" for entry in provider_entries):
                LOGGER.warning(
                    "CUDAExecutionProvider initialisation failed: %s. Falling back to CPUExecutionProvider.",
                    error,
                )
                cpu_only = [
                    entry
                    for entry in provider_entries
                    if _provider_name(entry) == "CPUExecutionProvider"
                ] or ["CPUExecutionProvider"]
                try:
                    self.inner = ort.InferenceSession(
                        str(self.model_path),
                        sess_options=session_options,
                        providers=cpu_only,
                    )
                except Exception as cpu_error:  # pragma: no cover - depends on onnxruntime
                    message = (
                        f"Failed to load ONNX model {spec.key} from {self.model_path}: {cpu_error}"
                    )
                    raise ModelUnavailableError(message) from cpu_error
            else:
                message = f"Failed to load ONNX model {spec.key} from {self.model_path}: {error}"
                raise ModelUnavailableError(message) from error
        self.input_name = self.inner.get_inputs()[0].name
        providers_available = self.inner.get_providers()
        self.providers_available = tuple(providers_available)
        self.primary_provider = (
            providers_available[0] if providers_available else "CPUExecutionProvider"
        )
        LOGGER.debug(
            "Initialised BackgroundRemovalSession id=%s providers=%s",
            hex(id(getattr(self, "inner", self))),
            self.providers_available,
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


def _initialise_session(
    model_key: str,
    *,
    providers: Sequence[ProviderEntry] | None,
    model_dir: Path,
    max_performance: bool,
    disable_cuda_graph: bool,
) -> BackgroundRemovalSession:
    """Return a freshly initialised :class:`BackgroundRemovalSession`."""

    resolved_dir = model_dir.expanduser()
    resolved_dir.mkdir(parents=True, exist_ok=True)
    spec = _resolve_spec(model_key)
    provider_entries = _normalise_providers(
        list(providers or detect_providers()),
        max_performance=max_performance,
    )
    try:
        model_path = _download_model(spec, resolved_dir)
    except ModelUnavailableError:
        raise
    except Exception as error:  # pragma: no cover - defensive wrapper
        message = f"Failed to prepare model {spec.key}: {error}"
        raise ModelUnavailableError(message) from error

    return BackgroundRemovalSession(
        spec,
        model_path,
        providers=provider_entries,
        max_performance=max_performance,
        disable_cuda_graph=disable_cuda_graph,
    )


def get_session(
    model_key: str,
    *,
    providers: Sequence[ProviderEntry] | None = None,
    model_dir: Path | None = None,
    max_performance: bool = False,
    thread_isolated: bool = False,
) -> BackgroundRemovalSession:
    """Return a cached :class:`BackgroundRemovalSession` for ``model_key``.

    When ``thread_isolated`` is :data:`True` a unique session instance is
    created per calling thread to avoid sharing CUDA streams across workers.
    """

    resolved_dir = Path(model_dir or _default_model_dir()).expanduser()
    thread_key = threading.get_ident() if thread_isolated else None

    provider_entries: Sequence[ProviderEntry] | None
    if providers is None:
        provider_entries = None
    else:
        provider_entries = tuple(providers)
    provider_signature = _provider_options_signature(provider_entries)

    cache_key: CacheKey = (
        resolved_dir,
        model_key,
        thread_key,
        bool(max_performance),
        provider_signature,
    )

    with _SESSION_CACHE_LOCK:
        cached = _SESSION_CACHE.get(cache_key)
    if cached is not None:
        if thread_isolated and not getattr(cached, "_thread_isolated", False):
            cached._thread_isolated = True  # type: ignore[attr-defined]
        elif not thread_isolated and getattr(cached, "_thread_isolated", False):
            cached._thread_isolated = False  # type: ignore[attr-defined]
        return cached

    session = _initialise_session(
        model_key,
        providers=provider_entries,
        model_dir=resolved_dir,
        max_performance=max_performance,
        disable_cuda_graph=thread_isolated,
    )
    if thread_isolated:
        session._thread_isolated = True  # type: ignore[attr-defined]
        LOGGER.debug(
            "Created isolated ONNX session for thread %s",
            threading.current_thread().name,
        )
    else:
        session._thread_isolated = False  # type: ignore[attr-defined]
    with _SESSION_CACHE_LOCK:
        _SESSION_CACHE[cache_key] = session
    return session


class SessionPool:
    """Pool of reusable :class:`BackgroundRemovalSession` instances."""

    def __init__(
        self,
        model_key: str,
        *,
        size: int,
        providers: Sequence[ProviderEntry] | None = None,
        model_dir: Path | None = None,
        max_performance: bool = False,
        disable_cuda_graph: bool = False,
    ) -> None:
        if size <= 0:
            raise ValueError("SessionPool size must be a positive integer")
        self._model_key = model_key
        self._model_dir = Path(model_dir or _default_model_dir()).expanduser()
        self._providers: Sequence[ProviderEntry] | None = tuple(providers or ()) or None
        self._max_performance = bool(max_performance)
        self._disable_cuda_graph = bool(disable_cuda_graph)
        self._sessions: list[BackgroundRemovalSession] = []
        self._queue: queue.LifoQueue[BackgroundRemovalSession] = queue.LifoQueue()
        self._lock = threading.Lock()
        self._closed = False
        for index in range(size):
            session = _initialise_session(
                model_key,
                providers=self._providers,
                model_dir=self._model_dir,
                max_performance=self._max_performance,
                disable_cuda_graph=self._disable_cuda_graph,
            )
            self._sessions.append(session)
            self._queue.put(session)
            session_id = hex(id(getattr(session, "inner", session)))
            LOGGER.debug("SessionPool initialised session #%s id=%s", index, session_id)

    def acquire(self) -> SessionLease:
        """Return a context manager leasing a session from the pool."""

        return SessionLease(self)

    def _acquire(self) -> BackgroundRemovalSession:
        if self._closed:
            raise RuntimeError("SessionPool has been closed")
        session = self._queue.get()
        LOGGER.debug(
            "Leased session id=%s to thread=%s",
            hex(id(getattr(session, "inner", session))),
            threading.current_thread().name,
        )
        session._from_pool = True  # type: ignore[attr-defined]
        return session

    def _release(self, session: BackgroundRemovalSession) -> None:
        if self._closed:
            release_session(session)
            return
        session._from_pool = False  # type: ignore[attr-defined]
        self._queue.put(session)
        LOGGER.debug(
            "Returned session id=%s to pool by thread=%s",
            hex(id(getattr(session, "inner", session))),
            threading.current_thread().name,
        )

    def close(self) -> None:
        """Release all sessions managed by the pool."""

        with self._lock:
            if self._closed:
                return
            self._closed = True
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        for session in self._sessions:
            release_session(session)
        self._sessions.clear()

    def __enter__(self) -> SessionPool:
        return self

    def __exit__(self, exc_type, exc, exc_tb) -> None:
        self.close()


class SessionLease(contextlib.AbstractContextManager[BackgroundRemovalSession]):
    """Context manager for leasing a session from :class:`SessionPool`."""

    def __init__(self, pool: SessionPool) -> None:
        self._pool = pool
        self._session: BackgroundRemovalSession | None = None

    def __enter__(self) -> BackgroundRemovalSession:
        self._session = self._pool._acquire()
        return self._session

    def __exit__(self, exc_type, exc, exc_tb) -> None:
        if self._session is not None:
            self._pool._release(self._session)
            self._session = None

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
    "SessionPool",
    "SessionLease",
    "release_session",
]
