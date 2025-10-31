"""Tests for the shared processing pipeline."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from bgremover_core.config import Config
from bgremover_core.models.specs import ModelSpec
from bgremover_core.processing import pipeline
from bgremover_core.utils.gpu_memory import GpuMemorySnapshot


class StubSession:
    def __init__(self, spec: ModelSpec) -> None:
        self.spec = spec
        self.input_name = "input"
        self.providers_available = ("CPUExecutionProvider",)
        self.primary_provider = self.providers_available[0]
        self.model_path = Path("/tmp/model.onnx")

    def run(self, _tensor):  # pragma: no cover - simple stub
        width, height = self.spec.input_size
        return np.ones((1, height, width), dtype=np.float32)


@pytest.fixture()
def stub_session(monkeypatch: pytest.MonkeyPatch) -> None:
    spec = ModelSpec(
        key="test-model",
        input_size=(32, 32),
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
    )
    session = StubSession(spec)
    monkeypatch.setattr(pipeline, "_prepare_session", lambda *_args, **_kwargs: session)


def test_remove_background_returns_rgba_array(stub_session: None) -> None:
    config = Config(model_dir=Path("/tmp/models"))
    image = np.zeros((32, 32, 4), dtype=np.uint8)
    result = pipeline.remove_background(image, "test-model", config=config)
    assert result.shape == (32, 32, 4)
    assert result.dtype == np.uint8


def test_process_image_returns_processing_result(stub_session: None) -> None:
    config = Config(model_dir=Path("/tmp/models"))
    image = np.zeros((32, 32, 4), dtype=np.uint8)
    result = pipeline.process_image(image, model_key="test-model", config=config)
    assert isinstance(result, pipeline.ProcessingResult)
    assert result.image.size == (32, 32)
    assert result.mask.shape == (32, 32)
    assert result.alpha.shape == (32, 32)


def test_remove_background_failure_raises_pipeline_error() -> None:
    config = Config(model_dir=Path("/tmp/models"))
    with pytest.raises(pipeline.PipelineError):
        pipeline.remove_background(np.zeros((10, 10), dtype=np.uint8), "test-model", config=config)


def test_process_folder_reports_results(tmp_path: Path, stub_session: None) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    image_path = input_dir / "sample.png"
    Image.new("RGBA", (32, 32), color=(255, 0, 0, 255)).save(image_path)

    report = pipeline.process_folder(
        input_dir,
        pattern="*.png",
        model_key="test-model",
        config=Config(model_dir=tmp_path),
    )
    assert report.total == 1
    assert report.successes == 1
    assert report.failures == 0
    # Validate that the generated file path matches expectations and that the
    # processed image is actually written to disk.
    path_out = report.entries[0].path_out
    assert path_out is not None
    assert path_out.exists()
    assert path_out.stem.endswith("_no_bg")

    # Attempt to open the generated file to catch silent write failures.
    with Image.open(path_out) as processed_image:
        processed_image.verify()


def test_process_folder_clamps_workers_for_gpu(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog):
    class GpuSession(StubSession):
        def __init__(self, spec: ModelSpec) -> None:
            super().__init__(spec)
            self.providers_available = ("CUDAExecutionProvider",)
            self.primary_provider = "CUDAExecutionProvider"

    spec = ModelSpec(
        key="test-model",
        input_size=(32, 32),
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
    )
    session = GpuSession(spec)
    monkeypatch.setattr(pipeline, "_prepare_session", lambda *_args, **_kwargs: session)

    entries: list[pipeline.ReportEntry] = []

    def fake_process(path: Path, **kwargs):
        options = kwargs["options"]
        entries.append(
            pipeline.ReportEntry(path_in=path, path_out=path, success=True, elapsed_ms=0.0)
        )
        assert options.max_workers == 1
        return entries[-1]

    monkeypatch.setattr(pipeline, "_process_single_path", fake_process)

    class FailExecutor:
        def __init__(self, *_args, **_kwargs) -> None:  # pragma: no cover - defensive
            raise AssertionError("ThreadPoolExecutor should not be used for GPU providers")

    monkeypatch.setattr(pipeline, "ThreadPoolExecutor", FailExecutor)

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    image_path = input_dir / "sample.png"
    Image.new("RGBA", (32, 32), color=(255, 0, 0, 255)).save(image_path)

    caplog.set_level("INFO")

    report = pipeline.process_folder(
        input_dir,
        pattern="*.png",
        model_key="test-model",
        config=Config(model_dir=tmp_path),
        max_workers=4,
    )

    assert report.total == 1
    assert entries
    assert "falling back to a single worker" in caplog.text


def test_process_folder_uses_thread_pool_when_memory_allows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class GpuSession(StubSession):
        def __init__(self, spec: ModelSpec) -> None:
            super().__init__(spec)
            self.providers_available = ("CUDAExecutionProvider",)
            self.primary_provider = "CUDAExecutionProvider"

    spec = ModelSpec(
        key="test-model",
        input_size=(32, 32),
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
    )
    session = GpuSession(spec)
    monkeypatch.setattr(pipeline, "_prepare_session", lambda *_args, **_kwargs: session)
    monkeypatch.setattr(pipeline, "recommend_worker_count", lambda *_args, **_kwargs: 2)
    monkeypatch.setattr(
        pipeline,
        "query_gpu_memory",
        lambda: GpuMemorySnapshot(total=8 * 1024**3, free=8 * 1024**3),
    )

    created_entries: list[pipeline.ReportEntry] = []

    def fake_process(path: Path, **_: object) -> pipeline.ReportEntry:
        entry = pipeline.ReportEntry(path_in=path, path_out=path, success=True, elapsed_ms=0.0)
        created_entries.append(entry)
        return entry

    monkeypatch.setattr(pipeline, "_process_single_path", fake_process)

    created_executors: list[ThreadPoolExecutor] = []

    class TrackingExecutor(ThreadPoolExecutor):
        def __init__(self, *args, **kwargs):
            self.max_workers_seen = kwargs.get("max_workers")
            super().__init__(*args, **kwargs)
            created_executors.append(self)

    monkeypatch.setattr(pipeline, "ThreadPoolExecutor", TrackingExecutor)

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    for index in range(2):
        Image.new("RGBA", (32, 32), color=(255, 0, 0, 255)).save(input_dir / f"sample_{index}.png")

    report = pipeline.process_folder(
        input_dir,
        pattern="*.png",
        model_key="test-model",
        config=Config(model_dir=tmp_path),
        max_workers=4,
    )

    assert report.total == 2
    assert created_entries
    assert created_executors
    assert created_executors[0].max_workers_seen == 2


def test_process_folder_releases_sessions_between_images(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = ModelSpec(
        key="test-model",
        input_size=(32, 32),
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
    )

    def fake_prepare(*_args, **_kwargs) -> StubSession:
        return StubSession(spec)

    release_calls: list[StubSession] = []

    def fake_release(session) -> None:
        release_calls.append(session)

    def fake_loaded_image(*_args, **_kwargs):
        return SimpleNamespace(image=Image.new("RGBA", (32, 32)))

    monkeypatch.setattr(pipeline, "_prepare_session", fake_prepare)
    monkeypatch.setattr(pipeline, "release_session", fake_release)
    monkeypatch.setattr(pipeline, "_process_loaded_image", fake_loaded_image)

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    for index in range(2):
        Image.new("RGBA", (32, 32), color=(255, 0, 0, 255)).save(input_dir / f"sample_{index}.png")

    report = pipeline.process_folder(
        input_dir,
        pattern="*.png",
        model_key="test-model",
        config=Config(model_dir=tmp_path),
    )

    assert report.total == 2
    assert len(release_calls) == 2
    assert release_calls[0] is not release_calls[1]


def test_gpu_memory_error_retries_on_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """GPU memory errors should trigger automatic CPU inference retries."""

    spec = ModelSpec(
        key="test-model",
        input_size=(16, 16),
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
    )

    class ErroringSession(StubSession):
        def __init__(self, model_spec: ModelSpec) -> None:
            super().__init__(model_spec)
            self.providers_available = ("CUDAExecutionProvider",)
            self.primary_provider = "CUDAExecutionProvider"
            self.model_path = Path("/tmp/model.onnx")
            self.inner = SimpleNamespace(run=self._run)

        def _run(self, *_args, **_kwargs):  # pragma: no cover - deterministic failure
            raise RuntimeError("BFCArena Out of memory")

    gpu_session = ErroringSession(spec)
    tensor = np.ones((1, 3, 16, 16), dtype=np.float32)

    created_cpu_sessions: list[object] = []

    class CpuFallbackSession:
        def __init__(self, session_spec: ModelSpec, model_path: Path, providers: list[str]):
            assert providers == ["CPUExecutionProvider"]
            assert session_spec is spec
            assert model_path == gpu_session.model_path
            self.spec = session_spec
            self.model_path = model_path
            self.providers_available = ("CPUExecutionProvider",)
            self.primary_provider = "CPUExecutionProvider"
            self.input_name = "input"

        def run(self, _: np.ndarray) -> np.ndarray:  # pragma: no cover - deterministic output
            width, height = self.spec.input_size
            return np.full((1, height, width), 2.0, dtype=np.float32)

    def fake_background_session(session_spec, model_path, providers):
        session = CpuFallbackSession(session_spec, model_path, providers)
        created_cpu_sessions.append(session)
        return session

    release_calls: list[object] = []

    monkeypatch.setattr(pipeline, "BackgroundRemovalSession", fake_background_session)
    monkeypatch.setattr(pipeline, "release_session", lambda session: release_calls.append(session))

    result = pipeline._run_session_with_cpu_fallback(gpu_session, tensor)

    assert created_cpu_sessions, "CPU fallback session should be instantiated"
    assert release_calls and release_calls[0] is created_cpu_sessions[0]
    assert result.shape == (1, 16, 16)
    assert np.all(result == 2.0)
