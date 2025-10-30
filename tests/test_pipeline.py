"""Tests for the shared processing pipeline."""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from bgremover_core.config import Config
from bgremover_core.models.specs import ModelSpec
from bgremover_core.processing import pipeline


class StubSession:
    def __init__(self, spec: ModelSpec) -> None:
        self.spec = spec
        self.input_name = "input"
        self.providers_available = ("CPUExecutionProvider",)

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
    assert report.entries[0].path_out is not None


def test_prepare_session_normalises_custom_providers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Explicit provider hints should be normalised before session creation."""

    config = Config(model_dir=tmp_path)
    observed: dict[str, object] = {}

    def fake_detect_providers(hints: Sequence[str] | None = None) -> list[str]:
        observed["hints"] = tuple(hints or ())
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]

    def fake_get_session(
        model_key: str,
        *,
        providers: Sequence[str],
        model_dir: Path,
    ) -> str:
        observed["providers"] = list(providers)
        observed["model_dir"] = model_dir
        return "session"

    monkeypatch.setattr(pipeline, "detect_providers", fake_detect_providers)
    monkeypatch.setattr(pipeline, "get_session", fake_get_session)

    result = pipeline._prepare_session(
        "test-model",
        config=config,
        providers=["cuda"],
    )

    assert result == "session"
    assert observed["hints"] == ("cuda",)
    assert observed["providers"] == ["CUDAExecutionProvider", "CPUExecutionProvider"]
    assert observed["model_dir"] == tmp_path
