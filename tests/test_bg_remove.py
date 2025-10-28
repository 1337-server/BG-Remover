"""Tests for the background removal helpers."""
from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from app.services import bg_remove


def _stub_session(providers: list[str]) -> object:
    """Return a simple stub object that mimics an ONNX session."""

    class _StubSession:
        def __init__(self, provider_list: list[str]) -> None:
            self.providers = provider_list

    return _StubSession(providers)


def test_create_session_uses_gpu_priority(monkeypatch: pytest.MonkeyPatch) -> None:
    """GPU providers should be ordered TensorRT → CUDA → CPU."""

    recorded_providers: list = []

    def fake_new_session(model_name: str, providers: list) -> object:
        recorded_providers[:] = providers
        first_provider = providers[0][0] if isinstance(providers[0], tuple) else providers[0]
        return _stub_session([first_provider])

    monkeypatch.setattr(bg_remove, "_rembg_new_session", fake_new_session)
    monkeypatch.setattr(bg_remove, "_REMBG_IMPORT_ERROR", None)
    monkeypatch.setattr(bg_remove.runtime_compat, "ensure_runtime_ready", lambda: None)
    monkeypatch.setattr(
        bg_remove.accelerator,
        "onnx_providers_available",
        lambda: ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    monkeypatch.setattr(bg_remove.accelerator, "detect_gpu_name", lambda: "NVIDIA GeForce RTX 5090")
    monkeypatch.setattr(bg_remove.accelerator, "is_rtx_50xx", lambda name: True)

    context = bg_remove.create_session(config={})

    assert recorded_providers[0][0] == "TensorrtExecutionProvider"
    assert recorded_providers[1][0] == "CUDAExecutionProvider"
    assert recorded_providers[-1] == "CPUExecutionProvider"
    assert context.runtime == "cuda"
    assert context.provider == "TensorrtExecutionProvider"
    assert context.warning is None


def test_create_session_falls_back_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """Failures initialising GPU providers should fall back to CPU with a warning."""

    call_sequence: list[list] = []

    def fake_new_session(model_name: str, providers: list) -> object:
        call_sequence.append(providers)
        first_provider = providers[0][0] if isinstance(providers[0], tuple) else providers[0]
        if first_provider in {"TensorrtExecutionProvider", "CUDAExecutionProvider"}:
            raise RuntimeError("GPU provider failed")
        return _stub_session([first_provider])

    monkeypatch.setattr(bg_remove, "_rembg_new_session", fake_new_session)
    monkeypatch.setattr(bg_remove, "_REMBG_IMPORT_ERROR", None)
    monkeypatch.setattr(bg_remove.runtime_compat, "ensure_runtime_ready", lambda: None)
    monkeypatch.setattr(
        bg_remove.accelerator,
        "onnx_providers_available",
        lambda: ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    monkeypatch.setattr(bg_remove.accelerator, "detect_gpu_name", lambda: "NVIDIA GeForce RTX 5090")
    monkeypatch.setattr(bg_remove.accelerator, "is_rtx_50xx", lambda name: True)
    monkeypatch.setattr(bg_remove, "_CPU_WARNING_LOGGED", False)

    context = bg_remove.create_session(config={})

    assert len(call_sequence) == 3
    assert context.runtime == "cpu"
    assert context.provider == "CPUExecutionProvider"
    assert context.warning is not None


def test_build_colorkey_mask_detects_foreground() -> None:
    """Colour-key mask should identify a dark subject on a light background."""

    image = Image.new("RGB", (100, 100), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.rectangle((30, 30, 70, 70), fill=(0, 0, 0))

    mask = bg_remove.build_colorkey_mask(image, tolerance=20)
    assert mask is not None
    assert mask.shape == (100, 100)
    assert mask[0, 0] == 0
    assert mask[50, 50] == 255


def test_get_output_format_spec_handles_aliases() -> None:
    """Output format lookups should accept dotted and mixed-case aliases."""

    spec = bg_remove.get_output_format_spec(".JPEG")
    assert spec.key == "jpg"
    assert spec.mime_type == "image/jpeg"


@pytest.mark.parametrize("recursive", [False, True])
def test_remove_bg_folder_invokes_processing(tmp_path: Path, recursive: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    """Folder helper should return results for supported images only."""

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "image.jpg").write_bytes(b"fake")
    nested = input_dir / "nested"
    nested.mkdir()
    (nested / "skip.txt").write_text("not an image")
    (nested / "photo.png").write_bytes(b"fake")

    outputs: list[Path] = []

    def fake_remove_bg_file(*args, **kwargs):  # type: ignore[override]
        source = Path(args[0])
        destination = Path(args[1])
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"png")
        outputs.append(destination)
        return bg_remove.RemovalResult(source, destination, True, None, 1.0)

    monkeypatch.setattr(bg_remove, "remove_bg_file", fake_remove_bg_file)

    result_list = bg_remove.remove_bg_folder(
        input_dir,
        output_dir=tmp_path / "output",
        session=object(),
        recursive=recursive,
    )

    expected_count = 2 if recursive else 1
    assert len(result_list) == expected_count
    assert all(result.success for result in result_list)
    assert all(output.suffix == ".png" for output in outputs)


def test_remove_bg_folder_default_output_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Default folder processing should place files under ``cwd / 'output'``."""

    monkeypatch.chdir(tmp_path)
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "example.jpg").write_bytes(b"fake")

    destinations: list[Path] = []

    def fake_remove_bg_file(*args, **kwargs):  # type: ignore[override]
        source = Path(args[0])
        destination = Path(args[1])
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"png")
        destinations.append(destination)
        return bg_remove.RemovalResult(source, destination, True, None, 1.0)

    monkeypatch.setattr(bg_remove, "remove_bg_file", fake_remove_bg_file)

    results = bg_remove.remove_bg_folder(input_dir, session=object())

    expected_destination = tmp_path / "output" / "example.png"
    assert destinations == [expected_destination]
    assert results[0].path_out == expected_destination


def test_remove_bg_file_default_output_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Single-file processing should default to ``cwd / 'output'``."""

    monkeypatch.chdir(tmp_path)
    input_file = tmp_path / "sample.jpg"
    image = Image.new("RGB", (4, 4), color=(255, 0, 0))
    image.save(input_file)

    monkeypatch.setattr(bg_remove, "_get_session", lambda session=None, **_: object())

    def fake_run_rembg(image: Image.Image, session: object) -> Image.Image:
        """Return a solid opaque mask for deterministic behaviour."""

        return Image.new("RGBA", image.size, color=(255, 255, 255, 255))

    monkeypatch.setattr(bg_remove, "_run_rembg", fake_run_rembg)

    result = bg_remove.remove_bg_file(input_file)

    expected_output = tmp_path / "output" / "sample.png"
    assert result.success
    assert result.path_out == expected_output
    assert expected_output.exists()


def test_remove_bg_file_respects_requested_format(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicit output formats should control the exported file type."""

    monkeypatch.chdir(tmp_path)
    input_file = tmp_path / "transparent.png"
    image = Image.new("RGBA", (4, 4), color=(0, 128, 255, 200))
    image.save(input_file)

    monkeypatch.setattr(bg_remove, "_get_session", lambda session=None, **_: object())

    def fake_run_rembg(image: Image.Image, session: object) -> Image.Image:
        """Return a semi-transparent mask for deterministic output."""

        return Image.new("RGBA", image.size, color=(255, 255, 255, 200))

    monkeypatch.setattr(bg_remove, "_run_rembg", fake_run_rembg)

    result = bg_remove.remove_bg_file(input_file, output_format="jpg")

    assert result.success
    assert result.path_out is not None
    assert result.path_out.suffix == ".jpg"

    with Image.open(result.path_out) as exported:
        assert exported.mode == "RGB"


def test_remove_bg_file_emits_callbacks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Progress and preview callbacks should receive updates during processing."""

    monkeypatch.chdir(tmp_path)
    input_file = tmp_path / "sample.jpg"
    image = Image.new("RGB", (4, 4), color=(255, 0, 0))
    image.save(input_file)

    monkeypatch.setattr(bg_remove, "_get_session", lambda session=None, **_: object())

    def fake_run_rembg(image: Image.Image, session: object) -> Image.Image:
        return Image.new("RGBA", image.size, color=(255, 255, 255, 128))

    monkeypatch.setattr(bg_remove, "_run_rembg", fake_run_rembg)

    progress_events: list[tuple[str, float]] = []
    preview_events: list[str] = []

    def progress_callback(stage: str, percent: float) -> None:
        progress_events.append((stage, percent))

    def preview_callback(preview_image: Image.Image, stage: str) -> None:
        assert isinstance(preview_image, Image.Image)
        preview_events.append(stage)

    result = bg_remove.remove_bg_file(
        input_file,
        progress_callback=progress_callback,
        preview_callback=preview_callback,
    )

    assert result.success
    assert any(stage == "mask" for stage, _ in progress_events)
    assert any(stage == "refined" for stage in preview_events)
