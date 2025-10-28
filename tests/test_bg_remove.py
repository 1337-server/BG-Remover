from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image, ImageDraw

from app.services import bg_remove


class _FakeSession:
    """Lightweight ONNX Runtime session stub used for unit tests."""

    def __init__(self, providers: list[str], value: float = 0.5) -> None:
        self._providers = providers
        self._value = value
        self.run_calls = 0

    def get_providers(self) -> list[str]:
        return list(self._providers)

    def get_inputs(self) -> list[SimpleNamespace]:
        return [SimpleNamespace(name="input", shape=[1, 3, 320, 320], type="tensor(float)")]

    def run(self, *_args, **_kwargs) -> list[np.ndarray]:
        self.run_calls += 1
        data = np.full((1, 1, 320, 320), self._value, dtype=np.float32)
        return [data]


def _reset_session_state() -> None:
    """Reset module-level caches to ensure deterministic test outcomes."""

    bg_remove._SESSION_CONTEXT = None
    bg_remove._SESSION_CONFIG_SIGNATURE = None
    bg_remove._SESSION_POOLS.clear()
    bg_remove._PRELOADED_SIGNATURES.clear()
    bg_remove._STARTUP_LOGGED_SIGNATURES.clear()
    bg_remove._SESSION_METADATA.clear()
    bg_remove._CPU_WARNING_LOGGED = False
    bg_remove._CUDA_HINT_LOGGED = False


def test_create_session_uses_gpu_priority(monkeypatch: pytest.MonkeyPatch) -> None:
    """GPU providers should be selected when available."""

    _reset_session_state()

    fake_session = _FakeSession(["CUDAExecutionProvider", "CPUExecutionProvider"])

    monkeypatch.setattr(bg_remove, "_PRELOAD_MODEL_NAMES", ("u2net",))
    monkeypatch.setattr(bg_remove.runtime_compat, "ensure_runtime_ready", lambda: None)
    monkeypatch.setattr(bg_remove.model_registry, "preload_models", lambda **_: {"u2net": fake_session})
    monkeypatch.setattr(bg_remove.model_registry, "get_session", lambda name: fake_session)
    monkeypatch.setattr(
        bg_remove.model_registry,
        "get_available_providers",
        lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    monkeypatch.setattr(bg_remove.accelerator, "detect_gpu_name", lambda: "NVIDIA GeForce RTX 5090")
    monkeypatch.setattr(bg_remove.accelerator, "is_rtx_50xx", lambda _: True)

    context = bg_remove.create_session(config={"BG_ACCELERATOR": "cuda"})

    assert context.runtime == "cuda"
    assert context.provider == "CUDAExecutionProvider"
    assert context.warning is None
    assert bg_remove._SESSION_METADATA[id(fake_session)]["model"] == "u2net"
    assert fake_session.run_calls >= 1, "warm-up should invoke at least one inference"


def test_create_session_falls_back_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """When CUDA is unavailable the context should report a CPU fallback."""

    _reset_session_state()

    fake_session = _FakeSession(["CPUExecutionProvider"], value=0.0)

    monkeypatch.setattr(bg_remove, "_PRELOAD_MODEL_NAMES", ("u2net",))
    monkeypatch.setattr(bg_remove.runtime_compat, "ensure_runtime_ready", lambda: None)
    monkeypatch.setattr(bg_remove.model_registry, "preload_models", lambda **_: {"u2net": fake_session})
    monkeypatch.setattr(bg_remove.model_registry, "get_session", lambda name: fake_session)
    monkeypatch.setattr(bg_remove.model_registry, "get_available_providers", lambda: ["CPUExecutionProvider"])
    monkeypatch.setattr(bg_remove.accelerator, "detect_gpu_name", lambda: "NVIDIA GeForce RTX 5090")
    monkeypatch.setattr(bg_remove.accelerator, "is_rtx_50xx", lambda _: True)

    context = bg_remove.create_session(config={"BG_ACCELERATOR": "cuda", "BG_WARN_ON_CPU": True})

    assert context.runtime == "cpu"
    assert context.provider == "CPUExecutionProvider"
    assert context.warning == "GPU requested but unavailable — falling back to CPU"


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
def test_remove_bg_folder_invokes_processing(
    tmp_path: Path, recursive: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
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
        session=_FakeSession(["CPUExecutionProvider"]),
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

    results = bg_remove.remove_bg_folder(input_dir, session=_FakeSession(["CPUExecutionProvider"]))

    expected_destination = tmp_path / "output" / "example.png"
    assert destinations == [expected_destination]
    assert results[0].path_out == expected_destination


def test_remove_bg_file_sanitises_mask(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Inference results containing NaNs should be clamped before writing the output."""

    monkeypatch.chdir(tmp_path)
    source = tmp_path / "source.png"
    Image.new("RGB", (8, 8), color=(200, 200, 200)).save(source)

    class _NoisySession(_FakeSession):
        def run(self, *_args, **_kwargs) -> list[np.ndarray]:
            data = np.array([[[[0.0, np.nan], [np.inf, -np.inf]]]], dtype=np.float32)
            return [data]

    session = _NoisySession(["CPUExecutionProvider"])
    bg_remove._SESSION_METADATA[id(session)] = {
        "model": "u2net",
        "provider": "CPUExecutionProvider",
        "runtime": "cpu",
    }

    result = bg_remove.remove_bg_file(
        source,
        session=session,
        alpha_matting=False,
        use_colorkey_fallback=False,
        feather_radius=0,
    )

    assert result.success
    assert result.path_out is not None
    output_image = Image.open(result.path_out)
    alpha = np.asarray(output_image.split()[-1])
    assert alpha.min() >= 0
    assert alpha.max() <= 255
