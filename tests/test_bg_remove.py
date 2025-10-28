"""Unit tests for the simplified background removal helpers."""
from __future__ import annotations

import io
from pathlib import Path

import pytest
from PIL import Image

import bg_removal as bg_remove


def _make_image_bytes(color: tuple[int, int, int, int] = (255, 0, 0, 255)) -> bytes:
    image = Image.new("RGBA", (4, 4), color=color)
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    return buffer.getvalue()


def test_get_output_format_spec_handles_extensions() -> None:
    spec = bg_remove.get_output_format_spec(".PNG")
    assert spec.key == "png"
    assert spec.mime_type == "image/png"


def test_remove_background_bytes_returns_result(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummySession:
        providers_available = ("CPUExecutionProvider",)

    call_count = 0

    def fake_predict(image: Image.Image, session: DummySession) -> Image.Image:
        nonlocal call_count
        call_count += 1
        mask = Image.new("L", image.size, color=255)
        return mask

    monkeypatch.setattr(bg_remove, "_load_session", lambda model_name: DummySession())
    monkeypatch.setattr(bg_remove, "_predict_mask", fake_predict)

    result = bg_remove.remove_background_bytes(_make_image_bytes())
    assert result.success
    assert result.format_spec.key == "png"
    assert call_count == 1
    restored = Image.open(io.BytesIO(result.as_bytes()))
    assert restored.mode in {"RGBA", "RGB"}


def test_remove_bg_file_writes_to_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "source.png"
    Image.new("RGBA", (2, 2), color=(0, 128, 255, 255)).save(source, "PNG")

    class DummySession:
        providers_available = ("CPUExecutionProvider",)

    def fake_predict(image: Image.Image, session: DummySession) -> Image.Image:
        return Image.new("L", image.size, color=255)

    monkeypatch.setattr(bg_remove, "_load_session", lambda model_name: DummySession())
    monkeypatch.setattr(bg_remove, "_predict_mask", fake_predict)

    result = bg_remove.remove_bg_file(source, tmp_path)
    assert result.path_out is not None
    assert result.path_out.exists()
    assert result.path_out.suffix == ".png"


def test_encode_result_image_returns_data_url(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummySession:
        providers_available = ("CPUExecutionProvider",)

    monkeypatch.setattr(bg_remove, "_load_session", lambda model_name: DummySession())
    monkeypatch.setattr(
        bg_remove,
        "_predict_mask",
        lambda image, session: Image.new("L", image.size, color=255),
    )
    result = bg_remove.remove_background_bytes(_make_image_bytes())
    data_url = bg_remove.encode_result_image(result)
    assert data_url.startswith("data:image/png;base64,")
