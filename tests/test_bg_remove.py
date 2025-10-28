"""Tests for the background removal helpers."""
from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from app.services import bg_remove


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

    monkeypatch.setattr(bg_remove, "_get_session", lambda session=None: object())

    def fake_run_rembg(image: Image.Image, session: object) -> Image.Image:
        """Return a solid opaque mask for deterministic behaviour."""

        return Image.new("RGBA", image.size, color=(255, 255, 255, 255))

    monkeypatch.setattr(bg_remove, "_run_rembg", fake_run_rembg)

    result = bg_remove.remove_bg_file(input_file)

    expected_output = tmp_path / "output" / "sample.png"
    assert result.success
    assert result.path_out == expected_output
    assert expected_output.exists()
