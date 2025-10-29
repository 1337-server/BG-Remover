"""CLI integration tests."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from bgremover_core.config import Config
from bgremover_core.processing.pipeline import Report, ReportEntry
from runtimes.cli import bgr_cli


@pytest.fixture()
def sample_image(tmp_path: Path) -> Path:
    path = tmp_path / "input.png"
    Image.new("RGBA", (4, 4), color=(255, 0, 0, 255)).save(path)
    return path


def test_cli_single_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    sample_image: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_path = tmp_path / "output.png"

    monkeypatch.setattr(bgr_cli, "load_config", lambda: Config(model_dir=tmp_path))
    monkeypatch.setattr(
        bgr_cli,
        "remove_background",
        lambda array, model_key, config, feather_radius: np.ones_like(array),
    )

    exit_code = bgr_cli.main([
        "remove",
        "--input",
        str(sample_image),
        "--output",
        str(output_path),
    ])
    assert exit_code == 0
    assert output_path.exists()
    stdout, _ = capsys.readouterr()
    assert "✓" in stdout


def test_cli_batch_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    input_dir = tmp_path / "images"
    input_dir.mkdir()
    (input_dir / "image.png").write_bytes(b"notanimage")

    def fake_process_folder(*_args, **_kwargs):
        entry = ReportEntry(
            path_in=input_dir / "image.png",
            path_out=None,
            success=False,
            elapsed_ms=0.0,
            error="fail",
        )
        return Report([entry])

    monkeypatch.setattr(bgr_cli, "load_config", lambda: Config(model_dir=tmp_path))
    monkeypatch.setattr(bgr_cli, "process_folder", fake_process_folder)

    exit_code = bgr_cli.main([
        "remove",
        "--input",
        str(input_dir),
        "--batch",
    ])
    assert exit_code == 1
    stdout, _ = capsys.readouterr()
    assert "fail" in stdout
