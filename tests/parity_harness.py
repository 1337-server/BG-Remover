"""Temporary parity harness comparing CLI and GUI processing pipelines."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from bgremover_core import Config, load_config
from bgremover_core.background_remover import process_image
from bgremover_core.io.image_io import image_to_numpy
from bgremover_core.processing.pipeline import ProcessingResult
from runtimes.gui.bg_remover_gui import run_gui_pipeline_for_parity


def _save_mask(mask: np.ndarray, path: Path) -> Path:
    """Persist ``mask`` as an ``L`` mode PNG at ``path``."""

    image = Image.fromarray((mask * 255.0).round().astype(np.uint8), mode="L")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, "PNG", optimize=True)
    return path


def _log_debug(prefix: str, result: ProcessingResult) -> None:
    """Print per-stage statistics recorded in ``result`` for diagnostics."""

    debug = result.debug
    print(
        f"{prefix} PRE  shape={debug.pre_shape} min={debug.pre_min:.6f} max={debug.pre_max:.6f}"
    )
    print(
        f"{prefix} RUN  logits_shape={debug.logits_shape} "
        f"min={debug.logits_min:.6f} max={debug.logits_max:.6f}"
    )
    print(
        f"{prefix} POST1 mask_min={debug.mask_pre_min:.6f} mask_max={debug.mask_pre_max:.6f}"
    )
    print(
        f"{prefix} POST2 mask_min={debug.mask_post_min:.6f} mask_max={debug.mask_post_max:.6f}"
    )


def _run_cli_pipeline(image: Image.Image, *, config: Config, model_key: str) -> ProcessingResult:
    """Execute the CLI pipeline by calling :func:`process_image`."""

    array = image_to_numpy(image)
    return process_image(array, model_key=model_key, config=config)


def run_parity_check(
    image_path: Path,
    *,
    output_dir: Path | None = None,
    model_key: str | None = None,
    config: Config | None = None,
) -> None:
    """Compare CLI and GUI masks for ``image_path`` and write diagnostic files."""

    active_config = config or load_config()
    resolved_model = model_key or active_config.default_model
    with Image.open(image_path) as source:
        source_image = source.convert("RGBA")
    cli_result = _run_cli_pipeline(source_image, config=active_config, model_key=resolved_model)
    gui_result = run_gui_pipeline_for_parity(
        source_image,
        config=active_config,
        model_key=resolved_model,
    )
    destination = output_dir or image_path.parent
    destination.mkdir(parents=True, exist_ok=True)
    cli_mask_path = _save_mask(cli_result.mask, destination / "mask_cli.png")
    gui_mask_path = _save_mask(gui_result.mask, destination / "mask_gui.png")
    print(f"CLI mask saved to {cli_mask_path}")
    print(f"GUI mask saved to {gui_mask_path}")
    _log_debug("CLI", cli_result)
    _log_debug("GUI", gui_result)


def _build_parser() -> argparse.ArgumentParser:
    """Return an argument parser for the parity harness CLI."""

    parser = argparse.ArgumentParser(description="CLI/GUI parity diagnostic harness")
    parser.add_argument("image", type=Path, help="Input image to process")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory where diagnostic masks will be written",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model key to use for both pipelines (defaults to configuration)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for running the parity harness from the command line."""

    parser = _build_parser()
    args = parser.parse_args(argv)
    config = load_config()
    run_parity_check(
        args.image.expanduser(),
        output_dir=args.output_dir.expanduser() if args.output_dir else None,
        model_key=args.model,
        config=config,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
