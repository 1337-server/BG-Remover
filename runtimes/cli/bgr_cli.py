"""Command line interface for the background remover project."""
from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from PIL import Image

from bgremover_core import (
    Config,
    ConfigError,
    init_logging,
    load_config,
    persist_config,
    process_folder,
    remove_background,
)
from bgremover_core.io.image_io import image_to_numpy, save_image_to_path

STATUS_SUCCESS = "✓"
STATUS_FAILURE = "✗"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Background remover utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    remove_parser = subparsers.add_parser(
        "remove",
        help="Remove backgrounds from images",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    remove_parser.add_argument("--input", required=True, help="Input file or folder path")
    remove_parser.add_argument("--output", help="Output file or directory path")
    remove_parser.add_argument("--model", help="Model key to use for inference")
    remove_parser.add_argument(
        "--batch",
        action="store_true",
        help="Process all supported images in the input directory",
    )
    remove_parser.add_argument(
        "--provider",
        action="append",
        choices=["cuda", "cpu", "directml"],
        help="Hint a preferred execution provider (can be used multiple times)",
    )
    remove_parser.add_argument("--model-dir", help="Custom directory to store model weights")
    remove_parser.add_argument(
        "--persist-config",
        action="store_true",
        help="Persist configuration changes such as --model-dir for future runs",
    )
    remove_parser.add_argument(
        "--feather-radius",
        type=int,
        default=3,
        help="Feather radius applied to the final alpha matte",
    )
    return parser


def _apply_overrides(config: Config, args: argparse.Namespace) -> Config:
    updates = {}
    if args.model_dir:
        updates["model_dir"] = Path(args.model_dir).expanduser()
    if args.model:
        updates["default_model"] = args.model
    if args.provider:
        hints: list[str] = []
        for item in args.provider:
            if item == "cuda":
                hints.append("CUDAExecutionProvider")
            elif item == "cpu":
                hints.append("CPUExecutionProvider")
            elif item == "directml":
                hints.append("DmlExecutionProvider")
        updates["provider_hints"] = tuple(hints)
    return config.with_updates(**updates) if updates else config


def _persist_if_requested(config: Config, args: argparse.Namespace) -> None:
    if args.persist_config:
        try:
            persist_config(config)
        except ConfigError as error:  # pragma: no cover - defensive logging
            print(f"Failed to persist configuration: {error}", file=sys.stderr)


def _load_image(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return image_to_numpy(image)


def _handle_single(args: argparse.Namespace, config: Config) -> int:
    input_path = Path(args.input).expanduser()
    if not input_path.exists() or not input_path.is_file():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        return 2

    output_path = (
        Path(args.output).expanduser()
        if args.output
        else input_path.with_stem(f"{input_path.stem}_no_bg")
    )
    try:
        image_array = _load_image(input_path)
        result_array = remove_background(
            image_array,
            args.model or config.default_model,
            config=config,
            feather_radius=args.feather_radius,
        )
        result_image = Image.fromarray(result_array)
        save_image_to_path(result_image, output_path)
        print(f"{STATUS_SUCCESS} {input_path} → {output_path}")
        return 0
    except Exception as error:  # pragma: no cover - defensive logging
        print(f"{STATUS_FAILURE} {input_path}: {error}", file=sys.stderr)
        return 1


def _format_row(entry: dict[str, object | None]) -> str:
    """Return a formatted CLI table row for a batch processing entry."""

    status = STATUS_SUCCESS if entry.get("success") else STATUS_FAILURE
    input_text = str(entry.get("input") or "")
    output_text = str(entry.get("output") or "")
    elapsed = entry.get("elapsed_ms")
    if isinstance(elapsed, int | float):
        timing = f"{float(elapsed):.1f}"
    else:
        timing = "-"
    error = str(entry.get("error") or "")
    return f"{status} | {input_text} | {output_text} | {timing} ms | {error}"


def _handle_batch(args: argparse.Namespace, config: Config) -> int:
    input_path = Path(args.input).expanduser()
    if not input_path.exists() or not input_path.is_dir():
        print(f"Input directory not found: {input_path}", file=sys.stderr)
        return 2
    report = process_folder(
        input_path,
        args.output,
        "*",
        model_key=args.model or config.default_model,
        config=config,
        feather_radius=args.feather_radius,
    )
    print("Status | Input | Output | Time | Error")
    print("-" * 80)
    for row in report.to_rows():
        print(_format_row(row))
    print("-" * 80)
    print(f"Processed {report.total} files: {report.successes} succeeded, {report.failures} failed")
    return 0 if report.failures == 0 else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    config = _apply_overrides(load_config(), args)
    init_logging(config.log_level)
    _persist_if_requested(config, args)

    if args.command == "remove":
        if args.batch:
            return _handle_batch(args, config)
        return _handle_single(args, config)
    return 1


if __name__ == "__main__":
    sys.exit(main())
