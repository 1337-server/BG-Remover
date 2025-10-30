"""Command line interface for the background remover project."""
from __future__ import annotations

import argparse
import gc
import inspect
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
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
from bgremover_core.paths import CONFIG_FILE, INPUT_DIR, OUTPUT_DIR

STATUS_SUCCESS = "✓"
STATUS_FAILURE = "✗"


class _StoreWithFlag(argparse.Action):
    """Store an argument value and flag that it was provided explicitly."""

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: str,
        option_string: str | None = None,
    ) -> None:
        setattr(namespace, self.dest, values)
        setattr(namespace, f"{self.dest}_provided", True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Background remover utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    remove_parser = subparsers.add_parser(
        "remove",
        help="Remove backgrounds from images",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    remove_parser.set_defaults(
        input_provided=False,
        output_provided=False,
    )
    remove_parser.add_argument(
        "-i",
        "--input",
        default=str(INPUT_DIR),
        action=_StoreWithFlag,
        help="Input file or directory path.",
    )
    remove_parser.add_argument(
        "-o",
        "--output",
        default=str(OUTPUT_DIR),
        action=_StoreWithFlag,
        help="Output file or directory path.",
    )
    remove_parser.add_argument(
        "-c",
        "--config",
        default=str(CONFIG_FILE),
        action=_StoreWithFlag,
        help="Config file path.",
    )
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

    if getattr(args, "output_provided", False):
        output_candidate = Path(args.output).expanduser()
        if output_candidate.suffix:
            output_path = output_candidate
        else:
            output_candidate.mkdir(parents=True, exist_ok=True)
            suffix = input_path.suffix or ".png"
            output_path = output_candidate / f"{input_path.stem}_no_bg{suffix}"
    else:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        suffix = input_path.suffix or ".png"
        output_path = OUTPUT_DIR / f"{input_path.stem}_no_bg{suffix}"
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
        # Release GPU allocations immediately after finishing a single image.
        gc.collect()
        torch.cuda.empty_cache()
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
    if getattr(args, "output_provided", False):
        batch_output_dir = Path(args.output).expanduser()
    else:
        batch_output_dir = OUTPUT_DIR
    batch_output_dir.mkdir(parents=True, exist_ok=True)
    report = process_folder(
        input_path,
        batch_output_dir,
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
    config_path = Path(getattr(args, "config", str(CONFIG_FILE))).expanduser()
    load_config_fn = load_config
    signature = inspect.signature(load_config_fn)
    if "config_path" in signature.parameters:
        base_config = load_config_fn(config_path=config_path)
    else:  # pragma: no cover - compatibility with patched tests
        base_config = load_config_fn()
    config = _apply_overrides(base_config, args)
    init_logging(config.log_level)
    _persist_if_requested(config, args)

    if args.command == "remove":
        if args.batch:
            return _handle_batch(args, config)
        return _handle_single(args, config)
    return 1


if __name__ == "__main__":
    sys.exit(main())
