"""Command-line interface for the background remover utilities."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

from app.services import runtime_compat
from app.services.bg_remove import (
    RemovalResult,
    ensure_global_session,
    get_output_format_spec,
    list_output_format_choices,
    remove_bg_file,
    remove_bg_folder,
)


def parse_args() -> argparse.Namespace:
    """Parse and return command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Remove image backgrounds using rembg with optional alpha matting.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    default_format = get_output_format_spec(None).key
    format_choices = ", ".join(choice.upper() for choice in list_output_format_choices())

    def parse_format_argument(value: str) -> str:
        """Normalise CLI format arguments and validate support."""

        try:
            return get_output_format_spec(value).key
        except ValueError as exc:
            raise argparse.ArgumentTypeError(str(exc)) from exc

    parser.add_argument(
        "input",
        nargs="?",
        help="Image file or folder to process. Defaults to the current directory.",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output directory. Defaults to ./output within the current working directory.",
    )
    parser.add_argument(
        "--alpha-matting",
        action="store_true",
        help="Enable alpha matting for better hair and fur details.",
    )
    parser.add_argument("--am-foreground", type=int, default=240, help="Alpha matting foreground threshold.")
    parser.add_argument("--am-background", type=int, default=10, help="Alpha matting background threshold.")
    parser.add_argument("--am-erode", type=int, default=10, help="Alpha matting erode size.")
    parser.add_argument(
        "--colorkey-tolerance",
        type=int,
        default=14,
        help="Tolerance for solid-background colour key fallback.",
    )
    parser.add_argument(
        "--feather-radius",
        type=int,
        default=3,
        help="Feather radius for smoothing alpha edges (requires OpenCV).",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Process folders recursively instead of only the top level.",
    )
    parser.add_argument(
        "--no-colorkey-fallback",
        action="store_true",
        help="Disable the solid-colour background fallback.",
    )
    parser.add_argument(
        "--format",
        "-f",
        dest="output_format",
        type=parse_format_argument,
        help=(
            "Specify the output format. Supported values: "
            f"{format_choices}. Defaults to {default_format.upper()}."
        ),
    )
    parser.add_argument(
        "--accelerator",
        choices=["auto", "cuda", "cpu"],
        default=None,
        help="Select the execution accelerator (overrides BG_ACCELERATOR).",
    )
    parser.add_argument(
        "--cuda-device-id",
        type=int,
        default=None,
        help="Select the CUDA device id when using GPU acceleration.",
    )
    parser.add_argument(
        "--no-warn-on-cpu",
        action="store_true",
        help="Disable CPU fallback warnings.",
    )
    return parser.parse_args()


def _print_result(result: RemovalResult) -> None:
    """Output a short human-readable summary for a processed file."""

    status = "✅" if result.success else "❌"
    if result.success and result.path_out:
        message = f"{status} {result.path_in} → {result.path_out} ({result.timing_ms:.1f} ms)"
    else:
        message = f"{status} {result.path_in}: {result.error or 'unknown error'}"
    print(message)


def main() -> None:
    """Entry point for the CLI."""

    args = parse_args()
    input_path = Path(args.input).expanduser() if args.input else Path.cwd()
    runtime_compat.ensure_runtime_ready()

    config_overrides: Dict[str, Any] = {}
    if args.accelerator:
        config_overrides["BG_ACCELERATOR"] = args.accelerator
    if args.cuda_device_id is not None:
        config_overrides["BG_CUDA_DEVICE_ID"] = args.cuda_device_id
    if args.no_warn_on_cpu:
        config_overrides["BG_WARN_ON_CPU"] = False

    ensure_global_session(config=config_overrides or None)

    if input_path.is_dir():
        results = remove_bg_folder(
            input_path,
            args.output,
            output_format=args.output_format,
            recursive=args.recursive,
            alpha_matting=args.alpha_matting,
            am_foreground=args.am_foreground,
            am_background=args.am_background,
            am_erode=args.am_erode,
            use_colorkey_fallback=not args.no_colorkey_fallback,
            colorkey_tolerance=args.colorkey_tolerance,
            feather_radius=args.feather_radius,
        )
        for result in results:
            _print_result(result)
        successes = sum(1 for result in results if result.success)
        print(f"\nDone. {successes}/{len(results)} images processed successfully.")
    elif input_path.is_file():
        output = args.output
        result = remove_bg_file(
            input_path,
            output,
            alpha_matting=args.alpha_matting,
            am_foreground=args.am_foreground,
            am_background=args.am_background,
            am_erode=args.am_erode,
            use_colorkey_fallback=not args.no_colorkey_fallback,
            colorkey_tolerance=args.colorkey_tolerance,
            feather_radius=args.feather_radius,
            output_format=args.output_format,
        )
        _print_result(result)
        if not result.success:
            sys.exit(1)
    else:
        sys.exit(f"Input path not found: {input_path}")


if __name__ == "__main__":
    main()
