"""Command-line interface for the background removal helper."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from bg_removal import (
    DEFAULT_MODEL_NAME,
    RemovalResult,
    ensure_global_session,
    get_output_format_spec,
    list_output_format_choices,
    remove_bg_file,
    remove_bg_folder,
)

MODEL_CHOICES: dict[str, str] = {
    "general": "isnet-general-use",
    "human": "u2net_human_seg",
    "object": "u2net",
    "anime": "isnet-anime",
}
DEFAULT_MODEL_KEY = next(iter(MODEL_CHOICES))


def parse_args() -> argparse.Namespace:
    """Parse and return command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Remove image backgrounds with optional alpha matting and folder batching.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    default_format = get_output_format_spec(None).key
    format_choices = ", ".join(choice.upper() for choice in list_output_format_choices())

    def parse_format_argument(value: str) -> str:
        """Normalise CLI format arguments and validate support."""

        try:
            return get_output_format_spec(value).key
        except ValueError as exc:  # pragma: no cover - delegated to argparse
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
        "--model",
        choices=sorted(MODEL_CHOICES.keys()),
        default=DEFAULT_MODEL_KEY,
        help="Background removal model to use.",
    )
    parser.add_argument(
        "--alpha-matting",
        action="store_true",
        help="Enable alpha matting for detailed hair and fur handling.",
    )
    parser.add_argument(
        "--am-foreground",
        type=int,
        default=240,
        help="Alpha matting foreground threshold in the 0-255 range.",
    )
    parser.add_argument(
        "--am-background",
        type=int,
        default=10,
        help="Alpha matting background threshold in the 0-255 range.",
    )
    parser.add_argument("--am-erode", type=int, default=10, help="Alpha matting erode size (0-255).")
    parser.add_argument(
        "--colorkey-tolerance",
        type=int,
        default=14,
        help="Tolerance for solid-colour background fallback (0-255).",
    )
    parser.add_argument(
        "--feather-radius",
        type=int,
        default=3,
        help="Feather radius for smoothing alpha edges in pixels.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="When processing a directory, include all subdirectories.",
    )
    parser.add_argument(
        "--no-colorkey-fallback",
        action="store_true",
        help="Disable the solid-colour background helper when detecting green/blue screens.",
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
    model_key = args.model or DEFAULT_MODEL_KEY
    model_name = MODEL_CHOICES.get(model_key, DEFAULT_MODEL_NAME)

    ensure_global_session(model_name)

    if input_path.is_dir():
        results = remove_bg_folder(
            input_path,
            args.output,
            output_format=args.output_format,
            model_name=model_name,
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
        if successes != len(results):
            sys.exit(1)
    elif input_path.is_file():
        result = remove_bg_file(
            input_path,
            args.output,
            output_format=args.output_format,
            model_name=model_name,
            alpha_matting=args.alpha_matting,
            am_foreground=args.am_foreground,
            am_background=args.am_background,
            am_erode=args.am_erode,
            use_colorkey_fallback=not args.no_colorkey_fallback,
            colorkey_tolerance=args.colorkey_tolerance,
            feather_radius=args.feather_radius,
        )
        _print_result(result)
        if not result.success:
            sys.exit(1)
    else:
        sys.exit(f"Input path not found: {input_path}")


if __name__ == "__main__":
    main()
