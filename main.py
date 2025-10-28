"""Command-line interface for the background removal service."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from app.services.bg_remove import (
    RemovalResult,
    ensure_global_session,
    remove_bg_file,
    remove_bg_folder,
)


def parse_args() -> argparse.Namespace:
    """Parse and return command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Remove image backgrounds using rembg with optional alpha matting.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
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
    ensure_global_session()

    if input_path.is_dir():
        results = remove_bg_folder(
            input_path,
            args.output,
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
        )
        _print_result(result)
        if not result.success:
            sys.exit(1)
    else:
        sys.exit(f"Input path not found: {input_path}")


if __name__ == "__main__":
    main()
