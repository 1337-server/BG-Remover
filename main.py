"""Simple command line interface for the background removal helper."""
from __future__ import annotations

import argparse
from pathlib import Path

from app.services.bg_remove import (
    ensure_global_session,
    get_output_format_spec,
    list_output_format_choices,
    remove_bg_file,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Remove the background from a single image.")
    parser.add_argument("input", help="Path to the input image")
    parser.add_argument("--output", "-o", help="Destination file or directory")

    format_choices = list_output_format_choices()

    def _parse_format(value: str) -> str:
        try:
            return get_output_format_spec(value).key
        except ValueError as exc:  # pragma: no cover - delegated to argparse error handling
            raise argparse.ArgumentTypeError(str(exc)) from exc

    parser.add_argument(
        "--format",
        "-f",
        dest="output_format",
        type=_parse_format,
        choices=format_choices,
        help="Output format key",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for the CLI command."""

    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input path not found: {input_path}")

    ensure_global_session()
    result = remove_bg_file(input_path, args.output, output_format=args.output_format)
    output_path = result.path_out or Path(args.output or "")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
