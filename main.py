"""Command-line interface for the background remover utilities."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable, Optional

import click
from rich.console import Console

from app.services import runtime_compat
from app.services.bg_remove import (
    RemovalResult,
    ensure_global_session,
    get_accelerator_status,
    get_output_format_spec,
    list_output_format_choices,
    remove_bg_file,
    remove_bg_folder,
)


def _env_bool(name: str, default: bool) -> bool:
    """Return a boolean environment variable value with ``default`` fallback."""

    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    """Return an integer environment variable value with ``default`` fallback."""

    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        return int(raw_value)
    except ValueError:
        click.secho(
            f"Environment variable {name} is not a valid integer: {raw_value}",
            fg="yellow",
            err=True,
        )
        return default


DEFAULT_ACCELERATOR = os.getenv("BG_ACCELERATOR", "auto")
DEFAULT_CUDA_DEVICE_ID = _env_int("BG_CUDA_DEVICE_ID", 0)
DEFAULT_WARN_ON_CPU = _env_bool("BG_WARN_ON_CPU", True)


class OutputFormatParam(click.ParamType):
    """Click parameter type that validates rembg output formats."""

    name = "format"

    def convert(  # type: ignore[override]
        self, value: Any, param: Optional[click.Parameter], ctx: Optional[click.Context]
    ) -> str:
        """Normalise CLI format arguments and validate support."""

        try:
            spec = get_output_format_spec(value)
        except ValueError as exc:  # pragma: no cover - validation is deterministic
            self.fail(str(exc), param, ctx)
        return spec.key


FORMAT_PARAM = OutputFormatParam()
CONSOLE = Console()


def _print_result(result: RemovalResult) -> None:
    """Output a short human-readable summary for a processed file."""

    status_icon = "✅" if result.success else "❌"
    if result.success and result.path_out:
        message = f"{status_icon} {result.path_in} → {result.path_out} ({result.timing_ms:.1f} ms)"
    else:
        message = f"{status_icon} {result.path_in}: {result.error or 'unknown error'}"
    CONSOLE.print(message)


def _summarise_results(results: Iterable[RemovalResult]) -> str:
    """Return a compact text summary for a sequence of results."""

    results_list = list(results)
    total = len(results_list)
    successes = sum(1 for item in results_list if item.success)
    return f"Done. {successes}/{total} images processed successfully."


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("input_path", required=False, type=click.Path(path_type=Path, resolve_path=True))
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path, resolve_path=True),
    help="Output directory. Defaults to ./output within the current working directory.",
)
@click.option("--alpha-matting", is_flag=True, help="Enable alpha matting for better hair and fur details.")
@click.option("--am-foreground", type=int, default=240, show_default=True, help="Alpha matting foreground threshold.")
@click.option("--am-background", type=int, default=10, show_default=True, help="Alpha matting background threshold.")
@click.option("--am-erode", type=int, default=10, show_default=True, help="Alpha matting erode size.")
@click.option(
    "--colorkey-tolerance",
    type=int,
    default=14,
    show_default=True,
    help="Tolerance for solid-background colour key fallback.",
)
@click.option(
    "--feather-radius",
    type=int,
    default=3,
    show_default=True,
    help="Feather radius for smoothing alpha edges (requires OpenCV).",
)
@click.option("--recursive", is_flag=True, help="Process folders recursively instead of only the top level.")
@click.option(
    "--no-colorkey-fallback",
    is_flag=True,
    help="Disable the solid-colour background fallback.",
)
@click.option(
    "--format",
    "-f",
    "output_format",
    type=FORMAT_PARAM,
    help=(
        "Specify the output format. Supported values: "
        + ", ".join(choice.upper() for choice in list_output_format_choices())
    ),
)
@click.option(
    "--accelerator",
    type=click.Choice(["auto", "cuda", "cpu"], case_sensitive=False),
    default=DEFAULT_ACCELERATOR,
    show_default=True,
    help="Select the execution provider. Defaults to environment BG_ACCELERATOR or 'auto'.",
)
@click.option(
    "--cuda-device-id",
    type=int,
    default=DEFAULT_CUDA_DEVICE_ID,
    show_default=True,
    help="Select the CUDA device ID when using GPU acceleration.",
)
@click.option(
    "--no-warn-on-cpu",
    is_flag=True,
    default=not DEFAULT_WARN_ON_CPU,
    help="Suppress warnings when the application falls back to CPU execution.",
)
def main(
    input_path: Optional[Path],
    output: Optional[Path],
    alpha_matting: bool,
    am_foreground: int,
    am_background: int,
    am_erode: int,
    colorkey_tolerance: int,
    feather_radius: int,
    recursive: bool,
    no_colorkey_fallback: bool,
    output_format: Optional[str],
    accelerator: str,
    cuda_device_id: int,
    no_warn_on_cpu: bool,
) -> None:
    """Entry point for the CLI."""

    input_path = input_path or Path.cwd()
    accelerator_mode = accelerator.strip().lower() if accelerator else "auto"
    warn_on_cpu = not no_warn_on_cpu

    config_overrides = {
        "BG_ACCELERATOR": accelerator_mode,
        "BG_CUDA_DEVICE_ID": cuda_device_id,
        "BG_WARN_ON_CPU": warn_on_cpu,
    }

    runtime_compat.ensure_runtime_ready()
    ensure_global_session(config=config_overrides)
    status = get_accelerator_status()

    CONSOLE.print(f"[bold]Execution provider:[/] {status.provider_description}")
    if status.gpu_name:
        flair = " (RTX 50-series)" if status.rtx_50_series else ""
        CONSOLE.print(f"GPU detected: {status.gpu_name}{flair}")
    if status.warning and warn_on_cpu:
        CONSOLE.print(f"[yellow]{status.warning}[/]")

    if input_path.is_dir():
        results = remove_bg_folder(
            input_path,
            output,
            output_format=output_format,
            recursive=recursive,
            alpha_matting=alpha_matting,
            am_foreground=am_foreground,
            am_background=am_background,
            am_erode=am_erode,
            use_colorkey_fallback=not no_colorkey_fallback,
            colorkey_tolerance=colorkey_tolerance,
            feather_radius=feather_radius,
        )
        for result in results:
            _print_result(result)
        click.echo(f"\n{_summarise_results(results)}")
        return

    if input_path.is_file():
        destination = output
        result = remove_bg_file(
            input_path,
            destination,
            alpha_matting=alpha_matting,
            am_foreground=am_foreground,
            am_background=am_background,
            am_erode=am_erode,
            use_colorkey_fallback=not no_colorkey_fallback,
            colorkey_tolerance=colorkey_tolerance,
            feather_radius=feather_radius,
            output_format=output_format,
        )
        _print_result(result)
        if not result.success:
            raise SystemExit(1)
        return

    raise SystemExit(f"Input path not found: {input_path}")


if __name__ == "__main__":
    main()
