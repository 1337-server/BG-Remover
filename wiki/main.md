# main.py — Command-Line Interface

## Overview
`main.py` exposes a command-line interface for running background removal tasks without a GUI. It wraps the shared processing helpers in user-friendly CLI options for single files or directories.

## Role in the Project
- Provides automation-friendly access for scripts, CI pipelines, or power users.
- Offers parity with UI options such as alpha matting, recursive directory processing, and output format selection.
- Reuses `bg_removal.py` for all heavy lifting.

## Key Components
- **`parse_args`**: Builds an `argparse.ArgumentParser` with switches for models, alpha matting thresholds, folder recursion, and output formats.
- **`main`**: Entry point that normalises paths, ensures sessions are ready, and delegates to `remove_bg_file` or `remove_bg_folder`.
- **`MODEL_CHOICES`**: Maps friendly CLI keys to model names accepted by `ensure_global_session`.
- **`_print_result`**: Pretty-prints individual outcomes with success/failure markers.

## Implementation Notes
- Invoked via `python main.py` or `python -m main` depending on packaging.
- Defaults to processing the current working directory when no input path is provided.
- Returns a non-zero exit code when any file in a batch fails—useful for automation.
- Uses `get_output_format_spec` to validate the `--format` argument against `bg_removal.py` definitions.

## Invocation Example
```bash
python main.py ./input --output ./output --model human --alpha-matting --format png
```

> **Interface Type:** CLI (shares backend with Flask and Tkinter UIs).
