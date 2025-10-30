# BG-Remover

BG-Remover is a multi-runtime background removal toolkit that exposes the same
ONNX Runtime-powered pipeline through a command-line interface (CLI), a Flask
web application, and a desktop GUI. Every runtime supports CPU-only execution
and GPU acceleration through CUDA, DirectML, or ROCm, automatically falling back
when a preferred provider is unavailable.

All runtimes share the same directory layout so assets and configuration stay in
sync regardless of how you launch the tool:

```
bg-remover/
├─ input/          # Default input images
├─ output/         # Processed results
├─ config/config.json
├─ runtimes/
│  ├─ gui/
│  ├─ flask/
│  └─ cli/
```

The shared `bgremover_core` package provides filesystem helpers, configuration
loading, logging, model management, and the processing pipeline. CLI, Flask, and
GUI entry points simply import and orchestrate these shared utilities.

## Quickstart

### Prerequisites

- Python 3.10+
- `pip install -r requirements.txt`
- Optional GPU acceleration: install an ONNX Runtime build for your platform
  (e.g. `onnxruntime-gpu`, `onnxruntime-directml`, or `onnxruntime-rocm`).
- Latest NVIDIA/AMD/Intel GPU drivers when using hardware acceleration.

### CLI runtime

```bash
python -m runtimes.cli.bgr_cli remove --input ./input/sample.png --output ./output/sample_no_bg.png
```

Key flags:

- `--batch` — process an entire folder; defaults to `bg-remover/input` and
  writes results to `bg-remover/output`.
- `--provider cuda|directml|cpu` — hint the preferred execution provider.
  The CLI gracefully falls back to CPU when a GPU provider fails to initialise.
- `--model <key>` — select any registered model (see the configuration
  reference for available keys).
- `--model-dir <path>` — choose where ONNX weights are cached.
- `--persist-config` — write CLI selections back to `config/config.json`.

Exit code `0` indicates success for every processed image. The batch command
prints a table with per-file status, timings, and any raised errors.

### Flask runtime

```bash
export FLASK_APP=runtimes.flask.app
python -m flask run --debug
```

The Flask UI now offers separate pages for single-image and batch/folder
processing. A navigation bar links both views, and each view shares the same
configuration stored at the project root. Highlights include:

- Embedded preview cards directly within the upload grid.
- Advanced settings spanning the full page width beneath the preview with
  tooltips describing valid ranges.
- Dark mode styling applied to the entire page (`body`, `app-root`, cards, and
  modals).
- Batch page designed for folders and ZIP uploads with consolidated downloads.

### GUI runtime

```bash
python -m runtimes.gui.bg_remover_gui
```

The GUI remembers the selected theme, advanced settings, and model directory.
Advanced panels are restored from the previous session and start collapsed to
keep the interface focused. Each advanced option now features a tooltip with the
expected range and hints. You can also choose where model weights are stored so
that cached downloads can be shared between runtimes or stored on fast local
media.

## Configuration

Configuration lives in `config/config.json`. All runtimes read and write to this
location and default to the project directories defined above. Environment
variables can override persisted options:

| Variable | Description |
| --- | --- |
| `BGR_CONFIG_PATH` | Custom path for `config.json` if you need a different project root. |
| `MODEL_DIR` | Override the shared model cache directory. |
| `BGR_DEFAULT_MODEL` | Change the default model key used when none is specified. |
| `BGR_PROVIDER_HINTS` | Comma-separated providers to prioritise (e.g. `CUDAExecutionProvider,CPUExecutionProvider`). |
| `BGR_LOGLEVEL` | Logging level for every runtime (default `INFO`). |

Persisted configuration is updated automatically when the GUI saves settings or
when the CLI runs with `--persist-config`.

## Model catalogue

| Model key | Input size | Notes |
|-----------|------------|-------|
| `isnet-general-use` | 1024×1024 | General purpose backgrounds |
| `u2net_human_seg` | 320×320 | Portrait-focused matte |
| `u2net` | 320×320 | Object isolation |
| `isnet-anime` | 1024×1024 | Illustration/anime scenes |
| `briaai/RMBG-2.0` | 1024×1024 | High-quality universal model |
| `matting-by-generation` | 1024×1024 | Portrait matting variant |
| `sam_segmentation_model` | 1024×1024 | Alias of BRIA 2.0 |
| `sam_vit_b_01ec64_encoder` / `decoder` | 1024×1024 | Segment Anything weights |

Weights download on demand and honour `HUGGINGFACEHUB_API_TOKEN` or
`HF_API_TOKEN` for private repositories.

## Logging

Structured logging is initialised across runtimes via
`bgremover_core.config.init_logging`. Logs stream to STDOUT/STDERR and to
`error.log` in the current working directory (or next to the packaged GUI
executable).

## Deployment

See [`deployment.md`](deployment.md) for runtime-specific build and packaging
instructions, including Docker usage and GUI bundling with PyInstaller.

## Testing

Run the consolidated test suite with:

```bash
python -m pytest -q
```

The tests cover provider detection, session creation, pipeline success/error
paths, CLI exit codes, Flask routes, and GUI helper logic.
