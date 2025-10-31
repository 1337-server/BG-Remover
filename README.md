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

Get up and running with the Flask Web UI in just a few commands:

```bash
git clone https://github.com/BG-Remover/BG-Remover.git
cd BG-Remover
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
pip install -r requirements.txt
python -m runtimes.flask.app
```

The development server binds to `http://127.0.0.1:5000` by default. Once you see
“Running on http://127.0.0.1:5000” in the console, open the URL in a browser to
access the single-image and batch upload pages.

### Optional extras

- Install a GPU-enabled ONNX Runtime build (for example,
  `pip install onnxruntime-gpu`, `onnxruntime-directml`, or `onnxruntime-rocm`)
  to accelerate inference on supported hardware.
- Install ``tkinterdnd2`` if you want drag-and-drop in the desktop GUI:

  ```bash
  pip install tkinterdnd2
  ```

The remainder of this README covers the installation process in more detail and
documents every runtime.

## Installation

### Prerequisites

- Python 3.10 or newer (the project is regularly validated on Python 3.12).
- Git (or another method to obtain the repository).
- Access to the internet for downloading Python packages and ONNX weights.

### Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
# On Windows PowerShell:
#   .venv\Scripts\Activate.ps1
```

### Install dependencies

```bash
pip install -r requirements.txt
```

Add a GPU-specific ONNX Runtime build if you want hardware acceleration.

## Running modes

Each runtime reads configuration from `config/config.json`, uses `input/` as the
default source of images, and writes results to `output/` unless otherwise
specified.

### Flask Web UI

#### Development server

```bash
flask --app runtimes.flask.app run --debug
```

#### Direct module execution

```bash
python -m runtimes.flask.app
```

The Flask runtime exposes dedicated pages for single-image and batch/folder
processing. A navigation bar links both views, which share the same persisted
settings. Highlights include embedded previews, full-width advanced settings
with tooltips, comprehensive dark mode styling, and consolidated downloads for
batch jobs.

For production deployments, run a WSGI server such as Gunicorn:

```bash
gunicorn "runtimes.flask.app:create_app()" --bind 0.0.0.0:8080 --workers 4
```

### CLI runtime

```bash
python -m runtimes.cli.bgr_cli remove --input ./input/sample.png --output ./output/sample_no_bg.png
```

Key flags:

- `--batch` — process an entire folder; defaults to `input/` and writes results
  to `output/`.
- `--provider cuda|directml|cpu` — hint the preferred execution provider. The
  CLI falls back to CPU when a GPU provider fails to initialise.
- `--model <key>` — select any registered model (see the configuration
  reference for available keys).
- `--model-dir <path>` — choose where ONNX weights are cached.
- `--persist-config` — write CLI selections back to `config/config.json`.

Exit code `0` indicates success for every processed image. Batch commands print
a table with per-file status, timings, and any raised errors.

### GUI runtime

```bash
python -m runtimes.gui.bg_remover_gui
```

Batch processing tips:

- Drag and drop folders or image files directly into the Batch tab to stage them
  for processing. Enable the "Include subfolders" toggle to scan dropped
  directories recursively.
- Drag-and-drop support for the Batch tab relies on the optional
  ``tkinterdnd2`` dependency.

The GUI remembers the selected theme, advanced settings, and model directory.
Advanced panels are restored from the previous session and start collapsed to
keep the interface focused. Each advanced option features a tooltip with the
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
