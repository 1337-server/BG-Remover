# Background Remover

Unified tooling for removing image backgrounds with ONNX Runtime. The project now separates a framework-agnostic core from dedicated runtimes so the CLI, Flask app, and ttkbootstrap GUI can share the same pipeline, configuration, and logging logic.

## Repository layout

```
.
├── bgremover_core/
│   ├── config.py
│   ├── io/
│   ├── models/
│   └── processing/
├── runtimes/
│   ├── cli/
│   ├── flask_app/
│   └── gui/
├── scripts/
├── tests/
├── Dockerfile
├── requirements.txt
└── README.md
```

The `bgremover_core` package exposes the shared pipeline (`processing.pipeline.remove_background`), filesystem helpers, model specifications, and configuration helpers. Each runtime imports exclusively from this core layer.

## Quickstart

### Prerequisites

* Python 3.12+
* `pip install -r requirements.txt`
* Optional GPU acceleration requires `onnxruntime-gpu` and NVIDIA drivers.

### CLI

```
python -m runtimes.cli.bgr_cli remove --input path/to/image.png --output result.png
```

Batch mode processes an entire folder and prints a summary table:

```
python -m runtimes.cli.bgr_cli remove --input ./photos --batch
```

Key options:

* `--model <key>` – one of `isnet-general-use`, `u2net`, `u2net_human_seg`, `isnet-anime`, `briaai/RMBG-2.0`, `matting-by-generation`, `sam_segmentation_model`.
* `--provider cuda|cpu|directml` – hint the preferred execution provider; defaults to automatic detection favouring CUDA.
* `--model-dir <path>` – override the ONNX model cache (persists when `--persist-config` is supplied).
* `--feather-radius` – control post-processing softness (0–50 px).

Exit status is `0` when every image succeeds and non-zero otherwise.

### Flask web app

```
python -m runtimes.flask_app.app
```

The app factory exposes `create_app()` for WSGI or Gunicorn deployments. The landing page offers tooltips for each option, a GPU/CPU badge that lists the active providers, and a model directory override field that applies only to the current session.

### GUI

```
python -m runtimes.gui.bg_remover_gui
```

The GUI mirrors the CLI options with single-image and batch tabs. It shows a coloured pill indicating GPU or CPU execution providers, allows selecting and persisting a custom model directory, and logs status lines for each processed file (failures render in red and trigger a message box).

Batch mode defaults to an `output` sibling next to the input directory when no destination is selected.

## Configuration & environment variables

`bgremover_core.config.load_config()` merges persisted settings with environment variables:

* `MODEL_DIR` – custom cache directory for ONNX models (default: `~/.cache/bg-remover/models`).
* `BGR_DEFAULT_MODEL` – fallback model key when none is supplied.
* `BGR_PROVIDER_HINTS` – comma-separated provider hints (e.g. `CUDAExecutionProvider,CPUExecutionProvider`).
* `BGR_LOGLEVEL` – root log level (`INFO`, `DEBUG`, etc.).

`persist_config()` writes settings to `~/.bgremover.json`. The GUI provides a “Save” action, while the CLI exposes `--persist-config`.

## Model catalogue

| Model key | Input size | Notes |
|-----------|------------|-------|
| `isnet-general-use` | 1024×1024 | General purpose |
| `u2net_human_seg` | 320×320 | Portrait focused |
| `u2net` | 320×320 | Object isolation |
| `isnet-anime` | 1024×1024 | Illustration/anime |
| `briaai/RMBG-2.0` | 1024×1024 | High-quality general scenes |
| `matting-by-generation` | 1024×1024 | Portrait matting variant |
| `sam_segmentation_model` | 1024×1024 | Alias of BRIA 2.0 |
| `sam_vit_b_01ec64_encoder` / `decoder` | 1024×1024 | Segment Anything weights |

Weights download automatically on first use. Hugging Face downloads honour `HUGGINGFACEHUB_API_TOKEN` or `HF_API_TOKEN` when private repos are required.

## Logging

`init_logging()` configures structured console logging and writes `error.log` in the current working directory (or alongside the packaged executable). GUI status entries mirror these logs and colourise failures.

## Migration guide

* Old modules such as `bg_removal.py`, `main.py`, and `app.py` have been replaced by the `bgremover_core` package and the runtime-specific entry points under `runtimes/`.
* CLI invocation is now `python -m runtimes.cli.bgr_cli remove ...`.
* Flask app factory lives at `runtimes.flask_app.app:create_app`.
* GUI entry point is `python -m runtimes.gui.bg_remover_gui`.

## Testing

Run the consolidated test suite with:

```
pytest -q
```

The new tests cover provider detection, session creation, pipeline happy/error paths, CLI exit codes, Flask routes, and GUI helper logic.

## Docker

See `deployment.md` for runtime-specific build arguments, compose examples, and GPU notes.
