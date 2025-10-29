# Background Remover Wiki

Welcome to the developer wiki for the background removal toolkit. This project exposes the same ONNX-based image segmentation pipeline through multiple user interfaces:

- **Flask web UI** (served from `app.py`)
- **Tkinter desktop UI** (implemented in `bg_remover_gui.py`)
- **Command-line interface** (`main.py`)

Common inference and model-management logic lives in `bg_removal.py` with supporting modules like `model_configs.py` and `requests_shim.py`.

## Quick Start

- Install dependencies with `pip install -r requirements.txt` (or use the Docker/Docker Compose files).
- Download ONNX models by running either UI once or calling `python -m bg_removal --help` to trigger lazy setup.
- Launch your preferred interface:
  - Flask UI: `python app.py --dev`
  - Tkinter UI: `python bg_remover_gui.py`
  - CLI: `python main.py --help`

## Module Reference

- [app.py](app.md) — Flask routes, background tasks, and download registries for the browser UI. *(Flask web UI)*
- [bg_removal.py](bg_removal.md) — Core ONNX inference engine, downloads, and file-processing helpers.
- [bg_remover_gui.py](bg_remover_gui.md) — Tkinter/ttkbootstrap desktop application. *(Tkinter desktop GUI)*
- [main.py](main.md) — Command-line entry point for batch and single-image processing.
- [model_configs.py](model_configs.md) — Shared schema describing advanced model parameters.
- [requests_shim.py](requests_shim.md) — Minimal fallback for `requests` during restricted deployments.

Refer to each page for detailed documentation, usage notes, and integration tips.
