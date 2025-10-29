# bg_removal.py — Core Processing Engine

## Overview
`bg_removal.py` implements the ONNX Runtime processing pipeline shared by all interfaces. It downloads models, prepares inference sessions, and exposes helpers for per-file and batch background removal.

## Role in the Project
- Centralises model discovery, download, and caching logic.
- Provides file- and folder-level processing utilities consumed by the CLI, Flask UI, and Tkinter UI.
- Offers status reporting for downloads and runtime capability checks (e.g., accelerator availability).

## Key Components
- **`OutputFormat` dataclass** and **`OUTPUT_FORMATS`** tuple define supported export options (PNG, WebP, JPG) and normalisation helpers.
- **Download management**: `DownloadStatus`, `ensure_models_downloaded`, `prefetch_model_async`, and related locking infrastructure keep model assets available.
- **Processing helpers**: `remove_bg_file`, `remove_bg_folder`, `remove_bg_bytes`, and `encode_result_image` drive inference and image post-processing.
- **Session lifecycle**: `ensure_global_session`, `get_runtime_payload`, and `_load_session` coordinate ONNX Runtime providers and reuse sessions per model.
- **Result dataclasses**: `RemovalResult` captures timing, output paths, and errors for each processed file.

## Implementation Notes
- Relies on `onnxruntime`, `numpy`, and `Pillow`; OpenCV (`cv2`) is optional for matting refinements.
- Supports fallbacks when `requests` is unavailable by using `requests_shim`.
- Uses thread pools for concurrent batch processing—avoid long-lived global state in callbacks.
- Output formats enforce extension normalisation via `OutputFormat.normalise_filename`.
- Environment variables (`ONNXRUNTIME_EXECUTION_PROVIDERS`, etc.) influence accelerator selection.

## Invocation Example
```python
from pathlib import Path
from bg_removal import remove_bg_file

result = remove_bg_file(Path("input.png"), "output", alpha_matting=True)
if result.success:
    print("Saved to", result.path_out)
else:
    print("Failed:", result.error)
```

> **Shared Logic:** Used by Flask web UI, Tkinter GUI, and CLI.
