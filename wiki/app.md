# app.py — Flask Web UI

## Overview
`app.py` hosts the Flask application that powers the browser-based background removal interface. It defines the web routes, download registries, and startup hooks that prepare ONNX models before serving requests.

## Role in the Project
- Serves HTML templates and static assets under `templates/` and `static/`.
- Exposes REST-like endpoints for image uploads, batch ZIP downloads, and status polling.
- Manages temporary files and registries shared with the frontend.
- Provides command-line helpers (`--dev`, `--host`, `--port`) for local development.

## Key Components
- **`create_app`**: Factory that initialises Flask, optionally runs model download/session warm-up, and registers routes.
- **Blueprint `image_converter_bp`**: Groups API routes for upload, preview, and download handling.
- **Registry dataclasses (`RegistryItem`, `ZipRegistryItem`)**: Track generated assets, TTLs, and cleanup timers.
- **Route handlers**: `/`, `/convert`, `/status/<task_id>`, `/download/<file_id>`, `/zip/<task_id>` and others coordinate background processing and responses.
- **Background helpers**: Functions like `_schedule_registry_eviction` and `_create_zip_async` offload long-running work.

## Implementation Notes
- Requires `ensure_models_downloaded()` and `ensure_global_session()` from `bg_removal.py` before serving requests.
- Uses thread locks to guard registries; avoid blocking operations inside these sections.
- Cleans temporary files on app teardown via `atexit` handlers; remember to register new registries for cleanup.
- Integrates with the Jinja templates for configuration payloads (`MODEL_CONFIGS_FRONTEND`, `FORMAT_OPTIONS`).

## Invocation Example
```bash
# Development server with auto-reload
python app.py --dev --host 0.0.0.0 --port 5000
```

> **UI Type:** Flask web UI.
