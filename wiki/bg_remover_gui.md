# bg_remover_gui.py — Tkinter Desktop GUI

## Overview
`bg_remover_gui.py` delivers a ttkbootstrap-powered desktop application that mirrors the web UI's options for local users. It supports drag-and-drop style single-image processing and batch folder jobs with live progress updates.

## Role in the Project
- Provides an offline-friendly GUI that does not require a web browser.
- Shares configuration defaults and model choices with the Flask UI to keep behaviour consistent.
- Surfaces runtime logs and error handling tailored for packaged executables (e.g., PyInstaller).

## Key Components
- **`REMOVAL_MODEL_OPTIONS`**: Declares the same model presets exposed by the web app, mapped to ONNX model names.
- **UI classes/functions**: `BackgroundRemovalApp`, worker threads, and event bindings manage form controls, previews, and job queues.
- **Helper utilities**: `resource_path`, `_ensure_pyinstaller_hidden_imports`, `_runtime_directory`, and `_write_traceback_to_log` support packaged deployments.
- **Processing workers**: Use `remove_bg_file` from `bg_removal.py` on background threads and communicate via queues for responsive UI updates.

## Implementation Notes
- Depends on `ttkbootstrap`, `Pillow`, and `watchdog` for the live folder monitor; ensure these extras are bundled when shipping installers.
- Logs fatal errors to `error.log` in the runtime directory so non-technical users can share diagnostics.
- Uses `ensure_models_downloaded()` and `ensure_global_session()` during startup to pre-warm inference sessions.
- Threaded workers must interact with Tkinter via queued callbacks; avoid direct widget updates from background threads.

## Invocation Example
```bash
python bg_remover_gui.py
```

> **UI Type:** Tkinter desktop GUI.
