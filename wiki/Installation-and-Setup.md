# Installation & Setup

> 📦 **Scope:** Configure BG-Remover for any runtime (CLI, Flask, GUI) using the
> shared project layout.

## Prerequisites

| Component | Notes |
| --- | --- |
| Python | 3.10 or later (the project is routinely verified on Python 3.12). Create a virtual environment for isolation. |
| Pip packages | `pip install -r requirements.txt` installs the shared core and runtime dependencies. |
| GPU acceleration | Install the appropriate ONNX Runtime build: `onnxruntime-gpu` (CUDA), `onnxruntime-directml`, or `onnxruntime-rocm`. |
| Drivers | Keep GPU drivers current (NVIDIA CUDA, AMD ROCm, Intel DML). |

Optional but recommended:

- `pyinstaller` when packaging the GUI runtime.
- Docker / Docker Compose for containerised deployments.

## Repository layout

BG-Remover repositories now expose a unified directory structure shared by every
runtime:

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

The first time you import `bgremover_core.paths`, these directories are created
automatically. Configuration written by any runtime is saved to
`config/config.json`, so the GUI, Flask app, and CLI stay aligned.

## Environment variables

Set environment variables to override the defaults without editing the config
file:

- `BGR_CONFIG_PATH` — custom path to `config.json` if the repo is embedded in
  another project.
- `MODEL_DIR` — path where ONNX models are cached.
- `BGR_PROVIDER_HINTS` — comma-separated list of providers to prioritise (e.g.
  `CUDAExecutionProvider,CPUExecutionProvider`).
- `BGR_DEFAULT_MODEL` — default model key when none is supplied.
- `BGR_LOGLEVEL` — global log level (INFO, DEBUG, WARNING, ...).

## Dependency installation checklist

1. **Clone the repository** (or download an archive) to the desired location.
2. **Create a virtual environment**:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. **Install Python packages**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Optional GPU extras**:

   ```bash
   pip install onnxruntime-gpu  # or directml / rocm variants
   ```

5. **Verify directories**: ensure `input/`, `output/`, and `config/` exist (they
   are generated automatically, but double-check for CI/deployment pipelines).
6. **Test a runtime** by launching the Flask server directly:

   ```bash
   python -m runtimes.flask.app
   ```

   When the console reports `Running on http://127.0.0.1:5000`, open the URL in a
   browser. You can also explore the [CLI Usage](CLI-Usage.md) and
   [Using the GUI](Using-the-GUI.md) guides.

## Updating dependencies

Use `pip install -U -r requirements.txt` to upgrade shared dependencies. When
packaging the GUI, rebuild the PyInstaller bundle after updating Python modules.
