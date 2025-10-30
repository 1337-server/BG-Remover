# Deployment

This document collects build, packaging, and hosting tips for every BG-Remover
runtime. All runtimes share the same project directories (`input/`, `output/`,
and `config/config.json`) so configuration persists regardless of how an
executable is launched.

## Building from source

Start by installing shared dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### CLI runtime

```bash
python -m runtimes.cli.bgr_cli remove --input ./input/example.png --output ./output/example_no_bg.png
```

Useful flags:

- `--batch` to process entire folders.
- `--provider cuda|directml|cpu` to request a specific execution provider.
- `--config` to load a custom `config.json` (defaults to `config/config.json`).
- `--persist-config` to store CLI preferences for future runs.

### Flask runtime

Run the development server with auto-reload:

```bash
export FLASK_APP=runtimes.flask.app
python -m flask run --debug
```

For production, create a WSGI server configuration:

```bash
gunicorn "runtimes.flask.app:create_app()" --bind 0.0.0.0:8080 --workers 4
```

The Flask UI ships with distinct pages for single-image and batch/folder
processing. The navigation bar links both views, each embedding a live preview
pane and advanced settings that stretch across the page beneath the preview.
Dark mode applies to the full layout, including the `<body>`, root element, and
all cards.

### GUI runtime

Launch the desktop GUI directly from source:

```bash
python -m runtimes.gui.bg_remover_gui
```

The GUI preserves dark/light mode selection across sessions, restores the full
set of advanced controls (collapsed by default), surfaces tooltips describing
valid ranges, and lets users choose the ONNX model storage directory.

## Packaging the GUI

Use PyInstaller to build a standalone executable with the project icon:

```bash
pyinstaller runtimes/gui/bg_remover_gui.py --noconfirm --onefile --windowed --icon=assets/icon.ico
```

The generated binary reads and writes configuration to `config/config.json` in
the project directory, so bundle that file alongside the executable when
redistributing. Ship the `input/` and `output/` folders to preserve defaults.

## Docker

The repository Dockerfile supports every runtime via the `RUNTIME` build
argument:

```bash
docker build -t bg-remover-cli --build-arg RUNTIME=cli .
docker build -t bg-remover-flask --build-arg RUNTIME=flask .
docker build -t bg-remover-gui --build-arg RUNTIME=gui .
```

### Running the CLI container

```bash
docker run --rm -it \
  -v $(pwd)/models:/models \
  -v $(pwd)/input:/app/input \
  -v $(pwd)/output:/app/output \
  bg-remover-cli python -m runtimes.cli.bgr_cli remove --input /app/input/sample.png
```

Mount `config/config.json` if you want to persist configuration across runs.

### Running the Flask container

```bash
docker run --rm -it \
  -p 8080:8080 \
  -v $(pwd)/models:/models \
  -v $(pwd)/input:/app/input \
  -v $(pwd)/output:/app/output \
  bg-remover-flask
```

The container entrypoint launches Gunicorn bound to `0.0.0.0:8080`. The batch
page mirrors the desktop workflow by accepting ZIP/folder uploads and listing
results beneath the preview grid.

### Optional Docker Compose

```yaml
services:
  bg-remover:
    build:
      context: .
      args:
        RUNTIME: flask
    image: bg-remover-flask
    ports:
      - "8080:8080"
    volumes:
      - ./models:/models
      - ./input:/app/input
      - ./output:/app/output
    environment:
      - MODEL_DIR=/models
      - BGR_PROVIDER_HINTS=CUDAExecutionProvider,CPUExecutionProvider
```

Enable GPU acceleration by installing the NVIDIA container toolkit and passing
`--gpus all` (or the ROCm equivalent) to `docker run`. ONNX Runtime automatically
falls back to CPU when a requested GPU provider fails.

## Local Dockerless deployment checklist

1. Create a Python virtual environment and install dependencies.
2. Copy or symlink `input/`, `output/`, and `config/` into your deployment
   location.
3. Launch the runtime of choice using the commands above.
4. Set `MODEL_DIR` if you want to share model caches across machines or disks.
