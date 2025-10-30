# Deployment

The repo ships with a single Dockerfile that builds any runtime. Use the `RUNTIME` build argument to choose between the CLI, Flask app, or GUI.

## Building images

```
docker build -t br-remover-cli --build-arg RUNTIME=cli .
docker build -t br-remover-flask --build-arg RUNTIME=flask .
docker build -t br-remover-gui --build-arg RUNTIME=gui .
```

The image configures `MODEL_DIR=/models`, so you can mount a persistent cache:

```
docker run --rm -it \
  -v $(pwd)/models:/models \
  br-remover-cli python -m runtimes.cli.bgr_cli remove --help
```

## Runtime commands

| Runtime | Default command |
|---------|-----------------|
| CLI | `python -m runtimes.cli.bgr_cli --help` |
| Flask | `gunicorn "runtimes.flask.app:create_app()" --bind 0.0.0.0:8080` |
| GUI | `python -m runtimes.gui.bg_remover_gui` (requires host display forwarding) |

The Flask variant exposes port `8080`. Map it to the host when running containers: `-p 8080:8080`.

### Flask UI highlights

* Drag & drop uploads, batch ZIP support, and a responsive layout optimised for desktops and tablets.
* Persistent preferences stored in the browser (theme, advanced settings) with optional server-side sync via `remember_preferences`.
* Dark/light theme toggle, GPU/CPU provider badge, and a background colour picker with transparency toggle.
* Live activity log, thumbnail previews, and a `/history` endpoint that surfaces all processed files for the current container.

By default processed assets are written to `<instance_path>/results`. Override this location via `OUTPUT_DIR` if you prefer to mount a dedicated volume:

```
docker run --rm -it \
  -p 8080:8080 \
  -v $(pwd)/models:/models \
  -v $(pwd)/web-results:/var/lib/bgremover/results \
  -e OUTPUT_DIR=/var/lib/bgremover/results \
  br-remover-flask
```

## Docker Compose example

```yaml
services:
  br-remover:
    build:
      context: .
      args:
        RUNTIME: flask
    image: br-remover-flask
    ports:
      - "8080:8080"
    volumes:
      - ./models:/models
      - ./web-results:/var/lib/bgremover/results
    environment:
      - MODEL_DIR=/models
      - BGR_LOGLEVEL=INFO
      - OUTPUT_DIR=/var/lib/bgremover/results
```

## GPU acceleration

* Install the NVIDIA container toolkit on the host.
* Build with the CUDA-enabled ONNX Runtime (e.g. `pip install onnxruntime-gpu`) or extend the image accordingly.
* Run containers with the appropriate runtime:

  ```
  docker run --rm -it \
    --gpus all \
    -e BGR_PROVIDER_HINTS=CUDAExecutionProvider,CPUExecutionProvider \
    -v $(pwd)/models:/models \
    br-remover-cli python -m runtimes.cli.bgr_cli remove --input sample.png --output result.png
  ```

The GUI image is intended for development. To use it in Docker you must configure X11 or Wayland forwarding (Linux) or rely on host execution.
