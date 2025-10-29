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
| Flask | `gunicorn "runtimes.flask_app.app:create_app()" --bind 0.0.0.0:8080` |
| GUI | `python -m runtimes.gui.bg_remover_gui` (requires host display forwarding) |

The Flask variant exposes port `8080`. Map it to the host when running containers: `-p 8080:8080`.

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
    environment:
      - MODEL_DIR=/models
      - BGR_LOGLEVEL=INFO
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
