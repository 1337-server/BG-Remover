## 🖼️ Background Remover (rembg + U²-Net)

This project combines a rich command-line tool and a small Flask UI to remove image backgrounds with
[rembg](https://github.com/danielgatis/rembg) (U²-Net based) while keeping memory usage and processing
times predictable. It can clean up single images, entire folders, or uploaded files, and always writes
PNG output with transparency preserved.

---

### 🚀 Features

* ✅ **rembg-powered masks** with optional alpha matting for detailed hair and fur handling.
* ✅ **Solid background fallback** (colour-key) plus configurable feathering when OpenCV is available.
* ✅ **Single-image CLI** and **folder batch mode** that respect EXIF orientation and reuse one model session.
* ✅ **Web interface** built with Flask featuring upload + server-folder workflows and ZIP downloads.
* ✅ Works on CPU or GPU, auto-orients input files, and limits oversized images to keep RAM usage stable.

---

### 📦 Installation

1. Clone this repository.
2. (Optional) Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # macOS/Linux
   .venv\Scripts\activate      # Windows
   ```
3. Install runtime dependencies (force-reinstall if upgrading from an older environment):

   ```bash
   # CPU-only installation (default)
   pip install --upgrade --force-reinstall -r requirements-cpu.txt

   # CUDA 12.x GPU installation
   pip install --upgrade --force-reinstall -r requirements-gpu.txt

   # Compatibility wrapper that defers to the CPU set
   pip install --upgrade --force-reinstall -r requirements.txt
   ```

The first run of either the CLI or web service initialises a single rembg session and caches the U²-Net
weights automatically.

---

### 🚦 Quick start

Follow these steps to try the background remover in a few minutes:

1. **Prepare the environment**

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # macOS/Linux
   .venv\Scripts\activate      # Windows
   # Choose one of the dependency sets
   pip install -r requirements-cpu.txt          # CPU only
   # pip install -r requirements-gpu.txt        # CUDA-enabled GPU
   ```

2. **Run the CLI for a single image**

   ```bash
   python main.py --input ./samples/cat.jpg
   ```

   The processed image is saved to `./output/cat.png`. Add `--output` to customise the destination
   directory.

3. **Batch-convert an entire folder**

   ```bash
   python main.py --input ./photos --recursive --output ./photos_cutout
   ```

   Nested folders are preserved when `--recursive` is provided.

4. **Launch the Flask web interface**

   ```bash
   python serve.py
   ```

   Open http://127.0.0.1:5000/image/remove-bg to upload a file or process a server-side folder from
   your browser.

5. **Optional: run everything in Docker**

   ```bash
   docker compose up --build                 # CPU-only image
   # docker compose --profile gpu up --build  # CUDA-enabled image (requires --gpus all)
   ```

   The service exposes port `5000`. The GPU profile assumes `nvidia-container-toolkit` is installed and
   the container is launched with GPU access enabled.

---

### ⚡ Accelerator configuration

Background removal automatically prefers CUDA when the `onnxruntime-gpu` build is installed and
the ONNX Runtime `CUDAExecutionProvider` is available. You can override or inspect the behaviour
with the following knobs:

* **Environment variables**
  * `BG_ACCELERATOR` – `auto` (default), `cuda`, or `cpu` to pin the runtime.
  * `BG_CUDA_DEVICE_ID` – CUDA device index to bind when using the GPU (default `0`).
  * `BG_WARN_ON_CPU` – `true` (default) to surface warnings when the app falls back to CPU.
* **CLI flags**
  * `python main.py --accelerator cuda --cuda-device-id 1`
  * `python serve.py --accelerator cpu --no-warn-on-cpu`
* **Health checks**
  * `GET /health/accelerator` returns the detected ONNX providers, selected runtime, GPU name, and RTX 50-series flag.

When the app detects that it is running on the CPU despite a compatible GPU, a dismissible banner
appears in the UI and a single log warning explains how to install `onnxruntime-gpu`, enable
`nvidia-container-toolkit`, or launch Docker with `--gpus all`.

---

### 🧠 Command-line usage

Run the CLI with:

```bash
python main.py --input path/to/image_or_folder --output optional/output/dir \
  --alpha-matting --am-foreground 240 --am-background 10 --am-erode 10 \
  --colorkey-tolerance 14 --feather-radius 3 --recursive
```

Key behaviour:

* Passing a **file** writes to `./output/<name>.png` (or to `--output` if given).
* Passing a **folder** produces PNGs under `./output` (or the directory from `--output`).
* Use `--alpha-matting` + thresholds for tricky edges, `--no-colorkey-fallback` to disable the
  solid-colour helper, and `--recursive` to process nested folders.
* Override accelerator selection with `--accelerator [auto|cuda|cpu]`, `--cuda-device-id`, and
  `--no-warn-on-cpu` to suppress CLI warnings.

---

### 🌐 Flask web interface

Start the web UI once the dependencies are installed:

```bash
python serve.py
# or
python -m app
```

Both entry points run the standard Flask development server on `0.0.0.0:5000`. The
interface processes each upload end-to-end and returns the final result once the
background removal is complete—no streaming or WebSocket connection is required.

Open http://127.0.0.1:5000/image/remove-bg in your browser to access:

* **Single Image** tab – upload an image, receive the processed file once complete, or fetch JSON payloads.
* **Folder Processing** tab – supply a server-side folder, optional output directory, recursive mode,
  alpha-matting settings, and request a ZIP bundle of the processed results.

The Flask app initialises a single rembg session on startup so repeated requests remain fast. Set
`BR_FORCE_CPU=1` to disable CUDA when troubleshooting GPU driver mismatches. The UI header always
shows the selected accelerator and GPU name, and displays a dismissible warning if the app is running
on the CPU because CUDA providers are missing.

---

### 🧪 Tests

The suite includes service-level regression tests. Run them with:

```bash
pytest
```

You can also sanity-check the project compiles with:

```bash
python -m compileall main.py app tests
```

---

### 📂 Project layout

* `main.py` – CLI entrypoint.
* `app/services/bg_remove.py` – rembg session management and folder/file helpers.
* `app/routes/image_converter.py` – Flask blueprint exposing the UI + JSON endpoints.
* `templates/` – Base template + background removal form.
* `tests/` – Pytest-based regression tests for the service utilities.

---

### 🧰 Credits

* **Model:** [U²-Net – Qin et al., Pattern Recognition 2020](https://github.com/xuebinqin/U-2-Net)
* **Background removal engine:** [rembg](https://github.com/danielgatis/rembg)
