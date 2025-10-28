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
   # CPU-only environments (default)
   pip install --upgrade --force-reinstall -r requirements-cpu.txt

   # NVIDIA GPU environments with CUDA 12.x drivers
   pip install --upgrade --force-reinstall -r requirements-gpu.txt
   ```

   The root `requirements.txt` file simply references the CPU variant for backward compatibility.

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

   # Choose the dependency set that matches your hardware
   pip install -r requirements-cpu.txt   # or requirements-gpu.txt
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
   # CPU image
   docker compose up --build

   # GPU image (requires nvidia-container-toolkit and --profile gpu)
   docker compose --profile gpu up --build
   ```

   The service exposes port `5000` and watches the `./data` volume for input/output folders. You can
   also build the images manually:

   ```bash
   docker build -f Dockerfile.cpu -t bg-remover:cpu .
   docker build -f Dockerfile.gpu -t bg-remover:gpu .
   docker run --rm -it -p 5000:5000 bg-remover:cpu
   docker run --rm -it --gpus all -p 5000:5000 bg-remover:gpu
   ```

---

### ⚙️ Accelerator configuration

The application selects the best available ONNXRuntime execution provider on startup:

* Set `BG_ACCELERATOR` to `auto` (default), `cuda`, or `cpu` to influence provider choice.
* Set `BG_CUDA_DEVICE_ID` (default `0`) to target a specific GPU when CUDA is available.
* Control warning behaviour with `BG_WARN_ON_CPU` (`true` by default). When enabled, the UI and logs
  display a banner if the app falls back to CPU execution because no GPU was detected or initialisation failed.

Both the CLI (`main.py`) and development server (`serve.py`) expose matching flags:

* `--accelerator [auto|cuda|cpu]`
* `--cuda-device-id <int>`
* `--no-warn-on-cpu`

The web UI surfaces the active accelerator in an info banner and exposes diagnostics via
`GET /health/accelerator`, which reports the detected providers, selected runtime, GPU name (if any),
and RTX 50-series detection flag for quick troubleshooting.

---

### 🧠 Command-line usage

Run the CLI with:

```bash
python main.py --input path/to/image_or_folder --output optional/output/dir \
  --alpha-matting --am-foreground 240 --am-background 10 --am-erode 10 \
  --colorkey-tolerance 14 --feather-radius 3 --recursive \
  --accelerator auto --cuda-device-id 0
```

Key behaviour:

* Passing a **file** writes to `./output/<name>.png` (or to `--output` if given).
* Passing a **folder** produces PNGs under `./output` (or the directory from `--output`).
* Use `--alpha-matting` + thresholds for tricky edges, `--no-colorkey-fallback` to disable the
  solid-colour helper, and `--recursive` to process nested folders.
* Add `--accelerator cuda` to force GPU usage or `--accelerator cpu` when debugging CUDA setups.
  Combine with `--no-warn-on-cpu` to silence CPU fallback warnings during benchmarks.

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

The Flask app initialises a single rembg session on startup so repeated requests remain fast. A
runtime banner highlights the active accelerator; when the service is running on CPU due to missing
CUDA support a dismissible warning is shown (controllable via `BG_WARN_ON_CPU`). Use the
`/health/accelerator` endpoint for a quick JSON snapshot of the configured providers.

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
