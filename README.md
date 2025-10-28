## 🖼️ Background Remover (rembg + U²-Net)

This project combines a rich command-line tool and a small Flask UI to remove image backgrounds with
[rembg](https://github.com/danielgatis/rembg) (U²-Net based) while keeping memory usage and processing
times predictable. It can clean up single images, entire folders, or uploaded files, and exports in
multiple formats (PNG, WebP, JPEG, BMP, TIFF) with transparency preserved whenever the format allows it.

---

### 🚀 Features

* ✅ **rembg-powered masks** with optional alpha matting for detailed hair and fur handling.
* ✅ **Solid background fallback** (colour-key) plus configurable feathering when OpenCV is available.
* ✅ **Single-image CLI** and **folder batch mode** that respect EXIF orientation and reuse one model session.
* ✅ **Flexible exports** with selectable PNG, WebP, JPEG, BMP, or TIFF output (alpha preserved when supported).
* ✅ **Web interface** built with Flask featuring upload + server-folder workflows, ZIP downloads, and previews.
* ✅ Works on CPU or GPU, auto-orients input files, and limits oversized images to keep RAM usage stable.

---

### 📦 Installation

1. Clone this repository.
2. Create and activate a virtual environment (PowerShell shown below):

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\activate
   ```

   > Tip: run `scripts\clean_venv.ps1` to recreate the environment from scratch.

3. Install the pinned dependency set (includes CUDA-enabled ONNX Runtime and PyTorch wheels):

   ```powershell
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

The first run of either the CLI or web service initialises a single ONNX Runtime session and caches the
U²-Net / ISNet weights automatically.

---

### 🚦 Quick start

The commands below assume Windows PowerShell, but the same steps work on other platforms with minor
syntax tweaks.

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
scripts\verify_gpu.ps1
python serve.py
```

The verification script prints the ONNX Runtime provider list and Torch CUDA status so you can confirm
that the `CUDAExecutionProvider` is active. If CUDA is unavailable the app automatically falls back to
the CPU and logs a single warning explaining how to install a compatible GPU wheel.

Troubleshooting tips:

* Clear conflicting CUDA settings by unsetting `CUDA_PATH` before installing the requirements.
* Delete stale virtual environments with `scripts\clean_venv.ps1` when upgrading dependencies.
* If the CUDA provider still refuses to load, run `onnxruntime.get_available_providers()` inside a
  Python shell to confirm the wheel matches your driver; a CPU-only wheel continues to work.

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
  --format webp --alpha-matting --am-foreground 240 --am-background 10 --am-erode 10 \
  --colorkey-tolerance 14 --feather-radius 3 --recursive
```

Key behaviour:

* Passing a **file** writes to `./output/<name>.<format>` (or to `--output` if given).
* Passing a **folder** produces results under `./output` (or the directory from `--output`).
* Use `--alpha-matting` + thresholds for tricky edges, `--no-colorkey-fallback` to disable the
  solid-colour helper, and `--recursive` to process nested folders.
* Override accelerator selection with `--accelerator [auto|cuda|cpu]`, `--cuda-device-id`, and
  `--no-warn-on-cpu` to suppress CLI warnings.
* Specify `--format [png|webp|jpg|bmp|tiff]` to control the export type; unsupported values raise
  a clear validation error.

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

* **Single Image** tab – upload an image, receive the processed file once complete, request alternate
  formats, or fetch JSON payloads.
* **Folder Processing** tab – supply a server-side folder, optional output directory, recursive mode,
  alpha-matting settings, ZIP bundle downloads, and preview-size controls.

The Flask app initialises a single ONNX Runtime session on startup so repeated requests remain fast. Set
`BR_FORCE_CPU=1` to disable CUDA when troubleshooting GPU driver mismatches. The UI header always
shows the selected accelerator and GPU name, and displays a dismissible warning if the app is running
on the CPU because CUDA providers are missing.

---

### 🧪 Tests

The suite includes service-level regression tests. Run them with:

```bash
pytest
```

Static analysis helpers are included:

```bash
ruff check .
mypy app
python -m compileall main.py app tests
```

---

### 📂 Project layout

* `main.py` – CLI entrypoint.
* `app/services/model_registry.py` – shared ONNX Runtime session preloading and warm-up utilities.
* `app/services/bg_remove.py` – background removal helpers built on preloaded ONNX Runtime sessions.
* `app/routes/image_converter.py` – Flask blueprint exposing the UI + JSON endpoints.
* `templates/` – Base template + background removal form.
* `tests/` – Pytest-based regression tests for the service utilities.

---

### 🧰 Credits

* **Model:** [U²-Net – Qin et al., Pattern Recognition 2020](https://github.com/xuebinqin/U-2-Net)
* **Background removal engine:** [rembg](https://github.com/danielgatis/rembg)
