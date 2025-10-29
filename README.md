## 🖼️ Background Remover (U²-Net ONNX)

This project combines a rich command-line tool and a small Flask UI to remove image backgrounds using
direct ONNX Runtime sessions powered by the U²-Net family of models. It can clean up single images,
entire folders, or uploaded files, and exports in multiple formats (PNG, WebP, JPEG) with transparency
preserved whenever the format allows it.

---

### 🚀 Features

* ✅ **ONNX Runtime-powered masks** with optional alpha matting for detailed hair and fur handling.
* ✅ **Solid background fallback** (colour-key) plus configurable feathering when OpenCV is available.
* ✅ **Single-image CLI** and **folder batch mode** that respect EXIF orientation and reuse one model session.
* ✅ **Flexible exports** with selectable PNG, WebP, or JPEG output (alpha preserved when supported).
* ✅ **Multiple removal models** covering general scenes, portraits, products, and anime-style artwork.
* ✅ **Web interface** built with Flask featuring upload + server-folder workflows, ZIP downloads, previews, and guided help.
* ✅ **Desktop GUI** powered by ttkbootstrap with single-image and cancellable folder batch processing.
* ✅ Runs entirely on CPU, auto-orients input files, limits oversized images to keep RAM usage stable, and caches downloaded weights.

### 🎯 Model catalogue

The application automatically fetches the required ONNX weights when they are
first used. General-purpose defaults rely on the IS-Net family, while the
advanced options now point to actively maintained BRIA releases to avoid the
previous 404 errors:

| UI option | Model key | Source |
|-----------|-----------|--------|
| General | `isnet-general-use` | GitHub release (danielgatis/rembg) |
| High-quality General | `briaai/RMBG-2.0` | Hugging Face (`briaai/RMBG-2.0`) |
| Portrait Matting | `matting-by-generation` | Hugging Face (`briaai/BRIA-RMBG-1.4`) |
| Complex Scene | `sam_segmentation_model` | Hugging Face (`briaai/RMBG-2.0`) |
| Human | `u2net_human_seg` | GitHub release (danielgatis/rembg) |
| Object | `u2net` | GitHub release (danielgatis/rembg) |
| Anime | `isnet-anime` | GitHub release (danielgatis/rembg) |

---

### 📦 Installation

1. Clone this repository.
2. Create and activate a virtual environment (PowerShell shown below):

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\activate
   ```

   > Tip: run `scripts\clean_venv.ps1` to recreate the environment from scratch.

3. Install the pinned dependency set (CPU-only ONNX Runtime build):

   ```powershell
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

The first run of either the CLI or web service initialises a single ONNX Runtime session and caches the
U²-Net / ISNet weights automatically in `~/.u2net`.

---

### 🚦 Quick start

The commands below assume Windows PowerShell, but the same steps work on other platforms with minor
syntax tweaks.

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

Troubleshooting tips:

* Delete stale virtual environments with `scripts\clean_venv.ps1` when upgrading dependencies.

---

### 🧠 Command-line usage

Run the CLI with:

```bash
python main.py --input path/to/image_or_folder --output optional/output/dir \
  --model human --format webp --alpha-matting --am-foreground 240 --am-background 10 \
  --am-erode 10 --colorkey-tolerance 14 --feather-radius 3 --recursive
```

Key behaviour:

* Passing a **file** writes to `./output/<name>.<format>` (or to `--output` if given).
* Passing a **folder** produces results under `./output` (or the directory from `--output`).
* Use `--alpha-matting` + thresholds for tricky edges, `--model` to pick between general, human,
  object, or anime-focused weights, `--no-colorkey-fallback` to disable the solid-colour helper, and
  `--recursive` to process nested folders.
* Specify `--format [png|webp|jpg]` to control the export type; unsupported values raise a clear
  validation error.

---

### 🌐 Flask web interface

Start the web UI once the dependencies are installed:

```bash
python app.py
# or
python -m app
```

Both entry points run the standard Flask development server on `0.0.0.0:5000`. The
interface processes each upload end-to-end and returns the final result once the
background removal is complete—no streaming or WebSocket connection is required.

Open http://127.0.0.1:5000/image/remove-bg in your browser to access:

* **Single Image** tab – upload an image, receive the processed file once complete, request alternate
  formats, or fetch JSON payloads. Preview size, feathering, and destination directory can be tuned
  before submitting.
* **Folder Processing** tab – supply a server-side folder, optional output directory, recursive mode,
  alpha-matting settings, ZIP bundle downloads, and preview-size controls. Processed batches can be
  fetched as individual files or as an on-demand ZIP archive.

The Flask app initialises a single ONNX Runtime session on startup so repeated requests remain fast. The
UI header displays the active CPU execution provider so you can confirm the model is ready before
processing uploads.

---

### 🪟 Desktop GUI

Launch the ttkbootstrap-based desktop client to work with local files:

```bash
python bg_remover_gui.py
```

Key capabilities:

* **Single-image mode** (default) – choose an image file, preview the foreground mask, and export the
  processed result without blocking the interface.
* **Folder mode** – point the app at a directory and it will enumerate supported images (PNG, JPG,
  WebP, and more), display the file count, and process them in a background thread with a live
  progress bar, ETA, and log console.
* **Safe exports** – each batch is written to an `output/` subfolder inside the chosen directory to
  preserve the original assets. Existing files can be skipped automatically.
* **Cancellable runs** – stop an in-progress folder job with the *Cancel Batch* button; the UI remains
  responsive while work continues in the background thread.

Toggle *Folder Batch* mode via the radio buttons to reveal the folder workflow controls. While a batch
is running the file picker is disabled, the current image name and thumbnail are shown, and progress
updates are appended to the footer console. All advanced options (model selection, alpha matting,
colour-key fallback, etc.) mirror those exposed in the Flask UI.

> **Tip:** The first time you launch the GUI it downloads the selected model weights to
> `~/.u2net`. Future runs reuse the cached files so processing starts immediately.

#### Packaging the desktop app

Bundle the Tkinter interface with the optimized helper script in `scripts/build_executable.py`:

```bash
python scripts/build_executable.py --entry-point bg_remover_gui.py --name BackgroundRemoverGUI
```

The wrapper enables PyInstaller's multi-core build mode, automatically includes the `static/` and
`templates/` assets, and compresses the output with UPX when the packer is installed. Pass
`--onefile` to create a single-binary build or `--clean` to discard existing `build/` and `dist/`
artifacts before compiling; use `--no-upx` if antivirus software objects to the compressed output.

The GUI now wraps its startup sequence in a safe handler that logs uncaught exceptions to
`error.log` beside the script or bundled executable. If a packaged run fails you will also see a
message box pointing to that log file for the full traceback.

---

### 🧪 Tests

The suite includes service-level regression tests. Run them with:

```bash
pytest
```

Static analysis helpers are included:

```bash
ruff check .
mypy app.py bg_removal.py main.py
python -m compileall main.py app.py bg_removal.py tests
```

---

### 📦 Deployment and executable builds

Need a zero-dependency distribution for end users? Follow the
[deployment guide](DEPLOYMENT.md) to package the Flask UI into a standalone executable with
PyInstaller. The walkthrough covers environment setup, the new `scripts/build_executable.py` helper,
and validation steps to make sure the bundled app serves the UI correctly.

---

### 📂 Project layout

* `app.py` – Flask application, routes, and dev-server entry point.
* `bg_removal.py` – ONNX Runtime session management plus background removal helpers.
* `main.py` – CLI entry point for batch processing.
* `templates/` – Base template + background removal form.
* `tests/` – Pytest-based regression tests for the service utilities.

---

### 🧰 Credits

* **Model:** [U²-Net – Qin et al., Pattern Recognition 2020](https://github.com/xuebinqin/U-2-Net)
* **Background removal engine:** [U²-Net – Qin et al., Pattern Recognition 2020](https://github.com/xuebinqin/U-2-Net)
