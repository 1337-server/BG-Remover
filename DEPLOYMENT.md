# 🚀 Deployment and Executable Packaging Guide

This guide covers the recommended approach for turning the background remover web UI into a standalone
executable that can be shipped to users without requiring them to manage Python environments. The
process relies on [PyInstaller](https://pyinstaller.org/en/stable/) to freeze the Flask application and
its assets into a redistributable bundle.

## 1. Prerequisites

- Python 3.11 or 3.12 installed on the build machine.
- 64-bit operating system (Windows, macOS, or Linux).
- Sufficient disk space for the ONNX Runtime libraries (~1.5 GB once packaged).

> **Tip:** Build the executable on the platform you plan to distribute for. PyInstaller targets the
> current operating system and architecture.

## 2. Set up the build environment

1. Clone the repository and create a fresh virtual environment:

   ```bash
   git clone https://github.com/<your-account>/br-remover.git
   cd br-remover
   python -m venv .venv
   source .venv/bin/activate  # Windows PowerShell: .\.venv\Scripts\Activate.ps1
   ```

2. Install the pinned dependency set, which now includes PyInstaller:

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

## 3. Build the executable

Run the packaging helper to generate a distributable build:

```bash
python scripts/build_executable.py
```

The helper optimizes PyInstaller for this project by reusing cached build artefacts, bundling the
templates and static assets automatically, and enabling parallel compilation for faster builds. When
[UPX](https://upx.github.io/) is available it also compresses the output for smaller downloads.

### Build Commands

`scripts/build_executable.py` is a helper script for packaging the Background Remover GUI into a
standalone executable using PyInstaller. It wraps the recommended options so you can focus on choosing
the right entry point and output layout for your release.

```bash
python scripts/build_executable.py [OPTIONS]
```

| Flag | Description | Example |
|------|-------------|---------|
| `--name NAME` | Sets the executable name (defaults to `br-remover`). | `--name br-remover-gui` |
| `--entry-point PATH` | Points PyInstaller at the module to freeze. Use `bg_remover_gui.py` for the desktop GUI or `app.py` for the Flask server. | `--entry-point bg_remover_gui.py` |
| `--dist-dir PATH` | Directory that receives the finished build. Useful when writing to fast storage. | `--dist-dir /mnt/ssd/dist` |
| `--build-dir PATH` | Workspace used for intermediate files; can live on a RAM disk for faster I/O. | `--build-dir /mnt/ramdisk/build` |
| `--clean` | Removes existing build/dist folders before compiling (clears caches). | `--clean` |
| `--onefile` | Produces a single-file executable instead of a folder bundle. | `--onefile` |
| `--no-upx` | Skips UPX compression if antivirus tools raise false positives. | `--no-upx` |
| `--help` | Displays the full CLI usage information. | `--help` |

#### Example build scenarios

- **Basic fast cached build:**

  ```bash
  python scripts/build_executable.py --entry-point bg_remover_gui.py --name br-remover-gui
  ```

- **Clean one-file build:**

  ```bash
  python scripts/build_executable.py --entry-point bg_remover_gui.py --name br-remover-gui --onefile --clean
  ```

- **Build without UPX compression:**

  ```bash
  python scripts/build_executable.py --entry-point bg_remover_gui.py --name br-remover-gui --no-upx
  ```

- **Build using a RAM disk:**

  ```bash
  python scripts/build_executable.py --entry-point bg_remover_gui.py --name br-remover-gui --build-dir /mnt/ramdisk/build --dist-dir /mnt/ramdisk/dist
  ```

- **Multi-core optimized build (e.g., Ryzen 9 9950X):**

  ```bash
  python scripts/build_executable.py --entry-point bg_remover_gui.py --name br-remover-gui
  ```

  > PyInstaller automatically leverages all available CPU cores because the helper appends `--parallel`.

##### Performance Tips

- Reuse cached build artefacts for quick iterations; only pass `--clean` when dependencies change.
- Keep UPX compression enabled for smaller binaries unless antivirus software objects (`--no-upx`).
- Direct the `--build-dir` and `--dist-dir` to a RAM disk or ultra-fast SSD to reduce I/O bottlenecks.

## 4. Run and verify the bundle

After a successful build you will find the executable at:

- **Windows:** `dist\br-remover\br-remover.exe`
- **macOS/Linux:** `dist/br-remover/br-remover`

Launch the binary and verify the UI:

```bash
./dist/br-remover/br-remover --host 127.0.0.1 --port 5000
```

When the server reports it is running, open <http://127.0.0.1:5000/image/remove-bg> in a browser. Upload
a sample image to confirm background removal completes without errors and the UI responds correctly.

> **Note:** The first run downloads the necessary ONNX model weights to the user's profile directory.
> Subsequent launches reuse the cached files and start significantly faster.

## 5. Package the desktop GUI (optional)

Prefer a native-feeling desktop app instead of the Flask web server? Reuse the same helper script with a
different entry point:

```bash
python scripts/build_executable.py --entry-point bg_remover_gui.py --name br-remover-gui
```

The generated bundle contains the ttkbootstrap-powered interface with single-image and batch folder
workflows. At runtime it creates an `output/` subfolder inside whichever directory you process to keep
the original images untouched.

All command-line flags described above continue to work—for example pass `--onefile` to emit a
single-binary distribution or `--clean` when you need to start from a fresh workspace. Regardless of
the packaging mode, the GUI records uncaught exceptions to an `error.log` file that lives next to the
executable and surfaces a message box with the location so crashes are no longer silent.

After building, double-click the executable (or run it from a terminal) to open the GUI window directly
without starting a local web server.

## 6. Distribute to end users

Share the contents of `dist/br-remover/` (or the single binary when using `--onefile`) with your users.
Provide the startup command above and highlight that the application serves the UI via a local web
server on port 5000 by default.

For convenience, you can include a small launcher script or shortcut that runs the executable and opens
the browser automatically.

---

Happy shipping! 🎉
