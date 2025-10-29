"""Helpers for producing a standalone executable for the background remover UI (optimized for multi-core builds)."""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from importlib import util
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DIST_DIR = PROJECT_ROOT / "dist"
DEFAULT_BUILD_DIR = PROJECT_ROOT / "build"


def _build_pyinstaller_command(
    *,
    entry_point: Path,
    name: str,
    dist_dir: Path,
    build_dir: Path,
    onefile: bool,
    use_upx: bool = True,
) -> list[str]:
    """Return the PyInstaller command for bundling ``entry_point`` with optimized settings."""

    add_data_sep = ";" if os.name == "nt" else ":"
    templates_dir = PROJECT_ROOT / "templates"
    static_dir = PROJECT_ROOT / "static"

    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--name", name,
        "--distpath", str(dist_dir),
        "--workpath", str(build_dir / "pyinstaller"),
        "--specpath", str(build_dir / "pyinstaller"),
        "--add-data", f"{templates_dir}{add_data_sep}templates",
        "--add-data", f"{static_dir}{add_data_sep}static",
        "--parallel",  # ✅ Use all CPU cores (PyInstaller ≥6.4)
    ]

    # Try to use UPX if available
    if use_upx:
        upx_path = shutil.which("upx")
        if upx_path:
            command.extend(["--upx-dir", str(Path(upx_path).parent)])
        else:
            print("[!] UPX not found — skipping compression. Install it for smaller, faster builds.")

    if onefile:
        command.append("--onefile")

    command.append(str(entry_point))
    return command


def build_executable(
    *,
    entry_point: Path,
    name: str,
    dist_dir: Path = DEFAULT_DIST_DIR,
    build_dir: Path = DEFAULT_BUILD_DIR,
    clean: bool = False,
    onefile: bool = False,
    use_upx: bool = True,
) -> Path:
    """Build the executable using PyInstaller and return its path."""

    if util.find_spec("PyInstaller") is None:
        raise ModuleNotFoundError(
            "PyInstaller is required. Install it with `pip install pyinstaller` before packaging.",
        )

    # ⚙️ Clean distribution directory, but reuse build cache for speed unless explicitly cleaned
    if clean:
        print("[*] Cleaning build and dist directories...")
        shutil.rmtree(dist_dir, ignore_errors=True)
        shutil.rmtree(build_dir, ignore_errors=True)
    else:
        print("[*] Reusing existing build cache for faster compilation...")

    dist_dir.mkdir(parents=True, exist_ok=True)
    build_dir.mkdir(parents=True, exist_ok=True)

    command = _build_pyinstaller_command(
        entry_point=entry_point,
        name=name,
        dist_dir=dist_dir,
        build_dir=build_dir,
        onefile=onefile,
        use_upx=use_upx,
    )

    print("[*] Running PyInstaller build command:")
    print("    " + " ".join(command))

    subprocess.run(command, check=True)

    # Determine the final executable path
    executable_path = dist_dir / name
    if not onefile:
        executable_path = executable_path / name

    if os.name == "nt":
        executable_path = executable_path.with_suffix(".exe")

    print(f"[*] Build completed successfully: {executable_path}")
    return executable_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Return parsed command line arguments for the executable builder."""

    parser = argparse.ArgumentParser(
        description="Build a standalone executable of the Flask background remover app (optimized).",
    )
    parser.add_argument(
        "--name",
        default="br-remover",
        help="Name for the generated executable.",
    )
    parser.add_argument(
        "--entry-point",
        default=str(PROJECT_ROOT / "app.py"),
        help="Path to the Python entry point to bundle.",
    )
    parser.add_argument(
        "--dist-dir",
        default=str(DEFAULT_DIST_DIR),
        help="Directory to store the built executable.",
    )
    parser.add_argument(
        "--build-dir",
        default=str(DEFAULT_BUILD_DIR),
        help="Temporary build directory (can be a RAM disk for faster I/O).",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Fully clean existing build artefacts before packaging (slower).",
    )
    parser.add_argument(
        "--onefile",
        action="store_true",
        help="Produce a single-file executable instead of a folder.",
    )
    parser.add_argument(
        "--no-upx",
        action="store_true",
        help="Disable UPX compression (use if you encounter antivirus false positives).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> Path:
    """Command line entry point for packaging the background remover UI."""

    args = parse_args(argv)
    entry_point = Path(args.entry_point).resolve()
    if not entry_point.exists():
        raise FileNotFoundError(f"Entry point {entry_point} does not exist")

    try:
        executable_path = build_executable(
            entry_point=entry_point,
            name=args.name,
            dist_dir=Path(args.dist_dir).resolve(),
            build_dir=Path(args.build_dir).resolve(),
            clean=args.clean,
            onefile=args.onefile,
            use_upx=not args.no_upx,
        )
    except ModuleNotFoundError as exc:
        raise SystemExit(str(exc)) from exc

    print(f"\n✅ Executable created at: {executable_path}")
    return executable_path


if __name__ == "__main__":
    main()
