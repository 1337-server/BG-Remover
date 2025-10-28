"""Command-line background remover powered by U\u00b2-Net."""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import requests
import torch
from PIL import Image

from u2net import U2NET, U2NETP

# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------
MODEL_URLS = {
    "full": [
        "https://huggingface.co/datasets/mesolitica/u2net/resolve/main/u2net.pth?download=1",
        "https://github.com/xuebinqin/U-2-Net/releases/download/v1.0/u2net.pth",
    ],
    "lite": [
        "https://huggingface.co/datasets/mesolitica/u2net/resolve/main/u2netp.pth?download=1",
        "https://github.com/xuebinqin/U-2-Net/releases/download/v1.0/u2netp.pth",
    ],
}
MODEL_FILENAMES = {
    "full": "u2net.pth",
    "lite": "u2netp.pth",
}
MODEL_MIN_BYTES = {
    "full": 50_000_000,  # ~= 50 MB, ensures a full download
    "lite": 2_000_000,  # ~= 2 MB
}
CHUNK_SIZE = 2 ** 20  # 1 MiB

SUPPORTED_EXTENSIONS = {
    ext.lower()
    for ext, format_name in Image.registered_extensions().items()
    if format_name
}
DEFAULT_OUTPUT_DIR_NAME = "output"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Remove image backgrounds using U\u00b2-Net.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        "-i",
        help="Input image file or directory. Defaults to the current working directory.",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Directory where processed images are stored.",
    )
    parser.add_argument(
        "--model",
        "-m",
        choices=sorted(MODEL_URLS.keys()),
        default="full",
        help="Select the model variant to use.",
    )
    return parser.parse_args()


def models_dir() -> Path:
    """Return the directory where model weights are stored."""
    base_dir = Path(__file__).resolve().parent
    return base_dir / "models"


def safe_torch_load(path: Path) -> Dict[str, torch.Tensor]:
    """Load a PyTorch state dictionary with broad version compatibility."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)  # type: ignore[arg-type]
    except TypeError:
        return torch.load(path, map_location="cpu")


def _remove_file_safely(path: Path) -> None:
    """Best-effort removal that copes with transient Windows locks."""
    try:
        path.unlink()
    except FileNotFoundError:
        return
    except PermissionError:
        # On Windows the file handle might still be held open briefly.
        import time

        time.sleep(1)
        try:
            path.unlink()
        except Exception:
            pass


def _download_to_path(url: str, destination: Path) -> None:
    """Stream a URL to a destination file using a temporary download."""
    temp_path = destination.with_suffix(destination.suffix + ".part")
    _remove_file_safely(temp_path)

    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        downloaded = 0

        with temp_path.open("wb") as file_obj:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                if not chunk:
                    continue
                file_obj.write(chunk)
                downloaded += len(chunk)
                if total:
                    percent = int(downloaded * 100 / total)
                    progress_bar = f"{percent:3d}%".rjust(4)
                    print(
                        f"\r   ↳ {progress_bar} ({downloaded / 1_048_576:.1f} MiB)",
                        end="",
                        flush=True,
                    )
        if total:
            print()

    temp_path.replace(destination)


def _looks_like_html(path: Path) -> bool:
    """Return True when a file starts with HTML characters (common download error)."""
    try:
        with path.open("rb") as file_obj:
            prefix = file_obj.read(2)
    except OSError:
        return True
    return prefix == b"<!"


def ensure_model_file(model_key: str) -> Path:
    """Ensure the selected model is present on disk and return its path."""
    models_path = models_dir()
    models_path.mkdir(parents=True, exist_ok=True)
    destination = models_path / MODEL_FILENAMES[model_key]

    if destination.exists() and destination.stat().st_size >= MODEL_MIN_BYTES[model_key]:
        if not _looks_like_html(destination):
            print(f"✅ Using cached model: {destination}")
            return destination
        print("⚠️ Cached model appears to be HTML. Re-downloading.")
        _remove_file_safely(destination)

    for url in MODEL_URLS[model_key]:
        print(f"⬇️  Downloading {model_key.upper()} model from {url}")
        try:
            _download_to_path(url, destination)
        except requests.RequestException as exc:
            print(f"⚠️  Download failed: {exc}")
            _remove_file_safely(destination)
            continue

        if destination.stat().st_size < MODEL_MIN_BYTES[model_key] or _looks_like_html(destination):
            print("⚠️  Downloaded file is invalid. Trying next mirror.")
            _remove_file_safely(destination)
            continue

        print(f"✅ Saved model to {destination}")
        return destination

    raise RuntimeError(
        "Unable to download the requested model. Please check your internet connection and try again."
    )


def load_model(model_key: str, device: torch.device) -> torch.nn.Module:
    """Download (if necessary) and load the selected U\u00b2-Net variant."""
    path = ensure_model_file(model_key)

    try:
        state_dict = safe_torch_load(path)
    except Exception as exc:  # pragma: no cover - highly unlikely
        print(f"⚠️  Failed to read cached model ({exc}). Re-downloading.")
        _remove_file_safely(path)
        path = ensure_model_file(model_key)
        state_dict = safe_torch_load(path)

    net: torch.nn.Module
    if model_key == "lite":
        net = U2NETP(3, 1)
    else:
        net = U2NET(3, 1)

    incompatible = net.load_state_dict(state_dict, strict=False)
    if incompatible.missing_keys:
        missing_preview = ", ".join(incompatible.missing_keys[:5])
        print(f"⚠️  Missing keys detected: {missing_preview}")
    if incompatible.unexpected_keys:
        unexpected_preview = ", ".join(incompatible.unexpected_keys[:5])
        print(f"⚠️  Unexpected keys detected: {unexpected_preview}")

    net.to(device)
    net.eval()
    return net


def _normalize_image(array: np.ndarray) -> np.ndarray:
    """Normalize an image array using ImageNet statistics."""
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    return (array - mean) / std


def process_image(
    image_path: Path,
    output_dir: Path,
    model: torch.nn.Module,
    device: torch.device,
) -> None:
    """Remove the background from a single image and save it as PNG."""
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as exc:  # pragma: no cover - depends on input data
        print(f"⚠️  Skipping {image_path.name}: {exc}")
        return

    original_size = image.size
    image_resized = image.resize((320, 320), Image.BILINEAR)
    image_np = np.asarray(image_resized, dtype=np.float32) / 255.0
    image_np = _normalize_image(image_np)
    tensor = torch.from_numpy(image_np.transpose(2, 0, 1)).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        prediction = outputs[0] if isinstance(outputs, (tuple, list)) else outputs

    prediction = prediction.squeeze().detach().cpu().numpy()
    max_val = prediction.max()
    min_val = prediction.min()
    if math.isclose(max_val, min_val):  # pragma: no cover - extremely rare
        alpha = np.zeros_like(prediction, dtype=np.uint8)
    else:
        alpha = ((prediction - min_val) / (max_val - min_val) * 255).astype(np.uint8)

    mask = Image.fromarray(alpha).resize(original_size, Image.BILINEAR)

    rgba = image.convert("RGBA")
    rgba.putalpha(mask)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{image_path.stem}.png"
    rgba.save(output_path)
    print(f"✅ Saved: {output_path}")


def _iter_image_files(path: Path) -> Iterable[Path]:
    """Yield supported image files from a path (recursively for directories)."""
    if path.is_dir():
        for candidate in sorted(path.rglob("*")):
            if candidate.is_file() and candidate.suffix.lower() in SUPPORTED_EXTENSIONS:
                yield candidate
    elif path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
        yield path
    else:
        raise FileNotFoundError(f"No supported images found at: {path}")


def main() -> None:
    """Entry point for the command-line tool."""
    args = parse_args()

    input_path = Path(args.input).expanduser() if args.input else Path.cwd()
    output_dir = Path(args.output).expanduser() if args.output else Path.cwd() / DEFAULT_OUTPUT_DIR_NAME

    if not input_path.exists():
        raise SystemExit(f"❌ Input path does not exist: {input_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"📂 Input: {input_path}")
    print(f"💾 Output: {output_dir}")
    print(f"⚙️  Model: {args.model.upper()}")
    print(f"💻 Device: {'GPU' if device.type == 'cuda' else 'CPU'}")

    model = load_model(args.model, device)

    try:
        image_paths = list(_iter_image_files(input_path))
    except FileNotFoundError as exc:
        raise SystemExit(str(exc))

    if not image_paths:
        raise SystemExit("❌ No supported image files were found.")

    total = len(image_paths)
    print(f"\n▶ Processing {total} file(s)...")
    for index, image_path in enumerate(image_paths, start=1):
        print(f"🖼️  [{index}/{total}] {image_path.name}")
        process_image(image_path, output_dir, model, device)

    print("\n🏁 Done.")


if __name__ == "__main__":
    main()
