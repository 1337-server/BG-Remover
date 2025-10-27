import os
import sys
import argparse
import numpy as np
import torch
import requests
from PIL import Image
from u2net_model import U2NET, U2NETP

# Model URLs (official sources)
MODEL_URLS = {
    # Full model (~176 MB)
    "full": "https://drive.google.com/uc?id=1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ&export=download",
    # Lite model (~4.7 MB)
    "lite": "https://drive.google.com/uc?id=1rbSTGKAE-MTxBYHd-51l2hMOQPT_7EPy&export=download",
}
MODEL_PATHS = {
    "full": "models/u2net.pth",
    "lite": "models/u2netp.pth",
}

def download_model(model_key):
    """Download the selected U2NET model from Google Drive if missing."""
    import re

    url = MODEL_URLS[model_key]
    out_path = MODEL_PATHS[model_key]

    if os.path.exists(out_path) and os.path.getsize(out_path) > 1e6:
        print(f"[✔] Found model '{out_path}', skipping download.")
        return out_path

    print(f"[↓] Downloading {model_key.upper()} model...")

    # Handle Google Drive large file download
    session = requests.Session()
    response = session.get(url, stream=True)
    confirm_token = None

    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            confirm_token = value
            break

    if confirm_token:
        params = {"confirm": confirm_token}
        response = session.get(url, params=params, stream=True)

    total = int(response.headers.get("content-length", 0))
    # Verify it’s a real PyTorch file (not HTML)
    with open(out_path, "rb") as f:
        first_bytes = f.read(2)

    if first_bytes == b'<!':
        try:
            os.remove(out_path)
        except PermissionError:
            # File might still be locked briefly on Windows
            import time
            time.sleep(1)
            try:
                os.remove(out_path)
            except Exception:
                pass
        sys.exit("[✖] Error: downloaded file is HTML, not a model. Try running again later.")

    # Verify it’s a real PyTorch file (not HTML)
    with open(out_path, "rb") as f:
        if f.read(2) == b'<!':
            os.remove(out_path)
            sys.exit("[✖] Error: downloaded file is HTML, not a model. Try running again later.")

    return out_path

def load_model(model_key, device):
    from u2net import U2NET, U2NETP

    if model_key == "lite":
        net = U2NETP(3, 1)
    else:
        net = U2NET(3, 1)

    path = download_model(model_key)
    net.load_state_dict(torch.load(path, map_location=device))
    net.to(device).eval()
    return net

def process_image(img_path, output_dir, model, device):
    """Remove background from one image and save PNG."""
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception as e:
        print(f"[!] Skipping '{img_path}': {e}", file=sys.stderr)
        return

    orig_size = img.size
    img_resized = img.resize((320, 320))
    img_np = np.array(img_resized).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_np = (img_np - mean) / std
    img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).unsqueeze(0).to(device)

    with torch.no_grad():
        d_outputs = model(img_tensor)
        pred = d_outputs[0] if isinstance(d_outputs, (tuple, list)) else d_outputs

    pred = pred.squeeze().cpu().numpy()
    ma, mi = pred.max(), pred.min()
    mask = ((pred - mi) / (ma - mi + 1e-8) * 255).astype(np.uint8)
    mask = Image.fromarray(mask).resize(orig_size, Image.BILINEAR)

    rgba = img.copy().convert("RGBA")
    rgba.putalpha(mask)

    base = os.path.splitext(os.path.basename(img_path))[0]
    out_path = os.path.join(output_dir, f"{base}.png")
    rgba.save(out_path)
    print(f"[✔] Saved: {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Remove image backgrounds using U²-Net.")
    parser.add_argument("--input", "-i", help="Input file or directory (default: current directory)")
    parser.add_argument("--output", "-o", help="Output directory (default: ./output)")
    parser.add_argument("--model", "-m", choices=["full", "lite"], default="full",
                        help="Choose 'full' (better quality) or 'lite' (faster, smaller). Default=full")
    args = parser.parse_args()

    # Default behavior if not provided
    cwd = os.getcwd()
    input_path = args.input or cwd
    output_dir = args.output or os.path.join(cwd, "output")

    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[📂] Input: {input_path}")
    print(f"[💾] Output: {output_dir}")
    print(f"[⚙️ ] Model: {args.model.upper()}")
    print(f"[💻] Device: {'GPU' if device.type == 'cuda' else 'CPU'}")

    model = load_model(args.model, device)

    if os.path.isdir(input_path):
        files = [os.path.join(input_path, f) for f in os.listdir(input_path)]
    else:
        files = [input_path]

    print(f"\n[▶] Processing {len(files)} file(s)...")
    for path in files:
        if os.path.isfile(path):
            process_image(path, output_dir, model, device)
    print("\n[🏁] Done.")

if __name__ == "__main__":
    main()
