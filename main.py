import os
import sys
import argparse
import numpy as np
import torch
import requests
from PIL import Image

# URL for U^2-Net pre-trained weights (Google Drive direct link)
MODEL_URL = "https://drive.google.com/uc?export=download&id=1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ"
MODEL_PATH = "u2net.pth"

def download_model(url=MODEL_URL, output_path=MODEL_PATH):
    """Download the U^2-Net model weights if not present."""
    try:
        print(f"Downloading U^2-Net model weights to '{output_path}'...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        # Google Drive links may require confirmation for large files
        # We handle the download in chunks:
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print("Model downloaded successfully.")
    except Exception as e:
        sys.exit(f"Error downloading model: {e}")

def load_u2net_model(device):
    """Load U^2-Net model (download weights if necessary). Returns the model."""
    # Define U^2-Net architecture (nested U-structure) in PyTorch
    # -- Omitted: Detailed architecture definition for brevity --
    # We assume classes U2NET and U2NETP are defined (from the official repo or included).
    # For example, one could import U2NET from a module or define it here.
    from u2net_model import U2NET  # This assumes a module or prior definition of the model.

    # Download weights if not present
    if not os.path.exists(MODEL_PATH):
        download_model()

    # Initialize model and load weights
    model = U2NET(in_ch=3, out_ch=1)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    except Exception as e:
        sys.exit(f"Failed to load U^2-Net model weights: {e}")
    model.to(device)
    model.eval()
    return model

def process_image(img_path, output_dir, model, device):
    """Process a single image: remove background and save PNG with transparency."""
    try:
        # Load image with Pillow
        img = Image.open(img_path).convert("RGB")
    except Exception as e:
        print(f"Skipping '{img_path}': {e}", file=sys.stderr)
        return  # skip this file

    # Prepare image for model – convert to tensor and normalize
    img_np = np.array(img).astype(np.float32)  # shape (H, W, 3)
    # Optional: Resize to 320x320 for model input (as recommended):contentReference[oaicite:7]{index=7}
    orig_size = img_np.shape[:2]  # original (H, W)
    input_size = (320, 320)
    if orig_size != input_size:
        # resize image numpy array to 320x320
        img_resized = Image.fromarray(img_np.astype('uint8')).resize(input_size, Image.BILINEAR)
        img_np = np.array(img_resized).astype(np.float32)
    # Normalize the image (scale 0-1 and apply ImageNet mean/std)
    img_np /= 255.0
    # ImageNet mean and std for RGB
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_np = (img_np - mean) / std
    # Transpose to CHW format for PyTorch and add batch dimension
    img_tensor = torch.from_numpy(img_np.transpose(2,0,1)).unsqueeze(0).to(device)

    # Run inference
    with torch.no_grad():
        outputs = model(img_tensor)            # forward pass
        # The model returns a tuple: (d0, d1, ..., d6). Use d0 (fused output):contentReference[oaicite:8]{index=8}.
        if isinstance(outputs, tuple) or isinstance(outputs, list):
            pred_mask = outputs[0]
        else:
            pred_mask = outputs  # in case model returns single tensor
    # pred_mask shape: [1,1,H,W]
    pred_mask = pred_mask.squeeze().cpu().numpy()  # HxW, probability map

    # Normalize mask to [0,255]
    # Apply min-max normalization:contentReference[oaicite:9]{index=9}
    ma, mi = pred_mask.max(), pred_mask.min()
    mask_norm = ((pred_mask - mi) / (ma - mi + 1e-8)) * 255  # scale to [0,255]
    mask_norm = mask_norm.astype(np.uint8)

    # If we resized for input, resize mask back to original image size
    if orig_size != input_size:
        mask_norm_img = Image.fromarray(mask_norm, mode='L').resize((orig_size[1], orig_size[0]), Image.BILINEAR)
        mask_norm = np.array(mask_norm_img)

    # Combine original image with mask to create RGBA output
    rgba = img.copy().convert("RGBA")        # get original image in RGBA
    alpha = Image.fromarray(mask_norm, mode='L')  # create alpha Image from mask
    rgba.putalpha(alpha)                     # put alpha channel:contentReference[oaicite:10]{index=10}
    # Determine output filename (same base name, .png extension)
    file_name = os.path.basename(img_path)
    base, _ = os.path.splitext(file_name)
    out_path = os.path.join(output_dir, base + ".png")
    try:
        rgba.save(out_path)
        print(f"Saved output: {out_path}")
    except Exception as e:
        print(f"Failed to save '{out_path}': {e}", file=sys.stderr)

def main():
    parser = argparse.ArgumentParser(description="Remove image background using U^2-Net.")
    parser.add_argument("--input", "-i", required=True, help="Input image file or directory of images")
    parser.add_argument("--output", "-o", required=True, help="Output directory for processed images")
    args = parser.parse_args()

    input_path = args.input
    output_dir = args.output
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Set device (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the U^2-Net model (download if necessary)
    model = load_u2net_model(device)

    # Process single image or all images in a directory
    if os.path.isdir(input_path):
        # Iterate through files in directory
        for filename in os.listdir(input_path):
            img_file = os.path.join(input_path, filename)
            process_image(img_file, output_dir, model, device)
    else:
        # Single file
        process_image(input_path, output_dir, model, device)

if __name__ == "__main__":
    main()
