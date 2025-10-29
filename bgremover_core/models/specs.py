"""Model specifications used by the background removal pipeline."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Descriptor describing a supported ONNX segmentation model."""

    key: str
    input_size: tuple[int, int]
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    normalisation_scale: float = 255.0
    checksum_md5: str | None = None
    url: str | None = None
    huggingface_repo: str | None = None
    huggingface_filename: str | None = None
    huggingface_revision: str = "main"
    local_filename: str | None = None


MODEL_SPECS: Mapping[str, ModelSpec] = {
    "isnet-general-use": ModelSpec(
        key="isnet-general-use",
        url="https://github.com/danielgatis/rembg/releases/download/v0.0.0/isnet-general-use.onnx",
        input_size=(1024, 1024),
        mean=(0.5, 0.5, 0.5),
        std=(1.0, 1.0, 1.0),
        checksum_md5="fc16ebd8b0c10d971d3513d564d01e29",
    ),
    "u2net": ModelSpec(
        key="u2net",
        url="https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx",
        input_size=(320, 320),
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        checksum_md5="60024c5c889badc19c04ad937298a77b",
    ),
    "u2net_human_seg": ModelSpec(
        key="u2net_human_seg",
        url="https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net_human_seg.onnx",
        input_size=(320, 320),
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        checksum_md5="c09ddc2e0104f800e3e1bb4652583d1f",
    ),
    "isnet-anime": ModelSpec(
        key="isnet-anime",
        input_size=(1024, 1024),
        mean=(0.485, 0.456, 0.406),
        std=(1.0, 1.0, 1.0),
        checksum_md5="6f184e756bb3bd901c8849220a83e38e",
        url="https://github.com/danielgatis/rembg/releases/download/v0.0.0/isnet-anime.onnx",
    ),
    "briaai/RMBG-2.0": ModelSpec(
        key="briaai/RMBG-2.0",
        input_size=(1024, 1024),
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        huggingface_repo="briaai/RMBG-2.0",
        huggingface_filename="RMBG-2.0.onnx",
    ),
    "matting-by-generation": ModelSpec(
        key="matting-by-generation",
        input_size=(1024, 1024),
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        huggingface_repo="briaai/RMBG-1.4",
        huggingface_filename="onnx/model.onnx",
    ),
    "sam_segmentation_model": ModelSpec(
        key="sam_segmentation_model",
        input_size=(1024, 1024),
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        huggingface_repo="briaai/RMBG-2.0",
        huggingface_filename="RMBG-2.0.onnx",
        local_filename="briaai/RMBG-2.0.onnx",
    ),
    "sam_vit_b_01ec64_encoder": ModelSpec(
        key="sam_vit_b_01ec64_encoder",
        input_size=(1024, 1024),
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        huggingface_repo="microsoft/segment-anything-model-webnn",
        huggingface_filename="sam_vit_b_01ec64.encoder-fp16.onnx",
    ),
    "sam_vit_b_01ec64_decoder": ModelSpec(
        key="sam_vit_b_01ec64_decoder",
        input_size=(1024, 1024),
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        huggingface_repo="microsoft/segment-anything-model-webnn",
        huggingface_filename="sam_vit_b_01ec64.decoder-fp16.onnx",
    ),
}

__all__ = ["MODEL_SPECS", "ModelSpec"]
