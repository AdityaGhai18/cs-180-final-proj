"""SAM 3 helpers for the real-time tracking experiment (kept separate from src/image_pipeline.py)."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

SAM_MODEL = "facebook/sam3"


def choose_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_sam3(device: torch.device):
    """Load SAM 3 (image model) once; returns (model, processor). Needs approved HF access."""
    from transformers import Sam3Model, Sam3Processor

    # Load the image processor explicitly; AutoProcessor can resolve the video processor.
    processor = Sam3Processor.from_pretrained(SAM_MODEL, token=True)
    model = Sam3Model.from_pretrained(SAM_MODEL, token=True).to(device).eval()
    return model, processor


def sam3_instances(
    model, processor, image: Image.Image, prompt: str, threshold: float = 0.5
) -> tuple[np.ndarray, np.ndarray]:
    """All instances matching the text prompt: (N x H x W bool masks, N scores)."""
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    inputs = processor(images=image, text=prompt, return_tensors="pt")
    inputs = {name: value.to(device, dtype=dtype) if value.is_floating_point() else value.to(device)
              for name, value in inputs.items()}
    with torch.inference_mode():
        outputs = model(**inputs)
    results = processor.post_process_instance_segmentation(
        outputs, threshold=threshold, mask_threshold=0.5, target_sizes=inputs["original_sizes"].tolist(),
    )[0]
    masks = results["masks"].cpu().numpy().astype(bool).reshape(-1, image.height, image.width)
    return masks, results["scores"].float().cpu().numpy()
