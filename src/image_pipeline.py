"""Image + texture + text prompt -> SAM 3 mask, relative depth, and a textured composite.

The composite is a naive baseline: the texture is tiled inside the mask and
shaded by the original image's brightness. It does not yet use depth to warp
the texture onto the object's surface.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# Allow unsupported MPS operations to fall back to CPU on Apple Silicon.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoImageProcessor,
    DepthAnythingForDepthEstimation,
    Sam3Model,
    Sam3Processor,
)

SAM_MODEL = "facebook/sam3"
DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"


def choose_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def run_sam3(image: Image.Image, prompt: str, device: torch.device) -> tuple[np.ndarray, float]:
    """Use SAM 3's text prompt; return the highest-confidence instance mask and its score."""
    # Load the image processor explicitly. AutoProcessor can resolve the repo's
    # video processor, which routes the text prompt into image preprocessing.
    processor = Sam3Processor.from_pretrained(SAM_MODEL, token=True)
    model = Sam3Model.from_pretrained(SAM_MODEL, token=True).to(device).eval()
    inputs = processor(images=image, text=prompt, return_tensors="pt")
    inputs = {name: value.to(device) for name, value in inputs.items()}
    with torch.inference_mode():
        outputs = model(**inputs)
    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=0.5,
        mask_threshold=0.5,
        target_sizes=inputs["original_sizes"].tolist(),
    )[0]
    if len(results["masks"]) == 0:
        raise SystemExit(f"SAM 3 found no object matching the prompt {prompt!r}.")
    best = int(results["scores"].argmax().item())
    return results["masks"][best].cpu().numpy().astype(bool), float(results["scores"][best])


def run_depth(image: Image.Image, device: torch.device) -> np.ndarray:
    """Return a full-resolution relative-depth map (not metric distance)."""
    processor = AutoImageProcessor.from_pretrained(DEPTH_MODEL, use_fast=False)
    model = DepthAnythingForDepthEstimation.from_pretrained(DEPTH_MODEL).to(device).eval()
    inputs = processor(images=image, return_tensors="pt")
    inputs = {name: value.to(device) for name, value in inputs.items()}
    with torch.inference_mode():
        prediction = model(**inputs).predicted_depth
    full_size = torch.nn.functional.interpolate(
        prediction.unsqueeze(1), size=(image.height, image.width),
        mode="bicubic", align_corners=False,
    ).squeeze().float()
    return full_size.cpu().numpy()


def apply_texture(image: Image.Image, texture: Image.Image, mask: np.ndarray) -> np.ndarray:
    """Tile the texture inside the mask, shaded by the original image's brightness."""
    rgb = np.asarray(image, dtype=np.float32) / 255.0
    tex = np.asarray(texture, dtype=np.float32) / 255.0
    reps = (-(-rgb.shape[0] // tex.shape[0]), -(-rgb.shape[1] // tex.shape[1]), 1)
    tiled = np.tile(tex, reps)[: rgb.shape[0], : rgb.shape[1]]

    luminance = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    shading = luminance / max(float(luminance[mask].mean()), 1e-6)
    out = rgb.copy()
    out[mask] = np.clip(tiled[mask] * shading[mask, None], 0, 1)
    return (out * 255).astype(np.uint8)


def save_outputs(
    image: Image.Image, prompt: str, mask: np.ndarray, depth: np.ndarray,
    textured: np.ndarray, output_dir: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    rgb = np.asarray(image)
    mask_u8 = mask.astype(np.uint8) * 255
    Image.fromarray(mask_u8).save(output_dir / "sam_mask.png")
    Image.fromarray(textured).save(output_dir / "textured.png")

    low, high = np.percentile(depth, (2, 98))
    depth_vis = np.clip((depth - low) / max(float(high - low), 1e-6), 0, 1)
    plt.imsave(output_dir / "depth_relative.png", depth_vis, cmap="inferno")

    panels = [rgb, mask_u8, depth_vis, textured]
    titles = [f"Input · {prompt!r}", "SAM 3 text mask", "Relative depth", "Textured (naive)"]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5), constrained_layout=True)
    for ax, panel, title in zip(axes, panels, titles):
        ax.imshow(panel, cmap="inferno" if panel is depth_vis else "gray" if panel.ndim == 2 else None)
        ax.set_title(title)
        ax.axis("off")
    fig.savefig(output_dir / "pipeline_visualization.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path, help="Input still image")
    parser.add_argument("--texture", required=True, type=Path, help="Texture image to apply")
    parser.add_argument("--prompt", required=True, help="Text description for SAM 3, e.g. 'orange tabby cat'")
    parser.add_argument("--output", type=Path, default=Path("outputs/demo_run"))
    args = parser.parse_args()

    image = Image.open(args.image).convert("RGB")
    texture = Image.open(args.texture).convert("RGB")
    device = choose_device()
    print(f"Image: {image.width}x{image.height}; texture: {texture.width}x{texture.height}; device: {device}")

    print(f"Running SAM 3 with text prompt: {args.prompt!r}")
    try:
        mask, score = run_sam3(image, args.prompt, device)
    except OSError as exc:
        raise SystemExit(
            f"Could not load SAM 3 ({exc}). Make sure you accepted the model terms at "
            "https://huggingface.co/facebook/sam3 and are logged in to Hugging Face."
        ) from exc

    print("Running Depth Anything V2...")
    depth = run_depth(image, device)

    textured = apply_texture(image, texture, mask)
    save_outputs(image, args.prompt, mask, depth, textured, args.output)
    (args.output / "run.json").write_text(json.dumps({
        "image": str(args.image), "texture": str(args.texture), "prompt": args.prompt,
        "sam_score": round(score, 4), "mask_pixels": int(mask.sum()),
        "sam_model": SAM_MODEL, "depth_model": DEPTH_MODEL, "device": str(device),
        "depth_note": "Relative monocular depth; not metric distance.",
        "texture_note": "Naive tiled texture with luminance shading; no depth-based warp yet.",
    }, indent=2) + "\n")
    print(f"Saved mask, depth, textured composite, and visualization to {args.output}")


if __name__ == "__main__":
    main()
