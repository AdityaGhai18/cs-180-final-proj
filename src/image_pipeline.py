"""Segment a text-described object with SAM 3, then estimate relative depth."""

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
from transformers import AutoImageProcessor, DepthAnythingForDepthEstimation

SAM_MODEL = "facebook/sam3"
DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"


def choose_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def run_sam3(image: Image.Image, prompt: str, device: torch.device) -> np.ndarray:
    """Use SAM 3's native text prompt to get the highest-confidence instance mask."""
    from transformers import Sam3Model, Sam3Processor

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
        raise RuntimeError(f"SAM 3 found no object matching the prompt {prompt!r}.")
    best = int(results["scores"].argmax().item())
    return results["masks"][best].cpu().numpy().astype(bool)


def run_box_sam(image: Image.Image, box: list[int], device: torch.device) -> np.ndarray:
    """Run the original public SAM ViT-B using a pixel-coordinate box prompt."""
    from transformers import SamModel, SamProcessor

    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    model = SamModel.from_pretrained("facebook/sam-vit-base").to(device).eval()
    inputs = processor(images=image, input_boxes=[[box]], return_tensors="pt")
    model_inputs = {
        name: inputs[name].to(device=device, dtype=torch.float32)
        for name in ("pixel_values", "input_boxes")
        if name in inputs
    }
    with torch.inference_mode():
        outputs = model(**model_inputs)
    masks = processor.post_process_masks(
        outputs.pred_masks,
        inputs["original_sizes"],
        inputs["reshaped_input_sizes"],
        binarize=True,
    )[0][0]
    best = int(outputs.iou_scores[0, 0].argmax().item())
    return masks[best].cpu().numpy().astype(bool)


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


def save_outputs(
    image: Image.Image, prompt: str, mask: np.ndarray, depth: np.ndarray, output_dir: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    rgb = np.asarray(image.convert("RGB"))
    mask_u8 = mask.astype(np.uint8) * 255
    Image.fromarray(mask_u8).save(output_dir / "sam_mask.png")

    low, high = np.percentile(depth, (2, 98))
    depth_vis = np.clip((depth - low) / max(float(high - low), 1e-6), 0, 1)
    plt.imsave(output_dir / "depth_relative.png", depth_vis, cmap="inferno")

    overlay = rgb.copy()
    green = np.zeros_like(rgb)
    green[..., 1] = 220
    overlay[mask] = (0.48 * rgb[mask] + 0.52 * green[mask]).astype(np.uint8)
    panels = [rgb, mask_u8, depth_vis, overlay]
    titles = [f"Input · {prompt}", "SAM 3 text mask", "Relative depth", "Mask overlay"]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5), constrained_layout=True)
    for ax, panel, title in zip(axes, panels, titles):
        ax.imshow(panel, cmap="gray" if panel.ndim == 2 else None)
        ax.set_title(title)
        ax.axis("off")
    fig.savefig(output_dir / "pipeline_visualization.png", dpi=160)
    plt.close(fig)

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path, help="Input still image")
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument("--prompt", help="Text description for SAM 3, e.g. 'orange tabby cat'")
    prompt_group.add_argument(
        "--box", nargs=4, type=int, metavar=("X1", "Y1", "X2", "Y2"),
        help="SAM ViT-B box prompt in image pixel coordinates",
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/demo_run"))
    args = parser.parse_args()

    image = Image.open(args.image).convert("RGB")
    device = choose_device()
    print(f"Image: {image.width}x{image.height}; device: {device}")
    if args.prompt is not None:
        segmentation_model = SAM_MODEL
        input_label = f"text prompt: {args.prompt}"
        print(f"Running SAM 3 with text prompt: {args.prompt!r}")
    else:
        x1, y1, x2, y2 = args.box
        if not (0 <= x1 < x2 <= image.width and 0 <= y1 < y2 <= image.height):
            parser.error(f"--box must be within the image bounds (0..{image.width}, 0..{image.height})")
        segmentation_model = "facebook/sam-vit-base"
        input_label = f"box prompt: {args.box}"
        print(f"Running SAM ViT-B with box prompt: {args.box}")
    try:
        if args.prompt is not None:
            mask = run_sam3(image, args.prompt, device)
        else:
            mask = run_box_sam(image, args.box, device)
    except OSError as exc:
        detail = f"\nUnderlying download/processor error: {exc.__cause__}" if exc.__cause__ else ""
        raise SystemExit(f"Could not load segmentation model files: {exc}{detail}") from exc

    print("Running Depth Anything V2...")
    try:
        depth = run_depth(image, device)
    except OSError as exc:
        raise SystemExit(f"SAM 3 mask is computed, but depth model files could not load: {exc}") from exc

    save_outputs(image, input_label, mask, depth, args.output)
    (args.output / "run.json").write_text(json.dumps({
        "input": str(args.image), "prompt": args.prompt, "box_xyxy": args.box,
        "sam_model": segmentation_model, "depth_model": DEPTH_MODEL,
        "device": str(device),
        "depth_note": "Relative monocular depth; not metric distance.",
    }, indent=2) + "\n")
    print(f"Saved mask, relative depth, and visualization to {args.output}")


if __name__ == "__main__":
    main()
