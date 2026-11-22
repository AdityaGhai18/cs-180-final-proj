"""Pack pipeline outputs into explorer/data.json for the interactive explorer page.

Usage: python src/build_explorer.py outputs/tabby_cat outputs/running_shoe ...
Each run folder must contain run.json, sam_mask.png, depth.npy, depth_relative.png
and textured.png (written by src/image_pipeline.py).
"""

from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PANEL_PX = 560      # longest side of the 2D panel images
OBJECT_STRIDE = 4   # keep every Nth pixel (in x and y) inside the mask
BACKGROUND_STRIDE = 10


def jpeg_data_uri(image: Image.Image) -> str:
    image = image.convert("RGB")
    image.thumbnail((PANEL_PX, PANEL_PX), Image.LANCZOS)
    buf = io.BytesIO()
    image.save(buf, "JPEG", quality=82)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def b64(array: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(array).tobytes()).decode()


def sample(mask: np.ndarray, stride: int) -> tuple[np.ndarray, np.ndarray]:
    grid = np.zeros_like(mask)
    grid[::stride, ::stride] = True
    ys, xs = np.nonzero(mask & grid)
    return xs, ys


def pack_run(run_dir: Path) -> dict:
    meta = json.loads((run_dir / "run.json").read_text())
    image = Image.open(ROOT / meta["image"]).convert("RGB")
    rgb = np.asarray(image)
    textured = np.asarray(Image.open(run_dir / "textured.png").convert("RGB"))
    mask = np.asarray(Image.open(run_dir / "sam_mask.png")) > 127
    depth = np.load(run_dir / "depth.npy").astype(np.float32)

    # Same 2nd–98th percentile stretch the pipeline uses for depth_relative.png.
    low, high = np.percentile(depth, (2, 98))
    depth01 = np.clip((depth - low) / max(float(high - low), 1e-6), 0, 1)
    depth_u8 = np.round(depth01 * 255).astype(np.uint8)

    ox, oy = sample(mask, OBJECT_STRIDE)
    bx, by = sample(~mask, BACKGROUND_STRIDE)
    depth_on_object = np.where(mask[..., None], rgb, (rgb * 0.18).astype(np.uint8))

    return {
        "id": run_dir.name,
        "prompt": meta["prompt"],
        "image_path": meta["image"],
        "texture_path": meta["texture"],
        "sam_model": meta["sam_model"],
        "depth_model": meta["depth_model"],
        "device": meta["device"],
        "sam_score": meta["sam_score"],
        "width": image.width,
        "height": image.height,
        "mask_coverage": float(mask.mean()),
        "mean_depth_object": float(depth01[mask].mean()),
        "mean_depth_background": float(depth01[~mask].mean()),
        "object_stride": OBJECT_STRIDE,
        "background_stride": BACKGROUND_STRIDE,
        "panels": {
            "input": jpeg_data_uri(image),
            "mask": jpeg_data_uri(Image.open(run_dir / "sam_mask.png")),
            "depth": jpeg_data_uri(Image.open(run_dir / "depth_relative.png")),
            "isolated": jpeg_data_uri(Image.fromarray(depth_on_object)),
            "textured": jpeg_data_uri(Image.fromarray(textured)),
        },
        "object": {
            "count": int(len(ox)),
            "xy": b64(np.stack([ox, oy], 1).astype("<u2")),
            "z": b64(depth_u8[oy, ox]),
            "rgb": b64(rgb[oy, ox]),
            "tex": b64(textured[oy, ox]),
        },
        "background": {
            "count": int(len(bx)),
            "xy": b64(np.stack([bx, by], 1).astype("<u2")),
            "z": b64(depth_u8[by, bx]),
            "rgb": b64(rgb[by, bx]),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path, help="Pipeline output folders")
    parser.add_argument("--out", type=Path, default=ROOT / "explorer" / "data.json")
    args = parser.parse_args()

    records = [pack_run(run) for run in args.runs]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # One array, one record per line.
    args.out.write_text("[\n" + ",\n".join(json.dumps(r) for r in records) + "\n]\n")
    for r in records:
        print(f"{r['id']}: {r['object']['count']} object points, {r['background']['count']} background points")
    print(f"Wrote {args.out} ({args.out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
