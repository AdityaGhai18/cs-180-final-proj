# CS 180: Texture Wrapping with Text-Prompted Segmentation

## Current checkpoint

`src/image_pipeline.py` takes three inputs: a still image, a texture image, and
a text prompt naming the object. SAM 3 segments the object from the text
prompt, Depth Anything V2 Small estimates relative depth, and the texture is
composited into the mask. The composite is a naive baseline: the texture is
tiled flat and shaded by the original image's brightness, without using depth
to warp it. Depth-aware warping and video are the next steps.

## Setup

Use Python 3.10–3.12. In this repository:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

SAM 3's model weights are gated on Hugging Face. Visit
https://huggingface.co/facebook/sam3, sign in, request access, and wait for
approval. Then authenticate once:

```bash
hf auth login
```

## Run

```bash
python src/image_pipeline.py \
  --image data/images/02_tabby_cat.png \
  --texture data/textures/procedural_orange_peel.png \
  --prompt "orange tabby cat" \
  --output outputs/tabby_cat
```

Each output folder contains `sam_mask.png`, `depth_relative.png`,
`textured.png`, `pipeline_visualization.png`, and `run.json`. The first run
downloads model weights. If SAM 3 finds several matching instances, the script
keeps the highest-confidence one. Depth Anything's output is relative depth,
not distance in meters.
