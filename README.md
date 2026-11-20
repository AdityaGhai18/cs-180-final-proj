# CS 180: Text-Prompted Object Segmentation and Depth

## Current checkpoint

`src/image_pipeline.py` accepts either a still image plus a text prompt for SAM
3, or a still image plus a box prompt for SAM ViT-B. Both paths produce an
object mask and then a relative depth map with Depth Anything V2 Small. SAM 3's
text prompt is the intended path and requires model access approval; the
box-prompt path is a local fallback that does not depend on SAM 3 access. This
checkpoint does not do texture warping, mesh construction, or video processing.

## Setup

Use Python 3.10–3.12. In this repository:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

SAM 3's model weights require access through Hugging Face. Visit
https://huggingface.co/facebook/sam3, sign in, and accept the model terms. Then
authenticate once from this environment:

```bash
hf auth login
```

## Run now with box prompts

```bash
python src/image_pipeline.py --image data/images/02_tabby_cat.png --box 250 70 970 1100 --output outputs/tabby_cat
python src/image_pipeline.py --image data/images/03_running_shoe.png --box 70 230 1200 950 --output outputs/running_shoe
python src/image_pipeline.py --image data/images/04_monstera_plant.png --box 140 30 1130 1100 --output outputs/monstera_plant
```

When SAM 3 access is approved, run the text-prompt version with:

```bash
python src/image_pipeline.py --image data/images/02_tabby_cat.png --prompt "orange tabby cat" --output outputs/tabby_cat_text
```

Outputs are `sam_mask.png`, `depth_relative.png`, `pipeline_visualization.png`,
and `run.json` in each output folder. The first run downloads model weights.
SAM 3 returns masks for matching instances; this script saves the highest
confidence mask. Depth Anything's output is relative depth, not distance in
meters.
