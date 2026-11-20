# Project status · text-prompt segmentation and depth

## Current checkpoint

The image pipeline supports both SAM 3's native text prompt and a SAM ViT-B
pixel-box prompt. The text mode is retained for when gated SAM 3 access is
approved. The box mode runs without SAM 3 access and still produces a mask
followed by Depth Anything V2 relative depth. Texture projection, mesh
construction, and video processing are later project steps.

## Run

Install the updated requirements, then run `src/image_pipeline.py` with
`--image` and exactly one of `--box` or `--prompt`. Example commands are in
`README.md`. The box mode downloads public SAM ViT-B weights; text mode requires
approved Hugging Face access to SAM 3.
