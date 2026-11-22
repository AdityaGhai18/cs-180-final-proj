# Project status · texture wrapping

## Current checkpoint

The image pipeline takes an image, a texture, and a text prompt. SAM 3 produces
the object mask from the prompt, Depth Anything V2 produces relative depth, and
a naive composite tiles the texture inside the mask with luminance shading.

## Next

- Use depth (surface normals) to warp the texture onto the object instead of tiling it flat.
- Extend from single images to video with temporally consistent texture.

## Run

See `README.md`. SAM 3 requires approved Hugging Face access to `facebook/sam3`.
