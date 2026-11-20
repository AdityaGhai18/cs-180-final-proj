"""Create a tiny original demo image and material tile for local smoke runs.

These are procedural illustrations, not downloaded photographs or training data.
They exist so the repo has inspectable inputs before external assets are chosen.
"""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


ROOT = Path(__file__).resolve().parents[1]


def make_scene(path: Path) -> None:
    width, height = 640, 360
    # Smooth sky and a distant green strip make the foreground shape easy to see.
    y = np.linspace(0, 1, height)[:, None, None]
    top = np.array([89, 167, 213], dtype=np.float32)[None, None, :]
    bottom = np.array([211, 231, 239], dtype=np.float32)[None, None, :]
    sky = np.broadcast_to(top * (1 - y) + bottom * y, (height, width, 3)).copy()
    sky[282:, :, :] = np.array([112, 157, 93], dtype=np.uint8)
    image = Image.fromarray(sky.astype(np.uint8), "RGB")
    draw = ImageDraw.Draw(image)

    # A simple side-view toy airplane is the single intended foreground object.
    shadow = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    sd.polygon([(102, 196), (305, 180), (514, 193), (305, 208)], fill=(35, 51, 62, 65))
    shadow = shadow.filter(ImageFilter.GaussianBlur(9))
    image = Image.alpha_composite(image.convert("RGBA"), shadow).convert("RGB")
    draw = ImageDraw.Draw(image)
    body = [(112, 172), (216, 164), (420, 167), (516, 177), (422, 187), (220, 188), (112, 180)]
    wing = [(265, 173), (350, 120), (376, 120), (345, 175), (375, 185), (350, 187), (275, 183)]
    far_wing = [(270, 179), (346, 224), (368, 224), (344, 181)]
    tail = [(151, 171), (183, 142), (202, 143), (188, 174)]
    draw.polygon(far_wing, fill=(205, 210, 207), outline=(68, 85, 93))
    draw.polygon(wing, fill=(235, 238, 230), outline=(68, 85, 93))
    draw.polygon(tail, fill=(235, 238, 230), outline=(68, 85, 93))
    draw.polygon(body, fill=(244, 241, 218), outline=(61, 78, 88))
    draw.ellipse((210, 169, 225, 181), fill=(90, 158, 183))
    draw.ellipse((236, 168, 251, 181), fill=(90, 158, 183))
    draw.ellipse((262, 168, 277, 181), fill=(90, 158, 183))
    draw.line((510, 177, 532, 176), fill=(58, 73, 79), width=3)
    image.save(path)


def make_orange_peel(path: Path) -> None:
    size = 256
    rng = np.random.default_rng(180)
    base = np.empty((size, size, 3), dtype=np.float32)
    noise = rng.normal(0, 5, (size, size, 1))
    base[:] = np.array([231, 111, 23], dtype=np.float32)
    base += noise
    image = Image.fromarray(np.uint8(np.clip(base, 0, 255)), "RGB")
    draw = ImageDraw.Draw(image, "RGB")
    # Repeating shallow dimples suggest citrus peel without using external art.
    spacing = 17
    for row, cy in enumerate(range(8, size + spacing, spacing)):
        offset = spacing // 2 if row % 2 else 0
        for cx in range(8 + offset, size + spacing, spacing):
            r = int(rng.integers(3, 6))
            draw.ellipse((cx-r, cy-r, cx+r, cy+r), fill=(194, 78, 13), outline=(247, 153, 50), width=2)
            draw.point((cx-r//2, cy-r//2), fill=(255, 191, 91))
    image = image.filter(ImageFilter.GaussianBlur(0.35))
    image.save(path)


def main() -> None:
    images = ROOT / "data" / "images"
    textures = ROOT / "data" / "textures"
    images.mkdir(parents=True, exist_ok=True)
    textures.mkdir(parents=True, exist_ok=True)
    make_scene(images / "synthetic_airplane.png")
    make_orange_peel(textures / "procedural_orange_peel.png")
    print(f"Created {images / 'synthetic_airplane.png'}")
    print(f"Created {textures / 'procedural_orange_peel.png'}")


if __name__ == "__main__":
    main()
