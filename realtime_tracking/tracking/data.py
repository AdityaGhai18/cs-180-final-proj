"""Load a Ref-DAVIS17 video: RGB frames, per-object ground-truth masks, and text expressions."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1] / "data" / "ref_davis17"


@dataclass
class Sequence:
    name: str
    frames: list[np.ndarray]          # H x W x 3 uint8
    gt: dict[int, list[np.ndarray]]   # object id -> per-frame bool masks
    expressions: dict[int, list[str]] # object id -> referring expressions


def load_expressions(root: Path = ROOT) -> dict[str, dict[int, list[str]]]:
    """Both annotators' full-video expressions, keyed by video then object id."""
    out: dict[str, dict[int, list[str]]] = {}
    for f in sorted((root / "davis_text_annotations").glob("Davis17_annot*_full_video.txt")):
        for line in f.read_text(errors="replace").splitlines():
            m = re.match(r'(\S+)\s+(\d+)\s+"(.*)"', line.strip())
            if m:
                out.setdefault(m[1], {}).setdefault(int(m[2]), []).append(m[3])
    return out


def load_sequence(name: str, root: Path = ROOT) -> Sequence:
    img_dir = root / "JPEGImages/480p" / name
    ann_dir = root / "Annotations/480p" / name
    if not img_dir.exists():
        raise SystemExit(f"Missing {img_dir}. Run: python realtime_tracking/download_ref_davis.py {name}")
    frames = [np.asarray(Image.open(p).convert("RGB")) for p in sorted(img_dir.glob("*.jpg"))]
    labels = [np.asarray(Image.open(p)) for p in sorted(ann_dir.glob("*.png"))]  # palette index = object id
    ids = sorted(int(i) for i in np.unique(np.stack(labels)) if i != 0)
    gt = {i: [lab == i for lab in labels] for i in ids}
    return Sequence(name, frames, gt, load_expressions(root).get(name, {}))
