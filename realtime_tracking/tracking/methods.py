"""Model wrappers used by the layer-1 tracking experiment.

- Sam3Detector: text prompt -> instance masks (slow, text-aware).
- EdgeTamTracker: streaming memory tracker (fast, text-blind), with SAM 3 mask insertion.
- DinoMatcher: plain DINOv2 patch-feature label propagation (Caron et al. 2021 style baseline).
Every call returns its own wall time so the simulator can place it on a clock.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sam3_utils import load_sam3, sam3_instances  # noqa: E402

EDGETAM_MODEL = "yonigozlan/EdgeTAM-hf"
DINO_MODEL = "facebook/dinov2-small"


def sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return 1.0 if union == 0 else float(np.logical_and(a, b).sum() / union)


class Sam3Detector:
    def __init__(self, device: torch.device):
        self.device = device
        self.model, self.processor = load_sam3(device)
        if device.type in ("mps", "cuda"):
            # 16-bit halves memory. In 32-bit on an 18 GB Mac, per-call time grew 4 s -> 40 s as the
            # system started swapping; 16-bit stays at ~2.3 s with the same masks.
            self.model.half()

    def detect(self, frame: np.ndarray, prompt: str) -> tuple[np.ndarray, np.ndarray, float]:
        """Returns (masks N x H x W, scores N, seconds)."""
        t0 = time.perf_counter()
        masks, scores = sam3_instances(self.model, self.processor, Image.fromarray(frame), prompt)
        sync(self.device)
        return masks, scores, time.perf_counter() - t0

    @staticmethod
    def pick(masks: np.ndarray, scores: np.ndarray, reference: np.ndarray | None = None):
        """Highest score, or (with a reference mask) highest IoU to it. Returns (mask, score) or (None, 0)."""
        if len(masks) == 0:
            return None, 0.0
        if reference is None:
            i = int(scores.argmax())
        else:
            i = int(np.argmax([iou(m, reference) for m in masks]))
        return masks[i], float(scores[i])


class EdgeTamTracker:
    """Single-object streaming tracker. Prompted frames (frame 0, SAM 3 inserts) stay in memory;
    recent frames roll over (num_maskmem=7). max_cond_frame_num caps how many prompted frames are used."""

    def __init__(self, device: torch.device, max_cond_frames: int = 4):
        from transformers import AutoProcessor, EdgeTamVideoModel

        self.device = device
        self.processor = AutoProcessor.from_pretrained(EDGETAM_MODEL)
        self.model = EdgeTamVideoModel.from_pretrained(
            EDGETAM_MODEL, max_cond_frame_num=max_cond_frames).to(device).eval()
        self.session = None
        self.keep_frames: set[int] = set()  # tracker indices whose pixels must stay for a later insert
        # The tracker's memory looks up "previous N frame indices", so skipped video frames would leave
        # holes. We give it consecutive indices of its own and map video frame numbers onto them.
        self.to_internal: dict[int, int] = {}

    def _internal(self, video_idx: int) -> int:
        if video_idx not in self.to_internal:
            self.to_internal[video_idx] = len(self.to_internal)
        return self.to_internal[video_idx]

    def keep(self, video_idx: int) -> None:
        self.keep_frames.add(self.to_internal[video_idx])

    def release(self, video_idx: int) -> None:
        self.keep_frames.discard(self.to_internal.get(video_idx, -1))

    def _prep(self, frame: np.ndarray):
        inputs = self.processor(images=Image.fromarray(frame), device=self.device, return_tensors="pt")
        return inputs.pixel_values[0], inputs.original_sizes

    def _mask(self, out, original_sizes) -> np.ndarray:
        masks = self.processor.post_process_masks([out.pred_masks], original_sizes=original_sizes, binarize=True)[0]
        return masks[0, 0].cpu().numpy().astype(bool)

    def _prune(self, current: int) -> None:
        frames = self.session.processed_frames or {}
        for k in [k for k in frames if k != current and k not in self.keep_frames]:
            del frames[k]

    def start(self, frame_idx: int, frame: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, float]:
        t0 = time.perf_counter()
        self.session = self.processor.init_video_session(inference_device=self.device, dtype=torch.float32)
        self.to_internal, self.keep_frames = {}, set()
        frame_idx = self._internal(frame_idx)
        pixels, sizes = self._prep(frame)
        self.processor.add_inputs_to_inference_session(
            self.session, frame_idx=frame_idx, obj_ids=1, input_masks=mask, original_size=sizes[0])
        out = self.model(inference_session=self.session, frame=pixels, frame_idx=frame_idx)
        result = self._mask(out, sizes)
        sync(self.device)
        self._prune(frame_idx)
        return result, time.perf_counter() - t0

    def step(self, frame_idx: int, frame: np.ndarray) -> tuple[np.ndarray, float]:
        t0 = time.perf_counter()
        frame_idx = self._internal(frame_idx)
        pixels, sizes = self._prep(frame)
        out = self.model(inference_session=self.session, frame=pixels, frame_idx=frame_idx)
        result = self._mask(out, sizes)
        sync(self.device)
        self._prune(frame_idx)
        return result, time.perf_counter() - t0

    def insert_past(self, frame_idx: int, mask: np.ndarray, hard_reset: bool) -> float:
        """Insert a late SAM 3 mask for an already-tracked frame as a prompted (kept) memory entry.
        hard_reset also clears the recent (non-prompted) memory."""
        t0 = time.perf_counter()
        s, obj = self.session, 0
        frame_idx = self.to_internal[frame_idx]
        if hard_reset:
            s.output_dict_per_obj[obj]["non_cond_frame_outputs"].clear()
        else:
            s.output_dict_per_obj[obj]["non_cond_frame_outputs"].pop(frame_idx, None)
        # Without this, the library treats the frame as a correction and stores it as recent memory.
        s.frames_tracked_per_obj[obj].pop(frame_idx, None)
        h, w = mask.shape
        self.processor.add_inputs_to_inference_session(
            s, frame_idx=frame_idx, obj_ids=1, input_masks=mask, original_size=(h, w))
        self.model(inference_session=s, frame_idx=frame_idx)
        sync(self.device)
        self.keep_frames.discard(frame_idx)
        return time.perf_counter() - t0


class DinoMatcher:
    """Plain feature matching: each patch takes the label of its most similar patches in memory
    (frame 0 + last few frames), within a local window. No training beyond the frozen ViT."""

    def __init__(self, device: torch.device, n_recent: int = 3, topk: int = 5, radius: int = 8, temp: float = 0.1):
        from transformers import AutoModel

        self.device = device
        self.model = AutoModel.from_pretrained(DINO_MODEL).to(device).eval()
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        self.n_recent, self.topk, self.radius, self.temp = n_recent, topk, radius, temp
        self.memory: list[tuple[torch.Tensor, torch.Tensor]] = []

    def _features(self, frame: np.ndarray) -> torch.Tensor:
        h, w = frame.shape[:2]
        self.gh, self.gw = h // 14, w // 14
        x = torch.from_numpy(frame).to(self.device).permute(2, 0, 1)[None].float() / 255
        x = F.interpolate(x, size=(self.gh * 14, self.gw * 14), mode="bilinear", align_corners=False)
        with torch.inference_mode():
            tokens = self.model(pixel_values=(x - self.mean) / self.std).last_hidden_state[0, 1:]
        return F.normalize(tokens, dim=-1)  # (gh*gw) x C

    def _down(self, mask: np.ndarray) -> torch.Tensor:
        # On CPU: MPS adaptive pooling needs evenly divisible sizes (480/34 is not).
        m = torch.from_numpy(mask.astype(np.float32))[None, None]
        return F.adaptive_avg_pool2d(m, (self.gh, self.gw)).flatten().to(self.device)

    def _up(self, labels: torch.Tensor, shape) -> np.ndarray:
        up = F.interpolate(labels.view(1, 1, self.gh, self.gw), size=shape, mode="bilinear", align_corners=False)
        return (up[0, 0] > 0.5).cpu().numpy()

    def start(self, frame_idx: int, frame: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, float]:
        t0 = time.perf_counter()
        feats = self._features(frame)
        ys, xs = torch.meshgrid(torch.arange(self.gh), torch.arange(self.gw), indexing="ij")
        coords = torch.stack([ys.flatten(), xs.flatten()], 1).float().to(self.device)
        self.local = torch.cdist(coords, coords) <= self.radius  # P x P neighbourhood
        self.first = (feats, self._down(mask))
        self.memory = []
        sync(self.device)
        return mask, time.perf_counter() - t0

    def step(self, frame_idx: int, frame: np.ndarray) -> tuple[np.ndarray, float]:
        t0 = time.perf_counter()
        feats = self._features(frame)
        entries = [self.first] + self.memory
        sims = torch.cat([feats @ f.T for f, _ in entries], dim=1)                      # P x (P*M)
        sims = sims.masked_fill(~self.local.repeat(1, len(entries)), -1.0)
        labels_mem = torch.cat([lab for _, lab in entries])
        vals, idx = sims.topk(self.topk, dim=1)
        weights = torch.softmax(vals / self.temp, dim=1)
        labels = (weights * labels_mem[idx]).sum(1)
        self.memory = (self.memory + [(feats, labels)])[-self.n_recent:]
        result = self._up(labels, frame.shape[:2])
        sync(self.device)
        return result, time.perf_counter() - t0
