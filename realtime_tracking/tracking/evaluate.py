"""Layer-1 experiment: text-prompted tracking under a simulated real-time clock.

The video is replayed at 30 fps. Each method's compute is measured for real; when a step finishes,
the method takes the newest frame that has arrived (frames that arrived meanwhile are skipped, as
with a live camera; Li et al. ECCV 2020). The viewer sees only the processed frames, each with its
own mask, so a slow method shows up as a choppy, delayed video: accuracy = J&F on shown frames,
smoothness = fps, delay = latency.

Usage:
  python realtime_tracking/tracking/evaluate.py --video car-shadow --object 1 --prompt "a white car"
  python realtime_tracking/tracking/evaluate.py --video car-shadow --object 1 --init gt --methods B D   # no SAM 3
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sam3_utils import choose_device  # noqa: E402
from tracking.data import load_sequence  # noqa: E402
from tracking.plot import plot_over_time  # noqa: E402
from tracking.methods import DinoMatcher, EdgeTamTracker, Sam3Detector, iou  # noqa: E402

HERE = Path(__file__).resolve().parents[1]  # realtime_tracking/
FPS = 30.0


def arrival(k: int) -> float:
    """Simulated time frame k arrives. The clock starts (t = 0) when the one-time startup detection on
    frame 0 is done, and frame 1 arrives then; startup time is reported separately."""
    return (k - 1) / FPS
MIN_SAM3_SCORE = 0.5
HARD_RESET_IOU = 0.3


@dataclass
class Output:
    src: int        # frame the mask was computed on
    finish: float   # simulated time (s) the mask became available
    mask: np.ndarray


@dataclass
class Run:
    outputs: list[Output] = field(default_factory=list)
    step_times: list[float] = field(default_factory=list)
    sam3_calls: int = 0
    startup_s: float = 0.0
    inserts: int = 0
    hard_resets: int = 0
    events: list[str] = field(default_factory=list)
    insert_frames: list[int] = field(default_factory=list)  # frames where a SAM 3 correction took effect
    reset_frames: list[int] = field(default_factory=list)


# ---------------------------------------------------------------- method definitions
def parse_method(name: str) -> dict:
    """A | B | D | C-async<seconds> (C-async0 = re-check as soon as the previous check finishes)."""
    if name in ("A", "B", "D"):
        return {"kind": name}
    if name.startswith("C-async"):
        return {"kind": "C-async", "interval": float(name[7:])}
    raise SystemExit(f"Unknown method {name}")


def simulate(name: str, frames, prompt, init_mask, models, realtime: bool) -> Run:
    cfg, run = parse_method(name), Run()
    n = len(frames)
    det: Sam3Detector | None = models.get("sam3")

    def first_mask():
        if init_mask is not None:
            return init_mask, 0.0
        masks, scores, dt = det.detect(frames[0], prompt)
        run.sam3_calls += 1
        m, _ = Sam3Detector.pick(masks, scores)
        return (m if m is not None else np.zeros(frames[0].shape[:2], bool)), dt

    # ---- one-time startup on frame 0: SAM 3 finds the object (every method starts the same way)
    mask0, dt_det = first_mask()
    if cfg["kind"] == "A":
        tracker, m, dt_trk = None, mask0, 0.0
    else:
        tracker = models["dino"] if cfg["kind"] == "D" else models["edgetam"]
        m, dt_trk = tracker.start(0, frames[0], mask0)
    run.startup_s = dt_det + dt_trk
    t = 0.0
    run.outputs.append(Output(0, t, m))

    last = 0
    last_sam3_frame, last_launch, pending = 0, 0.0, None
    saved_tracker_mask: dict[int, np.ndarray] = {}

    while True:
        idx = last + 1 if not realtime else max(last + 1, math.floor(t * FPS + 1e-9) + 1)
        if idx >= n:
            break
        t = max(t, arrival(idx))  # wait for the frame to arrive
        step = 0.0

        if cfg["kind"] == "A":
            masks, scores, dt = det.detect(frames[idx], prompt)
            run.sam3_calls += 1
            m, _ = Sam3Detector.pick(masks, scores)
            m = m if m is not None else np.zeros(frames[idx].shape[:2], bool)
            step = dt

        elif cfg["kind"] in ("B", "D"):
            m, step = tracker.step(idx, frames[idx])

        else:  # C-async
            # 1) apply a finished background SAM 3 result (it describes frame x, which is in the past)
            if pending is not None:
                ready = t >= pending["ready_time"] if realtime else idx >= pending["ready_frame"]
                if ready:
                    x, sm, agree = pending["x"], pending["mask"], pending["iou"]
                    if sm is not None and pending["score"] >= MIN_SAM3_SCORE:
                        hard = agree < HARD_RESET_IOU
                        step += tracker.insert_past(x, sm, hard_reset=hard)
                        run.inserts += 1
                        run.hard_resets += int(hard)
                        run.insert_frames.append(idx)
                        if hard:
                            run.reset_frames.append(idx)
                        run.events.append(f"frame {idx}: inserted SAM 3 mask for frame {x} (IoU vs tracker {agree:.2f}{', hard reset' if hard else ''})")
                    else:
                        tracker.release(x)
                    pending = None
            # 2) normal tracker step
            m, dt = tracker.step(idx, frames[idx])
            step += dt
            saved_tracker_mask[idx] = m

        t += step
        run.step_times.append(step)
        run.outputs.append(Output(idx, t, m))
        last = idx

        # 3) C-async: launch background SAM 3 on the frame just processed (its time is NOT on the clock)
        if cfg["kind"] == "C-async" and pending is None and t - last_launch >= cfg["interval"]:
            masks, scores, dt = det.detect(frames[idx], prompt)
            run.sam3_calls += 1
            sm, score = Sam3Detector.pick(masks, scores, reference=saved_tracker_mask[idx])
            agree = iou(sm, saved_tracker_mask[idx]) if sm is not None else 0.0
            tracker.keep(idx)
            pending = {"x": idx, "mask": sm, "score": score, "iou": agree,
                       "ready_time": t + dt, "ready_frame": idx + math.ceil(dt * FPS)}
            last_launch = t

    return run


# ---------------------------------------------------------------- scoring
def boundary_f(pred: np.ndarray, gt: np.ndarray) -> float:
    """DAVIS boundary F-measure (tolerance 0.8% of the image diagonal)."""
    def edges(m):
        m = m.astype(np.uint8)
        return (m - cv2.erode(m, np.ones((3, 3), np.uint8))).astype(bool)
    if not pred.any() and not gt.any():
        return 1.0
    if not pred.any() or not gt.any():
        return 0.0
    r = math.ceil(0.008 * math.hypot(*gt.shape))
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    pb, gb = edges(pred), edges(gt)
    pd, gd = cv2.dilate(pb.astype(np.uint8), disk).astype(bool), cv2.dilate(gb.astype(np.uint8), disk).astype(bool)
    precision = (pb & gd).sum() / max(pb.sum(), 1)
    recall = (gb & pd).sum() / max(gb.sum(), 1)
    return 0.0 if precision + recall == 0 else float(2 * precision * recall / (precision + recall))


def score(run: Run, gt: list[np.ndarray]) -> dict:
    """Display design 2: the viewer sees only the frames a method processed, each with its own mask.
    Accuracy = J&F over those frames; smoothness = fps; delay = latency."""
    J = [iou(o.mask, gt[o.src]) for o in run.outputs]
    Fm = [boundary_f(o.mask, gt[o.src]) for o in run.outputs]
    lat = [1000 * (o.finish - arrival(o.src)) for o in run.outputs[1:]] or [0.0]  # frame 0 = startup
    return {
        "J&F": float(np.mean(J) + np.mean(Fm)) / 2, "J": float(np.mean(J)), "F": float(np.mean(Fm)),
        "fps": float(len(run.step_times) / sum(run.step_times)) if run.step_times else 0.0,
        "latency_p50_ms": float(np.percentile(lat, 50)), "latency_p95_ms": float(np.percentile(lat, 95)),
        "frames_shown": len(run.outputs), "frames_J<0.5": int(sum(j < 0.5 for j in J)),
        "sam3_calls": run.sam3_calls, "inserts": run.inserts, "hard_resets": run.hard_resets,
        "startup_ms": 1000 * run.startup_s,
        "per_frame": [{"frame": o.src, "J": j, "latency_ms": l}
                      for o, j, l in zip(run.outputs, J, [0.0] + lat[:len(run.outputs) - 1])],
        "insert_frames": run.insert_frames, "reset_frames": run.reset_frames, "events": run.events,
    }


# ---------------------------------------------------------------- outputs
def apply_texture(frame: np.ndarray, mask: np.ndarray, texture: np.ndarray) -> np.ndarray:
    """Minimal texture wrap (layer 1b): stretch the texture over the mask's bounding box and keep
    the scene's lighting by multiplying with the frame's brightness relative to the mask's mean.
    Not anchored to the surface, so it slides when the object deforms (that is layer 2)."""
    img = frame.copy()
    if not mask.any():
        return img
    ys, xs = np.nonzero(mask)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    tex = cv2.resize(texture, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    lum = frame[y0:y1, x0:x1].astype(np.float32).mean(axis=2, keepdims=True)
    m = mask[y0:y1, x0:x1]
    shade = lum / max(float(lum[m].mean()), 1.0)
    region = img[y0:y1, x0:x1]
    region[m] = np.clip(tex * shade, 0, 255)[m].astype(np.uint8)
    return img


def overlay(frame, mask, color, label, texture=None):
    if texture is not None:
        img = apply_texture(frame, mask, texture)
    else:
        img = frame.copy()
        img[mask] = (0.45 * img[mask] + 0.55 * np.array(color)).astype(np.uint8)
    img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))
    cv2.rectangle(img, (0, 0), (img.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(img, label, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def write_video(path: Path, frames, gt, runs: dict, texture=None):
    """Grid at 30 fps: live ground truth first, then each method showing its most recent processed
    frame (with that frame's mask) until the next one is ready, so slow methods look choppy and late."""
    tiles_per_frame = []
    for k, frame in enumerate(frames):
        tiles = [overlay(frame, gt[k], (255, 255, 255), "Live camera (ground truth)", texture)]
        for name, (run, res) in runs.items():
            ready = [o for o in run.outputs if o.finish <= arrival(k) + 1e-9]
            if ready:
                o = ready[-1]
                label = f"{name} | {res['fps']:.1f} fps | {1000 * (k - o.src) / FPS:.0f} ms behind"
                tiles.append(overlay(frames[o.src], o.mask, (40, 200, 90), label, texture))
            else:
                blank = np.zeros_like(frame)
                tiles.append(overlay(blank, np.zeros(frame.shape[:2], bool), (0, 0, 0), f"{name} | starting...", None))
        while len(tiles) % 3:
            tiles.append(np.zeros_like(tiles[0]))
        rows = [np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)]
        tiles_per_frame.append(np.vstack(rows))
    h, w = tiles_per_frame[0].shape[:2]
    for codec in ("avc1", "mp4v"):
        vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), FPS, (w, h))
        if vw.isOpened():
            break
    for img in tiles_per_frame:
        vw.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    vw.release()


# ---------------------------------------------------------------- main
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", default="bmx-trees")
    p.add_argument("--object", type=int, default=2)
    p.add_argument("--prompt", help="defaults to the first Ref-DAVIS17 expression for the object")
    p.add_argument("--methods", nargs="+", default=["A", "B", "C-async2", "D"])
    p.add_argument("--init", choices=["sam3", "gt"], default="sam3", help="gt = debug without SAM 3")
    p.add_argument("--out", type=Path, default=HERE / "outputs")
    p.add_argument("--texture", type=Path, default=HERE.parent / "data/textures/procedural_orange_peel.png")
    args = p.parse_args()

    seq = load_sequence(args.video)
    gt = seq.gt[args.object]
    prompt = args.prompt or seq.expressions[args.object][0]
    device = choose_device()
    print(f"{args.video} object {args.object}: {len(seq.frames)} frames, prompt {prompt!r}, device {device}")

    needs_sam3 = args.init == "sam3" or any(m == "A" or m.startswith("C") for m in args.methods)
    models = {}
    if needs_sam3:
        models["sam3"] = Sam3Detector(device)
    if any(m != "A" and m != "D" for m in args.methods):
        models["edgetam"] = EdgeTamTracker(device)
    if "D" in args.methods:
        models["dino"] = DinoMatcher(device)

    # warm-up so the first timed call isn't paying for kernel compilation
    for _ in range(2):
        if "sam3" in models:
            models["sam3"].detect(seq.frames[0], prompt)
        for key in ("edgetam", "dino"):
            if key in models:
                models[key].start(0, seq.frames[0], gt[0])
                models[key].step(1, seq.frames[1])

    init_mask = gt[0] if args.init == "gt" else None
    out = args.out / f"{args.video}_obj{args.object}"
    out.mkdir(parents=True, exist_ok=True)
    results, runs = {}, {}
    for name in args.methods:
        run = simulate(name, seq.frames, prompt, init_mask, models, realtime=True)
        res = score(run, gt)
        results[name], runs[name] = res, (run, res)
        print(f"{name:10s} J&F {res['J&F']:.3f} | {res['fps']:5.1f} fps | latency p50 {res['latency_p50_ms']:6.0f} ms "
              f"p95 {res['latency_p95_ms']:6.0f} ms | startup {res['startup_ms']:5.0f} ms | shown {res['frames_shown']}/{len(gt)} frames | "
              f"SAM3 calls {res['sam3_calls']} inserts {res['inserts']} resets {res['hard_resets']}")

    summary = {"video": args.video, "object": args.object, "prompt": prompt, "device": str(device),
               "init": args.init, "fps_clock": FPS, "results": results}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    cols = ["J&F", "J", "F", "fps", "latency_p50_ms", "latency_p95_ms", "startup_ms", "frames_shown", "frames_J<0.5",
            "sam3_calls", "inserts", "hard_resets"]
    with open(out / "results.csv", "w") as f:
        f.write("method," + ",".join(cols) + "\n")
        for name, r in results.items():
            f.write(name + "," + ",".join(f"{r[c]:.4g}" for c in cols) + "\n")
    plot_over_time(out / "summary.json", out / "over_time.png")
    write_video(out / "side_by_side.mp4", seq.frames, gt, runs)
    texture = np.asarray(Image.open(args.texture).convert("RGB"))
    write_video(out / "side_by_side_textured.mp4", seq.frames, gt, runs, texture)
    print(f"Saved results to {out}")


if __name__ == "__main__":
    main()
