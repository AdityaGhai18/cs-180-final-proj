# Real-time text-prompted tracking + texture (layer 1)

Self-contained experiment: can we follow an object chosen by a text prompt in live video fast
enough to texture it in real time? Nothing outside this folder is modified; it only reuses the
shared texture `data/textures/procedural_orange_peel.png`.

## The four modes

| Mode | What it does |
|---|---|
| **A** · SAM 3 every frame | Text prompt on every frame. Most accurate per frame, very slow. |
| **B** · SAM 3 once + EdgeTAM | SAM 3 finds the object on frame 0, then the fast EdgeTAM tracker follows it. |
| **C** · B + background re-check | Like B, but every 2 s SAM 3 re-checks alongside the tracker and corrects its memory when the answer arrives. The video never waits. |
| **D** · SAM 3 once + DINO matching | Like B, but tracks by plain DINOv2 feature similarity (baseline). |

## How it's scored

The video is replayed at 30 fps. Each method's compute is timed for real; when it finishes a frame
it takes the newest one that has arrived, skipping the rest, like a live camera. The viewer sees
only the frames a method finished, each with its own mask, so slow methods look choppy and late.

- **J&F**: accuracy against Ref-DAVIS17's hand-drawn masks, on the frames shown.
- **fps**: how smooth the output is (30 = real time).
- **Latency p50 / p95**: delay from a frame arriving to its result; a high p95 means freezes.
- **Startup**: the one-time SAM 3 detection on frame 0 (reported separately).

## Run

From the repo root, with the repo's environment active:

```bash
pip install -r realtime_tracking/requirements.txt
hf auth login                                         # SAM 3 needs approved access
python realtime_tracking/download_ref_davis.py car-shadow
python realtime_tracking/tracking/evaluate.py --video car-shadow --object 1 --prompt "a white car"
```

Outputs go to `realtime_tracking/outputs/<video>_obj<id>/`: `results.csv`, `summary.json`, and
side-by-side videos (`side_by_side.mp4`, `side_by_side_textured.mp4`) showing the live camera next
to each method at its real speed.

## First result (car-shadow, "a white car", M3 Pro, 18 GB)

| Mode | J&F | fps | Latency p50 |
|---|---|---|---|
| A | 0.96 | 0.4 | 2.4 s |
| B | 0.99 | 5.0 | 169 ms |
| C | 0.99 | 6.2 | 152 ms |
| D | 0.93 | 4.3 | 186 ms |

- Accuracy is fine; speed is the problem (5–7 fps vs a 30 fps target).
- The 40-frame clip ends before C's first re-check returns, so B and C behave the same here; the fps
  gap is run-to-run noise (about ±20% on the Mac). C's background SAM 3 is also treated as free in
  the simulation even though it shares the GPU, so its real speed should be at or below B's.

## Notes

- **Use short noun phrases.** SAM 3 found nothing for the dataset sentence "a white car crossing a
  signal" but found the car for "a white car" (IoU 0.98).
- **SAM 3 runs in 16-bit on the GPU.** In 32-bit on an 18 GB Mac, time per frame grew from 4 s to
  40 s as memory filled and the system swapped; 16-bit stays at about 2.3 s with the same masks.
- EdgeTAM is given consecutive internal frame numbers, because its memory looks up "the previous N
  frames" and skipped frames would otherwise leave holes.
