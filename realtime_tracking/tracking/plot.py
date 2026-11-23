"""Plot accuracy and delay over time from a summary.json (no models needed).

Usage: python realtime_tracking/tracking/plot.py realtime_tracking/outputs/car-shadow_obj1/summary.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def plot_over_time(summary_path: Path, out_path: Path) -> None:
    """Top: IoU of each frame a method showed (dots = frames it actually produced).
    Bottom: how late each shown frame was. Markers: ● SAM 3 correction, × hard reset."""
    s = json.loads(Path(summary_path).read_text())
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, constrained_layout=True,
                                   gridspec_kw={"height_ratios": [2, 1]})
    for name, r in s["results"].items():
        frames = [p["frame"] for p in r["per_frame"]]
        J = {p["frame"]: p["J"] for p in r["per_frame"]}
        line, = ax1.plot(frames, list(J.values()), "-", marker=".", ms=4, lw=1.4,
                         label=f"{name}: J&F {r['J&F']:.2f}, {r['fps']:.1f} fps")
        c = line.get_color()
        for f in r.get("insert_frames", []):
            if f in J:
                ax1.plot(f, J[f], "o", ms=7, color=c, mec="white", mew=1)
        for f in r.get("reset_frames", []):
            if f in J:
                ax1.plot(f, J[f], "x", ms=9, color=c, mew=2)
        ax2.plot(frames, [p["latency_ms"] for p in r["per_frame"]], "-", marker=".", ms=4, lw=1.2, color=c)
    ax1.set_ylabel("IoU of shown frame")
    ax1.set_ylim(0, 1.02)
    ax1.set_title(f"{s['video']} · \"{s['prompt']}\" · dots = frames each method could process "
                  f"(● SAM 3 correction, × hard reset)", fontsize=10)
    ax1.legend(fontsize=8, ncol=2, loc="lower right")
    ax2.set_ylabel("latency (ms)")
    ax2.set_xlabel("video frame (30 fps)")
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    p = Path(sys.argv[1])
    plot_over_time(p, p.with_name("over_time.png"))
    print(f"Saved {p.with_name('over_time.png')}")
