"""Show what the physical-state signal looks like and how low/high-pass splits it.

For a real pour clip and a real slice clip, plot total force F and center-of-pressure x,
each as: raw signal + its SLOW (low-pass) component, and separately the FAST (high-pass)
component (fast = raw - slow). Makes visible WHY we model the fast part:
  - pour force = big slow RAMP (grip/fill) + small fast jitter
  - slice CoP  = slow drift + clear fast OSCILLATION (the strokes)

    python scripts/plot_signal_decomposition.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.train_action_dynamics import slow_fast  # noqa: E402

ROOT = "data/actionsense_states"
DS = 3          # 30 Hz -> 10 Hz (as the model uses)
CUT = 0.4       # low-pass cutoff Hz


def first_clip(substr):
    for r in (json.loads(l) for l in open(os.path.join(ROOT, "manifest.jsonl"))):
        if substr.lower() in r["label"].lower():
            st = np.load(os.path.join(ROOT, f"state_{r['idx']}.npy"))[::DS]
            h = int(np.argmax(st[:, :, 0].mean(0)))     # active hand
            return st[:, h, 0], st[:, h, 1], st[:, h, 2], 30.0 / DS   # F, x, y, fps
    raise SystemExit(f"no clip for {substr}")


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    clips = {"POUR": first_clip("Pour"), "SLICE": first_clip("Slice a cucumber")}
    fig, axes = plt.subplots(4, 2, figsize=(13, 10))
    for col, (name, (F, x, y, fps)) in enumerate(clips.items()):
        t = np.arange(len(F)) / fps
        Fs, Ff = slow_fast(F, fps, CUT)
        xs, xf = slow_fast(x, fps, CUT)
        # row0: force raw + slow
        ax = axes[0, col]
        ax.plot(t, F, color="0.6", lw=1, label="raw F")
        ax.plot(t, Fs, "C0", lw=2.2, label="slow (low-pass, grip/ramp)")
        ax.set_title(f"{name}: total force F"); ax.legend(fontsize=8); ax.set_ylabel("force")
        # row1: force fast
        ax = axes[1, col]
        ax.plot(t, Ff, "C1", lw=1.2); ax.axhline(0, color="0.8", lw=0.8)
        ax.set_title("F  fast = raw - slow (the modulation we model)"); ax.set_ylabel("F_fast")
        # row2: CoP_x raw + slow
        ax = axes[2, col]
        ax.plot(t, x, color="0.6", lw=1, label="raw CoP_x")
        ax.plot(t, xs, "C0", lw=2.2, label="slow")
        ax.set_title("center-of-pressure x"); ax.legend(fontsize=8); ax.set_ylabel("x  [-1,1]")
        # row3: CoP_x fast
        ax = axes[3, col]
        ax.plot(t, xf, "C1", lw=1.2); ax.axhline(0, color="0.8", lw=0.8)
        ax.set_title("CoP_x  fast (the stroke oscillation)"); ax.set_ylabel("x_fast")
        ax.set_xlabel("time (s)")
    fig.suptitle("Physical-state signal and its slow (low-pass) / fast (high-pass) split", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = "docs/actionsense/signal_decomposition.png"
    fig.savefig(out, dpi=120)
    print(f"[done] {out}")
    # quick numbers
    for name, (F, x, y, fps) in clips.items():
        Fs, Ff = slow_fast(F, fps, CUT); xs, xf = slow_fast(x, fps, CUT)
        print(f"{name}: F slow-range={Fs.ptp():.0f} fast-std={Ff.std():.0f} | "
              f"CoP_x slow-range={xs.ptp():.3f} fast-std={xf.std():.3f}")


if __name__ == "__main__":
    main()
