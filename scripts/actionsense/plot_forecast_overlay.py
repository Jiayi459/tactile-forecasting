"""Real tactile signal vs each history model's 1 s forecast — one figure per channel.

To avoid clutter, every history model is drawn in its OWN subplot (real + that one forecast),
and the three channels (F, CoP-x, CoP-y) go to three separate figures. Grid per figure:
rows = past-context (1/2/3/5/10 s), cols = (test clip x hand). x-axis in seconds; y-axis labelled
with units. Writes docs/forecast_<F|CoPx|CoPy>.png (results_summary.png untouched).

    python scripts/actionsense/plot_forecast_overlay.py --actions Slice,Peel --input-mode raw
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.actionsense import action_dynamics as AD  # noqa: E402

# (channel index, filename tag, y-axis label with units)
CHANNELS = [
    (0, "F",    "fast total force (sensor units, a.u.)"),
    (1, "CoPx", "fast CoP-x (normalized grid, -1..1)"),
    (2, "CoPy", "fast CoP-y (normalized grid, -1..1)"),
]


def forecast_all(model, norm, clip, t_in, t_out):
    """Rolling non-overlapping 1 s forecast, all 3 channels -> (frame_idx, mu (len,3))."""
    import torch
    feat, targ, aid = clip
    fn = norm.nx(feat)
    ts, mus = [], []
    model.eval()
    with torch.no_grad():
        for a in range(t_in, feat.shape[0] - t_out, t_out):
            x = torch.tensor(fn[a - t_in:a][None])
            yl = torch.tensor(norm.ny(targ[a - 1])[None])
            mu, _ = model(x, torch.tensor([aid]), yl, t_out)
            ts.append(np.arange(a, a + t_out)); mus.append(norm.dy(mu[0].numpy()))
    return np.concatenate(ts), np.concatenate(mus, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/actionsense_states")
    ap.add_argument("--actions", default="Slice,Peel")
    ap.add_argument("--input-mode", default="raw")
    ap.add_argument("--downsample", type=int, default=3)
    ap.add_argument("--cut", type=float, default=0.4)
    ap.add_argument("--pasts", default="1,2,3,5,10")
    ap.add_argument("--future-sec", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--n-clips", type=int, default=2)
    ap.add_argument("--out-prefix", default="docs/forecast")
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    subs = [s.strip() for s in args.actions.split(",")]
    fps = 30.0 / args.downsample
    t_out = int(round(args.future_sec * fps))
    pasts = [float(p) for p in args.pasts.split(",")]

    ref = AD.load_pooled(args.root, subs, args.downsample, args.cut, input_mode=args.input_mode, hand="left")
    _, test_ids = AD.split_train_test(len(ref), seed=1)
    max_tin = int(round(max(pasts) * fps))
    viz_ids = [i for i in test_ids if ref[i][0].shape[0] >= max_tin + t_out][:args.n_clips]
    print(f"visualizing test clips {viz_ids} (both hands); training {len(pasts)} history models/hand")

    results = {}   # (hand, clip) -> (true (T,3), {past: (frame_idx, mu(len,3))})
    for hand in ["left", "right"]:
        data = AD.load_pooled(args.root, subs, args.downsample, args.cut,
                              input_mode=args.input_mode, hand=hand)
        train = [d for i, d in enumerate(data) if i not in test_ids]
        models = {p: AD.train(train, len(subs), int(round(p * fps)), t_out, epochs=args.epochs)
                  for p in pasts}                                   # train once per history
        for ci in viz_ids:
            preds = {p: forecast_all(m, norm, data[ci], int(round(p * fps)), t_out)
                     for p, (m, norm) in models.items()}
            results[(hand, ci)] = (data[ci][1], preds)
        print(f"  hand={hand} done")

    cols = [(ci, hand) for ci in viz_ids for hand in ["left", "right"]]   # 2 clips x 2 hands
    for k, tag, ylabel in CHANNELS:
        fig, axes = plt.subplots(len(pasts), len(cols), figsize=(3.4 * len(cols), 2.2 * len(pasts)),
                                 squeeze=False, sharex="col")
        for ri, p in enumerate(pasts):
            for ciX, (ci, hand) in enumerate(cols):
                ax = axes[ri, ciX]
                true, preds = results[(hand, ci)]
                tt = np.arange(true.shape[0]) / fps
                ax.plot(tt, true[:, k], "k-", lw=1.3, label="real")
                fidx, mu = preds[p]
                ax.plot(fidx / fps, mu[:, k], "-", color="C1", lw=1.3, label=f"{p:.0f}s-hist forecast")
                if ri == 0:
                    ax.set_title(f"clip {ci} / {hand} hand", fontsize=9)
                if ciX == 0:
                    ax.set_ylabel(f"{p:.0f}s history\n{ylabel}", fontsize=8)
                if ri == len(pasts) - 1:
                    ax.set_xlabel("time (s)", fontsize=9)
                if ri == 0 and ciX == 0:
                    ax.legend(fontsize=7, loc="upper right")
                ax.grid(alpha=.25)
        fig.suptitle(f"{ylabel}  —  real vs 1 s forecast per history model ({args.input_mode} input, {args.actions})",
                     fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        out = f"{args.out_prefix}_{tag}.png"
        fig.savefig(out, dpi=110)
        print(f"[done] {out}")


if __name__ == "__main__":
    main()
