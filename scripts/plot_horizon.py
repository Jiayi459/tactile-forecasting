"""Time-horizon anchor plot for the CALIBRATED high-pass model.

For one test clip and one anchor, show — per channel (F, CoP-x, CoP-y) —
  * the PAST HISTORY window the model actually consumed (grey),
  * the TRUE future 1 s (black),
  * the model's 1 s FORECAST + calibrated +/-2 sigma band (blue),
  * the PERSISTENCE baseline = repeat last value (grey dashed).
A vertical line marks "now" (the forecast origin). This is the current
AUTOREGRESSIVE forecast (decoder rolls out t_out 0.1 s steps).

    python scripts/plot_horizon.py --input-mode highpass --hand right --history 3
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.actionsense import action_dynamics as AD  # noqa: E402

CHANNELS = [
    (0, "F",    "fast total force (a.u.)"),
    (1, "CoP-x", "fast CoP-x (grid, -1..1)"),
    (2, "CoP-y", "fast CoP-y (grid, -1..1)"),
]


def forecast_anchor(model, norm, clip, t_in, t_out, anchor, sigma_scale):
    """Forecast the t_out frames after `anchor`; return mu(t_out,3), sd(t_out,3)."""
    import torch
    feat, targ, aid = clip
    x = torch.tensor(norm.nx(feat[anchor - t_in:anchor])[None])
    yl = torch.tensor(norm.ny(targ[anchor - 1])[None])
    model.eval()
    with torch.no_grad():
        mu, lv = model(x, torch.tensor([aid]), yl, t_out)
    mu = norm.dy(mu[0].numpy())
    sd = np.exp(0.5 * lv[0].numpy()) * norm.sd_y * sigma_scale   # de-normalize + calibrate
    return mu, sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/actionsense_states")
    ap.add_argument("--actions", default="Slice,Peel")
    ap.add_argument("--input-mode", default="highpass")
    ap.add_argument("--hand", default="right")
    ap.add_argument("--history", type=float, default=3.0, help="past-context seconds")
    ap.add_argument("--future-sec", type=float, default=1.0)
    ap.add_argument("--downsample", type=int, default=3)
    ap.add_argument("--cut", type=float, default=0.4)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--clip-rank", type=int, default=0, help="which eligible test clip (0=first)")
    ap.add_argument("--out", default="docs/actionsense/horizon_highpass.png")
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    subs = [s.strip() for s in args.actions.split(",")]
    fps = 30.0 / args.downsample
    t_in = int(round(args.history * fps))
    t_out = int(round(args.future_sec * fps))

    data = AD.load_pooled(args.root, subs, args.downsample, args.cut,
                          input_mode=args.input_mode, hand=args.hand)
    tr_ids, te_ids = AD.split_train_test(len(data), seed=1)
    # hold out a validation slice of TRAIN for sigma calibration (never touch test)
    val = [data[i] for i in tr_ids[::5]]
    train = [data[i] for i in tr_ids if i not in set(tr_ids[::5])]

    print(f"training {args.input_mode}/{args.hand}  hist={args.history}s (t_in={t_in}) "
          f"future={args.future_sec}s (t_out={t_out})  on {len(train)} clips")
    model, norm = AD.train(train, len(subs), t_in, t_out, epochs=args.epochs)
    s = AD.calibrate_sigma(model, norm, val, t_in, t_out, target=0.95)
    print(f"calibrated sigma_scale = {s:.3f}")

    # pick an eligible test clip long enough to show history + horizon
    elig = [i for i in te_ids if data[i][0].shape[0] >= t_in + t_out + 2]
    ci = elig[args.clip_rank]
    clip = data[ci]
    T = clip[0].shape[0]
    anchor = t_in + (T - t_in - t_out) // 2          # a mid-clip anchor
    targ = clip[1]
    mu, sd = forecast_anchor(model, norm, clip, t_in, t_out, anchor, s)
    pers = np.repeat(targ[anchor - 1][None], t_out, 0)      # persistence = last value

    thist = (np.arange(anchor - t_in, anchor)) / fps
    tfut = (np.arange(anchor, anchor + t_out)) / fps
    now = anchor / fps

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    for j, (k, tag, ylabel) in enumerate(CHANNELS):
        ax = axes[j]
        ax.plot(thist, targ[anchor - t_in:anchor, k], color="0.5", lw=1.4,
                label=f"history ({args.history:.0f}s input)")
        ax.plot(tfut, targ[anchor:anchor + t_out, k], "k-", lw=2.0, label="true future")
        ax.plot(tfut, mu[:, k], "-", color="C0", lw=2.0, label="forecast (mean)")
        ax.fill_between(tfut, mu[:, k] - 2 * sd[:, k], mu[:, k] + 2 * sd[:, k],
                        color="C0", alpha=.2, label="forecast +/-2sigma (calibrated)")
        ax.plot(tfut, pers[:, k], "--", color="0.4", lw=1.6, label="persistence")
        ax.axvline(now, color="r", lw=1, ls=":")
        ax.set_title(f"{tag}"); ax.set_xlabel("time (s)"); ax.set_ylabel(ylabel)
        ax.grid(alpha=.25)
        if j == 0:
            ax.legend(fontsize=8, loc="upper left")
    fig.suptitle(f"Calibrated {args.input_mode} forecast — {args.hand} hand, {args.history:.0f}s history "
                 f"-> 1s ahead (clip {ci}, {args.actions}) | autoregressive", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=120)
    print(f"[done] {args.out}")


if __name__ == "__main__":
    main()
