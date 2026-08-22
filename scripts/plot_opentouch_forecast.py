"""Forecast curves for OpenTouch: history, truth, and each model's 1 s prediction.

One row per clip, one column per channel (F, CoP-x, CoP-y). The x axis is time in seconds
and covers BOTH the history the models were given and the horizon they predicted, so the
forecast can be read against the run-up that produced it rather than in isolation.

The probGRU's band is +-2 sigma from its log-variance head. That head is trained (it is
half the Gaussian NLL) but never scored -- the frozen harness measures point error only --
so this plot is the one place the model's own uncertainty is visible. A band that does not
widen with the horizon, or that ignores where the signal actually moves, says the
probabilistic half learned nothing, and no MSE table would reveal that.

Reads what --save-preds wrote; it does not train, so it costs seconds rather than the GPU
hours the forecasts themselves cost.

    python scripts/plot_opentouch_forecast.py --preds runs/preds --n 4
    python scripts/plot_opentouch_forecast.py --preds runs/preds --clips 812,1503 --origin 0.5
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np

STYLE = {"prob_gru": ("tab:red", "-"), "ar": ("tab:blue", "--"),
         "persistence": ("0.45", ":"), "seasonal": ("tab:green", "-."),
         "gru_aggregate": ("tab:purple", "-")}


def load(path):
    z = np.load(path, allow_pickle=True)
    models = sorted(k[3:] for k in z.files if k.startswith("mu_"))
    return z, models


def pick_origin(origins, frac):
    """Which forecast origin to draw: frac of the way through the clip's origins."""
    if len(origins) == 0:
        return None
    return int(np.clip(round(frac * (len(origins) - 1)), 0, len(origins) - 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True, help="directory written by --save-preds")
    ap.add_argument("--clips", help="comma-separated clip indices; default: the first --n")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--origin", type=float, default=0.5,
                    help="which forecast origin to draw, as a fraction of the clip (0..1)")
    ap.add_argument("--history-s", type=float, default=3.0,
                    help="seconds of run-up to draw before the origin")
    ap.add_argument("--out", default="docs/opentouch/raw/opentouch_forecast.png")
    ap.add_argument("--per-channel", action="store_true",
                    help="one figure per channel (out stem gets _F_R etc.) instead of one "
                         "figure with the channels side by side")
    ap.add_argument("--compare", metavar="DIR",
                    help="a second --save-preds directory; its prob_gru mean is overlaid so "
                         "two arms can be read on the same clip and origin")
    ap.add_argument("--compare-label", default="compare")
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.preds, "clip_*.npz")))
    if not files:
        raise SystemExit(f"no clip_*.npz under {a.preds} -- run with --save-preds first")
    if a.clips:
        want = {int(c) for c in a.clips.split(",")}
        files = [f for f in files
                 if int(os.path.basename(f)[5:-4]) in want] or files[:a.n]
    else:
        files = files[:a.n]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z0, _ = load(files[0])
    all_chans = [str(c) for c in z0["channels"]]
    groups = ([[i] for i in range(len(all_chans))] if a.per_channel
              else [list(range(len(all_chans)))])
    stem, ext = os.path.splitext(a.out)
    for g in groups:
        out = f"{stem}_{all_chans[g[0]]}{ext}" if a.per_channel else a.out
        draw(files, g, all_chans, a, plt, out)


def draw(files, cols_idx, all_chans, a, plt, out):
    chans = [all_chans[i] for i in cols_idx]
    fig, axes = plt.subplots(len(files), len(chans), squeeze=False,
                             figsize=(6.2 * len(chans) if len(chans) == 1
                                      else 4.6 * len(chans), 2.9 * len(files)))

    for row, path in enumerate(files):
        z, models = load(path)
        y, origins, fps = z["y"], z["origins"], float(z["fps"])
        idx = os.path.basename(path)[5:-4]
        j = pick_origin(origins, a.origin)
        if j is None:
            continue
        t0 = int(origins[j])                       # last observed frame
        H = z[f"mu_{models[0]}"].shape[1]
        lo = max(0, t0 - int(round(a.history_s * fps)))
        t_hist = np.arange(lo, t0 + 1) / fps
        t_fut = np.arange(t0 + 1, t0 + 1 + H) / fps

        other = None
        if a.compare:
            p2 = os.path.join(a.compare, os.path.basename(path))
            if os.path.exists(p2):
                other = np.load(p2, allow_pickle=True)

        for col, ci in enumerate(cols_idx):
            ax = axes[row][col]
            ax.plot(t_hist, y[lo:t0 + 1, ci], color="k", lw=1.2, label="history")
            ax.plot(t_fut, y[t0 + 1:t0 + 1 + H, ci], color="k", lw=1.2, ls="-",
                    alpha=0.35, label="truth (future)")
            ax.axvline(t0 / fps, color="0.7", lw=0.8)

            for m in models:
                c, ls = STYLE.get(m, ("tab:orange", "-"))
                ax.plot(t_fut, z[f"mu_{m}"][j, :, ci], color=c, ls=ls, lw=1.3, label=m)
                if f"sigma_{m}" in z.files:
                    sg = z[f"sigma_{m}"][j, :, ci]
                    ax.fill_between(t_fut, z[f"mu_{m}"][j, :, ci] - 2 * sg,
                                    z[f"mu_{m}"][j, :, ci] + 2 * sg,
                                    color=c, alpha=0.15, lw=0, label=f"{m} ±2σ")
            # A second arm on the same clip and the same origin: the only honest way to see
            # what an input change did, since a metric table cannot show where it changed.
            if other is not None and "mu_prob_gru" in other.files \
                    and other["mu_prob_gru"].shape[0] > j:
                ax.plot(t_fut, other["mu_prob_gru"][j, :, ci], color="tab:brown", ls=(0, (3, 1, 1, 1)),
                        lw=1.3, label=f"prob_gru ({a.compare_label})")
            if row == 0:
                ax.set_title(all_chans[ci], fontsize=10)
            if col == 0:
                ax.set_ylabel(f"[{idx}] {z['action']}\n{z['object_name']}", fontsize=8)
            if row == len(files) - 1:
                ax.set_xlabel("time (s)")
            ax.tick_params(labelsize=7)

    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=min(len(l), 7), fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(f"OpenTouch 1 s forecasts — {', '.join(chans)} — grey line = forecast "
                 f"origin; band = probGRU ±2σ", fontsize=10, y=0.955)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}  ({len(files)} clips, origin frac {a.origin})")


if __name__ == "__main__":
    raise SystemExit(main())
