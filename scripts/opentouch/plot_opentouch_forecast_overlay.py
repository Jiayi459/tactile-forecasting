"""Real signal vs each model's 1 s forecast — one figure per channel, OpenTouch.

Deliberately mirrors scripts/actionsense/plot_forecast_overlay.py, the ActionSense figure that writes
docs/forecast_{F,CoPx,CoPy}.png, so the two sensors can be read side by side without the
reader re-learning a layout. What is copied:

  * ONE FIGURE PER CHANNEL, named with the same tags (F, CoPx, CoPy), because F and CoP
    differ by orders of magnitude and sharing an axis wastes the range of both.
  * ONE MODEL PER SUBPLOT: real signal plus that one forecast. The original says why --
    overlaying every model in one axes is unreadable -- and it holds more here, where four
    models sit within a few percent of each other.
  * THE WHOLE CLIP, as rolling NON-OVERLAPPING 1 s forecasts stitched into a continuous
    line (the original's `range(t_in, T - t_out, t_out)`). A single origin shows one
    snapshot; this shows whether a model tracks the signal for the clip's whole duration,
    which is the question a forecaster is actually being asked.
  * x in seconds, y labelled with units, columns are test clips.

WHAT DIFFERS, AND WHY IT HAS TO. The ActionSense figure puts PAST-CONTEXT on the rows,
training one model per history length inside the script. Our runs train {1,2,3} s in
select_history but keep only the one VAL chose and discard the rest, so those forecasts do
not exist to plot. Rows are therefore the MODELS (persistence / seasonal / ar / prob_gru),
which is the same "one per subplot" idea applied to the axis we actually have. Saving the
sweep's predictions in a future run would let --rows history reproduce the original exactly;
it is nearly free, since those models are already trained.

Reads what --save-preds wrote. No training, no GPU.

    python scripts/opentouch/plot_opentouch_forecast_overlay.py --preds runs/preds --n-clips 3
    python scripts/opentouch/plot_opentouch_forecast_overlay.py --preds runs/preds \
        --compare runs/preds_df --compare-label raw+df
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np

# (filename tag, y-axis label with units) -- channel index comes from the saved `channels`
LABELS = {
    "F_R":    ("F",    "total force (sensor units, a.u.)"),
    "CoPx_R": ("CoPx", "CoP-x (normalized grid, -1..1)"),
    "CoPy_R": ("CoPy", "CoP-y (normalized grid, -1..1)"),
}
STYLE = {"prob_gru": "C3", "ar": "C0", "persistence": "0.45", "seasonal": "C2",
         "gru_aggregate": "C4"}


def rolling(z, model, H):
    """Non-overlapping 1 s forecasts across the whole clip -> (frame_idx, mu, sigma).

    The saved predictions are at every origin (stride 1); taking every H-th reproduces the
    original's non-overlapping walk instead of drawing H overlapping copies of each frame.

    sigma comes back too, and is None for the models that do not produce one. Plotting mu
    alone made the probabilistic models look like they had failed to represent the signal's
    fluctuation: after the D1 correction the target oscillates fast, mu is nearly flat -- as
    the MSE-optimal point forecast of a mostly-unpredictable signal must be -- and the band
    that says "and it wanders this far" was simply never drawn. Measured coverage is 96.6%,
    so the truth is inside the band almost always; a figure without it understates what the
    model knows and invites the wrong fix."""
    ors, mu = z["origins"], z[f"mu_{model}"]
    sg = z[f"sigma_{model}"] if f"sigma_{model}" in z.files else None
    if len(ors) == 0:
        return np.zeros(0, int), np.zeros((0, mu.shape[-1])), None
    sel = range(0, len(ors), H)
    idx = np.concatenate([np.arange(ors[j] + 1, ors[j] + 1 + H) for j in sel])
    val = np.concatenate([mu[j] for j in sel], 0)
    sig = np.concatenate([sg[j] for j in sel], 0) if sg is not None else None
    return idx, val, sig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default="runs/preds")
    ap.add_argument("--n-clips", type=int, default=3)
    ap.add_argument("--clips", help="explicit comma-separated clip indices")
    ap.add_argument("--band", action="store_true",
                    help="shade ±2σ where the model provides it (probGRU). Without it the "
                         "figure shows only the mean and understates what the model knows.")
    ap.add_argument("--sample", action="store_true",
                    help="also draw one trajectory drawn from the predictive distribution, "
                         "which is what should be compared to the real line's roughness")
    ap.add_argument("--min-seconds", type=float, default=3.0,
                    help="skip clips shorter than this; a two-second clip shows one segment")
    ap.add_argument("--compare", metavar="DIR", help="second arm, overlaid dashed")
    ap.add_argument("--compare-label", default="compare")
    ap.add_argument("--out-prefix", default="docs/opentouch_forecast")
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.preds, "clip_*.npz")),
                   key=lambda p: int(os.path.basename(p)[5:-4]))
    if not files:
        raise SystemExit(f"no clip_*.npz under {a.preds}")
    if a.clips:
        want = {int(c) for c in a.clips.split(",")}
        files = [f for f in files if int(os.path.basename(f)[5:-4]) in want]
    else:
        keep = []
        for f in files:
            z = np.load(f, allow_pickle=True)
            if len(z["origins"]) and len(z["y"]) / float(z["fps"]) >= a.min_seconds:
                keep.append(f)
            if len(keep) >= a.n_clips:
                break
        files = keep
    if not files:
        raise SystemExit("no clip long enough; lower --min-seconds")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z0 = np.load(files[0], allow_pickle=True)
    chans = [str(c) for c in z0["channels"]]
    models = sorted(k[3:] for k in z0.files if k.startswith("mu_"))
    H = z0[f"mu_{models[0]}"].shape[1]

    for k, ch in enumerate(chans):
        tag, ylabel = LABELS.get(ch, (ch, ch))
        fig, axes = plt.subplots(len(models), len(files), squeeze=False, sharex="col",
                                 figsize=(3.6 * len(files), 2.2 * len(models)))
        for ri, m in enumerate(models):
            for ci, path in enumerate(files):
                ax = axes[ri, ci]
                z = np.load(path, allow_pickle=True)
                y, fps = z["y"], float(z["fps"])
                tt = np.arange(len(y)) / fps
                ax.plot(tt, y[:, k], "k-", lw=1.3, label="real")
                idx, mu, sig = rolling(z, m, H)
                if sig is not None and a.band:
                    ax.fill_between(idx / fps, mu[:, k] - 2 * sig[:, k],
                                    mu[:, k] + 2 * sig[:, k],
                                    color=STYLE.get(m, "C1"), alpha=0.18, lw=0,
                                    label=f"{m} ±2σ")
                ax.plot(idx / fps, mu[:, k], "-", color=STYLE.get(m, "C1"), lw=1.3,
                        label=f"{m} 1 s forecast")
                if sig is not None and a.sample:
                    # One draw from the model's own predictive distribution. mu answers
                    # "where is it heading"; a draw shows the spread around that, and only
                    # the draw is on the same footing as the black line's roughness. Seeded
                    # per (clip, model) so redrawing the figure does not silently change it.
                    #
                    # The noise is INDEPENDENT PER FRAME, hence the label: the head predicts
                    # a diagonal Gaussian, so there is no joint distribution across the
                    # horizon to sample a trajectory from. The real signal is not white --
                    # measured r(1) is 0.318 for F -- so this draw is JAGGIER than any real
                    # second of data, and it is a picture of the marginal spread rather than
                    # of a plausible realisation.
                    rng = np.random.default_rng(abs(hash((os.path.basename(path), m))) % 2**32)
                    dr = mu[:, k] + sig[:, k] * rng.standard_normal(len(mu))
                    ax.plot(idx / fps, dr, "-", color=STYLE.get(m, "C1"), lw=0.7,
                            alpha=0.55, label=f"{m} draw (indep/frame)")
                ax.text(0.015, 0.93, m, transform=ax.transAxes, fontsize=8,
                        color=STYLE.get(m, "C1"), fontweight="bold", va="top")
                if a.compare:
                    p2 = os.path.join(a.compare, os.path.basename(path))
                    if os.path.exists(p2):
                        z2 = np.load(p2, allow_pickle=True)
                        if f"mu_{m}" in z2.files:
                            i2, m2, _ = rolling(z2, m, H)
                            ax.plot(i2 / fps, m2[:, k], "--", color="tab:brown", lw=1.1,
                                    label=f"{m} ({a.compare_label})")
                if ri == 0:
                    ax.set_title(f"clip {os.path.basename(path)[5:-4]} — {z['action']}",
                                 fontsize=9)
                if ci == 0:
                    ax.set_ylabel(f"{m}\n{ylabel}", fontsize=8)
                if ri == len(models) - 1:
                    ax.set_xlabel("time (s)", fontsize=9)
                # A legend PER ROW, not one for the figure. The figure-level version took
                # its handles from axes[0][0], so it named only the alphabetically first
                # model and the whole plot read as if it showed that one model alone.
                if ci == 0:
                    ax.legend(fontsize=6, loc="upper right")
                ax.grid(alpha=.25)
                ax.tick_params(labelsize=7)
        fig.suptitle(f"OpenTouch {tag}: real vs rolling 1 s forecast, one model per row",
                     fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        out = f"{a.out_prefix}_{tag}.png"
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        fig.savefig(out, dpi=110)
        plt.close(fig)
        print(f"[done] {out}")


if __name__ == "__main__":
    raise SystemExit(main())
