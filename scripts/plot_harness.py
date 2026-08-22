"""Plot the frozen-harness results table (pure visualization; no evaluation).

Reads docs/actionsense/harness_baselines.csv (written by eval_harness/evaluate.py) and produces:
  docs/actionsense/harness_skill_bars.png   per-channel bar chart of FULL-horizon skill vs persistence
  docs/actionsense/harness_skill_curves.png per-horizon-step skill curves (x = lead time, y = SS)

    python scripts/plot_harness.py
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.actionsense.eval_harness.config import load_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="docs/actionsense/harness_baselines.csv")
    ap.add_argument("--ref", default="persistence", help="skill baseline to plot against")
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fps = load_config().fps
    df = pd.read_csv(args.csv)
    metric = f"SS_vs_{args.ref}"
    models = [m for m in ["seasonal", "ar"] if m in df.model.unique()]
    channels = list(dict.fromkeys(df.channel))          # preserve order

    # ---- (1) per-channel full-horizon skill bars ----
    agg = df[(df.horizon_step == "all") & (df.metric == metric)]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(channels)); w = 0.8 / max(len(models), 1)
    for k, m in enumerate(models):
        vals = [agg[(agg.model == m) & (agg.channel == c)].value.iloc[0] for c in channels]
        ax.bar(x + k * w, vals, w, label=m)
    ax.axhline(0, color="0.6", lw=.8)
    ax.set_xticks(x + w * (len(models) - 1) / 2); ax.set_xticklabels(channels)
    ax.set_ylabel(f"skill vs {args.ref}  (1 - MSE/MSE_ref)")
    ax.set_title("Full-horizon (1 s) skill by channel"); ax.legend(); ax.grid(alpha=.3, axis="y")
    fig.tight_layout(); fig.savefig("docs/actionsense/harness_skill_bars.png", dpi=120)
    print("[done] docs/actionsense/harness_skill_bars.png")

    # ---- (2) per-horizon-step skill curves, one subplot per channel ----
    steps = sorted(int(s) for s in df.horizon_step.unique() if s != "all")
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True)
    for ci, c in enumerate(channels):
        ax = axes[ci // 3, ci % 3]
        for m in models:
            sub = df[(df.model == m) & (df.channel == c) & (df.metric == metric)
                     & (df.horizon_step != "all")]
            sub = sub.assign(h=sub.horizon_step.astype(int)).sort_values("h")
            ax.plot(sub.h / fps, sub.value, "-o", ms=3, label=m)
        ax.axhline(0, color="0.6", lw=.8)
        ax.set_title(c); ax.grid(alpha=.3)
        if ci % 3 == 0:
            ax.set_ylabel(f"SS vs {args.ref}")
        if ci // 3 == 1:
            ax.set_xlabel("lead time (s)")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Skill vs lead time, per channel", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig("docs/actionsense/harness_skill_curves.png", dpi=120)
    print("[done] docs/actionsense/harness_skill_curves.png")

    # ---- (3) MSE and MAE error curves, one subplot per channel, one line per baseline ----
    all_models = [m for m in ["persistence", "seasonal", "ar"] if m in df.model.unique()]
    styles = {"persistence": dict(ls="-", lw=2.0), "seasonal": dict(ls="--", lw=2.4, alpha=.7),
              "ar": dict(ls="-", lw=2.0)}
    units = {"F_L": "sensor units^2", "F_R": "sensor units^2"}   # CoP -> grid^2
    for metric in ["MSE", "MAE"]:
        fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True)
        for ci, c in enumerate(channels):
            ax = axes[ci // 3, ci % 3]
            for m in all_models:
                sub = df[(df.model == m) & (df.channel == c) & (df.metric == metric)
                         & (df.horizon_step != "all")]
                sub = sub.assign(h=sub.horizon_step.astype(int)).sort_values("h")
                ax.plot(sub.h / fps, sub.value, marker="o", ms=3, label=m, **styles[m])
            u = units.get(c, "grid^2") if metric == "MSE" else (units.get(c, "grid").replace("^2", ""))
            ax.set_title(c); ax.grid(alpha=.3); ax.ticklabel_format(axis="y", scilimits=(-2, 3))
            if ci % 3 == 0:
                ax.set_ylabel(f"{metric} ({u})")
            if ci // 3 == 1:
                ax.set_xlabel("lead time (s)")
        axes[0, 0].legend(fontsize=8, title="(seasonal == persistence)")
        fig.suptitle(f"{metric} vs lead time, per channel (3 baselines)", fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        out = f"docs/harness_{metric.lower()}_curves.png"
        fig.savefig(out, dpi=120); print(f"[done] {out}")


if __name__ == "__main__":
    main()
