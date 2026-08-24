"""Plot the tactile-map CV results (flatten vs CNN): skill vs history + coverage.

Pure visualization of docs/actionsense/tactile_map_cv_results.csv (written by scripts/actionsense/train_tactile_map.py,
the 5-fold probabilistic CV). Draws:
  docs/actionsense/tactile_map_skill_vs_history.png   mean skill vs persistence, flatten vs cnn, per history
  docs/actionsense/tactile_map_coverage.png           band coverage vs history (raw vs calibrated)

    python scripts/actionsense/plot_tactile_map.py
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="docs/actionsense/tactile_map_cv_results.csv")
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = list(csv.DictReader(open(args.csv)))
    encoders = sorted(set(r["encoder"] for r in rows))
    hists = sorted(set(float(r["history_s"]) for r in rows))
    by = defaultdict(list)
    for r in rows:
        by[(r["encoder"], float(r["history_s"]))].append(r)

    def mean_skill(e, h):                       # avg over forecast steps of the per-step mean skill
        return np.mean([float(r["mean_skill"]) for r in by[(e, h)]])

    def cov(e, h, key):
        return float(by[(e, h)][0][key])        # constant across steps

    # ---- (1) skill vs history ----
    fig, ax = plt.subplots(figsize=(8, 5))
    for e, c in zip(encoders, ["C0", "C1", "C2"]):
        ax.plot(hists, [mean_skill(e, h) for h in hists], "-o", color=c, lw=2, label=e)
    ax.axhline(0, color="0.5", lw=1, label="persistence")
    ax.set_xscale("log"); ax.set_xticks(hists); ax.set_xticklabels([f"{h:.0f}s" for h in hists])
    ax.set_xlabel("input history"); ax.set_ylabel("mean skill vs persistence (5-fold CV)")
    ax.set_title("Tactile-map -> F/CoP (probabilistic, 5-fold CV): CNN vs flatten")
    ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig("docs/actionsense/tactile_map_skill_vs_history.png", dpi=120)
    print("[done] docs/actionsense/tactile_map_skill_vs_history.png")

    # ---- (2) coverage vs history (raw solid, calibrated dashed) ----
    fig, ax = plt.subplots(figsize=(8, 5))
    for e, c in zip(encoders, ["C0", "C1", "C2"]):
        ax.plot(hists, [cov(e, h, "coverage_raw") for h in hists], "-o", color=c, lw=2, label=f"{e} raw")
        ax.plot(hists, [cov(e, h, "coverage_cal") for h in hists], "--", color=c, lw=2, alpha=.7)
    ax.axhline(0.95, color="r", ls=":", lw=1, label="ideal 0.95")
    ax.set_xscale("log"); ax.set_xticks(hists); ax.set_xticklabels([f"{h:.0f}s" for h in hists])
    ax.set_xlabel("input history"); ax.set_ylabel("coverage @ 2sd")
    ax.set_title("Band coverage (solid=raw, dashed=calibrated)")
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig("docs/actionsense/tactile_map_coverage.png", dpi=120)
    print("[done] docs/actionsense/tactile_map_coverage.png")


if __name__ == "__main__":
    main()
