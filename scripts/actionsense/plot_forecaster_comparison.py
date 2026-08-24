"""Four-way forecaster comparison on the raw 6-dim F/CoP target (all same target/split/protocol).

Reads the CV result CSVs (aggregate + map encoders) and overlays the harness linear-AR reference.
    docs/actionsense/forecaster_comparison.png  mean skill vs persistence, per input-history.

    python scripts/actionsense/plot_forecaster_comparison.py
"""
from __future__ import annotations

import csv
from collections import defaultdict

import numpy as np

AR_SKILL = 0.166          # linear AR on the SAME 5-fold CV as the GRU (raw 6-dim; history-agnostic).
#                           (The frozen-split harness AR was +0.180; +0.166 is the protocol-matched value.)


def load(path):
    by = defaultdict(list)
    try:
        for r in csv.DictReader(open(path)):
            by[(r["encoder"], float(r["history_s"]))].append(float(r["mean_skill"]))
    except FileNotFoundError:
        pass
    return by


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = {**load("docs/actionsense/tactile_map_cv_results.csv"), **load("docs/actionsense/tactile_map_cv_results_aggregate.csv")}
    hists = [1.0, 3.0, 10.0]

    def skill(enc):
        return [np.mean(data.get((enc, h), [np.nan])) for h in hists]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.axhline(AR_SKILL, color="C3", ls="--", lw=2, label=f"linear AR (aggregate, 5-fold) = +{AR_SKILL:.3f}")
    for enc, c, lbl in [("aggregate", "C0", "GRU (aggregate F/CoP input)"),
                        ("cnn", "C1", "CNN (tactile map input)"),
                        ("flatten", "C2", "flatten (tactile map input)")]:
        ax.plot(hists, skill(enc), "-o", color=c, lw=2, label=lbl)
    ax.axhline(0, color="0.5", lw=1, label="persistence")
    ax.set_xscale("log"); ax.set_xticks(hists); ax.set_xticklabels([f"{h:.0f}s" for h in hists])
    ax.set_xlabel("input history"); ax.set_ylabel("mean skill vs persistence (5-fold CV)")
    ax.set_title("Forecasting the raw 6-dim F/CoP: linear AR > GRU-aggregate > CNN-map > flatten-map")
    ax.legend(fontsize=9); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig("docs/actionsense/forecaster_comparison.png", dpi=120)
    print("[done] docs/actionsense/forecaster_comparison.png")


if __name__ == "__main__":
    main()
