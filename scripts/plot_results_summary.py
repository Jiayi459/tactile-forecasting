"""Plot the action-dynamics results CSV (skill by history / forecast-step / channel / hand).

Pure visualization of docs/actionsense/action_dynamics_results.csv (written by train_action_dynamics.py) —
no training. Four panels:
  (a) mean skill vs history-length, one line per (input_mode, hand)
  (b) per-channel skill (F, CoP-x, CoP-y) vs history, for one input_mode/hand
  (c) skill vs forecast-step (horizon curve), one line per history, for one input_mode/hand
  (d) coverage vs history (raw, and calibrated if present)

    python scripts/plot_results_summary.py
    python scripts/plot_results_summary.py --focus-mode raw --focus-hand right
"""
from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="docs/actionsense/action_dynamics_results.csv")
    ap.add_argument("--focus-mode", default="raw", help="input_mode for panels (b)/(c)")
    ap.add_argument("--focus-hand", default="right", help="hand for panels (b)/(c)")
    ap.add_argument("--out", default="docs/actionsense/results_summary.png")
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = list(csv.DictReader(open(args.csv)))
    modes = sorted(set(r["input_mode"] for r in rows))
    hands = sorted(set(r["hand"] for r in rows))
    hists = sorted(set(float(r["history_s"]) for r in rows))
    steps = sorted(set(float(r["forecast_step_s"]) for r in rows))
    has_cal = "coverage_cal" in rows[0]
    f = float
    by = defaultdict(list)                       # (mode,hand,hist) -> rows
    for r in rows:
        by[(r["input_mode"], r["hand"], f(r["history_s"]))].append(r)

    def mean_skill(mode, hand, h):               # avg over steps of the per-step mean skill
        return np.mean([f(r["mean_skill"]) for r in by[(mode, hand, h)]])

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # (a) skill vs history, per (mode, hand)
    ax = axes[0, 0]
    for mode in modes:
        for hand in hands:
            ax.plot(hists, [mean_skill(mode, hand, h) for h in hists], "-o", ms=4,
                    label=f"{mode}/{hand}")
    ax.set_xlabel("past-context (s)"); ax.set_ylabel("mean skill vs persistence")
    ax.set_title("(a) skill vs history — input_mode × hand"); ax.legend(fontsize=8); ax.grid(alpha=.3)

    # (b) per-channel skill vs history for the focus config
    ax = axes[0, 1]
    for j, ch in enumerate(["F_skill", "x_skill", "y_skill"]):
        vals = [np.mean([f(r[ch]) for r in by[(args.focus_mode, args.focus_hand, h)]]) for h in hists]
        ax.plot(hists, vals, "-o", ms=4, label={"F_skill": "F (force)", "x_skill": "CoP-x",
                                                "y_skill": "CoP-y"}[ch])
    ax.set_xlabel("past-context (s)"); ax.set_ylabel("skill")
    ax.set_title(f"(b) per-channel skill — {args.focus_mode}/{args.focus_hand}")
    ax.legend(fontsize=8); ax.grid(alpha=.3)

    # (c) skill vs forecast-step, per history, focus config
    ax = axes[1, 0]
    for h in hists:
        rs = sorted(by[(args.focus_mode, args.focus_hand, h)], key=lambda r: f(r["forecast_step_s"]))
        ax.plot([f(r["forecast_step_s"]) for r in rs], [f(r["mean_skill"]) for r in rs],
                "-o", ms=3, label=f"{h:.0f}s past")
    ax.axhline(0, color="0.7", lw=.8)
    ax.set_xlabel("forecast lead time (s)"); ax.set_ylabel("mean skill")
    ax.set_title(f"(c) skill vs horizon — {args.focus_mode}/{args.focus_hand}")
    ax.legend(fontsize=8); ax.grid(alpha=.3)

    # (d) coverage vs history
    ax = axes[1, 1]
    covkey = "coverage_raw" if has_cal else "coverage"
    for mode in modes:
        for hand in hands:
            ax.plot(hists, [np.mean([f(r[covkey]) for r in by[(mode, hand, h)]]) for h in hists],
                    "-o", ms=4, label=f"{mode}/{hand} raw")
    if has_cal:
        for mode in modes:
            for hand in hands:
                ax.plot(hists, [np.mean([f(r["coverage_cal"]) for r in by[(mode, hand, h)]]) for h in hists],
                        "--", alpha=.6)
    ax.axhline(0.95, color="r", lw=1, ls=":", label="ideal 0.95")
    ax.set_xlabel("past-context (s)"); ax.set_ylabel("coverage@2sd")
    ax.set_title("(d) band coverage" + (" (solid=raw, dashed=calibrated)" if has_cal else ""))
    ax.legend(fontsize=7); ax.grid(alpha=.3)

    fig.suptitle(f"Action-dynamics results — {os.path.basename(args.csv)}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=120)
    print(f"[done] {args.out}")


if __name__ == "__main__":
    main()
