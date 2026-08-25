"""Per-CATEGORY tactile predictability probe (training-free) over EgoTouch.

Merges scripts/egotouch/categorize_actions.py (verb -> action category) with
scripts/egotouch/tactile_predictability_probe.py (persistence / dynamics metrics) to answer:
"which category of action is easiest to predict from its own past tactile signal?"

No training / no torch required. For every EgoTouch trajectory it loads the 21x21
L+R pressure grids and computes, per sequence:

  RAW HARDNESS (lower = easier to predict in absolute terms; the "raw accuracy" view)
    persistence_nmse[h] = MSE(y[t+h], y[t]) / Var(y)

  LEARNABLE / STRUCTURED SIGNAL (higher = more repeatable structure a forecaster can win)
    periodicity         = max autocorr of total force at lag in [PERIOD_LO..PERIOD_HI]
      -> rhythmic / repeatable actions (wipe, slice, cut) score high.
    (NB: a naive constant-velocity skill proxy was tried and discarded -- h*velocity
     extrapolation blows up on impulsive tactile spikes and is noise-dominated; a real
     skill number needs a trained model, which runs on the CRC GPU, not here.)

  CONTACT DYNAMICS
    contact_migration   = 1 - IoU(active-taxel mask at t, at t+h)  (0 = stable footprint)

  COMPOSITE
    predictability_index = z(-pers_nmse_h15) + z(periodicity) + z(-contact_migration)
      -> combines "slow to decorrelate", "repeatable", and "spatially stable".

Aggregates by verb category AND by temporal-pattern class (Axis B), then ranks.
Writes docs/actionsense/predictability_by_category.csv.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
from collections import defaultdict

import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.tactile_pixel.categories import categorize  # verb -> action category

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                    "datasets", "EgoTouch")
SCENES = ["Home", "Office", "Outdoor", "Retail", "Workbench"]
DOCS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "docs")

FPS = 30
HORIZONS = [1, 5, 15, 30]          # frames (33 / 167 / 500 / 1000 ms)
PERIOD_LO, PERIOD_HI = 10, 45      # lag window (0.33-1.5 s) for periodicity peak
ACTIVE_FRAC = 0.05                 # taxel "in contact" if > 5% of sequence max

# Axis B: temporal pattern class per verb category (a-priori hypothesis to be tested).
# B1 periodic  B2 quasi-static  B3 ramp/slide  B4 one-shot transition  B5 long composite
TEMPORAL_PATTERN = {
    "Grasp/Hold/Lift":        "B2 quasi-static",
    "Pick-up":                "B4 transition",
    "Place/Put-down":         "B4 transition",
    "Take/Retrieve":          "B4 transition",
    "Push/Pull/Drag/Slide":   "B3 ramp/slide",
    "Open/Close":             "B4 transition",
    "Fold/Cloth":             "B1 periodic",
    "Plug/Unplug/Insert":     "B4 transition",
    "Squeeze":                "B3 ramp/slide",
    "Pinch":                  "B2 quasi-static",
    "Twist/Turn/Rotate":      "B1 periodic",
    "Press/Click":            "B4 transition",
    "Spray":                  "B1 periodic",
    "Swing/Throw/Strike":     "B4 transition",
    "Play (games/sports)":    "B5 composite",
    "Wash/Clean":             "B1 periodic",
    "Cook/Prepare":           "B5 composite",
    "Organize/Arrange":       "B5 composite",
    "Cut":                    "B1 periodic",
    "Inflate/Deflate":        "B3 ramp/slide",
    "Use tool/appliance":     "B5 composite",
    "Buy/Shop":               "B5 composite",
    "Other":                  "Other",
}


def load_seq(pz):
    z = np.load(pz)
    L = np.nan_to_num(z["left_pressure_grid"], nan=0.0)
    R = np.nan_to_num(z["right_pressure_grid"], nan=0.0)
    return np.stack([L, R], axis=1).astype(np.float32)  # (T, 2, 21, 21)


def seq_metrics(s):
    """Return dict of scalar predictability metrics for one (T,2,21,21) sequence."""
    T = s.shape[0]
    var = float(np.var(s)) + 1e-8
    out = {}
    # persistence nMSE per horizon (raw hardness / decorrelation rate)
    for h in HORIZONS:
        if T <= h + 1:
            continue
        a, b = s[:-h], s[h:]
        out[f"pers_nmse_h{h}"] = float(np.mean((b - a) ** 2)) / var
    # periodicity: autocorr of total force at lag in [PERIOD_LO, PERIOD_HI]
    f = s.reshape(T, -1).sum(1)
    if f.std() > 1e-6:
        f0 = f - f.mean()
        best = 0.0
        for lag in range(PERIOD_LO, min(PERIOD_HI, T - 1) + 1):
            a, b = f0[:-lag], f0[lag:]
            denom = np.sqrt((a * a).sum() * (b * b).sum()) + 1e-8
            r = float((a * b).sum() / denom)
            best = max(best, r)
        out["periodicity"] = best
    # contact migration at h=15 (1 - IoU of active masks)
    h = 15
    if T > h:
        thr = ACTIVE_FRAC * float(s.max() + 1e-8)
        m0 = s[:-h] > thr
        m1 = s[h:] > thr
        inter = np.logical_and(m0, m1).sum(axis=(1, 2, 3))
        union = np.logical_or(m0, m1).sum(axis=(1, 2, 3)).astype(np.float64)
        iou = np.where(union > 0, inter / np.maximum(union, 1), 1.0)
        out["contact_migration"] = float(1.0 - iou.mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-per-task", type=int, default=15,
                    help="cap trajectories per task for speed (0 = all)")
    ap.add_argument("--min-frames", type=int, default=35)
    ap.add_argument("--out", default=os.path.join(DOCS, "predictability_by_category.csv"))
    args = ap.parse_args()

    # gather (category, temporal_pattern, npz_path)
    items = []
    for sc in SCENES:
        sp = os.path.join(ROOT, sc)
        if not os.path.isdir(sp):
            continue
        for t in sorted(os.listdir(sp)):
            tp = os.path.join(sp, t)
            if not os.path.isdir(tp) or t == "metadata":
                continue
            cat = categorize(t)
            tpat = TEMPORAL_PATTERN.get(cat, "Other")
            npzs = sorted(glob.glob(os.path.join(tp, "*", "pressure_grids.npz")))
            if args.max_per_task:
                npzs = npzs[:args.max_per_task]
            for p in npzs:
                items.append((cat, tpat, p))
    print(f"Probing {len(items)} trajectories "
          f"(max_per_task={args.max_per_task or 'all'}, min_frames={args.min_frames})")

    per_cat = defaultdict(list)
    per_pat = defaultdict(list)
    n_ok = 0
    for i, (cat, tpat, p) in enumerate(items):
        try:
            s = load_seq(p)
        except Exception as e:
            print(f"  skip {p}: {e}")
            continue
        if s.shape[0] < args.min_frames:
            continue
        m = seq_metrics(s)
        if not m:
            continue
        per_cat[cat].append(m)
        per_pat[tpat].append(m)
        n_ok += 1
        if (i + 1) % 250 == 0:
            print(f"  ...{i + 1}/{len(items)}")
    print(f"Usable sequences: {n_ok}\n")

    metric_keys = ["pers_nmse_h1", "pers_nmse_h15", "pers_nmse_h30",
                   "periodicity", "contact_migration"]

    def agg(groups):
        rows = {}
        for g, lst in groups.items():
            row = {"n": len(lst)}
            for k in metric_keys:
                vals = [d[k] for d in lst if k in d and np.isfinite(d[k])]
                row[k] = float(np.mean(vals)) if vals else float("nan")
            rows[g] = row
        return rows

    def add_index(rows):
        """predictability_index = z(-persH15) + z(periodicity) + z(-migration)."""
        def zcol(key, sign):
            vals = np.array([sign * rows[g][key] for g in rows], float)
            mu, sd = np.nanmean(vals), np.nanstd(vals) + 1e-8
            return {g: (sign * rows[g][key] - mu) / sd for g in rows}
        zp = zcol("pers_nmse_h15", -1.0)
        zper = zcol("periodicity", 1.0)
        zm = zcol("contact_migration", -1.0)
        for g in rows:
            rows[g]["pi"] = zp[g] + zper[g] + zm[g]
        return rows

    cat_rows = add_index(agg(per_cat))
    pat_rows = add_index(agg(per_pat))

    def show(title, rows):
        print(f"===== {title} (ranked by predictability_index PI, higher=easier) =====")
        hdr = f"{'group':<24}{'n':>5}{'persH1':>8}{'persH15':>9}{'persH30':>9}" \
              f"{'period':>8}{'migr15':>8}{'PI':>7}"
        print(hdr)
        print("-" * len(hdr))
        order = sorted(rows, key=lambda g: -rows[g]["pi"])
        for g in order:
            r = rows[g]
            print(f"{g:<24}{r['n']:>5}{r['pers_nmse_h1']:>8.3f}{r['pers_nmse_h15']:>9.3f}"
                  f"{r['pers_nmse_h30']:>9.3f}{r['periodicity']:>8.3f}"
                  f"{r['contact_migration']:>8.3f}{r['pi']:>7.2f}")
        print()

    show("BY TEMPORAL-PATTERN CLASS (Axis B)", pat_rows)
    show("BY ACTION CATEGORY (verb)", cat_rows)

    os.makedirs(DOCS, exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["grouping", "group", "n"] + metric_keys + ["predictability_index"])
        for g, r in sorted(pat_rows.items(), key=lambda kv: -kv[1]["pi"]):
            w.writerow(["temporal_pattern", g, r["n"]]
                       + [f"{r[k]:.5f}" for k in metric_keys] + [f"{r['pi']:.4f}"])
        for g, r in sorted(cat_rows.items(), key=lambda kv: -kv[1]["pi"]):
            w.writerow(["action_category", g, r["n"]]
                       + [f"{r[k]:.5f}" for k in metric_keys] + [f"{r['pi']:.4f}"])
    print(f"[done] wrote {args.out}")


if __name__ == "__main__":
    main()
