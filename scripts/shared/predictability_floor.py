#!/usr/bin/env python3
"""How much of a target's variation survives to the forecast horizon — for ANY of the sensors.

WHY THIS IS SHARED. d256's forecasts can look far more convincing than OpenTouch's while
scoring a third of its skill (SESSION_LOG 2026-08-28续2). Skill is a ratio against
persistence, so it moves with how hard persistence is to beat, not with how accurate the
model is -- and the two datasets' tables cannot settle which signal is actually harder,
because they use different estimators. `docs/skill_comparison.md` says so outright: the
frame-pooled and clip-balanced skills "disagree in sign" on F. Comparing them is a mistake I
made once.

This measures ONE quantity with ONE definition on whichever cache it is pointed at:

    R = E[(y[t+H] - y[t])^2] / (2 * Var(y))

For white noise E[(y[t+H]-y[t])^2] = 2*Var(y), so R ~ 1 means the horizon has decorrelated
the signal completely: persistence is then no better than predicting the mean, and it is
easy to beat. R << 1 means the signal barely moves over the horizon, persistence is close to
right, and beating it is hard. So R is, directly, how much room there is above persistence --
which is what a skill number is measured in.

rho1 is the lag-1 autocorrelation of the deviations, reported alongside because R alone
cannot separate "smooth" from "slow drift plus noise".

Works on any `state_N.npy` cache of shape (T, C, 6): d256 and ActionSense have C=2 hands,
OpenTouch C=1. The horizon comes from that arm's own frozen config, so each is measured at
its own 1 s.

    python scripts/shared/predictability_floor.py --config configs/d256/eval_harness.yaml
    python scripts/shared/predictability_floor.py --config configs/opentouch/eval_harness_d1.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.actionsense.eval_harness.config import load_config  # noqa: E402

MOMENTS = 3          # F, CoPx, CoPy per hand; the shear moments are not a forecast target


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--limit", type=int, default=None, help="recordings to read (default all)")
    ap.add_argument("--csv", default=None,
                    help="append rows [sensor,channel,R,rho1,n_recordings] here, so "
                         "build_skill_comparison.py can read the numbers instead of having "
                         "them transcribed into the document by hand")
    ap.add_argument("--sensor", default=None, help="name for the --csv rows (default: config stem)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    root = cfg.abspath("states_root")
    man = os.path.join(root, "manifest.jsonl")
    if not os.path.exists(man):
        sys.exit(f"{man} missing -- this arm's cache is not built here")
    rows = [json.loads(l) for l in open(man) if l.strip()]
    if args.limit:
        rows = rows[:args.limit]

    H, ds = cfg.horizon, cfg.downsample
    names = cfg.channels
    acc = {c: [] for c in range(len(names))}
    used = 0
    for r in rows:
        p = os.path.join(root, f"state_{r['idx']}.npy")
        if not os.path.exists(p):
            p = os.path.join(root, f"clip_{r['idx']}.npy")     # ActionSense's naming
            if not os.path.exists(p):
                continue
        st = np.load(p)[::ds]                                   # (T', C, 6)
        T, C, _ = st.shape
        if T <= H + 2:
            continue
        y = np.concatenate([st[:, h, :MOMENTS] for h in range(C)], axis=1).astype(np.float64)
        if y.shape[1] != len(names):
            sys.exit(f"state {r['idx']} gives {y.shape[1]} channels, config names {len(names)}")
        used += 1
        for c in range(y.shape[1]):
            x = y[:, c]
            d = x - x.mean()
            var = float(d.var())
            if var <= 1e-12:
                continue
            acc[c].append((float(np.mean((x[H:] - x[:-H]) ** 2)) / (2 * var),
                           float(np.mean(d[1:] * d[:-1]) / var)))

    print(f"{os.path.basename(args.config)}   fps {cfg.fps:g}  H {H} steps ({H/cfg.fps:.2f} s)"
          f"   {used} recordings")
    print(f"  {'channel':10s} {'R':>7s} {'rho1':>7s}   R~1 = decorrelated by the horizon "
          f"(persistence weak, easy to beat)")
    for c, name in enumerate(names):
        if not acc[c]:
            continue
        a = np.array(acc[c])
        print(f"  {name:10s} {a[:,0].mean():7.3f} {a[:,1].mean():7.3f}")
    allR = np.concatenate([np.array(v)[:, 0] for v in acc.values() if v])
    print(f"  {'MEAN':10s} {allR.mean():7.3f}")

    if args.csv:
        import csv as _csv
        sensor = args.sensor or os.path.splitext(os.path.basename(args.config))[0]
        new = not os.path.exists(args.csv)
        with open(args.csv, "a", newline="") as fh:
            w = _csv.writer(fh)
            if new:
                w.writerow(["sensor", "channel", "R", "rho1", "n_recordings"])
            for c, name in enumerate(names):
                if not acc[c]:
                    continue
                a = np.array(acc[c])
                w.writerow([sensor, name, round(float(a[:, 0].mean()), 4),
                            round(float(a[:, 1].mean()), 4), used])
        print(f"  appended {len(names)} rows -> {args.csv}")


if __name__ == "__main__":
    main()
