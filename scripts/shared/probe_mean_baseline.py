#!/usr/bin/env python3
"""READ-ONLY: would a trivial mean-reversion baseline beat the models we report?

WHY ASK. The predictability floor came back at R = 1.041 for OpenTouch and 0.717 for d256
(docs/d256/diagnostics/predictability_floor_all.txt). Since MSE_persistence = 2R*Var, a
predictor that simply emits a constant near the signal's own level has MSE ~ Var and would
score skill ~ 1 - 1/(2R): about 0.52 on OpenTouch and 0.30 on d256. Both are ABOVE the AR and
probGRU numbers those arms report (0.367/0.386 and 0.087/-0.093).

If that holds with a causal mean, the reported skills are not evidence of learned dynamics --
they are evidence that persistence is a weak reference at a 1 s horizon, and the baseline set
(persistence, seasonal, AR) is missing the one baseline that would have shown it.

WHAT MAKES THIS AN HONEST TEST. The estimate above uses each recording's OWN mean, which is
not available when the forecast is made. This scores two causal variants instead:

  hist_mean    mean of everything observed up to the origin, y[0..t]
  hist_mean_w  mean of the last `--window` frames only, which tracks slow drift

Both are computed from history alone, at exactly the harness origins, against exactly the
persistence reference the tables use. No training, no model, no fitting.

Writes nothing.

    python scripts/shared/probe_mean_baseline.py --config configs/d256/eval_harness.yaml
    python scripts/shared/probe_mean_baseline.py --config configs/opentouch/eval_harness_d1.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.actionsense.eval_harness.baselines.base import origins  # noqa: E402
from src.actionsense.eval_harness.config import load_config  # noqa: E402

MOMENTS = 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--window", type=int, default=None,
                    help="frames for the windowed mean (default: eval.min_history)")
    ap.add_argument("--limit", type=int, default=None)
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
    W = args.window or cfg.raw["eval"]["min_history"]
    names = cfg.channels

    # Sum of squared errors, pooled over every (origin, horizon step) -- the frame-pooled
    # convention the driver tables use, so these numbers sit on the same scale as their skill.
    sse = {k: np.zeros(len(names)) for k in ("persistence", "hist_mean", "hist_mean_w")}
    n = 0
    used = 0
    for r in rows:
        p = os.path.join(root, f"state_{r['idx']}.npy")
        if not os.path.exists(p):
            p = os.path.join(root, f"clip_{r['idx']}.npy")
            if not os.path.exists(p):
                continue
        st = np.load(p)[::ds]
        T, C, _ = st.shape
        y = np.concatenate([st[:, h, :MOMENTS] for h in range(C)], axis=1).astype(np.float64)
        if y.shape[1] != len(names):
            continue
        o = origins(T, cfg)
        if not len(o):
            continue
        used += 1
        csum = np.cumsum(y, axis=0)
        for t in o:
            truth = y[t + 1:t + 1 + H]
            preds = {
                "persistence": y[t],
                "hist_mean": csum[t] / (t + 1),
                "hist_mean_w": y[max(0, t - W + 1):t + 1].mean(0),
            }
            for k, v in preds.items():
                sse[k] += ((truth - v[None, :]) ** 2).sum(0)
            n += H

    if not n:
        sys.exit("no origins -- check eval.min_history against the recordings")
    base = sse["persistence"]
    print(f"{os.path.basename(args.config)}   {used} recordings, {n} scored frames, "
          f"H={H}, window={W}")
    print(f"  {'baseline':14s} " + " ".join(f"{c:>9s}" for c in names))
    for k in ("hist_mean", "hist_mean_w"):
        sk = 1 - sse[k] / base
        print(f"  {k:14s} " + " ".join(f"{v:9.3f}" for v in sk))
    print(f"  {'(reference)':14s} " + " ".join(f"{0.0:9.3f}" for _ in names)
          + "   <- persistence, by definition")
    print("\n  positive => this trivial predictor beats persistence. Compare against the AR and")
    print("  probGRU skills in docs/skill_comparison.md: anything they do not clear is a")
    print("  baseline they should have been measured against.")


if __name__ == "__main__":
    main()
