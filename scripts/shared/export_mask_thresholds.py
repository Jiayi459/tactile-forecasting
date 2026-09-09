#!/usr/bin/env python3
"""Fit the CoP mask thresholds on TRAIN and freeze them to a json the scorer consumes.

The harness convention (`mask:` in every eval_harness.yaml) is that a CoP frame is excluded
iff its hand's RAW force is below the TRAIN per-hand `percentile`th percentile -- fit on TRAIN
only. The shared scorer's `--mask corpus` instead re-estimates that threshold from whatever
set it is scoring, which is transductive and drifts with the population being scored
(W4-1, 2026-09-09). This script computes the thresholds ONCE, from TRAIN, through the very
function the harness itself uses (`dataset.force_thresholds`), and writes them next to the
states so every arm -- baselines and neural alike -- is masked identically via
`score_preds_per_action.py --thresholds`.

    python scripts/shared/export_mask_thresholds.py --config configs/egotouch/eval_harness.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.actionsense.eval_harness.config import load_config  # noqa: E402
from src.actionsense.eval_harness.dataset import force_thresholds, load_group  # noqa: E402
from src.actionsense.eval_harness.splits import load_splits  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="eval_harness.yaml of the corpus")
    ap.add_argument("--out", default=None,
                    help="output json (default: <states_root>/mask_thresholds.json)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    tr = load_splits(cfg)["train"]
    train = load_group(cfg, tr)
    thr = force_thresholds(cfg, train)                     # cfg.force_idx order
    names = [str(cfg.channels[i]) for i in cfg.force_idx]
    out = args.out or os.path.join(cfg.abspath("states_root"), "mask_thresholds.json")
    payload = {
        "force_thresholds": {n: float(t) for n, t in zip(names, thr)},
        "channels": [str(c) for c in cfg.channels],
        "percentile": cfg.raw["mask"]["percentile"],
        "fit_on": "train",
        "n_train_recordings": len(tr),
        "config": cfg.path,
        "config_hash": cfg.config_hash,
    }
    with open(out, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"wrote {out}: " + ", ".join(f"{n}={t:.6g}" for n, t in zip(names, thr))
          + f"  (TRAIN pct {payload['percentile']}, n={len(tr)} recordings, cfg {cfg.config_hash})")


if __name__ == "__main__":
    main()
