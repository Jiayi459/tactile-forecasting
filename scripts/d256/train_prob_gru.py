#!/usr/bin/env python3
"""Train the d256 probGRU over LOSO folds and score it on the harness's own origins.

Two arms (SESSION_LOG 2026-08-24, OQ-D5):
    --arm none    action embedding collapsed to one id -> tactile-from-tactile only.
                  This is the number comparable to the OpenTouch and ActionSense arms.
    --arm class   embedding = label_idx -> the model is told which activity is being done.
                  A different question; never report it beside the other two without saying so.
Run both and the difference is what knowing the activity is worth.

Scoring goes through src.d256.evaluate.score_external, so the GRU is compared to
persistence/seasonal/AR on identical origins, identical mask, identical metrics.

    python scripts/d256/train_prob_gru.py --arm none
    python scripts/d256/train_prob_gru.py --arm class --folds 0 --epochs 5     # smoke
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.actionsense.eval_harness.config import load_config  # noqa: E402
from src.d256 import evaluate as E  # noqa: E402
from src.d256 import prob_gru as PG  # noqa: E402
from src.d256 import splits as S  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join("configs", "d256", "eval_harness.yaml"))
    ap.add_argument("--arm", default="none", choices=["none", "class"])
    ap.add_argument("--folds", default=None, help="comma list; default all")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--t-in", type=int, default=None)
    ap.add_argument("--features", default=None, choices=["raw", "raw+df"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None, help="run dir (default runs/d256_probgru_<arm>)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    hp = {"arm": args.arm, "seed": args.seed}
    for k, v in (("epochs", args.epochs), ("t_in", args.t_in), ("features", args.features)):
        if v is not None:
            hp[k] = v

    out_dir = args.out or os.path.join("runs", f"d256_probgru_{args.arm}")
    os.makedirs(out_dir, exist_ok=True)

    fs = S.folds(cfg)
    print(S.summarize(cfg, fs))
    if args.folds:
        keep = {int(x) for x in args.folds.split(",")}
        fs = [f for f in fs if f["fold"] in keep]

    per_fold, histories = [], {}
    for f in fs:
        print(f"\n[fold {f['fold']}] held out {f['held_out']}  arm={args.arm}", flush=True)
        ctx = E.fold_context(cfg, f)
        base_results, _, _ = E.run_fold(cfg, f, ctx)

        model, norm, fnorm, aids, hist = PG.train(cfg, f, hp, device=args.device)
        preds = PG.forecast(model, cfg, f["test"], hp, norm, fnorm, aids, device=args.device)
        gru = E.score_external(cfg, f, "probgru", preds, ctx)

        results = {**base_results, "probgru": gru}
        per_fold.append((f, results))
        histories[f["fold"]] = hist

        sk = [float(np.asarray(gru["ch_mse"])[c] / base_results["persistence"]["ch_mse"][c])
              for c in range(len(cfg.channels))]
        print("    skill vs persistence: " +
              " ".join(f"{cfg.channels[c]} {1 - sk[c]:+.3f}" for c in range(len(cfg.channels))))
        torch_path = os.path.join(out_dir, f"fold{f['fold']}.pt")
        try:
            import torch
            torch.save({"state_dict": model.state_dict(), "hp": {**PG.DEFAULT_HP, **hp},
                        "config_hash": cfg.config_hash, "fold": f["fold"],
                        "held_out": f["held_out"]}, torch_path)
        except Exception as exc:                       # noqa: BLE001
            print(f"    WARN could not save checkpoint: {exc}")

    rows = E.build_rows(cfg, per_fold)
    import pandas as pd
    csv = os.path.join(out_dir, "metrics.csv")
    pd.DataFrame(rows).to_csv(csv, index=False)
    with open(os.path.join(out_dir, "history.json"), "w") as fh:
        json.dump({"arm": args.arm, "hp": {**PG.DEFAULT_HP, **hp},
                   "config_hash": cfg.config_hash, "folds": histories}, fh, indent=2)

    print(f"\nwrote {len(rows)} rows -> {csv}")
    print()
    print(E.summarize(cfg, rows))
    print("\nNOTE: arm 'none' is the cross-dataset-comparable number. arm 'class' answers "
          "'how predictable GIVEN the activity' and is not comparable to the other arms.")


if __name__ == "__main__":
    main()
