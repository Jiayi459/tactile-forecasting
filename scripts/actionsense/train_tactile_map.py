"""Cross-validate the tactile-map -> F/CoP forecaster (flatten vs CNN x history), probabilistic.

Mirrors the F/CoP probGRU protocol: 5-fold CV by recording, probabilistic (mean+var) head, sigma
calibration on a held-out VAL subset, skill-vs-persistence + coverage per channel & forecast step.
Writes a tidy CSV. Heavy (encoders x histories x folds models) -> intended for CRC GPU.

    python scripts/actionsense/train_tactile_map.py                       # full CV sweep (GPU recommended)
    python scripts/actionsense/train_tactile_map.py --folds 2 --epochs 5  # quick local smoke
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.actionsense.eval_harness.config import load_config          # noqa: E402
from src.actionsense.tactile_map import train as T                    # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tm-config", default="configs/actionsense/tactile_map.yaml")
    ap.add_argument("--encoders", default=None)
    ap.add_argument("--histories", default=None)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--csv", default="docs/actionsense/tactile_map_cv_results.csv")
    args = ap.parse_args()

    cfg = load_config()
    tmc = yaml.safe_load(open(args.tm_config))
    tm = {**tmc["preprocess"], **tmc["model"], **tmc["optim"]}
    if args.epochs:
        tm["epochs"] = args.epochs
    encoders = args.encoders.split(",") if args.encoders else tmc["sweep"]["encoders"]
    histories = [float(h) for h in args.histories.split(",")] if args.histories \
        else tmc["sweep"]["histories_s"]
    fps = cfg.fps
    recs = T.recordings(cfg, require_maps=True)
    ch = cfg.channels
    print(f"harness fps={fps:.0f} horizon={cfg.horizon}  {len(recs)} map recordings  "
          f"encoders={encoders} histories(s)={histories}  folds={args.folds} epochs={tm['epochs']}\n")

    os.makedirs(os.path.dirname(args.csv), exist_ok=True)
    fcsv = open(args.csv, "w", newline=""); w = csv.writer(fcsv)
    w.writerow(["encoder", "history_s", "forecast_step_s", *[f"{c}_skill" for c in ch],
                "mean_skill", "coverage_raw", "coverage_cal",
                *[f"{c}_hausdorff" for c in ch], "hausdorff_ratio_vs_persistence"])

    print(f"{'enc':8}{'hist':>5} | {'meanSkill':>10} {'covRaw':>7} {'covCal':>7}"
          f" | {'Hausdorff':>9}")
    print("-" * 42)
    for enc in encoders:
        for hist in histories:
            t_in = int(round(hist * fps))
            cv = T.cross_validate(cfg, tm, enc, t_in, recs, folds=args.folds)
            skc, sks = cv["skill_ch"], cv["skill_step"]
            cr, cc = cv["coverage_raw"], cv["coverage_cal"]
            hd_m = np.nanmean(cv["hausdorff_ch"], axis=0)          # (6,)
            hdr_m = np.nanmean(cv["hausdorff_ratio_ch"], axis=0)
            skc_m, sks_m = skc.mean(0), sks.mean(0)            # (6,), (H,6)
            print(f"{enc:8}{hist:>4.0f}s | {float(skc_m.mean()):>+10.3f} {cr:>7.2f} {cc:>7.2f}"
                  f" | Hausdorff {float(np.nanmean(hd_m)):>6.3f} "
                  f"({float(np.nanmean(hdr_m)):.2f}x persistence)")
            for st in range(cfg.horizon):
                w.writerow([enc, hist, round((st + 1) / fps, 2),
                            *[f"{sks_m[st, j]:.4f}" for j in range(6)],
                            f"{sks_m[st].mean():.4f}", f"{cr:.3f}", f"{cc:.3f}",
                            # per-channel Hausdorff is per FOLD, not per step, so the same
                            # value repeats down the step rows rather than pretending to
                            # resolve with the horizon
                            *[f"{hd_m[j]:.4f}" for j in range(6)],
                            f"{float(np.nanmean(hdr_m)):.4f}"])
            fcsv.flush()
    fcsv.close()
    print(f"\n[csv] {args.csv}")


if __name__ == "__main__":
    main()
