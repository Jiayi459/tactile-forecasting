#!/usr/bin/env python3
"""EgoTouch forecaster sweep under the OFFICIAL split -- the single-split counterpart of
scripts/actionsense/train_tactile_map.py.

SAME MODELS, DIFFERENT PROTOCOL (design v1, SESSION_LOG 2026-09-08续7; rulings 2026-09-09):
the encoders, backbones and every optimization hyperparameter come verbatim from the shared
src/actionsense/tactile_map/ modules. What differs is forced by the dataset:

  * OFFICIAL SINGLE SPLIT, NOT 5-FOLD CV (Q-F0). Folds would mix the official test into
    training. Fit on train; checkpoint selection AND sigma calibration on the official val
    (both recording-balanced); test_seen and test_unseen predicted separately and never
    mixed into one directory.
  * THE MATRIX IS 8 MODELS (Q-F(b)): seq2seq x {aggregate, flatten, cnn} + probgru x
    aggregate, x histories {1, 3} s -- the combinations the other sensors actually ran.
  * in_shape (2, 21, 21) reaches the shared encoders as a constructor argument (Q-G).
  * baseline_frames = 0 -- the map input is released-as-is (OQ5).

The OTHER audit (W9) prints, per split, how many recordings ride on the near-untrained OTHER
action embedding -- ~52% of test_unseen by the official-split estimate, which is why the
unseen aggregate must be reported split by verb-known/verb-OTHER (only probgru consumes aids).

Selection report (Q-D): train_model records per-epoch val NLL under BOTH criteria; this driver
dumps selected_epoch vs selected_epoch_pooled per arm to selection_report.json, so one run
answers whether pooled selection would have kept a different checkpoint.

    python scripts/egotouch/train_tactile_map.py                     # full 8-model sweep
    python scripts/egotouch/train_tactile_map.py --pairs seq2seq/aggregate --histories 1 --epochs 3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.actionsense.eval_harness.config import load_config  # noqa: E402
from src.actionsense.eval_harness.dataset import Norm, load_target  # noqa: E402
from src.actionsense.eval_harness.splits import load_splits  # noqa: E402
from src.actionsense.tactile_map import data as D  # noqa: E402
from src.actionsense.tactile_map import train as T  # noqa: E402

# Both backbones x all three inputs. This was 4 arms until 2026-09-11, on the stated grounds
# that it matched "what the other sensors actually ran" -- which was wrong: OpenTouch's d1_pg
# run is literally "probGRU backbone, THREE input representations" (pg_cnn, pg_flatten,
# prob_gru), and ActionSense ran probGRU over aggregate and cnn. Dropping probgru x {cnn,
# flatten} also cost the backbone-vs-input comparison in docs/skill_comparison.md, which pairs
# each input's two backbones and cannot include EgoTouch without them.
DEFAULT_PAIRS = ("seq2seq/aggregate,seq2seq/flatten,seq2seq/cnn,"
                 "probgru/aggregate,probgru/flatten,probgru/cnn")


def other_audit(verbs, vocab, by_idx, splits: dict[str, list[int]]) -> dict[str, int]:
    """Per split: recordings whose verb rides the OTHER embedding. Printed, and returned for
    the report -- 'can accept an unseen id' is not 'generalizes', so the count must be loud."""
    out = {}
    for name, idxs in splits.items():
        n = sum(1 for i in idxs if D.aid_of(vocab, by_idx, i) == D.OTHER)
        out[name] = n
        print(f"  OTHER audit [{name}]: {n}/{len(idxs)} recordings on the OTHER embedding",
              flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--harness-config", default="configs/egotouch/eval_harness.yaml")
    ap.add_argument("--tm-config", default="configs/egotouch/tactile_map.yaml")
    ap.add_argument("--pairs", default=DEFAULT_PAIRS,
                    help="comma list of backbone/encoder (default: the ruled 8-model matrix)")
    ap.add_argument("--histories", default=None, help="comma seconds; default from tm-config")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-root", default="runs/egotouch_tactile_map")
    args = ap.parse_args()

    cfg = load_config(args.harness_config)
    tmc = yaml.safe_load(open(args.tm_config))
    tm = {**tmc["preprocess"], **tmc["model"], **tmc["optim"]}
    if args.epochs:
        tm["epochs"] = args.epochs
    histories = ([float(h) for h in args.histories.split(",")] if args.histories
                 else [float(h) for h in tmc["sweep"]["histories_s"]])
    pairs = [tuple(p.split("/")) for p in args.pairs.split(",")]

    sp = load_splits(cfg)
    tr, va, te = sp["train"], sp["val"], sp["test"]
    tu = sp.get("test_unseen", [])
    every = tr + va + te + tu
    print(f"[egotouch tactile_map] cfg {cfg.config_hash} | tm {args.tm_config} | "
          f"min_history={cfg.raw['eval']['min_history']} horizon={cfg.horizon} | "
          f"baseline_frames={tm['baseline_frames']} | val_criterion="
          f"{tm.get('val_criterion', 'clip_balanced')}", flush=True)
    print(f"  OFFICIAL split: train={len(tr)} val={len(va)} test_seen={len(te)} "
          f"test_unseen={len(tu)}  (corpus {len(every)})", flush=True)
    print(f"  matrix: {pairs} x histories {histories} s = {len(pairs) * len(histories)} models",
          flush=True)

    verbs = D.verbs_of(cfg, every)
    from src.actionsense.eval_harness.dataset import group_keys
    _g = group_keys(cfg, every)
    objects = {i: _g[i].split("-", 1)[1] if "-" in _g[i] else "" for i in every}
    vocab, by_idx = D.action_vocab(verbs, tr)
    print(f"  action vocab: {len(vocab)} ids ({len(vocab) - 1} verbs kept + OTHER)", flush=True)
    audit = other_audit(verbs, vocab, by_idx,
                        {"train": tr, "val": va, "test_seen": te, "test_unseen": tu})
    aids = {i: D.aid_of(vocab, by_idx, i) for i in every}

    os.makedirs(args.out_root, exist_ok=True)
    # Start from any existing report so an INCREMENTAL run -- adding arms to a sweep that
    # already wrote predictions into this directory -- does not erase the earlier arms'
    # selection records. save_predictions already merges the npz files that way; this makes
    # the report follow the same rule instead of silently disagreeing with them.
    rep_path = os.path.join(args.out_root, "selection_report.json")
    report = {"config_hash": cfg.config_hash, "other_audit": audit, "arms": {}}
    if os.path.exists(rep_path):
        with open(rep_path) as fh:
            prev = json.load(fh)
        if prev.get("config_hash") != cfg.config_hash:
            raise SystemExit(
                f"{rep_path} was written under config {prev.get('config_hash')} but this run "
                f"uses {cfg.config_hash}. Merging them would put two origin definitions in one "
                f"report; point --out-root somewhere fresh.")
        report["arms"] = prev.get("arms", {})
        print(f"  resuming report with {len(report['arms'])} arms already recorded", flush=True)
    t0 = time.time()
    for hist in histories:
        t_in = int(round(hist * cfg.fps))
        for backbone, encoder in pairs:
            arm = f"{encoder}_{backbone}"
            pg = backbone == "probgru"
            kw = dict(aids=aids, residual=not pg)
            tm_arm = {**tm, "backbone": backbone, "n_act": len(vocab)}

            if encoder == "aggregate":
                tnorm = Norm.from_train({i: load_target(cfg, i) for i in tr})
                mk = lambda ids: D.AggWindows({i: tnorm.z(load_target(cfg, i)) for i in ids},  # noqa: E731
                                              cfg, t_in, **kw)
                train_ds, val_ds = mk(tr), mk(va)
                test_ds, unseen_ds = mk(te), mk(tu)
            else:
                have = set(D.available_idxs(cfg, every))
                missing = [i for i in every if i not in have]
                if missing:
                    raise SystemExit(f"{len(missing)} recordings lack clip_<idx>.npy "
                                     f"(e.g. {missing[:5]}) -- re-run the extractor without "
                                     f"--no-clips rather than training on a silent subset")
                maps_tr, tgts_tr = D.load_raw(cfg, tr, tm["baseline_frames"])
                mnorm = D.MapNorm.from_train(maps_tr, tm["alpha"])
                tnorm = Norm.from_train(tgts_tr)
                tm_arm["in_shape"] = next(iter(maps_tr.values())).shape[1:]
                train_ds = D.MapWindows(D.normalize(maps_tr, mnorm),
                                        {i: tnorm.z(t) for i, t in tgts_tr.items()},
                                        cfg, t_in, **kw)
                mk = lambda ids: T._dataset(cfg, tm_arm, t_in, ids, mnorm, tnorm, **kw)  # noqa: E731
                val_ds, test_ds, unseen_ds = mk(va), mk(te), mk(tu)

            print(f"[{arm} t_in={t_in}] train {len(train_ds)} / val {len(val_ds)} / "
                  f"test_seen {len(test_ds)} / test_unseen {len(unseen_ds)} windows", flush=True)
            model = T.train_model(train_ds, val_ds, cfg, encoder, tm_arm, seed=args.seed,
                                  materialize=(encoder == "aggregate"))
            scale = T.calibrate_sigma(model, val_ds)
            report["arms"][f"{arm}_{hist:g}s"] = {
                "selected_epoch": model.selected_epoch,
                "selected_epoch_pooled": model.selected_epoch_pooled,
                "selection_differs": model.selection_differs,
                "val_criterion": model.val_criterion,
                "sigma_scale": scale,
                "val_curves": [[float(a), float(b)] for a, b in model.val_curves],
            }
            print(f"    selected epoch {model.selected_epoch} (pooled would pick "
                  f"{model.selected_epoch_pooled}; differs={model.selection_differs}) | "
                  f"sigma_scale {scale:.3f} | {time.time() - t0:.0f}s elapsed", flush=True)

            for name, ds in (("test_seen", test_ds), ("test_unseen", unseen_ds)):
                preds = T._per_recording(model, ds, tnorm, cfg.horizon)
                if preds:
                    out_dir = os.path.join(args.out_root, f"{name}_{hist:g}s")
                    T.save_predictions({arm: preds}, cfg, out_dir, verbs, objects)
            with open(rep_path, "w") as fh:
                json.dump(report, fh, indent=2)     # rewritten per arm: a crash loses nothing

    print(f"done: {len(report['arms'])} arms in {time.time() - t0:.0f}s -> {args.out_root}",
          flush=True)


if __name__ == "__main__":
    main()
