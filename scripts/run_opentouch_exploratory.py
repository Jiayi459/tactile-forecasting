"""OpenTouch run: classical baselines + a GRU arm, over the whole corpus.

STILL EXPLORATORY, THOUGH LESS SO THAN IT WAS. It now defaults to the real location-level
split (src/opentouch/splits.py, 2026-08-15) rather than the ad-hoc one it was born with,
and --folds runs the grouped cross-validation. What keeps it exploratory is D1: the raw
pressure carries a large DC offset (F sits near 750k and moves by ~4%; CoP barely leaves
the sensor centre), so a forecaster is mostly being asked to predict a constant.
Persistence looks strong and skill reads high for reasons unrelated to dynamics. Every row
is tagged exploratory=True until that is settled.

WHAT IT DOES NOT DO, ON PURPOSE
  * No smooth/abrupt grouping. The user's ruling (2026-08-15) is to train on everything and
    separate only when scoring, and evaluate.trait_rows() does that -- but it refuses while
    any TEST action is unadjudicated, and 36 such actions remain. Per-class numbers must
    wait for those verdicts or they make the verdicts post-hoc.
  * No edit to configs/opentouch/eval_harness.yaml. config_hash is the hash of that file;
    editing it to repoint states_root would silently break comparability with every run
    that came before. Point the path with a symlink instead:
        ln -sfn ~/opentouch/cache data/opentouch_states

    python scripts/run_opentouch_exploratory.py --folds 4 --save-preds runs/preds
    python scripts/run_opentouch_exploratory.py --model none          # baselines only, fast
    python scripts/run_opentouch_exploratory.py --epochs 5 --max-clips 300   # smoke test
"""
from __future__ import annotations

import argparse
import collections
import csv
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.actionsense.eval_harness.config import load_config          # noqa: E402
from src.opentouch import evaluate as EV                             # noqa: E402
from src.opentouch import metrics                                    # noqa: E402
from src.opentouch.dataset import (eligible_clips, group_keys,       # noqa: E402
                                   missing_groups)


def adhoc_split(cfg, clips, field, seed, frac=(0.6, 0.2, 0.2)):
    """Hold out whole GROUPS, never individual clips.

    Clips from one scene share an environment, an object set and a participant's habits,
    so a random clip-level split would put near-duplicates on both sides and inflate every
    score. Grouping by `field` is the weakest defensible substitute for the real split
    while splits.py is blocked -- it is not equivalent to it.

    A holdout unit is then MOVED BACK INTO TRAIN if it carries an AR fit group that TRAIN
    would otherwise never see. dataset.missing_groups() documents why the caller has to do
    this: AR fits per object_category, so a category living entirely inside a held-out
    scene makes AR.predict raise KeyError deep in ar.py instead of failing here. Moving
    the unit keeps the holdout at scene granularity (no clip-level leakage); it does bias
    the split toward TRAIN, which is one more reason these numbers are exploratory.
    """
    by = collections.defaultdict(list)
    for r in clips:
        by[(r.get(field) or "unknown").strip()].append(r["idx"])
    units = sorted(by)
    random.Random(seed).shuffle(units)

    n = sum(len(v) for v in by.values())
    want = [frac[0] * n, (frac[0] + frac[1]) * n]
    assign, acc = {}, 0
    for u in units:
        assign[u] = "train" if acc < want[0] else ("val" if acc < want[1] else "test")
        acc += len(by[u])

    moved = []
    while True:
        tr = [i for u, b in assign.items() if b == "train" for i in by[u]]
        gtr = set(group_keys(cfg, tr).values())
        culprit = next((u for u, b in assign.items() if b != "train"
                        and set(group_keys(cfg, by[u]).values()) - gtr), None)
        if culprit is None:
            break
        moved.append(culprit)
        assign[culprit] = "train"

    out = collections.defaultdict(list)
    for u, b in assign.items():
        out[b] += by[u]
    return {k: sorted(out[k]) for k in ("train", "val", "test")}, len(units), moved


def emit_rows(cfg, model_name, R, ref_results, exploratory_tag):
    rows, H, chans = [], cfg.horizon, cfg.channels
    for ci, ch in enumerate(chans):
        n = R["n"][ci]
        for h in range(H):
            for metric, val in (("MSE", R["hz_mse"][h, ci]), ("MAE", R["hz_mae"][h, ci])):
                rows.append((model_name, ch, h + 1, metric, float(val), int(n)))
            for b, RB in ref_results.items():
                rows.append((model_name, ch, h + 1, f"SS_vs_{b}",
                             float(metrics.skill(R["hz_mse"][h, ci], RB["hz_mse"][h, ci])),
                             int(n)))
        for metric, val in (("MSE", R["ch_mse"][ci]), ("MAE", R["ch_mae"][ci])):
            rows.append((model_name, ch, "all", metric, float(val), int(n)))
        for b, RB in ref_results.items():
            rows.append((model_name, ch, "all", f"SS_vs_{b}",
                         float(metrics.skill(R["ch_mse"][ci], RB["ch_mse"][ci])), int(n)))
    return [dict(zip(("model", "channel", "horizon_step", "metric", "value", "n_frames"), r),
                 config_hash=cfg.config_hash, exploratory=True, split=exploratory_tag)
            for r in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/opentouch/eval_harness.yaml")
    ap.add_argument("--gru-config", default="configs/opentouch/gru_aggregate.yaml")
    ap.add_argument("--split-mode", default="location",
                    choices=["location", "adhoc", "random"],
                    help="location = src/opentouch/splits.py, the real protocol; adhoc = "
                         "the field-level fallback this script used before splits.py existed")
    ap.add_argument("--split-field", default="scene",
                    help="--split-mode adhoc only: manifest field held out as a whole")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, help="override the GRU epochs (smoke runs)")
    ap.add_argument("--histories", help="override the history sweep, e.g. 1,2,3 (seconds)")
    ap.add_argument("--max-clips", type=int, help="subsample the corpus (smoke runs)")
    ap.add_argument("--model", default="prob_gru",
                    choices=["prob_gru", "gru_aggregate", "both", "none",
                             "flatten", "cnn", "map_aggregate", "map_all"],
                    help="prob_gru = the ActionSense probabilistic GRU (architecture and "
                         "Gaussian NLL verbatim); gru_aggregate = the deterministic arm")
    ap.add_argument("--skip-gru", action="store_true", help="alias for --model none")
    ap.add_argument("--device", help="torch device for prob_gru (default: cuda if present)")
    ap.add_argument("--weight-decay", type=float, default=0.0,
                    help="Adam weight decay for prob_gru (0 = ActionSense's setting)")
    ap.add_argument("--dropout", type=float, default=0.0,
                    help="dropout on the prob_gru heads (0 = the verbatim architecture)")
    ap.add_argument("--baseline-scope", default="shard",
                    choices=["shard", "trainval", "train"],
                    help="clips the map arms' per-taxel baseline is estimated from. 'shard' "
                         "uses that shard's own frames, which is the only scope that yields "
                         "an estimate for a wholly held-out location; it is transductive in "
                         "the inputs and never in the targets.")
    ap.add_argument("--select-on", default="nll", choices=["nll", "mse"],
                    help="VAL curve that picks the probGRU weights and input history. The "
                         "harness scores point error only, so 'mse' aligns selection with "
                         "reporting; 'nll' is the default that reproduces earlier runs.")
    ap.add_argument("--features", default="raw", choices=["raw", "raw+df"],
                    help="prob_gru inputs: raw = ActionSense's five verbatim; raw+df adds "
                         "dF/dt, the one view of force that carries no DC (ablation)")
    ap.add_argument("--folds", type=int,
                    help="run k grouped folds (every location held out once) instead of "
                         "a single split; location mode only")
    ap.add_argument("--save-preds", metavar="DIR",
                    help="dump per-clip forecasts (mean, and sigma for prob_gru) so "
                         "scripts/plot_opentouch_forecast.py can draw them without retraining")
    ap.add_argument("--save-model", metavar="DIR",
                    help="checkpoint the probGRU per split (weights, hyperparameters, "
                         "action vocabulary and both normalizers -- everything needed to "
                         "reproduce a forecast without retraining)")
    ap.add_argument("--out", default="docs/exploratory_opentouch.csv")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.folds:
        if args.split_mode != "location":
            raise SystemExit("--folds requires --split-mode location")
        return run_folds(cfg, args)
    clips = eligible_clips(cfg)
    if args.max_clips and len(clips) > args.max_clips:
        clips = random.Random(args.seed).sample(clips, args.max_clips)
    if args.split_mode == "random":
        # WITHIN-location clip-level split, for one purpose only: telling memorisation
        # apart from location shift. Both make VAL degrade under the location split, and
        # they need different fixes. If VAL still degrades from epoch 2 when TRAIN and VAL
        # come from the SAME places, the model is memorising; if it flattens, what it
        # learned was location-specific and no regulariser will supply the missing
        # diversity. NEVER a protocol split -- clips from one scene are near-duplicates.
        ids = sorted(r["idx"] for r in clips)
        rng = random.Random(args.seed)
        rng.shuffle(ids)
        n = len(ids)
        splits = {"train": sorted(ids[: int(0.6 * n)]),
                  "val": sorted(ids[int(0.6 * n): int(0.8 * n)]),
                  "test": sorted(ids[int(0.8 * n):])}
        tag = f"random-clip-seed{args.seed}"
        print(f"DIAGNOSTIC SPLIT (leaky by construction): {tag} | "
              f"train {len(splits['train'])} val {len(splits['val'])} "
              f"test {len(splits['test'])}")
        print("  Use it to read the training curve, never to report a score.")
    elif args.split_mode == "location":
        from src.opentouch import splits as SP
        splits = SP.build(cfg, seed=args.seed)
        tag = f"location-seed{args.seed}"
        print(SP.summarize(cfg, splits))
    else:
        splits, n_units, moved = adhoc_split(cfg, clips, args.split_field, args.seed)
        tag = f"adhoc-{args.split_field}-seed{args.seed}"
        print(f"eligible clips {len(clips)} | {args.split_field} units {n_units} | "
              f"train {len(splits['train'])} val {len(splits['val'])} "
              f"test {len(splits['test'])}")
        if moved:
            print(f"  moved into TRAIN to cover AR groups it would never have seen: {moved}")
    # splits.build() returns metadata keys (locations, seed, frac, unit) alongside the three
    # id lists, so only the buckets may be measured here.
    if min(len(splits[b]) for b in ("train", "val", "test")) == 0:
        raise SystemExit("a split came out empty -- too few holdout units; "
                         "try --split-mode adhoc, or a different --seed")

    # The check dataset.missing_groups() exists for: AR fits per group and raises KeyError
    # deep inside ar.py if asked to score one it never fit. splits.build() already asserts
    # it; repeated here because --split-mode adhoc does not go through it.
    check_groups(cfg, splits)
    rows, results = run_split(cfg, splits, args, tag)
    write_and_report(cfg, rows, results, args.out, tag)
    return 0


def run_folds(cfg, args):
    """k grouped folds, every location held out once. Reports the spread across folds,
    which is the point: with 12 lumpy locations a single split's TEST is 2-3 places, and
    'would this hold on other environments' is answerable only by rotating them."""
    from src.opentouch import splits as SP
    fs = SP.folds(cfg, args.folds, args.seed)
    print(SP.summarize_folds(cfg, fs))

    all_rows, per_fold = [], []
    for f in fs:
        print(f"\n########## fold {f['fold']}/{args.folds - 1}  "
              f"test = {', '.join(f['locations']['test'])} ##########")
        check_groups(cfg, f)
        tag = f"location-k{args.folds}-fold{f['fold']}-seed{args.seed}"
        rows, results = run_split(cfg, f, args, tag)
        for r in rows:
            r["fold"] = f["fold"]
        all_rows += rows
        per_fold.append(results)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(all_rows[0]))
        w.writeheader(); w.writerows(all_rows)
    print(f"\nwrote {args.out}  ({len(all_rows)} rows, {args.folds} folds)")

    models = [m for m in per_fold[0]]
    print(f"\n=== skill vs persistence across {args.folds} folds "
          f"(mean [min, max]; EXPLORATORY) ===")
    print(f"{'model':16s} " + " ".join(f"{c:>22s}" for c in cfg.channels))
    for m in models:
        if m == "persistence":
            continue
        cells = []
        for ci in range(len(cfg.channels)):
            v = [metrics.skill(R[m]["ch_mse"][ci], R["persistence"]["ch_mse"][ci])
                 for R in per_fold]
            cells.append(f"{sum(v) / len(v):8.4f} [{min(v):6.3f},{max(v):6.3f}]")
        print(f"{m:16s} " + " ".join(cells))
    print("\nRead the spread, not just the mean: a model that wins on three locations and "
          "loses on the fourth has not been shown to generalize across environments.\n"
          "EXPLORATORY: F is DC-dominated (D1 unresolved).")
    return 0


def check_groups(cfg, splits):
    """AR fits per group and raises KeyError deep in ar.py if asked to score one it never
    fit. splits.build()/folds() already assert this; repeated because --split-mode adhoc
    does not go through them."""
    tr = splits["train"]
    gtr = group_keys(cfg, tr, tr)
    for part in ("val", "test"):
        miss = missing_groups(gtr, group_keys(cfg, splits[part], tr))
        if miss:
            raise SystemExit(f"{part} carries AR groups absent from train: {sorted(miss)}")


def run_split(cfg, splits, args, tag):
    print("fitting baselines (persistence / seasonal / ar) ...")
    results, norm, extras = EV.fit_and_forecast(cfg, splits)
    rows = []
    for m in EV.MODELS:
        rows += emit_rows(cfg, m, results[m], results, tag)

    ext, sig, kept = {}, {}, {}
    want = [] if (args.skip_gru or args.model == "none") else (
        ["prob_gru", "gru_aggregate"] if args.model == "both"
        else (["map_aggregate", "flatten", "cnn"] if args.model == "map_all"
              else [args.model]))
    for which in want:
        import torch  # noqa: F401  (imported late so --model none works without it)
        gcfg = load_config(args.gru_config)
        hs = ([float(x) for x in args.histories.split(",")] if args.histories
              else gcfg.raw["sweep"]["histories_s"])

        if which in ("flatten", "cnn", "map_aggregate"):
            # The tactile_map family: three encoders behind one probabilistic head, so the
            # encoder is the only variable (src/opentouch/tactile_map.py).
            from src.opentouch import tactile_map as TM
            enc = "aggregate" if which == "map_aggregate" else which
            hp = dict(TM.DEFAULT_HP)
            if args.epochs:
                hp["epochs"] = args.epochs
            print(f"{which}: history sweep {hs} s on VAL by NLL (epochs={hp['epochs']}) ...")
            scores = {}
            for sec in hs:
                ti = max(1, int(round(sec * cfg.fps)))
                *_, h = TM.train(cfg, enc, splits["train"], splits["val"], ti, hp,
                                 norm=norm, device=args.device,
                                 base_scope=args.baseline_scope)
                scores[ti] = h["best_val_nll"]
            t_in = min(scores, key=scores.get)
            # NLL, not args.select_on: this family scores its sweep by best_val_nll above,
            # and printing the flag's name here would label the number with a criterion that
            # did not produce it. The knob reaches prob_gru only.
            print(f"  t_in={t_in} ({t_in / cfg.fps:.1f} s); val NLL {scores}")
            model, _, mnorm, hist = TM.train(cfg, enc, splits["train"], splits["val"],
                                             t_in, hp, norm=norm, device=args.device,
                                             base_scope=args.baseline_scope)
            # Every arm in this family is probabilistic (OQ-G overturned globally,
            # 2026-08-19), so every arm's sigma is saved -- a variance that is trained and
            # never recorded cannot be checked against the errors it claims to describe.
            preds, sd = TM.predict_with_sigma(
                model, cfg, enc, norm, mnorm, splits["test"], t_in,
                TM.scope_ids(cfg, splits["train"], splits["val"], args.baseline_scope))
            if args.save_preds:
                sig[which] = sd
            print(f"  best val NLL {hist['best_val_nll']:.6f}")
        elif which == "gru_aggregate":
            from src.opentouch import gru_aggregate as G
            hp = dict(gcfg.raw["model"], **gcfg.raw["optim"])
            if args.epochs:
                hp["epochs"] = args.epochs
            print(f"gru_aggregate: history sweep {hs} s on VAL (epochs={hp['epochs']}) ...")
            t_in, scores = G.select_history(cfg, splits["train"], splits["val"], hs, hp)
            print(f"  t_in={t_in} ({t_in / cfg.fps:.1f} s); val MSE {scores}")
            model, _, hist = G.train(cfg, splits["train"], splits["val"], t_in, hp, norm=norm)
            preds = G.predict(model, cfg, norm, splits["test"], t_in)
            print(f"  best val MSE {hist['best_val_mse']:.6f}")
        else:
            # ActionSense's probGRU: its own hyperparameters (hidden 48 / 80 epochs), not
            # gru_aggregate.yaml's -- those belong to the deterministic aggregate model.
            from src.opentouch import prob_gru as P
            hp = dict(P.DEFAULT_HP)
            hp["features"] = args.features
            hp["weight_decay"] = args.weight_decay
            hp["dropout"] = args.dropout
            hp["select_on"] = args.select_on
            if args.epochs:
                hp["epochs"] = args.epochs
            print(f"prob_gru: history sweep {hs} s on VAL by {args.select_on.upper()} "
                  f"(epochs={hp['epochs']}) ...")
            t_in, scores, kept = P.select_history(cfg, splits["train"], splits["val"], hs,
                                                  hp, device=args.device,
                                                  keep=bool(args.save_preds))
            print(f"  t_in={t_in} ({t_in / cfg.fps:.1f} s); val NLL {scores}")
            model, _, fnorm, vocab, by_idx, hist = P.train(
                cfg, splits["train"], splits["val"], t_in, hp, norm=norm, device=args.device)
            preds = P.predict(model, cfg, norm, fnorm, vocab, by_idx, splits["test"], t_in)
            print(f"  best val NLL {hist['best_val_nll']:.6f} | "
                  f"action vocab {hist['n_actions']} (incl. 'other') | "
                  f"features {hist['features']} ({hist['n_features']} dims) | "
                  f"wd {hist['weight_decay']:g} drop {hist['dropout']:g} | "
                  f"weights chosen on VAL {hist['selected_on_metric'].upper()} "
                  f"(min NLL @ epoch {hist.get('best_val_nll_epoch', '?')}, "
                  f"min MSE @ epoch {hist.get('best_val_mse_epoch', '?')})")

        R = EV.score_external(cfg, splits, which, preds, results, norm)
        rows += emit_rows(cfg, which, R, results, tag)
        results[which] = R
        ext[which] = preds
        if which == "prob_gru" and args.save_model:
            import torch
            os.makedirs(args.save_model, exist_ok=True)
            ck = os.path.join(args.save_model, f"prob_gru_{tag}.pt")
            torch.save({"state_dict": model.state_dict(), "hp": hp, "t_in": t_in,
                        "vocab": vocab, "by_idx": by_idx,
                        "norm": {"mean": norm.mean, "std": norm.std},
                        "fnorm": {"mean": fnorm.mean, "std": fnorm.std},
                        "channels": list(cfg.channels), "config_hash": cfg.config_hash,
                        "split": tag, "history": hist, "sweep": scores}, ck)
            print(f"  saved model -> {ck}")
        if which == "prob_gru" and args.save_preds:
            from src.opentouch import prob_gru as P
            sig[which] = {i: P.predict_with_sigma(model, cfg, norm, fnorm, vocab,
                                                  by_idx, i, t_in)[1]
                          for i in splits["test"]}
            # Every swept history, not only the one VAL chose: these models are already
            # trained, so their forecasts cost a prediction pass and make the
            # rows-are-history figure drawable without training anything twice.
            for th, (mh, nh, fh, vh, bh, _) in (kept or {}).items():
                save_history_preds(cfg, splits, nh, P.predict(mh, cfg, nh, fh, vh, bh,
                                                              splits["test"], th),
                                   os.path.join(args.save_preds, f"hist_{th}"), th)

    if args.save_preds:
        save_predictions(cfg, splits, norm, ext, sig, args.save_preds, tag)
    return rows, results


def save_history_preds(cfg, splits, norm, preds, out_dir, t_in):
    """One directory per swept history length, holding just the signal and that model's mean.

    Deliberately thinner than save_predictions: the baselines do not depend on the GRU's
    history length, so recomputing them per length would be waste, and the figure these feed
    draws one model per subplot anyway."""
    from src.opentouch.dataset import load_group
    os.makedirs(out_dir, exist_ok=True)
    test = load_group(cfg, splits["test"])
    rows = {r["idx"]: r for r in eligible_clips(cfg, actions=())}
    from src.opentouch.baselines import origins
    for i, Y in test.items():
        np.savez_compressed(
            os.path.join(out_dir, f"clip_{i}.npz"),
            y=np.asarray(Y, dtype=np.float64),
            origins=np.asarray(origins(len(Y), cfg)), fps=cfg.fps, t_in=t_in,
            action=rows.get(i, {}).get("action", ""),
            object_name=rows.get(i, {}).get("object_name", ""),
            channels=np.array(cfg.channels), mu_prob_gru=preds[i])
    print(f"  saved history t_in={t_in} forecasts -> {out_dir}")


def save_predictions(cfg, splits, norm, ext, sig, out_dir, tag):
    """Dump per-clip forecasts so a plot can be drawn without retraining.

    The first full run wrote only the metric table, which meant the four hours of GPU time
    could not produce a single forecast curve. Baseline predictions are recomputed here
    with clip provenance (predict_series_by_clip) rather than reconstructed from the
    concatenated arrays evaluate.fit_and_forecast returns.
    """
    from src.opentouch.baselines import origins
    from src.opentouch.dataset import load_group

    os.makedirs(out_dir, exist_ok=True)
    test = load_group(cfg, splits["test"])
    tr_ids = splits["train"]
    gte = group_keys(cfg, splits["test"], tr_ids)
    gtr = group_keys(cfg, tr_ids, tr_ids)
    train, val = load_group(cfg, tr_ids), load_group(cfg, splits["val"])
    gva = group_keys(cfg, splits["val"], tr_ids)

    per_clip = {}
    for name in EV.MODELS:
        bl = EV.CLASSES[name](cfg, norm)
        bl.fit(train, gtr); bl.select(val, gva, cfg.horizon)
        _, yh, ids = EV.BL.predict_series_by_clip(bl, test, gte, cfg)
        at = 0
        for i, Y in sorted(test.items()):
            n = len(origins(len(Y), cfg))
            per_clip.setdefault(i, {})[name] = yh[at:at + n]
            at += n
    for name, d in ext.items():
        for i, v in d.items():
            per_clip.setdefault(i, {})[name] = v

    rows = {r["idx"]: r for r in eligible_clips(cfg, actions=())}
    for i, models in per_clip.items():
        np.savez_compressed(
            os.path.join(out_dir, f"clip_{i}.npz"),
            y=np.asarray(test[i], dtype=np.float64),
            origins=np.asarray(origins(len(test[i]), cfg)),
            fps=cfg.fps, action=rows.get(i, {}).get("action", ""),
            object_name=rows.get(i, {}).get("object_name", ""),
            channels=np.array(cfg.channels), tag=tag,
            **{f"mu_{k}": v for k, v in models.items()},
            **{f"sigma_{k}": v[i] for k, v in sig.items() if i in v})
    print(f"  saved {len(per_clip)} clips of forecasts -> {out_dir}")


def write_and_report(cfg, rows, results, out, tag):
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {out}  ({len(rows)} rows)")

    print(f"\n=== full-horizon per-channel MSE (EXPLORATORY, split={tag}) ===")
    print(f"{'model':16s} " + " ".join(f"{c:>12s}" for c in cfg.channels))
    for m, R in results.items():
        print(f"{m:16s} " + " ".join(f"{R['ch_mse'][ci]:12.5f}"
                                     for ci in range(len(cfg.channels))))
    print(f"\n=== skill vs persistence (>0 is better; EXPLORATORY) ===")
    print(f"{'model':16s} " + " ".join(f"{c:>12s}" for c in cfg.channels))
    for m, R in results.items():
        if m == "persistence":
            continue
        print(f"{m:16s} " + " ".join(
            f"{metrics.skill(R['ch_mse'][ci], results['persistence']['ch_mse'][ci]):12.4f}"
            for ci in range(len(cfg.channels))))
    print("\nEXPLORATORY: ad-hoc split, not the frozen protocol; not reportable. "
          "F is DC-dominated (D1 unresolved), so these favour persistence for reasons "
          "unrelated to dynamics.")


if __name__ == "__main__":
    raise SystemExit(main())
