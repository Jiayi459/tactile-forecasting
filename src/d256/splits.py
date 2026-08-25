"""Leave-one-subject-out splits for d256 (user decision 2026-08-24, OQ-D3).

WHY LOSO, AND WHY d256 SPECIFICALLY. The shipped train/val is discarded: it covers 3 of 20
classes, holds out a different three per group, and puts S05 on both sides (docs/d256.md §7).
Of the three arms only d256 can do subject-held-out at all -- its manifest carries `subject`,
which ActionSense's does not (see the fit_scope note in configs/actionsense/eval_harness.yaml).
So this answers cross-person generalization, which the other two arms cannot.

THE SPLIT UNIT IS THE RECORDING, AND THAT IS NOT NEGOTIABLE. d256's shipped clips are stride-1
sliding windows sharing 15 of 16 frames, so any clip-level split puts near-duplicates on both
sides. Recordings (segments, as rebuilt by scripts/d256/extract_d256_states.py) are disjoint in
time; subjects are disjoint by construction. This module never reads signal values -- it only
partitions indices -- so it cannot leak.

VAL comes out of the TRAIN subjects, never the test subject: early stopping reads VAL, so a VAL
drawn from the held-out subject would select the checkpoint using the very person being tested.
"""
from __future__ import annotations

import json
import os

import numpy as np

from src.actionsense.eval_harness.config import Config
from .dataset import eligible_recordings, group_keys, manifest, missing_groups


def subjects(cfg: Config) -> list[str]:
    cfgd = cfg.raw["split"].get("subjects")
    return list(cfgd) if cfgd else sorted({r["subject"] for r in manifest(cfg)})


def _val_from_train(rows, train_subjects, val_frac, seed):
    """Hold out whole recordings from the TRAIN subjects for early stopping.

    Stratified by class so a fold's VAL cannot miss a class entirely just by shuffling, which
    matters here: the smallest classes have only a handful of recordings per subject.
    """
    rng = np.random.default_rng(seed)
    by_class: dict[int, list[int]] = {}
    for r in rows:
        if r["subject"] in train_subjects:
            by_class.setdefault(r["label_idx"], []).append(r["idx"])
    train, val = [], []
    for c in sorted(by_class):
        idx = np.array(sorted(by_class[c]))
        idx = idx[rng.permutation(len(idx))]
        n_val = int(round(val_frac * len(idx)))
        # Never take the last recording of a class into VAL -- TRAIN must still contain it,
        # or its AR group is unfitted and group_keys' "other" pooling is doing work it
        # should not have to do.
        n_val = min(n_val, max(0, len(idx) - 1))
        val += idx[:n_val].tolist()
        train += idx[n_val:].tolist()
    return sorted(train), sorted(val)


def folds(cfg: Config) -> list[dict]:
    """One fold per subject: that subject is TEST, the rest split into TRAIN/VAL."""
    rows = eligible_recordings(cfg)
    if not rows:
        raise ValueError("no eligible recordings -- check eval.min_history against "
                         "dataset.budget_table()")
    subs = subjects(cfg)
    sp = cfg.raw["split"]
    out = []
    for held in subs:
        test = sorted(r["idx"] for r in rows if r["subject"] == held)
        if not test:
            continue
        train_subs = [s for s in subs if s != held]
        train, val = _val_from_train(rows, set(train_subs), sp.get("val_frac", 0.2),
                                     sp.get("seed", 0))
        fold = {"fold": len(out), "held_out": held, "train_subjects": train_subs,
                "train": train, "val": val, "test": test,
                "n": len(rows), "protocol": "loso"}
        # AR is fit per group on TRAIN and then asked to score VAL/TEST. If a group reaches
        # scoring without a fit, baselines/ar.py raises KeyError deep in the run; assert here
        # instead, where the message names the fold.
        gt = group_keys(cfg, train, train_idxs=train)
        for name, part in (("val", val), ("test", test)):
            miss = missing_groups(gt, group_keys(cfg, part, train_idxs=train))
            if miss:
                raise ValueError(
                    f"fold {fold['fold']} (held out {held}): {name} contains AR groups TRAIN "
                    f"never fitted: {sorted(miss)}. Lower baselines.min_group_size or widen "
                    f"the 'other' pool in dataset.group_keys.")
        out.append(fold)
    return out


def summarize(cfg: Config, fs: list[dict]) -> str:
    rows = {r["idx"]: r for r in eligible_recordings(cfg)}
    lines = [f"LOSO, {len(fs)} folds, {len(rows)} eligible recordings "
             f"(min_history {cfg.raw['eval']['min_history']} + horizon {cfg.horizon})"]
    for f in fs:
        fr = lambda part: sum(rows[i]["T"] // cfg.downsample for i in part)   # noqa: E731
        ncls = lambda part: len({rows[i]["label_idx"] for i in part})          # noqa: E731
        lines.append(
            f"  fold {f['fold']} test={f['held_out']}: "
            f"train {len(f['train']):3d} rec / {fr(f['train']):6d} fr / {ncls(f['train']):2d} cls  "
            f"val {len(f['val']):3d} / {fr(f['val']):5d} / {ncls(f['val']):2d}  "
            f"test {len(f['test']):3d} / {fr(f['test']):5d} / {ncls(f['test']):2d}")
    return "\n".join(lines)


def save(cfg: Config, fs: list[dict], path: str | None = None) -> str:
    path = path or cfg.abspath("split_file")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({"protocol": "loso", "seed": cfg.raw["split"].get("seed", 0),
                   "config_hash": cfg.config_hash, "folds": fs}, f, indent=2)
    return path


def load(cfg: Config, path: str | None = None) -> list[dict]:
    with open(path or cfg.abspath("split_file")) as f:
        d = json.load(f)
    if d.get("config_hash") != cfg.config_hash:
        raise ValueError(
            f"split file was built under config_hash {d.get('config_hash')} but the config now "
            f"hashes to {cfg.config_hash}. eval.min_history or the rate changed, so the frozen "
            f"split no longer matches the protocol. Rebuild it deliberately.")
    return d["folds"]
