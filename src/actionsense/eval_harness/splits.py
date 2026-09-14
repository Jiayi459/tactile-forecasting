"""Frozen train/val/test split — the ONE split every evaluation imports.

Split is BY RECORDING (clip idx) so both hands of a recording always land in the same
split (no leakage). It is stratified by (action, object) so each of the 5 dish-classes
(slice-cucumber/potato/bread, peel-cucumber/potato) is split 60/20/20 independently.
Deterministic given `split.seed`. Written once to split_file; loaded read-only after.

LEAKAGE RULE: all fitting uses TRAIN, hyperparameter selection uses VAL, and TEST is
touched exactly once by evaluate.py. This module never looks at signal values — it only
partitions recording indices — so it cannot leak.
"""
from __future__ import annotations

import json
import os

import numpy as np

from .config import Config


def parse_label(label: str) -> tuple[str, str]:
    """'Slice a cucumber' -> ('slice', 'cucumber'). object = last whitespace token."""
    toks = label.strip().split()
    return toks[0].lower(), toks[-1].lower()


def _manifest(cfg: Config) -> list[dict]:
    root = cfg.abspath("states_root")
    with open(os.path.join(root, "manifest.jsonl")) as f:
        return [json.loads(l) for l in f]


def eligible_recordings(cfg: Config) -> list[dict]:
    """Recordings whose label matches a target action AND are long enough to yield >=1
    forecast origin at the configured rate/history/horizon."""
    actions = tuple(a.lower() for a in cfg.raw["actions"])
    ds = cfg.downsample
    need = cfg.raw["eval"]["min_history"] + cfg.origin_horizon   # == horizon unless ablating
    out = []
    for r in _manifest(cfg):
        if not r["label"].lower().startswith(actions):
            continue
        if (r["T"] // ds) >= need:
            out.append(r)
    return out


def make_splits(cfg: Config) -> dict:
    """Build the stratified 60/20/20 split. Deterministic; does not touch signals."""
    recs = eligible_recordings(cfg)
    frac = cfg.raw["split"]
    rng = np.random.default_rng(frac["seed"])
    groups: dict[tuple[str, str], list[int]] = {}
    for r in recs:
        groups.setdefault(parse_label(r["label"]), []).append(r["idx"])
    train, val, test = [], [], []
    for key in sorted(groups):
        idx = np.array(sorted(groups[key]))
        idx = idx[rng.permutation(len(idx))]
        n = len(idx)
        n_tr = int(round(frac["train"] * n))
        n_va = int(round(frac["val"] * n))
        train += idx[:n_tr].tolist()
        val += idx[n_tr:n_tr + n_va].tolist()
        test += idx[n_tr + n_va:].tolist()
    return {
        "train": sorted(train), "val": sorted(val), "test": sorted(test),
        "n": len(recs), "seed": frac["seed"], "fractions": [frac["train"], frac["val"], frac["test"]],
        "stratify": "action,object",
    }


def load_splits(cfg: Config, rebuild: bool = False) -> dict:
    """Load the frozen split, creating split_file on first use."""
    path = cfg.abspath("split_file")
    if rebuild or not os.path.exists(path):
        sp = make_splits(cfg)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(sp, f, indent=2)
        return sp
    with open(path) as f:
        return json.load(f)
