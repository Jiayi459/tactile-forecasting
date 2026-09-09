#!/usr/bin/env python3
"""EgoTouch reference ladder under the official split, exported for the ONE scorer (W6).

Fits seasonal + AR on the official TRAIN, selects on the official VAL
(recording-balanced, see baselines/ar.py), and writes per-recording forecast npz files --
the same layout `--save-preds` produces for the neural arms -- for test_seen and
test_unseen SEPARATELY. Persistence is not exported: the scorer synthesizes it from y and
origins, so it is identical for every arm by construction.

Why not evaluate.py: the harness's own CSV aggregates window-pooled, which the 2026-09-09
doctrine retired for anything that produces a claim. Exporting predictions and scoring them
through scripts/shared/score_preds_per_action.py --thresholds gives baselines and neural
models the same scorer, the same TRAIN-fitted mask, the same recordings, the same origins --
one scoring path, zero duplicated metric code.

Both fit scopes run (Q-K): `group` = (action, object), the primary; `global` = one pooled
"ALL" group, the robustness control -- EgoTouch's TRAIN has ~168 groups, many with 1-2
recordings, where ActionSense had 5. A group TRAIN never saw (all of test_unseen) routes to
the baselines' _GLOBAL fallback under the group scope.

    python scripts/egotouch/run_baselines.py                        # full
    python scripts/egotouch/run_baselines.py --arms ar --scopes group
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.actionsense.eval_harness.baselines.ar import AR  # noqa: E402
from src.actionsense.eval_harness.baselines.base import origins, predict_series  # noqa: E402
from src.actionsense.eval_harness.baselines.seasonal import SeasonalNaive  # noqa: E402
from src.actionsense.eval_harness.config import load_config  # noqa: E402
from src.actionsense.eval_harness.dataset import Norm, group_keys, load_group  # noqa: E402
from src.actionsense.eval_harness.splits import load_splits  # noqa: E402

ARMS = {"seasonal": SeasonalNaive, "ar": AR}


def export(out_dir: str, data: dict[int, np.ndarray], mus: dict[str, dict[int, np.ndarray]],
           verbs: dict[int, str], cfg) -> int:
    """One clip_<idx>.npz per recording with >=1 origin, mu_<arm> keys, overlay layout."""
    os.makedirs(out_dir, exist_ok=True)
    n = 0
    for i, Y in sorted(data.items()):
        ors = origins(len(Y), cfg)
        if not len(ors):
            continue
        arms = {f"mu_{name}": m[i] for name, m in mus.items() if i in m}
        np.savez_compressed(
            os.path.join(out_dir, f"clip_{i}.npz"),
            y=Y, origins=ors, fps=cfg.fps, action=verbs.get(i, ""), object_name="",
            channels=np.array([str(c) for c in cfg.channels]), tag="egotouch", **arms)
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/egotouch/eval_harness.yaml")
    ap.add_argument("--out-root", default="runs/egotouch_baselines")
    ap.add_argument("--arms", default="seasonal,ar")
    ap.add_argument("--scopes", default="group,global")
    args = ap.parse_args()

    cfg = load_config(args.config)
    sp = load_splits(cfg)
    tr, va, te = sp["train"], sp["val"], sp["test"]
    tu = sp.get("test_unseen", [])
    every = tr + va + te + tu
    groups = group_keys(cfg, every)

    print(f"[egotouch baselines] cfg {cfg.config_hash} ({cfg.path}) | "
          f"min_history={cfg.raw['eval']['min_history']} horizon={cfg.horizon}", flush=True)
    print(f"  official split: train={len(tr)} val={len(va)} test_seen={len(te)} "
          f"test_unseen={len(tu)}  (corpus {len(every)})", flush=True)

    train, val = load_group(cfg, tr), load_group(cfg, va)
    test, unseen = load_group(cfg, te), load_group(cfg, tu)
    for name, d in (("train", train), ("val", val), ("test_seen", test), ("test_unseen", unseen)):
        el = sum(1 for Y in d.values() if len(origins(len(Y), cfg)))
        print(f"  {name}: {el}/{len(d)} recordings yield >=1 origin", flush=True)
    norm = Norm.from_train(train)
    verbs = {i: parse_label_verb(groups[i]) for i in every}

    mus_seen: dict[str, dict[int, np.ndarray]] = {}
    mus_unseen: dict[str, dict[int, np.ndarray]] = {}
    for scope in args.scopes.split(","):
        gmap = dict(groups) if scope == "group" else {i: "ALL" for i in every}
        n_groups = len({gmap[i] for i in tr})
        for arm in args.arms.split(","):
            bl = ARMS[arm](cfg, norm)
            bl.fit(train, {i: gmap[i] for i in tr})
            bl.select(val, {i: gmap[i] for i in va}, cfg.horizon)
            if arm == "ar":
                import collections
                oc = collections.Counter(bl.order.values())
                print(f"  [{arm}/{scope}] {n_groups} TRAIN groups | selected orders: "
                      f"{dict(sorted(oc.items()))}", flush=True)
            else:
                fell = sum(1 for v in bl.periods.values() if v is None)
                print(f"  [{arm}/{scope}] {n_groups} TRAIN groups | "
                      f"{fell}/{len(bl.periods)} groups fell back to persistence", flush=True)
            for data, mus in ((test, mus_seen), (unseen, mus_unseen)):
                key = f"{arm}_{scope}"
                mus.setdefault(key, {})
                for i, Y in data.items():
                    _, yh = predict_series(bl, {i: Y}, {i: gmap[i]}, cfg)
                    if len(yh):
                        mus[key][i] = yh.astype(np.float32)

    n1 = export(os.path.join(args.out_root, "test_seen"), test, mus_seen, verbs, cfg)
    n2 = export(os.path.join(args.out_root, "test_unseen"), unseen, mus_unseen, verbs, cfg)
    print(f"  exported {n1} test_seen and {n2} test_unseen recordings -> {args.out_root}/"
          f" (arms: {sorted(mus_seen)})", flush=True)


def parse_label_verb(group: str) -> str:
    """group 'grasp-cola' -> 'grasp' (group_keys joins (action, object) with '-')."""
    return group.split("-")[0]


if __name__ == "__main__":
    main()
