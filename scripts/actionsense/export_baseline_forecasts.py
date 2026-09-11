"""Write ActionSense's classical baselines into a prediction directory, per recording.

The tactile_map path trains neural arms and scores them, but never materialises the
harness's own persistence / seasonal / AR forecasts, so a figure of "every model" could
only ever show the neural ones. This produces them on the SAME rolling origins, in the
same npz layout, so `scripts/shared/merge_preds.py` can put all of them in one file.

FIT ON TRAIN, PREDICT ON TEST. The baselines are fitted on the harness TRAIN split and their
hyperparameters selected on VAL, exactly as evaluate.py does it; only test recordings are
written out. Fitting on everything would be faster and would quietly turn the reference these
figures are read against into an oracle.

    python scripts/actionsense/export_baseline_forecasts.py --out runs/as_preds_baselines
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.actionsense.eval_harness import evaluate as EV                # noqa: E402
from src.actionsense.eval_harness.baselines import base as BL          # noqa: E402
from src.actionsense.eval_harness.config import load_config            # noqa: E402
from src.actionsense.eval_harness.dataset import Norm, group_keys, load_group  # noqa: E402
from src.actionsense.eval_harness.splits import load_splits, parse_label  # noqa: E402



def corpus_recordings(cfg) -> list[int]:
    """Manifest recordings that have both a state file and a raw map.

    Inlined rather than imported from tactile_map.train, which pulls torch in for a list of
    integers this script otherwise never needs.
    """
    root = cfg.abspath("states_root")
    idxs = []
    with open(os.path.join(root, "manifest.jsonl")) as fh:
        for line in fh:
            if line.strip():
                i = int(json.loads(line)["idx"])
                if os.path.exists(os.path.join(root, f"state_{i}.npy")) \
                        and os.path.exists(os.path.join(root, f"clip_{i}.npy")):
                    idxs.append(i)
    return sorted(idxs)


def corpus_folds(recs: list[int], folds: int, seed: int):
    """Reproduce EXACTLY the folds tactile_map.train.cross_validate assigns.

    That function draws `fold_of` from default_rng(seed) over the recording list, then splits
    each fold's train part into val/trn with default_rng(seed*100 + f). Both are reproduced
    here rather than re-randomised, because a baseline fitted on a different partition than
    the arm it is the reference for is not a reference -- it would have seen recordings the
    arm was tested on. Yields (trn, val, te) per fold.
    """
    fold_of = np.random.default_rng(seed).integers(0, folds, size=len(recs))
    for f in range(folds):
        te = [recs[i] for i in range(len(recs)) if fold_of[i] == f]
        tr = [recs[i] for i in range(len(recs)) if fold_of[i] != f]
        if len(te) < 1 or len(tr) < 4:          # the same guard cross_validate applies
            continue
        r2 = np.random.default_rng(seed * 100 + f)
        idx = r2.permutation(len(tr))
        nv = max(2, len(tr) // 6)
        yield [tr[i] for i in idx[nv:]], [tr[i] for i in idx[:nv]], te


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/actionsense/eval_harness.yaml")
    ap.add_argument("--out", required=True)
    ap.add_argument("--scope", default="frozen", choices=["frozen", "corpus"],
                    help="frozen = the harness's slice+peel split, fit on TRAIN and written "
                         "for TEST. corpus = all map-bearing recordings under the SAME 5-fold "
                         "CV the neural arms used, so every recording gets a forecast from a "
                         "fold that did not train on it, and the baselines become comparable "
                         "to the corpus runs instead of covering only two of fourteen actions.")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    cfg = load_config(a.config)
    if a.scope == "frozen":
        sp = load_splits(cfg)
        partitions = [(sp["train"], sp["val"], sp["test"])]
    else:
        recs = corpus_recordings(cfg)
        partitions = list(corpus_folds(recs, a.folds, a.seed))
        n_te = sum(len(p[2]) for p in partitions)
        print(f"corpus scope: {len(recs)} recordings, {len(partitions)} folds, "
              f"{n_te} test forecasts (every recording held out exactly once)")

    root = cfg.abspath("states_root")
    verbs = {}
    with open(os.path.join(root, "manifest.jsonl")) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                verbs[r["idx"]] = parse_label(r["label"])[0]

    per_clip: dict[int, dict] = {}
    truth: dict[int, np.ndarray] = {}
    for fi, (tr_idx, va_idx, te_idx) in enumerate(partitions):
        train, val, test = (load_group(cfg, ids) for ids in (tr_idx, va_idx, te_idx))
        gtr, gva, gte = (group_keys(cfg, ids) for ids in (tr_idx, va_idx, te_idx))
        # The baselines take the TRAIN-fitted Norm, as evaluate.fit_and_forecast constructs
        # them; AR in particular fits in normalised space. Refitted per fold, matching
        # cross_validate, which builds a fresh Norm from each fold's trn subset.
        norm = Norm.from_train(train)
        truth.update(test)
        for name in EV.MODELS:
            bl = EV.CLASSES[name](cfg, norm)
            bl.fit(train, gtr)
            bl.select(val, gva, cfg.horizon)
            for i, Y in sorted(test.items()):
                ors = BL.origins(len(Y), cfg)
                if not len(ors):
                    continue
                yh = np.stack([bl.predict(Y[:t + 1], cfg.horizon, gte[i]) for t in ors])
                per_clip.setdefault(i, {})[f"mu_{name}"] = yh.astype(np.float64)
        tag = "frozen" if a.scope == "frozen" else f"fold {fi + 1}/{len(partitions)}"
        print(f"  [{tag}] fit on {len(tr_idx)} / selected on {len(va_idx)} / "
              f"wrote {len(test)}  (running total {len(per_clip)})", flush=True)

    os.makedirs(a.out, exist_ok=True)
    for i, arrays in sorted(per_clip.items()):
        Y = truth[i]
        np.savez_compressed(
            os.path.join(a.out, f"clip_{i}.npz"),
            y=np.asarray(Y, dtype=np.float64),
            origins=BL.origins(len(Y), cfg), fps=cfg.fps,
            action=verbs.get(i, ""), object_name="",
            channels=np.array(cfg.channels), tag="actionsense-baselines", **arrays)
    print(f"wrote {len(per_clip)} recordings x {len(EV.MODELS)} baselines -> {a.out}")


if __name__ == "__main__":
    raise SystemExit(main())
