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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/actionsense/eval_harness.yaml")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    cfg = load_config(a.config)
    sp = load_splits(cfg)
    train, val, test = (load_group(cfg, sp[k]) for k in ("train", "val", "test"))
    gtr, gva, gte = (group_keys(cfg, sp[k]) for k in ("train", "val", "test"))

    root = cfg.abspath("states_root")
    verbs = {}
    with open(os.path.join(root, "manifest.jsonl")) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                verbs[r["idx"]] = parse_label(r["label"])[0]

    # The baselines take the TRAIN-fitted Norm, as evaluate.fit_and_forecast constructs them;
    # AR in particular fits in normalised space.
    norm = Norm.from_train(train)

    per_clip: dict[int, dict] = {}
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
        print(f"  {name}: {len(per_clip)} recordings", flush=True)

    os.makedirs(a.out, exist_ok=True)
    for i, arrays in sorted(per_clip.items()):
        Y = test[i]
        np.savez_compressed(
            os.path.join(a.out, f"clip_{i}.npz"),
            y=np.asarray(Y, dtype=np.float64),
            origins=BL.origins(len(Y), cfg), fps=cfg.fps,
            action=verbs.get(i, ""), object_name="",
            channels=np.array(cfg.channels), tag="actionsense-baselines", **arrays)
    print(f"wrote {len(per_clip)} recordings x {len(EV.MODELS)} baselines -> {a.out}")


if __name__ == "__main__":
    raise SystemExit(main())
