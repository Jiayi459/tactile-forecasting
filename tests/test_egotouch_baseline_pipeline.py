"""End-to-end E1 on a synthetic corpus: extractor-layout states -> run_baselines ->
export_mask_thresholds -> the shared scorer with --thresholds. All numpy; runs everywhere.

This is the integration contract for the single scoring path (W6/W7): baselines and the mask
thresholds flow through the same files and the same scorer the neural arms will use.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
PY = sys.executable


def _corpus(tmp_path):
    """Synthetic states dir + config: 3 seen groups + 1 unseen-only group, sine + noise."""
    states = tmp_path / "states"
    states.mkdir()
    rng = np.random.default_rng(0)
    rows, splits = [], {"train": [], "val": [], "test": [], "test_unseen": []}
    idx = 0
    plan = [("grasp cola", "train", 6), ("grasp cola", "val", 2), ("grasp cola", "test", 2),
            ("fold towel", "train", 6), ("fold towel", "val", 2), ("fold towel", "test", 2),
            ("wipe window", "test_unseen", 3)]           # group TRAIN never sees
    for label, split, n in plan:
        for _ in range(n):
            T = int(rng.integers(400, 700))              # raw frames @30Hz
            t = np.arange(T)
            st = np.zeros((T, 2, 6), np.float32)
            for hand in range(2):
                st[:, hand, 0] = 5 + np.sin(2 * np.pi * t / 90) + 0.1 * rng.normal(size=T)
                st[:, hand, 1:3] = 0.3 * np.sin(2 * np.pi * t[:, None] / 90 + hand) \
                    + 0.05 * rng.normal(size=(T, 2))
            np.save(states / f"state_{idx}.npy", st)
            rows.append({"idx": idx, "label": label, "cat": "x", "fps": 30.0, "T": T,
                         "features": 12, "has_clip": False})
            splits[split].append(idx)
            idx += 1
    with open(states / "manifest.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    json.dump(splits, open(states / "splits.json", "w"))

    cfg_path = tmp_path / "harness.yaml"
    cfg_path.write_text(f"""
target:
  channels: [F_L, CoPx_L, CoPy_L, F_R, CoPx_R, CoPy_R]
  force_idx: [0, 3]
  cop_idx: [1, 2, 4, 5]
rate: {{fps_raw: 30.0, downsample: 3, horizon_s: 1.0}}
mask: {{percentile: 5}}
baselines:
  fit_scope: group
  ar_orders: [2, 5]
  seasonal_period_min_s: 0.3
  seasonal_period_max_s: 3.0
  seasonal_min_autocorr: 0.1
split: {{train: 0.6, val: 0.2, test: 0.2, seed: 0}}
eval: {{stride: 1, min_history: 30, seed: 0}}
paths:
  states_root: {states}
  split_file: {states}/splits.json
  out_csv: {tmp_path}/out.csv
actions: [""]
""")
    return states, cfg_path


def test_baselines_export_and_shared_scorer_consume(tmp_path):
    states, cfg_path = _corpus(tmp_path)
    out_root = tmp_path / "runs"

    r = subprocess.run([PY, os.path.join(ROOT, "scripts/egotouch/run_baselines.py"),
                        "--config", str(cfg_path), "--out-root", str(out_root)],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    assert "test_unseen=3" in r.stdout          # the loud population line

    r = subprocess.run([PY, os.path.join(ROOT, "scripts/shared/export_mask_thresholds.py"),
                        "--config", str(cfg_path)], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    thr_file = states / "mask_thresholds.json"
    thr = json.load(open(thr_file))
    assert set(thr["force_thresholds"]) == {"F_L", "F_R"} and thr["fit_on"] == "train"

    # every exported recording carries all four arms over the harness origins
    seen = sorted((out_root / "test_seen").glob("clip_*.npz"))
    unseen = sorted((out_root / "test_unseen").glob("clip_*.npz"))
    assert len(seen) == 4 and len(unseen) == 3
    z = np.load(seen[0])
    arms = {k for k in z.files if k.startswith("mu_")}
    assert arms == {"mu_seasonal_group", "mu_ar_group", "mu_seasonal_global", "mu_ar_global"}
    n_or = len(z["origins"])
    for k in arms:
        assert z[k].shape == (n_or, 10, 6)

    # the unseen recordings went through the _GLOBAL fallback, not a KeyError
    zu = np.load(unseen[0])
    assert np.isfinite(zu["mu_ar_group"]).all()

    # and the ONE scorer scores both directories with the TRAIN-fitted mask
    for d in (out_root / "test_seen", out_root / "test_unseen"):
        r = subprocess.run([PY, os.path.join(ROOT, "scripts/shared/score_preds_per_action.py"),
                            "--preds", str(d), "--out", str(tmp_path / "scored"),
                            "--thresholds", str(thr_file)],
                           capture_output=True, text=True, cwd=ROOT)
        assert r.returncode == 0, r.stderr
    md = (tmp_path / "scored" / "test_seen.md").read_text()
    assert "TRAIN-fitted" in md                 # provenance lands in the report footer
