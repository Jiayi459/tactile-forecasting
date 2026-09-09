"""E2 integration on a synthetic 21x21 corpus: official-split trainer -> preds -> report.

torch-dependent: skips where torch cannot load (the dev Mac), runs on CRC. The E1 twin
(test_egotouch_baseline_pipeline.py) covers the numpy half everywhere.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
try:
    import torch  # noqa: F401
except Exception as exc:  # pragma: no cover
    pytest.skip(f"torch unavailable: {exc}", allow_module_level=True)

PY = sys.executable


def _corpus(tmp_path):
    states = tmp_path / "states"
    states.mkdir()
    rng = np.random.default_rng(0)
    rows, splits, idx = [], {"train": [], "val": [], "test": [], "test_unseen": []}, 0
    plan = [("grasp cola", "train", 5), ("grasp cola", "val", 2), ("grasp cola", "test", 2),
            ("fold towel", "train", 5), ("fold towel", "val", 2), ("fold towel", "test", 1),
            ("wipe window", "test_unseen", 2)]
    for label, split, n in plan:
        for _ in range(n):
            T = int(rng.integers(220, 400))
            clip = np.abs(rng.normal(size=(T, 2, 21, 21))).astype(np.float32)
            st = np.zeros((T, 2, 6), np.float32)
            st[:, :, 0] = clip.sum((2, 3))
            st[:, :, 1:3] = 0.2 * rng.normal(size=(T, 2, 2))
            np.save(states / f"state_{idx}.npy", st)
            np.save(states / f"clip_{idx}.npy", clip)
            rows.append({"idx": idx, "label": label, "cat": "x", "fps": 30.0, "T": T,
                         "features": 12, "has_clip": True})
            splits[split].append(idx)
            idx += 1
    with open(states / "manifest.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    json.dump(splits, open(states / "splits.json", "w"))
    (tmp_path / "harness.yaml").write_text(f"""
target:
  channels: [F_L, CoPx_L, CoPy_L, F_R, CoPx_R, CoPy_R]
  force_idx: [0, 3]
  cop_idx: [1, 2, 4, 5]
rate: {{fps_raw: 30.0, downsample: 3, horizon_s: 1.0}}
mask: {{percentile: 5}}
baselines:
  fit_scope: group
  ar_orders: [2]
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
    (tmp_path / "tm.yaml").write_text("""
preprocess: {baseline_frames: 0, alpha: 10.0}
model: {d: 16, hidden: 16}
optim: {epochs: 2, lr: 0.003, batch: 32, seed: 0}
sweep:
  encoders: [aggregate, flatten, cnn]
  histories_s: [1]
paths: {out_dir: unused}
""")
    return tmp_path


def test_official_split_trainer_end_to_end(tmp_path):
    root = _corpus(tmp_path)
    out = root / "runs"
    r = subprocess.run(
        [PY, os.path.join(ROOT, "scripts/egotouch/train_tactile_map.py"),
         "--harness-config", str(root / "harness.yaml"), "--tm-config", str(root / "tm.yaml"),
         "--pairs", "seq2seq/aggregate,seq2seq/cnn,probgru/aggregate",
         "--histories", "1", "--out-root", str(out)],
        capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    assert "OFFICIAL split: train=10 val=4 test_seen=3 test_unseen=2" in r.stdout
    assert "OTHER audit [test_unseen]: 2/2" in r.stdout     # 'wipe' never in TRAIN vocab

    rep = json.load(open(out / "selection_report.json"))
    assert set(rep["arms"]) == {"aggregate_seq2seq_1s", "cnn_seq2seq_1s", "aggregate_probgru_1s"}
    for a in rep["arms"].values():
        assert a["val_criterion"] == "clip_balanced"
        assert len(a["val_curves"]) == 2                    # both criteria, every epoch
        assert isinstance(a["selection_differs"], bool)

    seen = sorted((out / "test_seen_1s").glob("clip_*.npz"))
    unseen = sorted((out / "test_unseen_1s").glob("clip_*.npz"))
    assert len(seen) == 3 and len(unseen) == 2
    z = np.load(seen[0])
    mus = {k for k in z.files if k.startswith("mu_")}       # the three arms merged into one npz
    assert mus == {"mu_aggregate_seq2seq", "mu_cnn_seq2seq", "mu_aggregate_probgru"}
    assert z[next(iter(mus))].shape == (len(z["origins"]), 10, 6)
