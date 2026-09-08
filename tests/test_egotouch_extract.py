"""Contracts for the EgoTouch extractor that cannot be checked once the data is on CRC.

Everything here runs on synthetic grids, so it is the only place the three EgoTouch-specific
decisions (no baseline correction, NaN-mask-only, official split join) are pinned down.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.actionsense.eval_harness.splits import parse_label  # noqa: E402
from src.actionsense.physical_state import frame_state  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "extract_egotouch_states", os.path.join(ROOT, "scripts", "egotouch", "extract_egotouch_states.py"))
ex = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ex)


def _npz(tmp_path, scene, task, stamp, left, right, tactile_max=1.0):
    d = tmp_path / scene / task / stamp
    d.mkdir(parents=True)
    np.savez(d / "pressure_grids.npz", left_pressure_grid=left,
             right_pressure_grid=right, tactile_max=np.float32(tactile_max))
    return d


def _grid(t=6, seed=0):
    return np.abs(np.random.default_rng(seed).normal(size=(t, ex.GRID, ex.GRID))).astype(np.float32)


# --- the split join -------------------------------------------------------------------

def test_traj_key_strips_hdf5_and_keeps_three_components():
    p = "/data_all/intern10/tmp/TouchAnything-Dev/datasets/x/Home/pick_up_bottle/20260314_001018_378.hdf5"
    assert ex.traj_key(p) == "Home/pick_up_bottle/20260314_001018_378"


def test_traj_key_agrees_between_split_path_and_trajectory_dir():
    """The join only works if an .hdf5 path and a trajectory DIRECTORY reduce to the same key."""
    from_split = ex.traj_key("/anything/at/all/Home/grasp_cola/20260412_101136_379.hdf5")
    from_dir = ex.traj_key("datasets/EgoTouch/Home/grasp_cola/20260412_101136_379")
    assert from_split == from_dir == "Home/grasp_cola/20260412_101136_379"


def test_load_official_split_rejects_a_file_missing_a_split(tmp_path):
    p = tmp_path / "split.json"
    p.write_text(json.dumps({"train": [], "val": [], "test_seen": []}))   # no test_unseen
    with pytest.raises(SystemExit, match="test_unseen"):
        ex.load_official_split(str(p))


# --- the grids ------------------------------------------------------------------------

def test_load_clip_masks_nan_to_zero_and_stacks_left_then_right(tmp_path):
    left, right = _grid(seed=1), _grid(seed=2)
    left[0, 0, 0] = np.nan
    d = _npz(tmp_path, "Home", "grasp_cola", "t0", left, right)
    clip = ex.load_clip(str(d / "pressure_grids.npz"))
    assert clip.shape == (6, ex.HANDS, ex.GRID, ex.GRID)
    assert np.isfinite(clip).all()
    assert clip[0, 0, 0, 0] == 0.0
    np.testing.assert_allclose(clip[:, 1], right)          # hand order is (L, R)


def test_load_clip_rejects_a_grid_that_is_not_21x21(tmp_path):
    d = _npz(tmp_path, "Home", "grasp_cola", "t0",
             np.zeros((4, 32, 32), np.float32), np.zeros((4, 32, 32), np.float32))
    assert ex.load_clip(str(d / "pressure_grids.npz")) is None


# --- the decision that is easiest to lose ---------------------------------------------

def test_no_baseline_correction_is_applied(tmp_path):
    """released-as-is: the state must equal per-frame moments of the RAW grid.

    If someone reinstates baseline_pct, a constant pedestal would be subtracted and this fails.
    """
    from src.actionsense.physical_state import clip_states
    clip = np.stack([_grid(seed=3), _grid(seed=4)], axis=1) + 5.0    # big DC pedestal
    got = clip_states(clip, baseline_pct=None)
    want = np.stack([frame_state(clip[t]) for t in range(clip.shape[0])])
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6)
    assert got[..., 0].min() > 0   # the pedestal survives; it was NOT removed


# --- the label trick the frozen harness depends on ------------------------------------

@pytest.mark.parametrize("task,want", [
    ("grasp_cola", ("grasp", "cola")),
    ("pick_up_bottle", ("pick", "bottle")),
    ("open_close_window", ("open", "window")),
])
def test_manifest_label_round_trips_through_parse_label(task, want):
    assert parse_label(task.replace("_", " ")) == want


# --- end to end -----------------------------------------------------------------------

def test_extract_writes_official_splits_and_drops_unassigned(tmp_path, monkeypatch, capsys):
    root, out = tmp_path / "EgoTouch", tmp_path / "states"
    for scene, task, stamp in [("Home", "grasp_cola", "t0"), ("Home", "grasp_cola", "t1"),
                               ("Retail", "pick_up_fruit", "t2"), ("Home", "drag_chair", "t3")]:
        _npz(root, scene, task, stamp, _grid(seed=hash(stamp) % 99), _grid(seed=7))
    (root / "split.json").write_text(json.dumps({
        "train": ["/x/Home/grasp_cola/t0.hdf5"],
        "val": ["/x/Home/grasp_cola/t1.hdf5"],
        "test_seen": ["/x/Retail/pick_up_fruit/t2.hdf5"],
        "test_unseen": [],                       # t3 is deliberately in NO split
    }))
    monkeypatch.setattr(sys, "argv", ["x", "--root", str(root), "--out", str(out)])
    ex.main()

    sp = json.loads((out / "splits.json").read_text())
    rows = [json.loads(l) for l in (out / "manifest.jsonl").read_text().splitlines()]
    by_traj = {r["traj"]: r for r in rows}

    # t3 is in no official split, so it is DROPPED entirely (OQ-B): absent from the manifest,
    # absent from disk, and it must not have consumed an index.
    assert len(rows) == 3 and "t3" not in by_traj
    assert sp["dropped_unassigned"] == 1
    assert json.loads((out / "unassigned_dropped.json").read_text()) == ["Home/drag_chair/t3"]
    assert sorted(r["idx"] for r in rows) == [0, 1, 2]
    assert not (out / "state_3.npy").exists()

    assert sp["train"] == [by_traj["t0"]["idx"]]
    assert sp["test"] == [by_traj["t2"]["idx"]]          # test == official test_seen
    assert sp["retention"]["test_seen"] == [1, 1] and sp["retention"]["test_unseen"] == [0, 0]
    assert by_traj["t0"]["label"] == "grasp cola" and by_traj["t0"]["fps"] == 30.0
    assert by_traj["t0"]["split"] == "train"
    assert np.load(out / f"state_{by_traj['t0']['idx']}.npy").shape == (6, 2, 6)
    assert np.load(out / f"clip_{by_traj['t0']['idx']}.npy").shape == (6, 2, 21, 21)
    assert "DROPPED (1)" in capsys.readouterr().out
