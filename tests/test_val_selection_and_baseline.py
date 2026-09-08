"""The two shared-code changes of 2026-09-08: released-as-is map input, clip-balanced selection.

Both touch the live ActionSense path, so each is pinned by a test that fails if the behaviour
drifts back. torch is not installed on the dev machine, so these skip locally and are meant to
be run on CRC -- see SESSION_LOG for the handover.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# Not importorskip: torch is INSTALLED but broken on the dev Mac (dlopen fails with OSError,
# which importorskip does not catch). Catch everything, so the suite stays green here and these
# tests actually run wherever torch works.
try:
    import torch  # noqa: F401
except Exception as exc:  # pragma: no cover
    pytest.skip(f"torch unavailable: {exc}", allow_module_level=True)

from src.actionsense.eval_harness.config import Config  # noqa: E402
from src.actionsense.tactile_map.data import load_map  # noqa: E402
from src.actionsense.tactile_map.train import _balanced, _rec_ids  # noqa: E402


def _cfg(states_root: str) -> Config:
    return Config(raw={"rate": {"fps_raw": 30.0, "downsample": 3, "horizon_s": 1.0},
                       "paths": {"states_root": states_root}},
                  path="synthetic", config_hash="0" * 8)


# --- released-as-is map input ---------------------------------------------------------

def test_zero_baseline_frames_leaves_the_map_untouched(tmp_path):
    """baseline_frames=0 must skip the subtraction, not take the mean of an empty axis.

    Before the fix this returned all-NaN with only a RuntimeWarning: min(0, len) == 0, so
    clip[:0].mean(0) is the mean of nothing. The map arms would have trained on NaN.
    """
    clip = np.abs(np.random.default_rng(0).normal(size=(12, 2, 21, 21))).astype(np.float32) + 1.0
    np.save(tmp_path / "clip_0.npy", clip)
    got = load_map(_cfg(str(tmp_path)), 0, baseline_frames=0)
    assert np.isfinite(got).all(), "released-as-is map input must not be NaN"
    np.testing.assert_allclose(got, clip[::3], rtol=0, atol=0)


def test_positive_baseline_frames_still_subtracts(tmp_path):
    """The ActionSense path is unchanged -- the fix must not silently disable it there."""
    clip = np.ones((12, 2, 21, 21), np.float32) * 3.0
    clip[6:] = 5.0
    np.save(tmp_path / "clip_0.npy", clip)
    got = load_map(_cfg(str(tmp_path)), 0, baseline_frames=2)
    assert got[0].max() == 0.0            # first frames sit at their own baseline
    assert got[-1].max() == pytest.approx(2.0)


# --- clip-balanced checkpoint selection -----------------------------------------------

def test_balanced_ignores_how_many_windows_a_recording_contributes():
    """One long recording must not outvote many short ones.

    Mirrors the real skew: recording 0 contributes 1000 windows, recordings 1..9 contribute
    one each. Pooled NLL is whatever recording 0 says; balanced is the mean of ten equals.
    """
    per = np.concatenate([np.full(1000, 10.0), np.full(9, 1.0)])
    rec = np.concatenate([np.zeros(1000, np.int64), np.arange(1, 10, dtype=np.int64)])
    pooled = per.mean()
    assert pooled == pytest.approx(9.919, abs=1e-3)      # the long recording owns it
    assert _balanced(per, rec) == pytest.approx((10.0 + 9 * 1.0) / 10)


def test_balanced_equals_pooled_when_every_recording_has_one_window():
    per = np.array([1.0, 2.0, 3.0, 4.0])
    rec = np.arange(4, dtype=np.int64)
    assert _balanced(per, rec) == pytest.approx(per.mean())


def test_rec_ids_follows_dataset_order():
    """_materialize stacks ds[0..n-1] in order, so the ids must line up with that order."""
    class FakeDS:
        index = [(7, 0), (7, 1), (9, 0)]
    np.testing.assert_array_equal(_rec_ids(FakeDS()), np.array([7, 7, 9]))
