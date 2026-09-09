"""The scorer's TRAIN-fitted mask path (W7): frozen thresholds, not the evaluated set's own."""
from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
_spec = importlib.util.spec_from_file_location(
    "scorer", os.path.join(ROOT, "scripts", "shared", "score_preds_per_action.py"))
scorer = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(scorer)

CH = ("F_L", "CoPx_L", "CoPy_L", "F_R", "CoPx_R", "CoPy_R")


def _y():
    y = np.ones((4, 3, 6))
    y[0, :, 0] = 0.0        # window 0: left hand out of contact
    y[1, :, 3] = 0.0        # window 1: right hand out of contact
    return y


def test_thresholds_mask_each_hand_by_its_own_force():
    m = scorer.build_mask(_y(), "none", CH, thresholds={"F_L": 0.5, "F_R": 0.5})
    assert m[:, :, [0, 3]].all()                       # force channels never masked
    assert not m[0, :, [1, 2]].any() and m[0, :, [4, 5]].all()   # left CoP out, right in
    assert not m[1, :, [4, 5]].any() and m[1, :, [1, 2]].all()


def test_thresholds_are_frozen_not_reestimated():
    """The same thresholds mask the same values identically whatever population is scored --
    the property `--mask corpus` lacks (its percentile moves with the evaluated set)."""
    thr = {"F_L": 0.5, "F_R": 0.5}
    a = scorer.build_mask(_y(), "none", CH, thresholds=thr)
    b = scorer.build_mask(np.concatenate([_y()] + [np.full((20, 3, 6), 9.0)]), "none", CH,
                          thresholds=thr)[: len(a)]
    np.testing.assert_array_equal(a, b)
    corpus_a = scorer.build_mask(_y(), "corpus", CH)
    corpus_b = scorer.build_mask(np.concatenate([_y()] + [np.full((20, 3, 6), 9.0)]),
                                 "corpus", CH)[: len(a)]
    assert not np.array_equal(corpus_a, corpus_b)      # the transductive mode really drifts


def test_thresholds_override_mask_mode():
    none_ = scorer.build_mask(_y(), "none", CH, thresholds={"F_L": 0.5, "F_R": 0.5})
    corp = scorer.build_mask(_y(), "corpus", CH, thresholds={"F_L": 0.5, "F_R": 0.5})
    np.testing.assert_array_equal(none_, corp)


def test_missing_force_channel_fails_loudly():
    with pytest.raises(SystemExit, match="F_R"):
        scorer.build_mask(_y(), "none", CH, thresholds={"F_L": 0.5})
