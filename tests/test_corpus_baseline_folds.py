"""The corpus baselines must land on EXACTLY the folds the neural arms were trained under.

A baseline is only a reference if it never saw the recordings the arm it is compared against
was tested on. `export_baseline_forecasts.py --scope corpus` therefore reproduces the fold
assignment from `tactile_map.train.cross_validate` rather than drawing its own, and this file
pins that reproduction: the algorithm is a handful of seeded RNG calls, and a future edit to
either side could desynchronise them without any error being raised.

No torch import here -- the exporter does not need it, so neither should the test.
"""
import importlib.util
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

_PATH = os.path.join(os.path.dirname(__file__), "..", "scripts", "actionsense",
                     "export_baseline_forecasts.py")
_spec = importlib.util.spec_from_file_location("_ebf", _PATH)
EBF = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(EBF)


def reference_folds(recs, folds, seed):
    """cross_validate's partitioning, written out independently of the implementation."""
    fold_of = np.random.default_rng(seed).integers(0, folds, size=len(recs))
    out = []
    for f in range(folds):
        te = [recs[i] for i in range(len(recs)) if fold_of[i] == f]
        tr = [recs[i] for i in range(len(recs)) if fold_of[i] != f]
        if len(te) < 1 or len(tr) < 4:
            continue
        r2 = np.random.default_rng(seed * 100 + f)
        idx = r2.permutation(len(tr))
        nv = max(2, len(tr) // 6)
        out.append(([tr[i] for i in idx[nv:]], [tr[i] for i in idx[:nv]], te))
    return out


def test_folds_match_cross_validate():
    """The 299-recording corpus the runs actually used, and a couple of other sizes."""
    for n in (299, 100, 37):
        recs = list(range(n))
        got = list(EBF.corpus_folds(recs, 5, 0))
        want = reference_folds(recs, 5, 0)
        assert len(got) == len(want), (n, len(got), len(want))
        for (gt, gv, ge), (wt, wv, we) in zip(got, want):
            assert gt == wt and gv == wv and ge == we, n


def test_every_recording_is_held_out_exactly_once():
    recs = list(range(299))
    tests = [t for _, _, t in EBF.corpus_folds(recs, 5, 0)]
    flat = sorted(i for t in tests for i in t)
    assert flat == recs, "a recording was held out twice, or never"


def test_no_fold_trains_on_what_it_predicts():
    """The property the whole exercise exists for: no leakage into the reference."""
    recs = list(range(299))
    for trn, val, te in EBF.corpus_folds(recs, 5, 0):
        assert not (set(trn) & set(te)), "test recordings leaked into the baseline's fit set"
        assert not (set(val) & set(te)), "test recordings leaked into the selection set"
        assert not (set(trn) & set(val)), "fit and selection sets overlap"


def test_seed_actually_changes_the_partition():
    """A seed that is ignored would make the reproduction above vacuous."""
    recs = list(range(299))
    a = [t for _, _, t in EBF.corpus_folds(recs, 5, 0)]
    b = [t for _, _, t in EBF.corpus_folds(recs, 5, 1)]
    assert a != b
