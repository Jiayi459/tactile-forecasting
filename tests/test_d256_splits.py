"""LOSO split invariants for d256, on a fixture with the REAL recording geometry.

The lengths/subjects/classes below are the actual 166 recordings the extractor produced from
the corpus (SESSION_LOG 2026-08-25), so a fold that is degenerate on the real data is
degenerate here too. Signal values are synthetic -- these tests are about index partitioning,
which is the only thing splits.py is allowed to touch.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from src.actionsense.eval_harness.config import load_config
from src.d256 import dataset as D
from src.d256 import splits as S


def requires_torch():
    """Skip when torch cannot actually be loaded.

    Not pytest.importorskip: a broken-but-present install (missing dylib) raises OSError, not
    ImportError, so importorskip lets it through and the test fails for an environment reason
    rather than a code one.
    """
    try:
        import torch  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"torch unusable here: {type(exc).__name__}")

# (subject, label_idx, T) per recording -- the real geometry.
GEOM = [
    ("S01", 0, 357), ("S01", 0, 22), ("S01", 1, 243), ("S01", 1, 35), ("S01", 1, 71),
    ("S01", 3, 153), ("S01", 3, 77), ("S01", 4, 260), ("S01", 4, 77), ("S01", 4, 37),
    ("S01", 5, 144), ("S01", 5, 122), ("S01", 6, 217), ("S01", 7, 117), ("S01", 7, 25),
    ("S01", 7, 53), ("S01", 8, 80), ("S01", 8, 16), ("S01", 9, 105), ("S01", 9, 39),
    ("S01", 10, 53), ("S01", 10, 19), ("S01", 10, 20), ("S01", 11, 57), ("S01", 11, 28),
    ("S01", 12, 56), ("S01", 12, 17), ("S01", 13, 62), ("S01", 13, 38), ("S01", 14, 61),
    ("S01", 14, 29), ("S01", 15, 657), ("S01", 16, 647), ("S01", 17, 330), ("S01", 18, 1068),
    ("S01", 19, 1022), ("S02", 0, 290), ("S02", 0, 287), ("S02", 0, 131), ("S02", 1, 301),
    ("S02", 1, 64), ("S02", 2, 168), ("S02", 2, 122), ("S02", 2, 89), ("S02", 3, 205),
    ("S02", 3, 65), ("S02", 3, 93), ("S02", 4, 332), ("S02", 4, 76), ("S02", 5, 120),
    ("S02", 5, 33), ("S02", 6, 114), ("S02", 7, 72), ("S02", 7, 48), ("S02", 7, 69),
    ("S02", 8, 75), ("S02", 8, 36), ("S02", 9, 77), ("S02", 10, 35), ("S02", 10, 19),
    ("S02", 10, 34), ("S02", 11, 58), ("S02", 11, 16), ("S02", 12, 59), ("S02", 12, 31),
    ("S02", 13, 65), ("S02", 14, 56), ("S02", 14, 17), ("S02", 15, 1014), ("S02", 16, 685),
    ("S02", 17, 203), ("S02", 18, 859), ("S02", 19, 851), ("S03", 0, 227), ("S03", 0, 18),
    ("S03", 0, 24), ("S03", 0, 238), ("S03", 2, 42), ("S03", 7, 280), ("S03", 8, 156),
    ("S03", 8, 48), ("S03", 8, 52), ("S03", 9, 67), ("S03", 10, 50), ("S03", 10, 23),
    ("S03", 11, 93), ("S03", 11, 19), ("S03", 12, 86), ("S03", 12, 20), ("S03", 13, 98),
    ("S03", 13, 26), ("S03", 14, 83), ("S03", 15, 677), ("S03", 16, 651), ("S03", 17, 222),
    ("S03", 18, 667), ("S03", 19, 788), ("S04", 0, 195), ("S04", 0, 317), ("S04", 1, 273),
    ("S04", 1, 39), ("S04", 2, 106), ("S04", 2, 76), ("S04", 2, 117), ("S04", 3, 223),
    ("S04", 3, 79), ("S04", 3, 25), ("S04", 4, 359), ("S04", 4, 78), ("S04", 5, 150),
    ("S04", 5, 41), ("S04", 5, 20), ("S04", 6, 131), ("S04", 7, 89), ("S04", 7, 18),
    ("S04", 7, 16), ("S04", 8, 90), ("S04", 8, 16), ("S04", 9, 91), ("S04", 10, 45),
    ("S04", 11, 27), ("S04", 11, 34), ("S04", 12, 36), ("S04", 12, 25), ("S04", 13, 30),
    ("S04", 13, 21), ("S04", 14, 39), ("S04", 15, 681), ("S04", 16, 535), ("S04", 17, 268),
    ("S04", 18, 658), ("S04", 19, 1059), ("S05", 0, 218), ("S05", 0, 349), ("S05", 1, 225),
    ("S05", 1, 109), ("S05", 3, 98), ("S05", 3, 65), ("S05", 5, 68), ("S05", 5, 84),
    ("S05", 5, 69), ("S05", 7, 73), ("S05", 7, 56), ("S05", 7, 34), ("S05", 8, 106),
    ("S05", 8, 22), ("S05", 8, 30), ("S05", 9, 115), ("S05", 9, 28), ("S05", 10, 46),
    ("S05", 11, 92), ("S05", 12, 166), ("S05", 13, 64), ("S05", 14, 62), ("S05", 14, 28),
    ("S05", 15, 785), ("S05", 16, 996), ("S05", 17, 199), ("S05", 18, 972), ("S05", 19, 987),
    ("S05", 2, 183), ("S05", 2, 150), ("S05", 4, 325), ("S05", 4, 34), ("S05", 6, 88),
    ("S05", 6, 45),
]


@pytest.fixture(scope="module")
def cfg(tmp_path_factory):
    root = tmp_path_factory.mktemp("d256_states")
    rng = np.random.default_rng(0)
    with open(root / "manifest.jsonl", "w") as fh:
        for i, (subj, cls, T) in enumerate(GEOM):
            fh.write(json.dumps({"idx": i, "label": f"class {cls}", "label_idx": cls,
                                 "subject": subj, "session": cls, "group": "signals1",
                                 "orig_split": "train", "fps": 6.0, "T": T}) + "\n")
            st = np.zeros((T, 2, 6), dtype=np.float32)
            t = np.arange(T) / 6.0
            for h in range(2):
                st[:, h, 0] = 400 + 40 * np.sin(2 * np.pi * 0.4 * t + h) + rng.normal(0, 3, T)
                st[:, h, 1] = 0.05 * np.sin(2 * np.pi * 0.5 * t + h)
                st[:, h, 2] = 0.05 * np.cos(2 * np.pi * 0.3 * t + h)
            np.save(root / f"state_{i}.npy", st)

    src = open("configs/d256/eval_harness.yaml").read()
    path = root / "cfg.yaml"
    path.write_text(src.replace("states_root: data/d256_states", f"states_root: {root}")
                       .replace("split_file: data/d256_states/splits.json",
                                f"split_file: {root}/splits.json"))
    return load_config(str(path))


def test_geometry_matches_the_real_extraction(cfg):
    rows = D.manifest(cfg)
    assert len(rows) == 166
    assert sum(r["T"] for r in rows) == 30916      # cross-checks the segment model
    assert len({r["label_idx"] for r in rows}) == 20


def test_config_min_history_keeps_most_recordings(cfg):
    """The 4 s history is provisional in the config; this pins what it actually costs.

    136, not the 138 recorded on 2026-08-25: that count came from the off-by-one eligibility
    test, which admitted the two 30-frame segments that yield no windows. The budget table in
    SESSION_LOG was computed the same way and is high by two for every row.
    """
    assert len(D.eligible_recordings(cfg)) == 136


def test_folds_are_disjoint_and_subject_pure(cfg):
    for f in S.folds(cfg):
        tr, va, te = set(f["train"]), set(f["val"]), set(f["test"])
        assert not (tr & va) and not (tr & te) and not (va & te)
        rows = {r["idx"]: r for r in D.eligible_recordings(cfg)}
        # The held-out subject must not appear in TRAIN *or* VAL: VAL drives early stopping,
        # so leaking the test subject into it selects the checkpoint on the person being tested.
        assert f["held_out"] not in {rows[i]["subject"] for i in tr | va}
        assert {rows[i]["subject"] for i in te} == {f["held_out"]}


def test_every_class_survives_in_every_train_fold(cfg):
    rows = {r["idx"]: r for r in D.eligible_recordings(cfg)}
    for f in S.folds(cfg):
        assert len({rows[i]["label_idx"] for i in f["train"]}) == 20


def test_ar_groups_in_val_and_test_were_all_fitted_on_train(cfg):
    for f in S.folds(cfg):
        gt = D.group_keys(cfg, f["train"], train_idxs=f["train"])
        for part in ("val", "test"):
            assert not D.missing_groups(gt, D.group_keys(cfg, f[part], train_idxs=f["train"]))


def test_saved_split_refuses_to_load_under_a_changed_config(cfg, tmp_path):
    fs = S.folds(cfg)
    p = str(tmp_path / "splits.json")
    S.save(cfg, fs, p)
    assert len(S.load(cfg, p)) == 5
    d = json.load(open(p))
    d["config_hash"] = "deadbeefdeadbeef"
    json.dump(d, open(p, "w"))
    with pytest.raises(ValueError, match="config_hash"):
        S.load(cfg, p)


def test_target_is_six_dim_both_hands(cfg):
    y = D.load_target(cfg, 0)
    assert y.shape == (GEOM[0][2], 6)
    assert cfg.channels == ["F_L", "CoPx_L", "CoPy_L", "F_R", "CoPx_R", "CoPy_R"]


# --------------------------------------------------------------------- harness wiring --

def test_external_scoring_rejects_predictions_off_the_harness_origins(cfg):
    """A model that used its own window sampler must fail loudly, not be quietly compared on
    different data than the baselines were."""
    from src.actionsense.eval_harness.baselines.base import origins
    from src.d256 import evaluate as E

    fold = S.folds(cfg)[0]
    ctx = E.fold_context(cfg, fold)
    H, C = cfg.horizon, len(cfg.channels)

    wrong = {i: np.zeros((3, H, C)) for i in fold["test"]}
    with pytest.raises(ValueError, match="did not use the harness origins"):
        E.score_external(cfg, fold, "bogus", wrong, ctx)

    right = {i: np.zeros((len(origins(len(ctx["test"][i]), cfg)), H, C)) for i in fold["test"]}
    incomplete = {k: v for k, v in list(right.items())[1:]}
    with pytest.raises(KeyError):
        E.score_external(cfg, fold, "bogus", incomplete, ctx)

    out = E.score_external(cfg, fold, "bogus", right, ctx)
    assert out["ch_mse"].shape == (C,)


def test_the_two_arms_differ_only_in_the_action_vocabulary(cfg):
    """arm 'none' collapses the embedding so skill measures signal predictability alone; arm
    'class' hands the model the label. Anything else differing would make the ablation
    uninterpretable."""
    requires_torch()
    from src.d256 import prob_gru as PG

    ids = S.folds(cfg)[0]["train"]
    a_none, n_none = PG.action_ids(cfg, "none", ids)
    a_cls, n_cls = PG.action_ids(cfg, "class", ids)
    assert n_none == 1 and set(a_none.values()) == {0}
    assert n_cls == 20 and len(set(a_cls.values())) == 20
    with pytest.raises(ValueError, match="unknown arm"):
        PG.action_ids(cfg, "labels", ids)


def test_features_are_causal_and_correctly_dimensioned(cfg):
    """velocity[t] must not depend on any sample after t -- a forecast built on a
    future-peeking feature is not a forecast."""
    requires_torch()
    from src.d256 import prob_gru as PG

    Y = D.load_target(cfg, 0)
    f = PG.features(Y, cfg.fps)
    assert f.shape == (len(Y), 10)
    assert PG.features(Y, cfg.fps, with_df=True).shape == (len(Y), 12)
    assert np.allclose(f[0, 6:], 0.0)                     # v[0] = 0, no backward reach

    cut = 40
    f_trunc = PG.features(Y[:cut], cfg.fps)
    assert np.allclose(f[:cut], f_trunc)                  # truncating the future changes nothing


def test_a_recording_of_exactly_min_history_plus_horizon_is_not_eligible(cfg):
    """The half-open range makes T == min_history + horizon yield zero origins.

    Two of the corpus's 166 segments are exactly 30 frames. Deriving eligibility as
    `T >= min_history + horizon` admitted them, they produced no windows, and every LOSO run
    died in score_external with an opaque KeyError. Eligibility must agree with origins()
    exactly, at the boundary and not merely in general.
    """
    from src.actionsense.eval_harness.baselines.base import origins

    need = cfg.raw["eval"]["min_history"] + cfg.horizon
    assert len(origins(need, cfg)) == 0, "boundary assumption changed"
    assert len(origins(need + 1, cfg)) == 1

    rows = D.eligible_recordings(cfg)
    assert all(len(origins(r["T"] // cfg.downsample, cfg)) > 0 for r in rows)
    assert not any(r["T"] == need for r in rows), \
        f"a T == {need} recording is eligible but yields no windows"
    # The corpus really does contain the boundary case, so this test is not hypothetical.
    assert sum(1 for _, _, T in GEOM if T == need) == 2


def test_every_test_recording_in_every_fold_yields_windows(cfg):
    """What actually broke: forecast() had nothing to return for one test recording, and
    score_external demands one per recording. Assert the property directly, per fold."""
    from src.actionsense.eval_harness.baselines.base import origins

    rows = {r["idx"]: r for r in D.eligible_recordings(cfg)}
    for f in S.folds(cfg):
        for part in ("train", "val", "test"):
            for i in f[part]:
                assert len(origins(rows[i]["T"] // cfg.downsample, cfg)) > 0, \
                    f"fold {f['fold']} {part} recording {i} (T={rows[i]['T']}) yields no windows"
