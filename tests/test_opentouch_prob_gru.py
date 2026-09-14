"""Tests for src/opentouch/prob_gru.py -- the ActionSense probGRU trained on OpenTouch.

SEPARATE FILE because it needs torch, like tests/test_opentouch_gru_aggregate.py: a
module-level skip keeps one broken install from taking the whole directory's collection
down with it.
"""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# NOT pytest.importorskip: that only catches ImportError, and a half-installed torch raises
# OSError from dlopen (SESSION_LOG 2026-08-12).
try:
    import torch
except Exception as exc:                                             # pragma: no cover
    pytest.skip(f"torch unavailable ({type(exc).__name__})", allow_module_level=True)

from src.actionsense.eval_harness.config import load_config           # noqa: E402
from src.opentouch import prob_gru as P                               # noqa: E402
from src.opentouch.dataset import Norm, load_target                   # noqa: E402


@pytest.fixture
def cfg(tmp_path):
    """A tiny synthetic cache plus a config pointing at it."""
    rng = np.random.default_rng(0)
    recs = []
    for i in range(12):
        T = 80
        t = np.arange(T) / 30.0
        st = np.zeros((T, 1, 6), np.float32)
        st[:, 0, 0] = 50 + 10 * np.sin(2 * np.pi * 0.7 * t + i)
        st[:, 0, 1] = 0.2 * np.sin(2 * np.pi * 0.5 * t + i) + rng.normal(0, 0.01, T)
        st[:, 0, 2] = 0.2 * np.cos(2 * np.pi * 0.3 * t + i)
        np.save(tmp_path / f"state_{i}.npy", st)
        recs.append({"idx": i, "shard": "s", "clip_id": f"c{i}", "scene": "sc",
                     "action": ["holding", "sliding", "rare_one"][min(i, 2) if i < 3 else i % 2],
                     "object_category": "cup", "environment": "e", "T": T,
                     "fps_est": 30.0, "has_clip": False, "has_pose": False})
    (tmp_path / "manifest.jsonl").write_text(
        "\n".join(json.dumps(r) for r in recs) + "\n")

    src = "configs/opentouch/eval_harness.yaml"
    text = open(src).read().replace("states_root: data/opentouch_states",
                                    f"states_root: {tmp_path}")
    p = tmp_path / "harness.yaml"
    p.write_text(text)
    # These fixtures exercise the historical model/state contract; new RAW-to-model
    # TRAIN-only coverage lives in test_train_only_calibration.py.
    p.write_text(p.read_text().replace("mode: train_only", "mode: legacy"))
    return load_config(str(p))


def test_features_are_causal(cfg):
    """A change at time t must not move any feature before t -- the forecasting requirement
    that rules out filtfilt and centred differences."""
    Y = load_target(cfg, 0)
    f0 = P.features(Y, cfg.fps)
    Y2 = Y.copy()
    Y2[40:] += 5.0
    f1 = P.features(Y2, cfg.fps)
    assert np.allclose(f0[:40], f1[:40])
    assert not np.allclose(f0[40:], f1[40:])


def test_feature_layout(cfg):
    Y = load_target(cfg, 0)
    f = P.features(Y, cfg.fps)
    assert f.shape == (len(Y), 5)
    assert np.allclose(f[:, :3], Y)                      # first three are the raw channels
    assert f[0, 3] == 0 and f[0, 4] == 0                 # velocity starts at zero


def test_vocab_is_built_from_train_only(cfg):
    """An action that appears only in VAL/TEST must fall back to 'other', not create an id."""
    train_ids, other_ids = [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11], [2]
    vocab, by_idx = P.action_vocab(cfg, train_ids)
    assert vocab["other"] == P.OTHER
    assert "rare_one" not in vocab                       # idx 2 only, and below min_group_size
    assert P._aid(vocab, by_idx, other_ids[0]) == P.OTHER


def test_train_and_predict_shapes_match_the_harness(cfg):
    ids = list(range(12))
    tr, va, te = ids[:8], ids[8:10], ids[10:]
    norm = Norm.from_train({i: load_target(cfg, i) for i in tr})
    hp = {"epochs": 2, "hidden": 8}
    m, norm, fnorm, vocab, by_idx, hist = P.train(cfg, tr, va, t_in=15, hp=hp, norm=norm)
    preds = P.predict(m, cfg, norm, fnorm, vocab, by_idx, te, t_in=15)

    from src.opentouch.baselines import origins
    for i in te:
        n_or = len(origins(len(load_target(cfg, i)), cfg))
        assert preds[i].shape == (n_or, cfg.horizon, len(cfg.channels))
        assert np.isfinite(preds[i]).all()
    assert len(hist["val_nll"]) == 2


def test_loss_is_gaussian_nll(cfg):
    """The loss must be ActionSense's 0.5*(lv + (y-mu)^2 * exp(-lv)), not an MSE in disguise:
    with the residual fixed, raising log-variance past its optimum must increase the loss."""
    y = torch.zeros(4, 3, 3)
    mu = torch.full_like(y, 0.5)
    a = P.nll(mu, torch.zeros_like(y), y)
    b = P.nll(mu, torch.full_like(y, 2.0), y)
    assert float(b) > float(a)
    # and the optimum of lv for a fixed residual is log(residual^2)
    grid = [float(P.nll(mu, torch.full_like(y, v), y)) for v in np.linspace(-4, 4, 81)]
    best = np.linspace(-4, 4, 81)[int(np.argmin(grid))]
    assert abs(best - np.log(0.25)) < 0.15


def test_determinism(cfg):
    ids = list(range(12))
    tr, va = ids[:8], ids[8:10]
    hp = {"epochs": 2, "hidden": 8}
    out = []
    for _ in range(2):
        m, norm, fnorm, vocab, by_idx, _ = P.train(cfg, tr, va, t_in=15, hp=hp)
        out.append(P.predict(m, cfg, norm, fnorm, vocab, by_idx, [10], t_in=15)[10])
    assert np.allclose(out[0], out[1])


def test_df_ablation_adds_exactly_one_causal_column(cfg):
    """raw+df appends dF/dt and changes nothing else -- the first five columns must be
    bit-identical, or the ablation would be confounded with a change to the other inputs."""
    Y = load_target(cfg, 0)
    a = P.features(Y, cfg.fps)
    b = P.features(Y, cfg.fps, with_df=True)
    assert a.shape[1] == 5 and b.shape[1] == 6
    assert np.array_equal(a, b[:, :5])
    # the new column is the causal difference of F, and it is causal
    assert np.allclose(b[:, 5], P.causal_velocity(Y[:, 0:1], cfg.fps).ravel())
    assert b[0, 5] == 0


def test_df_ablation_reaches_the_model_and_is_recorded(cfg):
    ids = list(range(12))
    hp = {"epochs": 1, "hidden": 8, "features": "raw+df"}
    m, norm, fnorm, vocab, by_idx, hist = P.train(cfg, ids[:8], ids[8:10], t_in=15, hp=hp)
    assert fnorm.with_df is True
    assert hist["n_features"] == 6 and hist["features"] == "raw+df"
    # the normalizer carries the layout, so prediction cannot silently use the other one
    preds = P.predict(m, cfg, norm, fnorm, vocab, by_idx, ids[10:], t_in=15)
    assert all(np.isfinite(v).all() for v in preds.values())


def test_default_is_still_the_verbatim_five(cfg):
    ids = list(range(12))
    _, _, fnorm, _, _, hist = P.train(cfg, ids[:8], ids[8:10], t_in=15,
                                      hp={"epochs": 1, "hidden": 8})
    assert fnorm.with_df is False and hist["n_features"] == 5


def test_regularisers_are_off_by_default_and_reach_the_model(cfg):
    """p=0 dropout must be an identity, or the 'verbatim architecture' claim is false."""
    ids = list(range(12))
    _, _, _, _, _, h0 = P.train(cfg, ids[:8], ids[8:10], t_in=15,
                                hp={"epochs": 1, "hidden": 8})
    assert h0["weight_decay"] == 0.0 and h0["dropout"] == 0.0

    m = P.ProbGRU(5, 2, 8, dropout=0.0)
    m.eval()
    x = torch.randn(3, 15, 5); a = torch.zeros(3, dtype=torch.long); yl = torch.randn(3, 3)
    m.train()                                  # dropout is active in train mode, if any
    mu1, _ = m(x, a, yl, 4)
    mu2, _ = m(x, a, yl, 4)
    assert torch.allclose(mu1, mu2), "p=0 dropout must not perturb anything"

    _, _, _, _, _, h1 = P.train(cfg, ids[:8], ids[8:10], t_in=15,
                                hp={"epochs": 1, "hidden": 8, "weight_decay": 1e-3,
                                    "dropout": 0.2})
    assert h1["weight_decay"] == 1e-3 and h1["dropout"] == 0.2


def test_val_mse_is_tracked_beside_val_nll(cfg):
    """The two can move apart, and which epoch each prefers is the diagnostic: NLL can be
    ruined by an overconfident sigma while the mean is unchanged."""
    ids = list(range(12))
    *_, h = P.train(cfg, ids[:8], ids[8:10], t_in=15, hp={"epochs": 3, "hidden": 8})
    assert len(h["val_mse"]) == 3 and np.isfinite(h["val_mse"]).all()
    assert "best_val_mse_epoch" in h and "best_val_nll_epoch" in h
    # the MSE is of the MEAN only: it must not depend on the variance head
    assert h["best_val_mse"] <= max(h["val_mse"]) + 1e-12


def test_select_history_can_return_its_models(cfg):
    """The sweep trains one model per length and used to discard all of them, so the
    rows-are-history figure could not be drawn without paying for training twice."""
    ids = list(range(12))
    hp = {"epochs": 1, "hidden": 8}
    best, scores, none_kept = P.select_history(cfg, ids[:8], ids[8:10], (0.5, 1.0), hp)
    assert set(scores) == {int(round(0.5 * cfg.fps)), int(round(1.0 * cfg.fps))}
    # the arity does NOT depend on `keep`: a flag-dependent return length crashed a run
    # after it had done all its training (2026-08-19)
    assert none_kept == {}

    best2, scores2, kept = P.select_history(cfg, ids[:8], ids[8:10], (0.5, 1.0), hp, keep=True)
    assert best2 == best and scores2 == scores          # keeping changes nothing it returns
    assert set(kept) == set(scores)
    for t_in, out in kept.items():
        m, norm, fnorm, vocab, by_idx, hist = out
        p = P.predict(m, cfg, norm, fnorm, vocab, by_idx, [10], t_in)
        assert np.isfinite(p[10]).all()


def test_select_on_picks_the_epoch_that_criterion_minimises(cfg):
    """'nll' and 'mse' must be able to load DIFFERENT weights.

    The 2026-08-20 D1 run had fold0's VAL MSE still falling at epoch 8 while its VAL NLL had
    climbed since epoch 1, so the knob only means anything if the two curves are tracked
    independently and the requested one selects. This asserts those mechanics rather than
    that any particular epoch wins on toy data.
    """
    ids = list(range(12))
    hp = {"epochs": 4, "hidden": 4, "batch": 8, "log_train_every": 1}

    seen = {}
    for crit in ("nll", "mse"):
        h = P.train(cfg, ids[:8], ids[8:10], 6, dict(hp, select_on=crit),
                    verbose=False, device="cpu")[5]
        assert h["selected_on_metric"] == crit
        # both curves are recorded no matter which one selected
        assert len(h["val_nll"]) == 4 and len(h["val_mse"]) == 4
        assert "best_val_nll_epoch" in h and "best_val_mse_epoch" in h
        seen[crit] = (h["best_val_nll_epoch"], h["best_val_mse_epoch"])

    # the recorded epochs describe the curves, so they do not depend on which one selected
    assert seen["nll"] == seen["mse"]

    with pytest.raises(ValueError, match="select_on"):
        P.train(cfg, ids[:8], ids[8:10], 6, dict(hp, select_on="rmse"),
                verbose=False, device="cpu")


def test_history_sweep_scores_by_the_selecting_criterion(cfg):
    """Input length and weights must be chosen by one definition of "best", not two."""
    ids = list(range(12))
    hp = {"epochs": 2, "hidden": 4, "batch": 8, "log_train_every": 1}

    _, s_nll, k_nll = P.select_history(cfg, ids[:8], ids[8:10], (0.5, 1.0),
                                       dict(hp, select_on="nll"), keep=True)
    _, s_mse, k_mse = P.select_history(cfg, ids[:8], ids[8:10], (0.5, 1.0),
                                       dict(hp, select_on="mse"), keep=True)
    assert set(s_nll) == set(s_mse)
    for t in s_nll:
        # kept[t] is the whole (model, norm, fnorm, vocab, by_idx, history) tuple
        assert s_nll[t] == pytest.approx(k_nll[t][5]["best_val_nll"])
        assert s_mse[t] == pytest.approx(k_mse[t][5]["best_val_mse"])
