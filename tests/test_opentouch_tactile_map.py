"""Tests for src/opentouch/tactile_map.py -- flatten / cnn / aggregate on the map.

Needs torch, so it lives in its own file behind a module-level skip like the other two.
"""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    import torch
except Exception as exc:                                             # pragma: no cover
    pytest.skip(f"torch unavailable ({type(exc).__name__})", allow_module_level=True)

from src.actionsense.eval_harness.config import load_config           # noqa: E402
from src.opentouch import tactile_map as TM                           # noqa: E402
from src.opentouch.baselines import origins                           # noqa: E402
from src.opentouch.dataset import Norm, load_target                   # noqa: E402


@pytest.fixture
def cfg(tmp_path):
    rng = np.random.default_rng(0)
    recs = []
    for i in range(12):
        T = 90
        t = np.arange(T) / 30.0
        st = np.zeros((T, 1, 6), np.float32)
        st[:, 0, 0] = 50 + 10 * np.sin(2 * np.pi * 0.7 * t + i)
        st[:, 0, 1] = 0.2 * np.sin(2 * np.pi * 0.5 * t + i)
        st[:, 0, 2] = 0.2 * np.cos(2 * np.pi * 0.3 * t + i)
        np.save(tmp_path / f"state_{i}.npy", st)
        # a resting level plus a moving contact blob, like the real corpus
        yy, xx = np.mgrid[0:16, 0:16]
        m = np.full((T, 1, 16, 16), 3050.0, np.float16)
        for k in range(30, 60):
            m[k, 0] += (200 * np.exp(-((xx - 8) ** 2 + (yy - 8) ** 2) / 6)).astype(np.float16)
        np.save(tmp_path / f"clip_{i}.npy", m)
        recs.append({"idx": i, "shard": f"sh{i % 2}", "clip_id": f"c{i}", "scene": "s",
                     "action": "holding", "object_category": "cup", "environment": "e",
                     "T": T, "fps_est": 30.0, "has_clip": True, "has_pose": False})
    (tmp_path / "manifest.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    text = open("configs/opentouch/eval_harness.yaml").read().replace(
        "states_root: data/opentouch_states", f"states_root: {tmp_path}")
    p = tmp_path / "h.yaml"; p.write_text(text)
    return load_config(str(p))


def test_encoders_differ_only_in_the_encoder(cfg):
    """models.py's stated design: one GRU and one probabilistic head behind all three."""
    hp = dict(TM.DEFAULT_HP, d=8, hidden=8)
    ms = {e: TM.build_model(e, cfg, hp) for e in ("flatten", "cnn", "aggregate")}
    for e, m in ms.items():
        assert isinstance(m, TM.Seq2Seq)
        assert m.gru.hidden_size == 8 and m.mu.out_features == cfg.horizon * 3
        assert m.lv.out_features == cfg.horizon * 3          # every arm is probabilistic
    assert type(ms["flatten"].encoder) is TM.FlattenEncoder
    assert type(ms["cnn"].encoder) is TM.CNNEncoder


def test_grid_is_one_hand_sixteen(cfg):
    assert (TM.IN_CH, TM.GRID, TM.FLAT) == (1, 16, 256)
    x = torch.randn(2, 5, 1, 16, 16)
    for enc in (TM.FlattenEncoder(8), TM.CNNEncoder(8)):
        assert enc(x).shape == (2, 5, 8)


def test_baseline_is_the_per_taxel_median_per_shard(cfg):
    b = TM.taxel_baselines(cfg, list(range(12)))
    assert set(b) == {"sh0", "sh1"}
    # the planted resting level is 3050 and contact is a minority of frames, so the median
    # must sit at rest rather than somewhere between rest and contact
    for v in b.values():
        assert np.allclose(np.median(v), 3050.0, atol=2.0)
    m = TM.load_map(cfg, 0, b["sh0"])
    assert m.shape[1:] == (1, 16, 16) and (m >= 0).all()
    assert m[0].max() < 5.0 and m[45].max() > 50.0        # rest ~0 after removal, contact not


def test_windows_are_residual_and_left_padded(cfg):
    norm = Norm.from_train({i: load_target(cfg, i) for i in range(8)})
    inp, _ = TM.build_inputs(cfg, "aggregate", [0], [0], norm, 10.0)
    X, Y = TM.windows(cfg, [0], 40, inp, norm)
    n_or = len(origins(len(load_target(cfg, 0)), cfg))
    assert X.shape == (n_or, 40, 3) and Y.shape == (n_or, cfg.horizon, 3)
    # Derived, not a magic number. The first origin is min_history, and the window is
    # M[t-t_in+1 : t+1], so it holds min_history+1 real frames -- 16, not 15. The literal 25
    # here asserted 15 and had never been executed: the job's pytest gate listed only the
    # prob_gru and gru_aggregate files, so this one ran for the first time on 2026-08-22.
    pad = 40 - (cfg.raw["eval"]["min_history"] + 1)
    assert torch.allclose(X[0, :pad], torch.zeros(pad, 3))   # early origin is left-padded
    assert not torch.allclose(X[0, pad], torch.zeros(3))     # and real data starts right after
    z = norm.z(np.asarray(load_target(cfg, 0), dtype=np.float64))
    t = int(origins(len(z), cfg)[0])
    assert np.allclose(Y[0].numpy(), (z[t + 1:t + 1 + cfg.horizon] - z[t]), atol=1e-5)


@pytest.mark.parametrize("enc", ["aggregate", "flatten", "cnn"])
def test_train_and_predict_align_with_the_harness(cfg, enc):
    tr, va, te = list(range(8)), [8, 9], [10, 11]
    hp = dict(TM.DEFAULT_HP, d=8, hidden=8, epochs=1)
    m, norm, mnorm, hist = TM.train(cfg, enc, tr, va, t_in=15, hp=hp)
    preds = TM.predict(m, cfg, enc, norm, mnorm, te, 15, tr + va)
    for i in te:
        n_or = len(origins(len(load_target(cfg, i)), cfg))
        assert preds[i].shape == (n_or, cfg.horizon, 3)
        assert np.isfinite(preds[i]).all()
    assert hist["encoder"] == enc and len(hist["val_mse"]) == 1


def test_a_zero_residual_reproduces_persistence(cfg):
    """The residual convention is what makes 'predict nothing' equal persistence; if the
    anchor were dropped the arm would be worse than the baseline it is measured against."""
    tr, te = list(range(10)), [10]
    norm = Norm.from_train({i: load_target(cfg, i) for i in tr})
    m, _, mnorm, _ = TM.train(cfg, "aggregate", tr, [], t_in=15,
                              hp=dict(TM.DEFAULT_HP, d=4, hidden=4, epochs=1), norm=norm)
    with torch.no_grad():                       # force the residual head to output zero
        m.mu.weight.zero_(); m.mu.bias.zero_()
    preds = TM.predict(m, cfg, "aggregate", norm, mnorm, te, 15, tr)
    Y = np.asarray(load_target(cfg, te[0]), dtype=np.float64)
    ors = origins(len(Y), cfg)
    assert np.allclose(preds[te[0]], np.repeat(Y[ors][:, None, :], cfg.horizon, axis=1),
                       atol=1e-6)


def test_map_arms_refuse_to_run_without_maps(cfg, tmp_path):
    """A cache holding states but no clip_*.npy must fail, not report numbers from nothing.

    On 2026-08-22 the D1 cache had no maps, taxel_baselines skipped every missing file, and
    flatten and cnn each returned a full metric table -- identical to four decimals, sigma
    exactly 0 -- having trained on an empty input set. The aggregate arm reads state_*.npy
    and was unaffected, which is precisely why the failure was silent.
    """
    ids = [r["idx"] for r in
           (json.loads(l) for l in open(tmp_path / "manifest.jsonl"))]
    TM.taxel_baselines(cfg, ids)                     # maps present: fine

    for p in tmp_path.glob("clip_*.npy"):
        p.unlink()
    with pytest.raises(FileNotFoundError, match="clip"):
        TM.taxel_baselines(cfg, ids)

    # the aggregate arm reads state_*.npy and must still be usable without any map
    agg, mn = TM.build_inputs(cfg, "aggregate", ids[:4], ids[:4],
                              Norm.from_train({i: load_target(cfg, i) for i in ids[:4]}),
                              TM.DEFAULT_HP["alpha"])
    assert set(agg) == set(ids[:4]) and mn is None


def test_held_out_shard_gets_a_baseline_under_the_shard_scope(cfg, tmp_path):
    """The scope that made flatten and cnn return arrays of zeros, and the one that does not.

    The fixture puts clips in shards sh0/sh1 by parity. Holding out a whole shard is what
    location-held-out CV does, and under a TRAIN-only scope that shard has no taxel baseline
    at all, so every one of its clips is dropped.
    """
    ids = [r["idx"] for r in (json.loads(l) for l in open(tmp_path / "manifest.jsonl"))]
    train = [i for i in ids if i % 2 == 0]          # sh0 only
    held = [i for i in ids if i % 2 == 1]           # sh1, wholly unseen

    assert TM.scope_ids(cfg, train, [], "train") == sorted(train)
    assert set(TM.scope_ids(cfg, train, [], "shard")) == set(ids)   # reaches the held-out shard

    norm = Norm.from_train({i: load_target(cfg, i) for i in train})
    a = TM.DEFAULT_HP["alpha"]
    strict, _ = TM.build_inputs(cfg, "flatten", held, TM.scope_ids(cfg, train, [], "train"),
                                norm, a)
    assert strict == {}                              # every held-out clip dropped
    wide, _ = TM.build_inputs(cfg, "flatten", held, TM.scope_ids(cfg, train, [], "shard"),
                              norm, a)
    assert set(wide) == set(held)                    # all of them recovered

    with pytest.raises(ValueError, match="scope"):
        TM.scope_ids(cfg, train, [], "everything")


def test_a_clip_with_origins_and_no_input_raises_instead_of_predicting_zeros(cfg, tmp_path):
    """Zeros for a droppable input are a fabrication, and on 2026-08-22 they were scored."""
    ids = [r["idx"] for r in (json.loads(l) for l in open(tmp_path / "manifest.jsonl"))]
    train, held = [i for i in ids if i % 2 == 0], [i for i in ids if i % 2 == 1]
    norm = Norm.from_train({i: load_target(cfg, i) for i in train})
    hp = dict(TM.DEFAULT_HP, d=8, hidden=8, epochs=1, batch=8)
    model, _, mnorm, _ = TM.train(cfg, "flatten", train, train[:2], 15, hp, norm=norm,
                                  device="cpu", base_scope="shard")
    with pytest.raises(RuntimeError, match="no input"):
        TM.predict_with_sigma(model, cfg, "flatten", norm, mnorm, held, 15,
                              TM.scope_ids(cfg, train, [], "train"))
