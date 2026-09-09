"""AR/seasonal after the 2026-09-09 changes: balanced order selection, global fallback."""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.actionsense.eval_harness.baselines.ar import AR, GLOBAL  # noqa: E402
from src.actionsense.eval_harness.baselines.seasonal import SeasonalNaive  # noqa: E402
from src.actionsense.eval_harness.config import Config  # noqa: E402
from src.actionsense.eval_harness.dataset import Norm  # noqa: E402


def _cfg(min_history=30):
    return Config(raw={
        "rate": {"fps_raw": 30.0, "downsample": 3, "horizon_s": 1.0},
        "target": {"channels": list("abcdef"), "force_idx": [0, 3], "cop_idx": [1, 2, 4, 5]},
        "baselines": {"ar_orders": [1, 2], "seasonal_period_min_s": 0.3,
                      "seasonal_period_max_s": 3.0, "seasonal_min_autocorr": 0.1},
        "eval": {"min_history": min_history, "stride": 1, "seed": 0},
    }, path="synthetic", config_hash="0" * 8)


def _norm():
    return Norm(mean=np.zeros(6), std=np.ones(6))


def _series(T, seed, freq=0.0):
    rng = np.random.default_rng(seed)
    t = np.arange(T)[:, None]
    base = np.sin(2 * np.pi * freq * t) if freq else 0.0
    return (base + 0.1 * rng.normal(size=(T, 6))).astype(float)


# --- the fallback that used to be a KeyError ------------------------------------------

def test_ar_unknown_group_routes_to_global_instead_of_keyerror():
    cfg = _cfg()
    ar = AR(cfg, _norm())
    train = {0: _series(200, 0), 1: _series(200, 1)}
    ar.fit(train, {0: "grasp-cola", 1: "grasp-cola"})
    ar.select({0: _series(200, 2)}, {0: "grasp-cola"}, cfg.horizon)
    out = ar.predict(_series(120, 3), cfg.horizon, "wipe-window")   # group TRAIN never saw
    assert out.shape == (cfg.horizon, 6) and np.isfinite(out).all()
    assert GLOBAL in ar.coef and GLOBAL in ar.order


def test_ar_reserved_group_name_is_refused():
    cfg = _cfg()
    with pytest.raises(ValueError, match="reserved"):
        AR(cfg, _norm()).fit({0: _series(120, 0)}, {0: GLOBAL})


def test_seasonal_unknown_group_uses_global_period_not_persistence():
    cfg = _cfg()
    sn = SeasonalNaive(cfg, _norm())
    period = 10
    train = {i: _series(300, i, freq=1 / period) for i in range(3)}
    sn.fit(train, {i: "grasp-cola" for i in range(3)})
    assert sn.periods.get(GLOBAL) == period
    hist = _series(120, 9, freq=1 / period)
    out = sn.predict(hist, cfg.horizon, "wipe-window")
    pers = np.repeat(hist[-1:], cfg.horizon, axis=0)
    assert not np.allclose(out, pers)          # it used the global cycle, not persistence


# --- balanced order selection ---------------------------------------------------------

def test_ar_order_selection_is_recording_balanced():
    """One long AR(1)-favouring recording vs several short AR(2)-favouring ones.

    Pooled MSE would let the long recording pick the order for everyone; the balanced
    criterion averages one error per recording, so the majority of recordings wins.
    """
    cfg = _cfg(min_history=5)
    rng = np.random.default_rng(0)

    def ar2(T, phi1, phi2, seed, noise):
        r = np.random.default_rng(seed)
        z = np.zeros((T, 6))
        z[0] = r.normal(size=6); z[1] = r.normal(size=6)
        for k in range(2, T):
            z[k] = phi1 * z[k - 1] + phi2 * z[k - 2] + noise * r.normal(size=6)
        return z

    # short recordings: strongly AR(2) (needs order 2); long recording: white noise with a
    # tiny AR(1) pull, where the extra AR(2) lag only adds variance.
    short = {i: ar2(60, 0.1, 0.85, i, 0.05) for i in range(8)}
    long_ = {99: 0.05 * rng.normal(size=(4000, 6))}
    train = {**short, **long_}
    groups = {i: "g" for i in train}
    ar = AR(cfg, _norm())
    ar.fit(train, groups)

    val = {**{i: ar2(60, 0.1, 0.85, 100 + i, 0.05) for i in range(8)},
           99: 0.05 * np.random.default_rng(7).normal(size=(4000, 6))}
    ar.select(val, {i: "g" for i in val}, cfg.horizon)
    assert ar.order["g"] == 2      # the eight short recordings outvote the one long one
