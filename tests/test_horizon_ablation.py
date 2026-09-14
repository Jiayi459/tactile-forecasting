"""Horizon ablation (SESSION_LOG 2026-09-14).

(a) `eval.origin_horizon_s` keeps every horizon on ONE origin set, and each shared origin still
    has a full target window at its own horizon.
(b) without the key nothing changes: origins are exactly what the frozen harness always produced.
(c) an origin horizon shorter than the forecast horizon is refused, not silently truncated.
(d) the four ablation configs differ from the frozen harness ONLY in rate.horizon_s,
    eval.origin_horizon_s and paths.out_csv -- so a later edit to the frozen harness that is not
    carried into the ablation fails here instead of producing an ablation of a different pipeline.
"""
import copy
import os
import sys

import numpy as np
import pytest
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.actionsense.eval_harness.config import Config, load_config      # noqa: E402
from src.actionsense.eval_harness.baselines.base import origins          # noqa: E402

ABLATION = {0.2: "h0p2", 0.5: "h0p5", 0.8: "h0p8", 1.0: "h1p0"}


def cfg_with(horizon_s, origin_horizon_s=None, min_history=40):
    raw = {"rate": {"fps_raw": 30.0, "downsample": 3, "horizon_s": horizon_s},
           "eval": {"stride": 1, "min_history": min_history, "seed": 0}}
    if origin_horizon_s is not None:
        raw["eval"]["origin_horizon_s"] = origin_horizon_s
    return Config(raw=raw, path="test", config_hash="test")


def test_default_origins_unchanged():
    c = cfg_with(1.0)
    assert c.origin_horizon == c.horizon == 10
    np.testing.assert_array_equal(origins(200, c), np.arange(40, 190))


@pytest.mark.parametrize("h", sorted(ABLATION))
def test_shared_origins_identical_across_horizons(h):
    T = 200
    c = cfg_with(h, origin_horizon_s=1.0)
    ors = origins(T, c)
    np.testing.assert_array_equal(ors, origins(T, cfg_with(1.0)))
    assert (ors + c.horizon <= T - 1).all(), "a shared origin lacks a full target window"


def test_natural_origins_differ_without_the_switch():
    """What the switch prevents: a 2-step horizon would otherwise score 8 more origins."""
    assert len(origins(200, cfg_with(0.2))) == len(origins(200, cfg_with(1.0))) + 8


def test_origin_horizon_shorter_than_horizon_is_refused():
    with pytest.raises(ValueError, match="shorter than"):
        cfg_with(1.0, origin_horizon_s=0.5).origin_horizon


def _without_ablation_keys(d):
    d = copy.deepcopy(d)
    d["rate"].pop("horizon_s")
    d["eval"].pop("origin_horizon_s", None)
    d["paths"].pop("out_csv")
    return d


def test_ablation_configs_differ_from_frozen_harness_only_where_they_must():
    main = yaml.safe_load(open(os.path.join(ROOT, "configs/actionsense/eval_harness.yaml")))
    outs = set()
    for h, tag in ABLATION.items():
        path = os.path.join(ROOT, f"configs/actionsense/horizon_ablation/eval_harness_{tag}.yaml")
        c = load_config(path)
        assert c.raw["rate"]["horizon_s"] == h
        assert c.horizon == int(round(h * 10))
        assert c.raw["eval"]["origin_horizon_s"] == 1.0 and c.origin_horizon == 10
        outs.add(c.raw["paths"]["out_csv"])
        assert _without_ablation_keys(c.raw) == _without_ablation_keys(main), \
            f"{tag} drifted from configs/actionsense/eval_harness.yaml"
    assert len(outs) == len(ABLATION), "two horizons would overwrite one results table"
