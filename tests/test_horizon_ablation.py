"""Horizon ablation (SESSION_LOG 2026-09-14, extended 2026-09-15).

(a) `eval.origin_horizon_s` keeps every horizon on ONE origin set, and each shared origin still
    has a full target window at its own horizon.
(b) without the key nothing changes: origins are exactly what the frozen harness always produced.
(c) an origin horizon shorter than the forecast horizon is refused, not silently truncated.
(d) each corpus's ablation configs differ from that corpus's frozen harness ONLY in
    rate.horizon_s, eval.origin_horizon_s and paths.out_csv -- so a later edit to the frozen
    harness that is not carried into the ablation fails here instead of producing an ablation of a
    different pipeline. Every config of one sweep shares the sweep's longest horizon as its origin
    horizon, and no two write the same results table.
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

# corpus -> (frozen harness, {horizon_s: tag}, shared origin horizon in seconds)
SWEEPS = {
    "actionsense": ("configs/actionsense/eval_harness.yaml",
                    {0.2: "h0p2", 0.5: "h0p5", 0.8: "h0p8", 1.0: "h1p0", 2.0: "h2p0", 3.0: "h3p0"},
                    3.0),
    "egotouch": ("configs/egotouch/eval_harness.yaml",
                 {0.5: "h0p5", 1.0: "h1p0", 2.0: "h2p0", 3.0: "h3p0"},
                 3.0),
}


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


@pytest.mark.parametrize("h", [0.2, 0.5, 0.8, 1.0, 2.0, 3.0])
def test_shared_origins_identical_across_horizons(h):
    T = 200
    c = cfg_with(h, origin_horizon_s=3.0)
    ors = origins(T, c)
    np.testing.assert_array_equal(ors, origins(T, cfg_with(3.0)))
    assert (ors + c.horizon <= T - 1).all(), "a shared origin lacks a full target window"


def test_natural_origins_differ_without_the_switch():
    """What the switch prevents: a 2-step horizon would otherwise score 28 more origins than 3 s."""
    assert len(origins(200, cfg_with(0.2))) == len(origins(200, cfg_with(3.0))) + 28


def test_origin_horizon_shorter_than_horizon_is_refused():
    with pytest.raises(ValueError, match="shorter than"):
        cfg_with(3.0, origin_horizon_s=1.0).origin_horizon


def _without_ablation_keys(d):
    d = copy.deepcopy(d)
    d["rate"].pop("horizon_s")
    d["eval"].pop("origin_horizon_s", None)
    d["paths"].pop("out_csv")
    return d


@pytest.mark.parametrize("corpus", sorted(SWEEPS))
def test_ablation_configs_differ_from_frozen_harness_only_where_they_must(corpus):
    frozen, horizons, origin_s = SWEEPS[corpus]
    main = yaml.safe_load(open(os.path.join(ROOT, frozen)))
    outs = set()
    for h, tag in horizons.items():
        path = os.path.join(ROOT, f"configs/{corpus}/horizon_ablation/eval_harness_{tag}.yaml")
        c = load_config(path)
        assert c.raw["rate"]["horizon_s"] == h
        assert c.horizon == int(round(h * c.fps))
        assert c.raw["eval"]["origin_horizon_s"] == origin_s
        assert c.origin_horizon == int(round(origin_s * c.fps))
        outs.add(c.raw["paths"]["out_csv"])
        assert _without_ablation_keys(c.raw) == _without_ablation_keys(main), \
            f"{corpus}/{tag} drifted from {frozen}"
    assert len(outs) == len(horizons), "two horizons would overwrite one results table"
    present = sorted(f for f in os.listdir(os.path.join(ROOT, f"configs/{corpus}/horizon_ablation")))
    assert present == sorted(f"eval_harness_{t}.yaml" for t in horizons.values()), \
        f"stray or missing configs in configs/{corpus}/horizon_ablation: {present}"
