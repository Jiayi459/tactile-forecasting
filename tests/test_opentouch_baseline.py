"""Tests for src/opentouch/baseline.py -- D1's estimator and correction.

Pure numpy: runs everywhere. The estimator moved out of the report script so the script
that WRITES a corrected cache cannot drift from the one whose behaviour was measured.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.opentouch import baseline as B                              # noqa: E402


def _corpus(T=400, rest=3050.0, noise=1.2, seed=0):
    """A quantised sensor at rest most of the time, with a contact minority."""
    rng = np.random.default_rng(seed)
    f = np.round(rest + rng.normal(0, noise, (T, 256))).astype(np.float32)
    f[150:250, 120] += 400                          # one taxel loaded 25% of the time
    return f


def test_median_finds_rest_where_a_low_quantile_undershoots():
    f = _corpus()
    base, sigma = B.estimate(f)
    assert abs(np.median(base) - 3050.0) <= 1.0
    # the loaded taxel is still unloaded most of the time, so its median is rest too
    assert abs(base[120] - 3050.0) <= 2.0
    # a 5th percentile -- the rejected estimator -- sits BELOW rest, which is the
    # undershoot that rectifies noise into a positive offset after clipping
    assert np.percentile(f, 5, axis=0)[120] < base[120]


def test_sigma_is_floored_because_the_readings_are_integers():
    """Untouched quantised taxels sit on one value, so the bare MAD is 0 and the threshold
    would silently do nothing -- what happened on all 26 shards before the floor."""
    flat = np.full((200, 256), 3050.0, np.float32)
    base, sigma = B.estimate(flat)
    assert (sigma > 0).all() and np.allclose(sigma, 0.5 * B.quantum(flat))


def test_duty_cycle_reports_the_premise_rather_than_assuming_it():
    f = _corpus()
    base, sigma = B.estimate(f)
    duty = B.duty_cycle(f, base, sigma)
    assert duty[120] > 0.2                       # the loaded taxel is visibly loaded
    assert (duty <= 0.5).all()                   # but no taxel breaks the median's premise


def test_correction_removes_the_dc_and_keeps_the_contact():
    f = _corpus()
    base, sigma = B.estimate(f)
    clip = f.reshape(-1, 1, 16, 16)
    out = B.correct(clip, base, sigma, k=1.0)
    assert (out >= 0).all()
    rest_frames = out[:100].reshape(100, -1)
    assert rest_frames.mean() < 1.0              # rest collapses to ~0
    assert out[200, 0].max() > 300               # contact survives


def test_higher_k_removes_more_and_never_less():
    f = _corpus()
    base, sigma = B.estimate(f)
    clip = f.reshape(-1, 1, 16, 16)
    tot = [B.correct(clip, base, sigma, k).sum() for k in (0.0, 1.0, 3.0)]
    assert tot[0] >= tot[1] >= tot[2]


def test_moments_match_the_extractor_bit_for_bit():
    """A corrected cache is only comparable with the raw one if F/CoP are computed the same
    way, so this asserts against the extractor's own implementation rather than a copy."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("ex", "scripts/opentouch/extract_opentouch.py")
    ex = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ex)
    p = np.abs(np.random.default_rng(1).normal(5, 2, (30, 16, 16)))
    assert np.allclose(B.moments(p), ex.moments(p))
