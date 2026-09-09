"""The recording-balanced primitive: reduces to pooled at equal weights, resists skew."""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.actionsense.eval_harness.weighting import (  # noqa: E402
    recording_weights, weighted_mean, weighted_percentile)


def test_weights_give_each_recording_equal_total_mass():
    rec = np.array([0] * 1000 + [1, 2, 3])
    w = recording_weights(rec)
    assert w[rec == 0].sum() == pytest.approx(1.0)
    assert w.sum() == pytest.approx(4.0)


def test_weighted_mean_reduces_to_pooled_with_equal_weights():
    x = np.random.default_rng(0).normal(size=100)
    assert weighted_mean(x, np.ones(100)) == pytest.approx(x.mean())


def test_weighted_mean_is_recording_balanced_under_skew():
    # one long recording says 10, nine short ones say 1 -- the balanced answer is 1.9
    x = np.concatenate([np.full(1000, 10.0), np.full(9, 1.0)])
    w = recording_weights(np.concatenate([np.zeros(1000, int), np.arange(1, 10)]))
    assert weighted_mean(x, w) == pytest.approx(1.9)
    assert x.mean() == pytest.approx(9.919, abs=1e-3)    # what pooled would have said


def test_weighted_percentile_matches_numpy_at_equal_weights():
    x = np.random.default_rng(1).normal(size=5001)
    for q in (0, 5, 50, 95, 100):
        assert weighted_percentile(x, np.ones_like(x), q) == pytest.approx(
            np.percentile(x, q), abs=1e-12)


def test_weighted_percentile_under_skew_tracks_the_balanced_distribution():
    # long recording at value 0, nine singleton recordings at value 1:
    # balanced mass is 10% at 0 and 90% at 1 -> the weighted median is 1
    x = np.concatenate([np.zeros(1000), np.ones(9)])
    w = recording_weights(np.concatenate([np.zeros(1000, int), np.arange(1, 10)]))
    assert weighted_percentile(x, w, 50) == pytest.approx(1.0)
    assert np.percentile(x, 50) == 0.0                   # pooled says the opposite


def test_nan_and_zero_weight_are_ignored():
    x = np.array([np.nan, 1.0, 3.0])
    w = np.array([5.0, 1.0, 1.0])
    assert weighted_mean(x, w) == pytest.approx(2.0)
    assert weighted_percentile(np.array([1.0, 2.0]), np.array([0.0, 1.0]), 50) == pytest.approx(2.0)
