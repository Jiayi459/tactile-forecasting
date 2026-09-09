"""Recording-balanced aggregation -- the ONE primitive behind every "measurement" stage.

Doctrine (user ruling 2026-09-09): every stage that produces a CLAIM -- AR order selection,
checkpoint selection, sigma calibration, reported metrics, bootstrap CIs -- aggregates
recording-balanced; the training loss alone stays window-pooled (it is an objective, not a
measurement, and train-side skew is diluted to ~8% by corpus size).

The primitive: weight each window by 1 / n_windows(its recording), so every recording carries
equal total weight no matter how many rolling origins its length yields. With equal-length
recordings this reduces exactly to the pooled version, which is why adopting it corpus-wide is
safe: it can only differ where the skew it corrects actually exists.
"""
from __future__ import annotations

import numpy as np


def recording_weights(rec_ids: np.ndarray) -> np.ndarray:
    """(N,) window-level weights: 1 / n_windows(recording). Sums to the number of recordings."""
    rec_ids = np.asarray(rec_ids)
    _, inv, counts = np.unique(rec_ids, return_inverse=True, return_counts=True)
    return 1.0 / counts[inv]


def weighted_mean(x: np.ndarray, w: np.ndarray) -> float:
    x, w = np.asarray(x, dtype=float).ravel(), np.asarray(w, dtype=float).ravel()
    keep = np.isfinite(x)
    if not keep.any():
        return float("nan")
    return float(np.average(x[keep], weights=w[keep]))


def weighted_percentile(x: np.ndarray, w: np.ndarray, q: float) -> float:
    """Weighted analogue of np.percentile(x, q) (q in [0, 100], linear interpolation).

    Convention chosen so that EQUAL weights reproduce np.percentile's linear interpolation
    EXACTLY (positions cum_i = i, target q/100 * (n-1)); the weighted generalization keeps
    cum_i = mass strictly below observation i, with the target scaled to the top position.
    """
    x, w = np.asarray(x, dtype=float).ravel(), np.asarray(w, dtype=float).ravel()
    keep = np.isfinite(x) & (w > 0)
    if not keep.any():
        return float("nan")
    x, w = x[keep], w[keep]
    order = np.argsort(x)
    x, w = x[order], w[order]
    cum = np.cumsum(w) - w                       # mass strictly below each observation
    return float(np.interp(q / 100.0 * cum[-1], cum, x))
