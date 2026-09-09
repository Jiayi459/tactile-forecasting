"""Seasonal-naive baseline: y_hat[t+h] = y at the same phase, whole periods back:

    idx = (t + h) - ceil(h / T) * T          (always <= t  =>  strictly causal)

The period T is estimated PER GROUP (activity x object) from the dominant peak of the TRAIN
autocorrelation function, searched in a plausible motion-cycle range (config, seconds). If no
clear peak exists for a group (no local maximum, or peak autocorrelation below
`seasonal_min_autocorr`), that group FALLS BACK to persistence and a warning is logged. The
estimated T per group is stored in `self.periods` for the results table.

Estimation uses TRAIN only (constraint 2). Autocorrelation is computed per recording and
averaged (no cross-recording concatenation -> no boundary artifacts).

UNKNOWN GROUPS (2026-09-09, EgoTouch port): a group absent from TRAIN -- every test_unseen
group, by construction -- routes to a pooled `_GLOBAL` period estimated over all of TRAIN,
the same rule AR applies. Only a group TRAIN saw but found no clear cycle for falls back to
persistence. Corpora whose test groups all appear in TRAIN never take the GLOBAL path.
"""
from __future__ import annotations

import warnings

import numpy as np

from .ar import GLOBAL
from .base import Baseline, by_group


def _mean_autocorr(recs: list[np.ndarray], norm, max_lag: int) -> np.ndarray:
    """Length-(max_lag+1) autocorrelation, averaged over recordings & channels (lag 0..max_lag)."""
    acc = np.zeros(max_lag + 1)
    wsum = 0.0
    for Y in recs:
        z = norm.z(Y)
        z = z - z.mean(0)
        var = (z * z).mean(0) + 1e-12          # per channel
        n = len(z)
        if n <= max_lag + 1:
            continue
        w = n
        for m in range(max_lag + 1):
            ac = (z[m:] * z[:n - m]).mean(0) / var    # per-channel autocorr at lag m
            acc[m] += w * ac.mean()
        wsum += w
    return acc / max(wsum, 1e-9)


class SeasonalNaive(Baseline):
    name = "seasonal"

    def __init__(self, cfg, norm):
        super().__init__(cfg, norm)
        self.pmin, self.pmax = cfg.seasonal_range
        self.min_ac = cfg.raw["baselines"]["seasonal_min_autocorr"]
        self.periods: dict[str, int | None] = {}     # group -> period (None = fallback)

    def fit(self, train: dict[int, np.ndarray], groups: dict[int, str]) -> None:
        per_group = by_group(train, groups)
        if GLOBAL in per_group:
            raise ValueError(f"group name {GLOBAL!r} is reserved for the unknown-group fallback")
        per_group[GLOBAL] = dict(train)            # pooled period for groups TRAIN never saw
        for g, recs in per_group.items():
            ac = _mean_autocorr(list(recs.values()), self.norm, self.pmax + 1)
            # local maxima within [pmin, pmax] above the autocorr floor
            peaks = [(m, ac[m]) for m in range(self.pmin, self.pmax + 1)
                     if ac[m] > ac[m - 1] and ac[m] >= ac[m + 1] and ac[m] >= self.min_ac]
            if not peaks:
                warnings.warn(f"[seasonal] group '{g}': no clear cycle "
                              f"(no autocorr peak >= {self.min_ac} in range); using persistence.")
                self.periods[g] = None
                continue
            # "dominant" = the FUNDAMENTAL: smallest-lag peak within 95% of the tallest peak
            vmax = max(v for _, v in peaks)
            self.periods[g] = min(m for m, v in peaks if v >= 0.95 * vmax)

    def predict(self, hist: np.ndarray, H: int, group: str) -> np.ndarray:
        if group not in self.periods:
            group = GLOBAL                           # TRAIN never saw this group (test_unseen)
        m = self.periods.get(group)
        if not m:                                    # fallback -> persistence
            return np.repeat(hist[-1:], H, axis=0)
        t = len(hist) - 1
        out = np.empty((H, 6))
        for h in range(1, H + 1):
            k = -(-h // m)                            # ceil(h/m)
            idx = (t + h) - k * m
            out[h - 1] = hist[idx] if idx >= 0 else hist[-1]     # idx <= t always (causal)
        return out
