"""Linear AR baseline — per channel, fit PER GROUP (activity x object) on TRAIN.

    z[k] ~= b + sum_{i=1..p} phi_i * z[k-i]      (z = global TRAIN-normalized signal)

Coefficients come from statsmodels AutoReg (trend='c') fit on the group's TRAIN series (numpy
OLS fallback if statsmodels is unavailable). The order p is selected PER GROUP on VAL from the
config candidate set by iterated H-step normalized MSE. Multi-step forecasts are the standard
iterated recursion seeded by the last p observed values -> reads only `hist` (times <= t), causal.

Pooling note: a group's TRAIN recordings are concatenated for the AutoReg fit; with long clips
the few cross-recording lag rows are negligible.

Two 2026-09-09 changes, both from the EgoTouch port and both applying corpus-wide:

* ORDER SELECTION IS RECORDING-BALANCED (doctrine: every stage that produces a claim
  aggregates per recording). The old pooled MSE let a split's longest recordings choose the
  order for everyone -- on the ActionSense frozen split ten of 15 val recordings hold 84% of
  the windows. Selection now averages each recording's own H-step MSE. With equal-length
  recordings this is the same number as before.

* UNKNOWN GROUPS FALL BACK TO A GLOBAL FIT. `predict` indexed `self.order[group]` directly, a
  KeyError for any group absent from TRAIN -- which is every group of EgoTouch's test_unseen
  split, by construction. `fit` now also fits one pooled `_GLOBAL` model over all of TRAIN,
  and `predict` routes unknown groups to it. Corpora whose test groups all appear in TRAIN
  (ActionSense, OpenTouch) never take this path.
"""
from __future__ import annotations

import numpy as np

from ..weighting import weighted_mean
from .base import Baseline, by_group, predict_series

GLOBAL = "_GLOBAL"          # reserved: the pooled all-TRAIN fit unknown groups fall back to

try:
    from statsmodels.tsa.ar_model import AutoReg
    HAVE_SM = True
except Exception:                          # pragma: no cover
    HAVE_SM = False


def _fit_channel(series: np.ndarray, p: int) -> np.ndarray:
    """Fit AR(p) to a 1-D series -> [phi_1..phi_p, b]. statsmodels AutoReg, numpy OLS fallback."""
    if HAVE_SM and len(series) > p + 1:
        res = AutoReg(series, lags=p, trend="c", old_names=False).fit()
        pr = np.asarray(res.params)        # [const, L1, .., Lp]
        return np.concatenate([pr[1:1 + p], pr[:1]])
    # numpy OLS fallback: rows [z[k-1..k-p], 1] -> z[k]
    X, y = [], []
    for k in range(p, len(series)):
        X.append(np.concatenate([series[k - p:k][::-1], [1.0]]))
        y.append(series[k])
    if not X:
        return np.zeros(p + 1)
    coef, *_ = np.linalg.lstsq(np.asarray(X), np.asarray(y), rcond=None)
    return coef                            # [phi_1..phi_p, b]


class AR(Baseline):
    name = "ar"

    def __init__(self, cfg, norm):
        super().__init__(cfg, norm)
        self.orders = list(cfg.raw["baselines"]["ar_orders"])
        # coef[group][order] -> (6, p+1) = [phi_1..phi_p, b] per channel
        self.coef: dict[str, dict[int, np.ndarray]] = {}
        self.order: dict[str, int] = {}

    def fit(self, train: dict[int, np.ndarray], groups: dict[int, str]) -> None:
        per_group = by_group(train, groups)
        if GLOBAL in per_group:
            raise ValueError(f"group name {GLOBAL!r} is reserved for the unknown-group fallback")
        per_group[GLOBAL] = dict(train)            # pooled fit for groups TRAIN never saw
        for g, recs in per_group.items():
            z = np.concatenate([self.norm.z(Y) for Y in recs.values()], axis=0)   # (sumT,6)
            self.coef[g] = {}
            for p in self.orders:
                C = np.zeros((6, p + 1))
                for c in range(6):
                    C[c] = _fit_channel(z[:, c], p)
                self.coef[g][p] = C
            self.order[g] = self.orders[0]

    def select(self, val: dict[int, np.ndarray], groups: dict[int, str], H: int) -> None:
        vg = by_group(val, groups)
        global_best = self._best_order(val, groups, H)
        for g in self.coef:
            self.order[g] = self._best_order(vg.get(g, {}),
                                             {i: g for i in vg.get(g, {})}, H, default=global_best)

    def _best_order(self, data, groups, H, default=None):
        """Order minimizing RECORDING-BALANCED iterated H-step normalized MSE on `data`.

        Each recording contributes its own mean squared error once, however many origins its
        length yields; the order is the argmin of the mean over recordings. Empty -> default.
        """
        if not data:
            return default if default is not None else self.orders[0]
        # Sweep the order of every group this data can route to -- including GLOBAL, which is
        # where predict() sends a recording whose group TRAIN never saw.
        sweep = ({g for g in set(groups.values()) if g in self.coef} | {GLOBAL})
        best, best_err = default if default is not None else self.orders[0], np.inf
        for p in self.orders:
            for g in sweep:                           # temporarily set this order, measure
                self.order[g] = p
            errs = []
            for i, Y in data.items():
                yt, yh = predict_series(self, {i: Y}, {i: groups[i]}, self.cfg)
                if len(yt):
                    errs.append(float((((yh - yt) / self.norm.std) ** 2).mean()))
            err = weighted_mean(np.array(errs), np.ones(len(errs))) if errs else np.inf
            if err < best_err:
                best, best_err = p, err
        return best

    def predict(self, hist: np.ndarray, H: int, group: str) -> np.ndarray:
        if group not in self.coef:
            group = GLOBAL                            # TRAIN never saw this group (test_unseen)
        p = self.order[group]
        C = self.coef[group][p]                       # (6, p+1)
        z = self.norm.z(hist)
        phi, b = C[:, :p], C[:, p]
        buf = list(z[-p:]) if len(z) >= p else list(np.zeros((p - len(z), 6))) + list(z)
        out = np.empty((H, 6))
        for h in range(H):
            recent = np.stack(buf[-p:])[::-1]          # (p,6): lag1..lagp
            nxt = b + (phi * recent.T).sum(1)
            out[h] = nxt
            buf.append(nxt)
        return self.norm.unz(out)
