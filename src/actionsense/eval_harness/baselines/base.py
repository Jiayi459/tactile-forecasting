"""Baseline contract + the causal rolling-origin batcher (group-aware).

Contract: given `hist` = observations at times [0..t] (shape (t+1, 6)) and the recording's
fit `group`, `predict(hist, H, group)` returns (H, 6) for target times t+1..t+H. It MUST read
only `hist` (never a value at time > t). `predict_series` builds the evaluation tensor by
calling predict once per origin on the past slice Y[:t+1], so causality is structural.

Fitting is GROUP-scoped: `fit` receives TRAIN recordings + their group labels and estimates
per-group parameters (activity x object, or one "ALL" group under global scope). `select`
chooses hyperparameters on VAL. At predict time the recording's own group is used.
"""
from __future__ import annotations

import numpy as np

from ..config import Config
from ..dataset import Norm


class Baseline:
    name = "base"

    def __init__(self, cfg: Config, norm: Norm):
        self.cfg = cfg
        self.norm = norm

    def fit(self, train: dict[int, np.ndarray], groups: dict[int, str]) -> None:
        """Estimate per-group parameters from TRAIN only. Default: nothing to fit."""

    def select(self, val: dict[int, np.ndarray], groups: dict[int, str], H: int) -> None:
        """Select hyperparameters on VAL only. Default: nothing to select."""

    def predict(self, hist: np.ndarray, H: int, group: str) -> np.ndarray:
        raise NotImplementedError


def by_group(data: dict[int, np.ndarray], groups: dict[int, str]) -> dict[str, dict[int, np.ndarray]]:
    out: dict[str, dict[int, np.ndarray]] = {}
    for i, Y in data.items():
        out.setdefault(groups[i], {})[i] = Y
    return out


def origins(T: int, cfg: Config) -> np.ndarray:
    """Valid forecast origins t: enough history behind, full horizon ahead.

    The future required is `cfg.origin_horizon`, which equals `cfg.horizon` unless a horizon
    ablation pins it to its longest horizon so every horizon is scored on the same origins."""
    lo = cfg.raw["eval"]["min_history"]
    stride = cfg.raw["eval"]["stride"]
    hi = T - cfg.origin_horizon                # need t+H_origin <= T-1 (H_origin >= H)
    return np.arange(lo, hi, stride)


def predict_series(bl: Baseline, data: dict[int, np.ndarray], groups: dict[int, str],
                   cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    """Rolling-origin evaluation. Returns (ytrue, yhat), each (N_total, H, 6)."""
    H = cfg.horizon
    yts, yhs = [], []
    for i, Y in sorted(data.items()):
        g = groups[i]
        for t in origins(len(Y), cfg):
            yts.append(Y[t + 1:t + 1 + H])                    # target-time indexed
            yhs.append(bl.predict(Y[:t + 1], H, g))           # causal: only past passed in
    if not yts:
        return np.zeros((0, H, 6)), np.zeros((0, H, 6))
    return np.stack(yts), np.stack(yhs)
