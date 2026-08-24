"""Curve-shape metrics that both sensors' evaluations use.

Deliberately at the top of `src/` rather than inside either package. `src/opentouch/metrics.py`
is a FORK of the harness's, kept separate on purpose; this is the opposite case -- one
definition that must not drift between ActionSense and OpenTouch, since the whole point of
computing it on both is comparing them.
"""
from __future__ import annotations

import numpy as np


def hausdorff_curves(pred: np.ndarray, true: np.ndarray, H: int) -> np.ndarray:
    """Symmetric Hausdorff distance between two H-step curves, per forecast. -> (N,)

    Each forecast is a SET of points in (time, value): time is h/H, so the horizon spans
    [0,1], and `pred`/`true` are expected ALREADY SCALED (dividing by the truth's own
    standard deviation over the horizon makes the axes commensurate and the result
    dimensionless). Both choices are as arbitrary as any two-axis metric's; what matters is
    that they are identical for every model, so ratios between models mean something even
    where the absolute number does not.

    SHIFT-INVARIANT, which is why it may be computed on residual-over-persistence targets
    just as well as on absolute ones: adding the same constant to `pred` and `true` moves
    every point of both sets together and leaves every pairwise distance unchanged.

    WHY IT IS WORTH HAVING BESIDE MSE. MSE is pointwise, so a flat forecast through the
    middle of an oscillation scores far better than its shape deserves -- which is what
    every arm in this project does (SESSION_LOG 2026-08-20). Hausdorff asks how far the
    WORST point of one curve is from the whole of the other, so a flat line through a
    swinging signal is charged roughly the amplitude, and a model that tracks the swing is
    not. On a unit sine: perfect 0.000, phase-shifted by 0.4 rad 0.301, the mean 0.995.
    """
    t = np.arange(H) / H
    dt = (t[:, None] - t[None, :]) ** 2
    dv = (pred[:, :, None] - true[:, None, :]) ** 2
    d = np.sqrt(dt[None, :, :] + dv)
    return np.maximum(d.min(axis=2).max(axis=1), d.min(axis=1).max(axis=1))


def hausdorff_scaled(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """(N,H) curves in any units -> (N,) Hausdorff, each scaled by its own truth's sd.

    Forecasts whose truth is constant over the horizon are returned as NaN: there is no
    shape to compare against, and calling that a perfect match would flatter every model.
    """
    H = pred.shape[1]
    sd = true.std(axis=1)
    out = np.full(len(pred), np.nan)
    ok = sd > 0
    if ok.any():
        out[ok] = hausdorff_curves(pred[ok] / sd[ok, None], true[ok] / sd[ok, None], H)
    return out
