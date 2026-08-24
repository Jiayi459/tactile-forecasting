"""D1: the per-taxel baseline and the explicit noise threshold, as one implementation.

The estimator was developed and validated inside scripts/opentouch/opentouch_baseline_report.py; it
lives here now because a second caller (the script that writes the corrected cache) must
not be allowed to drift from the one whose behaviour was measured. Both import this.

THE RULING (user, 2026-08-15) and what the corpus said about it (2026-08-16):

  BASELINE = a CENTRAL statistic, per taxel, pooled per shard -- the median, not a low
  quantile. A low quantile is robust against contact contamination but systematically
  undershoots, and subtracting an undershoot then clipping at zero half-wave rectifies the
  residual noise into a positive offset that scales with the noise. The median is only the
  resting level if a taxel is unloaded more than half the time; that is not assumed but
  measured, and `duty_cycle` came out below 0.5 for every taxel in all 26 shards.

  NOISE = an explicit soft threshold, x <- max(x - (base + k*sigma), 0), with sigma
  estimated ONE SIDED (MAD over frames at or below the median, since contact can only push
  a reading up) and FLOORED at half a quantisation step. The floor is not cosmetic: the
  readings are integer counts, an untouched taxel sits on one value, so the bare MAD came
  out at exactly 0.00 on all 26 shards and silently disabled the threshold.

  k = 1 was chosen from the sweep, not assumed: the curve has a sharp knee there (it removes
  70% of the post-baseline residual, later steps a few percent each), and because sigma sits
  at its floor of half a count, k=1 means "keep readings at least one count above rest" --
  the smallest meaningful threshold on quantised data. k>=2 pushed the CoP range to nearly
  the full [-1,1] and made it non-monotonic, the signature of too few surviving cells.
"""
from __future__ import annotations

import collections
import json
import os

import numpy as np

MAD_TO_SIGMA = 1.4826
GRID = 16


def quantum(frames: np.ndarray) -> float:
    """The reading's quantisation step: the smallest positive gap between distinct values."""
    v = np.unique(frames)
    if v.size < 2:
        return 1.0
    d = np.diff(v)
    d = d[d > 0]
    return float(np.min(d)) if d.size else 1.0


def estimate(frames: np.ndarray, q: float | None = None):
    """(T,N) frames -> (baseline (N,), sigma (N,)). Median, and a one-sided robust scale."""
    base = np.median(frames, axis=0)
    below = np.where(frames <= base, frames, np.nan)
    mad = MAD_TO_SIGMA * np.nan_to_num(np.nanmedian(np.abs(below - base), axis=0), nan=0.0)
    q = quantum(frames) if q is None else q
    return base, np.maximum(mad, 0.5 * q)


def duty_cycle(frames: np.ndarray, base: np.ndarray, sigma: np.ndarray, k: float = 3.0):
    """Fraction of frames a taxel spends above base + k*sigma. Above 0.5 means its median is
    contaminated by contact and the whole estimator's premise fails for that taxel."""
    return (frames > base + k * sigma).mean(0)


def moments(p: np.ndarray) -> np.ndarray:
    """(T,16,16) pressure -> (T,6) [F, CoPx, CoPy, sxx, syy, sxy], coords in [-1,1].

    Byte-identical maths to scripts/opentouch/extract_opentouch.py::moments, so a corrected cache is
    comparable with the raw one channel for channel."""
    T, H, W = p.shape
    ys = np.linspace(-1.0, 1.0, H)[:, None]
    xs = np.linspace(-1.0, 1.0, W)[None, :]
    p = np.clip(p.astype(np.float64), 0.0, None)
    F = p.sum(axis=(1, 2))
    safe = np.where(F > 0, F, 1.0)
    cx = (p * xs).sum(axis=(1, 2)) / safe
    cy = (p * ys).sum(axis=(1, 2)) / safe
    dx = xs[None, :, :] - cx[:, None, None]
    dy = ys[None, :, :] - cy[:, None, None]
    sxx = (p * dx * dx).sum(axis=(1, 2)) / safe
    syy = (p * dy * dy).sum(axis=(1, 2)) / safe
    sxy = (p * dx * dy).sum(axis=(1, 2)) / safe
    out = np.stack([F, cx, cy, sxx, syy, sxy], axis=1)
    out[F <= 0, 1:] = 0.0
    return out


def manifest(cache: str) -> list[dict]:
    with open(os.path.join(cache, "manifest.jsonl")) as f:
        return [json.loads(l) for l in f if l.strip()]


def by_shard(rows) -> dict[str, list[int]]:
    out = collections.defaultdict(list)
    for r in rows:
        out[r.get("shard", "?")].append(r["idx"])
    return {k: sorted(v) for k, v in sorted(out.items())}


def shard_frames(cache: str, idxs: list[int], max_frames: int = 20000, stride: int = 3):
    out, n = [], 0
    for i in idxs:
        p = os.path.join(cache, f"clip_{i}.npy")
        if not os.path.exists(p):
            continue
        a = np.load(p).astype(np.float32).reshape(-1, GRID * GRID)[::stride]
        out.append(a); n += len(a)
        if n >= max_frames:
            break
    if not out:
        return np.zeros((0, GRID * GRID), np.float32)
    return np.concatenate(out, 0)[:max_frames]


def shard_baselines(cache: str, rows=None, max_frames: int = 20000, stride: int = 3):
    """-> {shard: (baseline (256,), sigma (256,))}, the per-shard estimate D1 validated."""
    rows = rows if rows is not None else manifest(cache)
    out = {}
    for sh, ids in by_shard(rows).items():
        fr = shard_frames(cache, ids, max_frames, stride)
        if len(fr):
            out[sh] = estimate(fr)
    return out


def correct(clip: np.ndarray, base: np.ndarray, sigma: np.ndarray, k: float) -> np.ndarray:
    """(T,1,16,16) raw -> (T,1,16,16) with the baseline and k*sigma removed, clipped at 0."""
    flat = clip.astype(np.float32).reshape(len(clip), -1)
    return np.clip(flat - (base + k * sigma)[None, :], 0.0, None).reshape(
        len(clip), 1, GRID, GRID)
