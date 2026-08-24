"""D1: what a per-taxel baseline and an explicit noise threshold do to F and CoP.

DESCRIPTIVE ONLY -- no model, no forecast, no R^2. It measures the correction so the
correction can be chosen, and it is deliberately incapable of reporting an outcome.

THE METHOD (user's ruling, 2026-08-15). Two effects that a single low quantile conflates
are separated, each with its own interpretable parameter:

  BASELINE = a CENTRAL statistic of the non-contact level, per taxel, pooled per shard.
    Not a low quantile. A low quantile is robust against contact contamination but
    systematically UNDERSHOOTS the resting level, and subtracting an undershoot then
    clipping at zero half-wave rectifies the residual noise into a spurious positive
    offset that scales with the noise -- the rectification bias this script measures.

    Non-contact frames are identified WITHOUT a second threshold, which would only move
    the arbitrariness somewhere less visible. A taxel that is loaded less than half the
    time has its resting level AT its own median over the shard, so the per-taxel median
    IS the central statistic of its non-contact segment. That assumption is not asserted:
    duty_cycle below reports, per taxel, the fraction of frames above baseline + k*sigma,
    and any taxel above 0.5 is flagged as one whose median is contaminated by contact.

  NOISE = an explicit soft threshold, X <- max(X - k*sigma, 0), with sigma estimated ONE
    SIDED: 1.4826 * MAD over the frames at or below the median, FLOORED at half a
    quantisation step. Contact can only push a reading up, so excluding the upper half
    keeps grasps out of the noise estimate; the floor is there because the readings are
    integers and the bare MAD came out at exactly zero on all 26 shards, which silently
    disabled the threshold. k is swept, not fixed, so its influence is visible.

THE THREE DIAGNOSTICS THE USER ASKED FOR
  1. Rectification bias: among taxels that should be idle (never loaded across the whole
     shard), what fraction of frames survive the correction as non-zero? Under a correct
     baseline this tends to zero as k grows; under an undershooting one it stays high,
     which is the failure mode a low quantile hides.
  2. Per shard: dead / stuck / saturated taxel counts. "The grid has 169 live taxels and
     dead cells read ~0" is in extract_opentouch.py's docstring and was already falsified
     on 2026-08-13 (every cell read ~2900), so the census is measured, not assumed.
  3. Cross-shard stability of the per-shard baselines: if a taxel's resting level moves
     between shards, the baseline must stay per shard and cannot be estimated once
     corpus-wide -- and a session-specific offset is exactly what a location-level split
     stops a model from memorising.

    python scripts/opentouch/opentouch_baseline_report.py --cache ~/opentouch/cache
    python scripts/opentouch/opentouch_baseline_report.py --k 0,1,2,3,5 --shards office_ml_p1,eat_ygf_p1
"""
from __future__ import annotations

import argparse
import collections
import json
import os

import numpy as np

# The estimator now lives in src/opentouch/baseline.py so the script that WRITES the
# corrected cache cannot drift from the one whose behaviour was measured here.
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.opentouch.baseline import estimate, quantum        # noqa: E402,F401

MAD_TO_SIGMA = 1.4826


def load_manifest(cache):
    return [json.loads(l) for l in open(os.path.join(cache, "manifest.jsonl"))]


def shard_frames(cache, rows, max_frames, stride):
    """(F, 256) subsampled pressure frames for one shard, plus the clips they came from."""
    out, used = [], 0
    for r in rows:
        p = os.path.join(cache, f"clip_{r['idx']}.npy")
        if not os.path.exists(p):
            continue
        a = np.load(p).astype(np.float32)[:, 0].reshape(-1, 256)[::stride]
        out.append(a)
        used += 1
        if sum(len(x) for x in out) >= max_frames:
            break
    if not out:
        return np.zeros((0, 256), np.float32), 0
    return np.concatenate(out, 0)[:max_frames], used




def moments(p):
    """(N,256) -> (F, cx, cy) with the same maths as extract_opentouch.moments."""
    q = np.clip(p, 0.0, None).reshape(-1, 16, 16).astype(np.float64)
    ys = np.linspace(-1, 1, 16)[:, None]
    xs = np.linspace(-1, 1, 16)[None, :]
    F = q.sum((1, 2))
    safe = np.where(F > 0, F, 1.0)
    return F, (q * xs).sum((1, 2)) / safe, (q * ys).sum((1, 2)) / safe


def census(frames, base, sigma):
    """dead / stuck / saturated counts. Measured, because the documented assumption that
    dead cells read ~0 is false on this corpus -- every shard reports zero dead cells.

    SATURATED is "rails at its own ceiling", not "reaches the pooled 99.9th percentile".
    The first version used the latter and flagged 212-256 of 256 cells on every shard,
    which measured only how tight the pooled distribution is. A railed cell sits AT its
    maximum for a sustained fraction of the recording; a cell that merely touches its max
    once has not saturated. Equality is EXACT rather than within a tolerance: these are
    quantised counts, a railed cell returns the same integer over and over, and a tolerance
    of half a sigma re-admitted half the grid on the first attempt."""
    rng = frames.max(0) - frames.min(0)
    at_max = (frames == frames.max(0)).mean(0)      # EXACT equality: a rail is a rail
    return {
        "dead": int((frames.max(0) <= 1e-6).sum()),                    # never reads
        "stuck": int(((rng <= 3 * sigma) & (frames.max(0) > 1e-6)).sum()),
        "saturated": int(((at_max > 0.05) & (rng > 3 * sigma)).sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=os.path.expanduser("~/opentouch/cache"))
    ap.add_argument("--k", default="0,1,2,3,5", help="soft-threshold multiples of sigma")
    ap.add_argument("--shards", help="comma-separated shard names; default: all")
    ap.add_argument("--max-frames", type=int, default=20000, help="per shard, subsampled")
    ap.add_argument("--stride", type=int, default=3, help="frame subsampling for the fit")
    a = ap.parse_args()
    ks = [float(x) for x in a.k.split(",")]

    man = load_manifest(a.cache)
    by_shard = collections.defaultdict(list)
    for r in man:
        by_shard[r.get("shard", "?")].append(r)
    shards = ([s.strip() for s in a.shards.split(",")] if a.shards
              else sorted(by_shard))

    bases = {}
    print(f"{'shard':26s} {'clips':>6s} {'frames':>7s} {'base med':>9s} "
          f"{'sigma med':>9s} {'dead':>5s} {'stuck':>6s} {'sat':>4s} {'duty>0.5':>9s}")
    for s in shards:
        frames, nclip = shard_frames(a.cache, by_shard[s], a.max_frames, a.stride)
        if len(frames) == 0:
            print(f"{s:26s} {'-- no clip_*.npy (extracted with --no-clips?)':>40s}")
            continue
        base, sigma = estimate(frames)
        bases[s] = base
        duty = ((frames > base + 3 * sigma).mean(0))
        c = census(frames, base, sigma)
        print(f"{s:26s} {nclip:6d} {len(frames):7d} {np.median(base):9.1f} "
              f"{np.median(sigma):9.2f} {c['dead']:5d} {c['stuck']:6d} {c['saturated']:4d} "
              f"{int((duty > 0.5).sum()):9d}")

    if not bases:
        raise SystemExit("no raw maps found -- this report needs clip_*.npy in the cache")

    print("\n=== effect on F and CoP (first shard shown in full, per k) ===")
    s0 = next(iter(bases))
    frames, _ = shard_frames(a.cache, by_shard[s0], a.max_frames, a.stride)
    base, sigma = estimate(frames)
    F0, cx0, cy0 = moments(frames)
    print(f"shard {s0}")
    print(f"{'k':>4s} {'F mean':>12s} {'F std':>10s} {'F cv':>7s} "
          f"{'CoPx range':>11s} {'CoPy range':>11s} {'rect. bias':>11s}")
    # Idle = smallest EXCURSION above its own baseline, not smallest maximum: with every
    # cell resting near 3050, ranking by max ranks the baselines, not the activity.
    exc = frames.max(0) - base
    idle = exc <= np.percentile(exc, 5)
    print(f"{'raw':>4s} {F0.mean():12.1f} {F0.std():10.1f} {F0.std()/max(F0.mean(),1e-9):7.4f} "
          f"{cx0.max()-cx0.min():11.4f} {cy0.max()-cy0.min():11.4f} {'--':>11s}")
    for k in ks:
        corr = np.clip(frames - (base + k * sigma), 0, None)
        F, cx, cy = moments(corr)
        rect = float((corr[:, idle] > 0).mean()) if idle.any() else float("nan")
        print(f"{k:4.1f} {F.mean():12.1f} {F.std():10.1f} "
              f"{F.std()/max(F.mean(),1e-9):7.4f} {cx.max()-cx.min():11.4f} "
              f"{cy.max()-cy.min():11.4f} {rect:11.4f}")

    if len(bases) > 1:
        B = np.stack([bases[s] for s in bases])                   # (S,256)
        spread = (np.percentile(B, 75, 0) - np.percentile(B, 25, 0))
        rel = spread / np.maximum(np.median(B, 0), 1e-9)
        print(f"\n=== cross-shard baseline stability ({len(bases)} shards) ===")
        print(f"per-taxel baseline IQR across shards: median {np.median(spread):.1f}, "
              f"p90 {np.percentile(spread, 90):.1f}")
        print(f"relative to the level:                median {np.median(rel):.3f}, "
              f"p90 {np.percentile(rel, 90):.3f}")
        print("Small relative spread means one corpus-wide baseline would do for most "
              "taxels; a heavy upper tail means it would not for the rest, and a residual "
              "session offset is exactly what a location-held-out model cannot memorise. "
              "Read the median and the p90 together -- they can disagree.")


if __name__ == "__main__":
    raise SystemExit(main())
