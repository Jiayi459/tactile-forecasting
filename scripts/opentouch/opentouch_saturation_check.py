"""Is the OpenTouch grid clipping during contact, or does it only look that way?

The D1 census flagged 135-241 of 256 taxels per shard as sitting at their own maximum for
more than 5% of frames (2026-08-16). If that is a real ceiling, F is a clipped quantity and
every force conclusion inherits the distortion -- so it is worth one dedicated measurement
rather than a footnote. The paper says the calibrated range is 0.02-50 kPa and that values
outside it "may saturate" (arXiv:2512.16842), which makes the question live but does not
answer it.

A HARDWARE RAIL AND A PLATEAU LOOK THE SAME PER CELL AND DIFFERENT IN AGGREGATE, so the
test is aggregate:

  1. A rail is ONE VALUE. Real converters stop at 4095, 1023, 65535 -- a specific number
     shared by every cell that reaches it. If each cell's maximum is its own arbitrary
     number, cells are merely peaking, not railing.
  2. A rail is SHARD-INVARIANT. It is a property of the electronics, not of the session.
  3. A rail is a PLATEAU IN TIME. A clipped press holds the ceiling for a run of frames; a
     noisy peak touches its maximum once. Run lengths separate these.
  4. A rail COSTS FORCE. The share of contact-time samples pinned at the ceiling bounds how
     much of F was truncated.

    python scripts/opentouch/opentouch_saturation_check.py --cache ~/opentouch/cache
"""
from __future__ import annotations

import argparse
import collections
import json
import os

import numpy as np


def load_manifest(cache):
    return [json.loads(l) for l in open(os.path.join(cache, "manifest.jsonl"))]


def shard_frames(cache, rows, max_frames, stride):
    out = []
    for r in rows:
        p = os.path.join(cache, f"clip_{r['idx']}.npy")
        if not os.path.exists(p):
            continue
        out.append(np.load(p).astype(np.float32)[:, 0].reshape(-1, 256)[::stride])
        if sum(len(x) for x in out) >= max_frames:
            break
    return np.concatenate(out, 0)[:max_frames] if out else np.zeros((0, 256), np.float32)


def run_lengths(mask_1d):
    """Lengths of consecutive True runs -- a clipped press plateaus, a noise peak does not."""
    if not mask_1d.any():
        return np.array([], dtype=int)
    d = np.diff(np.concatenate([[0], mask_1d.view(np.int8), [0]]))
    return np.flatnonzero(d < 0) - np.flatnonzero(d > 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=os.path.expanduser("~/opentouch/cache"))
    ap.add_argument("--max-frames", type=int, default=20000)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--shards", help="default: all")
    a = ap.parse_args()

    man = load_manifest(a.cache)
    by_shard = collections.defaultdict(list)
    for r in man:
        by_shard[r.get("shard", "?")].append(r)
    shards = ([s.strip() for s in a.shards.split(",")] if a.shards else sorted(by_shard))

    all_max, per_shard_max, runs, pinned_share, force_share = [], {}, [], [], []
    for s in shards:
        fr = shard_frames(a.cache, by_shard[s], a.max_frames, a.stride)
        if len(fr) == 0:
            continue
        cmax = fr.max(0)
        all_max.append(cmax)
        per_shard_max[s] = float(cmax.max())

        # cells that spend real time at their own ceiling
        at = fr == cmax
        hot = np.flatnonzero(at.mean(0) > 0.05)
        for c in hot[:40]:
            runs.append(run_lengths(at[:, c]))
        pinned_share.append(float(at[:, hot].mean()) if hot.size else 0.0)
        base = np.median(fr, axis=0)
        corr = np.clip(fr - base, 0, None)
        tot = corr.sum()
        force_share.append(float(corr[:, hot].sum() / tot) if hot.size and tot > 0 else 0.0)

    if not all_max:
        raise SystemExit("no raw maps in the cache")

    M = np.concatenate(all_max)
    print("=== 1. is there ONE ceiling, or 256 different peaks? ===")
    cnt = collections.Counter(M.tolist())
    top = cnt.most_common(8)
    print(f"{len(cnt)} distinct per-cell maxima over {M.size} (cell, shard) pairs")
    print(f"{'value':>10s} {'cells':>7s} {'share':>7s}")
    for v, n in top:
        print(f"{v:10.1f} {n:7d} {n / M.size:7.1%}")
    print(f"corpus max {M.max():.1f} | a converter rail would be a round power of two "
          f"(1023 / 4095 / 65535) shared by most of the cells above")

    print("\n=== 2. is the ceiling the same in every shard? ===")
    vals = np.array(list(per_shard_max.values()))
    print(f"per-shard maximum: min {vals.min():.1f} max {vals.max():.1f} "
          f"distinct {len(set(vals.tolist()))} of {len(vals)} shards")
    print("A hardware rail is a property of the electronics: it would be identical "
          "everywhere. Session-specific maxima mean these are peaks, not rails.")

    print("\n=== 3. does the ceiling hold, or is it touched once? ===")
    R = np.concatenate([r for r in runs if r.size]) if any(r.size for r in runs) else np.array([1])
    print(f"at-max run lengths over {len(runs)} flagged cells: "
          f"median {np.median(R):.0f}  p90 {np.percentile(R, 90):.0f}  max {R.max():.0f} frames")
    print(f"fraction of runs that are a single frame: {(R == 1).mean():.1%}")
    print("Clipping plateaus (a press stays against the ceiling for many frames). "
          "Runs of one frame are a maximum being touched, which every cell does.")

    print("\n=== 4. how much force could be truncated? ===")
    print(f"samples pinned at the ceiling, among flagged cells: "
          f"{np.mean(pinned_share):.2%} (mean over shards)")
    print(f"share of baseline-removed F carried by those cells: "
          f"{np.mean(force_share):.1%}")
    print("\nVERDICT depends on 1-3 together: one shared round value, identical across "
          "shards, held for runs of many frames = real clipping, and F is truncated. "
          "Many distinct session-specific maxima touched for a frame at a time = the D1 "
          "census criterion was measuring peaks, and there is nothing to correct.")


if __name__ == "__main__":
    raise SystemExit(main())
