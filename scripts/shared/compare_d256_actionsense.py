#!/usr/bin/env python3
"""Is d256 the same recordings as ActionSense, or an independent collection?

The question came up because d256's labels, sensor suite and subject codes are ActionSense's,
and pairing recordings by label gave a length ratio of 4.948 +- 0.085. That ratio turned out
to be weak evidence -- if d256 were independently collected at 6 Hz while ActionSense runs at
30 Hz, a 5x frame-count ratio follows from the rates alone and says nothing about provenance.

So compare the SIGNALS. F(t) = sum of the pressure grid is invariant in shape to the rescaling
d256 applied (values arrive in ~[0,1]), so correlation survives it. If d256 is ActionSense
decimated by 5, some ActionSense recording's F(t)[::5] will align with a d256 recording's F(t)
at correlation near 1. If it is an independent collection, no candidate will.

Correlation is computed over every lag, on z-scored windows, because the two caches need not
start at the same instant. The best over all ActionSense recordings of the same label is
reported, along with the best over recordings of a DIFFERENT label as a control -- a high
"match" that the control also reaches is not a match, just a smooth-signal artefact.

    python scripts/shared/compare_d256_actionsense.py --d256 data/d256_states \
        --actionsense data/actionsense_states
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def load_manifest(root, names=("manifest.jsonl",)):
    for n in names:
        p = os.path.join(root, n)
        if os.path.exists(p):
            return [json.loads(l) for l in open(p) if l.strip()]
    sys.exit(f"no manifest under {root}")


def series(root, idx, hand):
    """F(t) for one hand, from whichever form the cache stores.

    ActionSense keeps both `state_N.npy` (T,C,6) moments and `clip_N.npy` (T,C,32,32) raw
    grids; d256 keeps only the former. Take F from the moments when they exist, otherwise sum
    the grid -- which is the same quantity, since F is defined as the grid's sum.
    """
    for stem in ("state", "clip"):
        p = os.path.join(root, f"{stem}_{idx}.npy")
        if not os.path.exists(p):
            continue
        a = np.load(p)
        h = min(hand, a.shape[1] - 1)
        if a.ndim == 3:                       # (T, C, 6) moments
            return a[:, h, 0].astype(np.float64)
        if a.ndim == 4:                       # (T, C, H, W) raw grid
            return a[:, h].reshape(len(a), -1).sum(1).astype(np.float64)
    return None


def best_lag_corr(a, b):
    """Max Pearson correlation of z-scored `a` against every full-overlap window of `b`."""
    if len(a) < 8 or len(b) < len(a):
        return -1.0, 0
    a = (a - a.mean()) / (a.std() + 1e-12)
    best, at = -1.0, 0
    for s in range(0, len(b) - len(a) + 1):
        w = b[s:s + len(a)]
        sd = w.std()
        if sd < 1e-12:
            continue
        c = float(np.mean(a * ((w - w.mean()) / sd)))
        if c > best:
            best, at = c, s
    return best, at


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d256", default="data/d256_states")
    ap.add_argument("--actionsense", default="data/actionsense_states")
    ap.add_argument("--decimate", type=int, default=5)
    ap.add_argument("--n", type=int, default=6, help="d256 recordings to test")
    ap.add_argument("--hand", type=int, default=0)
    ap.add_argument("--probe", type=int, default=150,
                    help="frames of d256 to slide, taken from the middle. Matching whole "
                         "recordings fails when the two caches segmented activities at "
                         "slightly different boundaries -- a 1-3%% difference is enough to "
                         "make the d256 side longer and skip the candidate entirely. A short "
                         "probe from the middle is immune to that and still discriminative.")
    args = ap.parse_args()

    D = load_manifest(args.d256)
    A = load_manifest(args.actionsense)
    print(f"d256 {len(D)} recordings   ActionSense {len(A)} recordings   "
          f"decimate {args.decimate}x   hand {args.hand}")
    print(f"\n{'d256 rec':>26s} {'T':>5s} {'best same-label':>16s} {'best other-label':>17s}"
          f" {'n>.95':>7s} {'verdict':>9s}")

    hits = 0
    tested = 0
    for d in sorted(D, key=lambda r: -r["T"])[:args.n]:
        x = series(args.d256, d["idx"], args.hand)
        if x is None or len(x) < 16:
            continue
        if args.probe and len(x) > args.probe:
            m = (len(x) - args.probe) // 2
            x = x[m:m + args.probe]
        same, other, allc = -1.0, -1.0, []
        for a in A:
            y = series(args.actionsense, a["idx"], args.hand)
            if y is None:
                continue
            c, _ = best_lag_corr(x, y[::args.decimate])
            allc.append(c)
            if a["label"].strip().lower() == d["label"].strip().lower():
                same = max(same, c)
            else:
                other = max(other, c)
        tested += 1
        # UNIQUENESS, not a margin. What identifies a match is that exactly one candidate out
        # of hundreds reaches near-perfect correlation, and that it carries the right label.
        # A fixed margin fails here: F(t) on these activities is smooth enough that the runner-
        # up sits around 0.94 by generic similarity, which is high without meaning anything.
        n_hi = sum(1 for c in allc if c > 0.95)
        hit = same > 0.95 and n_hi == 1
        hits += hit
        print(f"{d['subject']+'/s'+str(d['session']):>26s} {d['T']:5d} {same:16.3f} "
              f"{other:17.3f} {n_hi:>7d} {'MATCH' if hit else '-':>9s}")

    print()
    if not tested:
        sys.exit("nothing testable -- check the caches")
    if hits:
        print(f"{hits}/{tested} d256 recordings match a UNIQUE ActionSense recording carrying")
        print("  the same label, at a correlation no other candidate in the corpus reaches.")
        print("  => those d256 recordings ARE ActionSense recordings, decimated.")
    else:
        print(f"0/{tested} matched. No ActionSense recording lines up with these d256 series")
        print("  above what a wrong-label candidate already achieves.")
        print("  => the signals are NOT the same recordings, whatever the labels share.")


if __name__ == "__main__":
    main()
