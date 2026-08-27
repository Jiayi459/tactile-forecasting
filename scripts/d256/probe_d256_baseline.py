#!/usr/bin/env python3
"""READ-ONLY: would OpenTouch's D1 per-taxel baseline correction work on d256?

WHY ASK. Inside OpenTouch -- same protocol, same folds, same data -- the only difference
between the `raw` and `d1` runs is this correction, and it took AR from 0.178 to 0.425 and
probGRU from 0.237 to 0.428. d256 applies no correction (OQ-D2) and its AR sits at 0.067,
in the same band as OpenTouch UNCORRECTED. That is suggestive enough to test, and the
raw->d1 delta is protocol-controlled, so it is evidence about the correction rather than
about the split.

WHAT D1 ACTUALLY IS, since this was misread once (SESSION_LOG 2026-08-26续3 proved a *global*
constant offset cannot change skill, and then wrongly concluded the correction was pointless):

    corrected = clip(raw - (base + k*sigma), 0, None)

`base` is a PER-TAXEL median, so removing it changes the spatial distribution and therefore
CoP. `k*sigma` is a per-taxel noise floor and the clip is a rectification, so taxels whose
reading is indistinguishable from noise drop out of `F = sum(p)` entirely. That is denoising,
not de-biasing, and the shift-invariance argument does not cover it.

THE THING THAT MIGHT NOT TRANSFER. `estimate()` floors sigma at half the QUANTISATION STEP,
found by `quantum()` as the smallest positive gap between distinct readings. OpenTouch's grids
are raw ADC counts, where that is a real LSB. d256's are pre-scaled to ~[0,1], and if that
rescaling was per-clip or involved interpolation the "step" may be an artefact of float
spacing rather than the sensor's resolution -- in which case sigma is meaningless and the
threshold arbitrary. This script measures that before anything is applied.

Writes nothing. Prints only.

    python scripts/d256/probe_d256_baseline.py --root ~/forcevision --n 8
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src import d256  # noqa: E402
from src.actionsense.physical_state import frame_state  # noqa: E402
from src.opentouch.baseline import estimate, quantum  # noqa: E402
from scripts.d256.extract_d256_states import rebuild  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(os.path.expanduser("~"), "forcevision"))
    ap.add_argument("--group", default="signals1")
    ap.add_argument("--n", type=int, default=8, help="cells to sample")
    ap.add_argument("--k", type=float, default=1.0, help="noise-threshold multiple (D1 used 1.0)")
    args = ap.parse_args()

    root = d256.root_of(args.root)
    cells = {}
    for p in d256.iter_paths(root, groups=(args.group,)):
        m = d256.parse_path(p)
        cells.setdefault((m["split"], m["subject"], m["session"]), []).append(m["clip_id"])
    keys = sorted(cells)[:: max(1, len(cells) // args.n)][:args.n]
    if not keys:
        sys.exit(f"no cells under {root}/{args.group}")

    print(f"sampling {len(keys)} cells from {args.group}\n")
    print(f"{'cell':22s} {'T':>5s} {'quantum':>10s} {'#levels':>8s} {'base med':>9s} "
          f"{'sigma med':>10s} {'kept%':>7s} {'F kept%':>8s} {'CoP shift':>10s}")

    agg = []
    for split, subj, sess in keys:
        cell_dir = os.path.join(root, args.group, split, subj, str(sess))
        n = len(cells[(split, subj, sess)])
        streams, _ = next(rebuild(cell_dir, n, verify=False))
        grids = np.stack([streams["tactile-glove-left"],
                          streams["tactile-glove-right"]], axis=1)      # (T,2,32,32)
        T = len(grids)
        flat = grids.reshape(T, -1).astype(np.float64)
        q = quantum(flat)
        levels = len(np.unique(flat))
        base, sigma = estimate(flat, q)

        corr = np.clip(flat - (base + args.k * sigma)[None, :], 0.0, None)
        kept = float((corr > 0).mean()) * 100                       # taxel-frames surviving
        f_raw, f_corr = flat.sum(1), corr.sum(1)
        f_kept = float(f_corr.sum() / f_raw.sum()) * 100

        st_raw = np.stack([frame_state(grids[t]) for t in range(T)])
        cg = corr.reshape(T, 2, 32, 32)
        st_cor = np.stack([frame_state(cg[t]) for t in range(T)])
        shift = float(np.nanmean(np.abs(st_cor[:, :, 1:3] - st_raw[:, :, 1:3])))

        print(f"{subj}/s{sess:<2d} {split:<10s} {T:5d} {q:10.3g} {levels:8d} "
              f"{np.median(base):9.4f} {np.median(sigma):10.5f} {kept:6.1f}% {f_kept:7.2f}% "
              f"{shift:10.4f}")
        agg.append((q, levels, kept, f_kept, shift))

    a = np.array(agg)
    print(f"\nmedian over cells: quantum {np.median(a[:,0]):.4g}   levels {np.median(a[:,1]):.0f}"
          f"   taxel-frames kept {np.median(a[:,2]):.1f}%"
          f"   F kept {np.median(a[:,3]):.2f}%   CoP shift {np.median(a[:,4]):.4f}")

    print("\n=== verdict ===")
    qs = a[:, 0]
    if np.allclose(qs, qs[0], rtol=1e-6):
        print(f"  quantum is IDENTICAL across cells ({qs[0]:.4g}) -> a real sensor LSB, shared")
        print("     electronics property. estimate()'s sigma floor is meaningful here.")
    else:
        print(f"  quantum VARIES across cells ({qs.min():.4g} .. {qs.max():.4g}) -> it is not a")
        print("     shared quantisation step. Most likely the pre-scaling to [0,1] was per-clip,")
        print("     so this is float spacing, not sensor resolution, and sigma's floor -- hence")
        print("     the whole threshold -- would be arbitrary. D1 must NOT be transplanted as is.")
    if np.median(a[:, 3]) < 1:
        print(f"  F drops to {np.median(a[:,3]):.2f}% of raw: the threshold erases essentially")
        print("     the whole signal. k is far too large for this scaling.")
    elif np.median(a[:, 3]) > 95:
        print(f"  F keeps {np.median(a[:,3]):.2f}%: the threshold removes almost nothing, so the")
        print("     correction would be close to a no-op and cannot explain OpenTouch's gain.")
    else:
        print(f"  F keeps {np.median(a[:,3]):.2f}% while {np.median(a[:,2]):.1f}% of taxel-frames")
        print("     survive -- the correction is doing real work rather than nothing or everything.")


if __name__ == "__main__":
    main()
