"""Merge several prediction directories into one, so a single figure can show every arm.

Each run writes its own `clip_<idx>.npz` holding the arms it trained. A figure comparing
backbones needs them side by side, and the alternative -- teaching the plotter to read N
directories and reconcile their origins -- puts the reconciliation somewhere it cannot be
inspected. Merging first makes the result a file you can look at.

NAME COLLISIONS ARE THE POINT OF `--prefix`. Two runs both call an arm `cnn` while meaning
different backbones behind it, so merging them unprefixed would silently keep whichever came
last. A prefix per source is required whenever any name appears twice.

WHAT MUST MATCH. The truth, the origins and the channel order are properties of the data, not
of the run, so they are asserted equal across sources rather than taken from the first. A
mismatch means the runs scored different things and no figure over both is meaningful.

    python scripts/shared/merge_preds.py --out runs/merged \\
        runs/preds_d1_pg:pg runs/preds_d1_map2:s2s
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sources", nargs="+",
                    help="DIR or DIR:PREFIX -- the prefix is prepended to every model name")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    srcs = []
    for s in a.sources:
        d, _, pre = s.partition(":")
        if not os.path.isdir(d):
            raise SystemExit(f"not a directory: {d}")
        srcs.append((d, pre))

    per_clip: dict[int, dict] = {}
    meta: dict[int, dict] = {}
    seen: dict[str, str] = {}
    for d, pre in srcs:
        files = sorted(glob.glob(os.path.join(d, "clip_*.npz")))
        if not files:
            raise SystemExit(f"no clip_*.npz under {d}")
        for path in files:
            i = int(os.path.basename(path)[5:-4])
            z = np.load(path, allow_pickle=True)
            if i in meta:
                m = meta[i]
                if not np.array_equal(m["origins"], z["origins"]):
                    raise SystemExit(f"clip {i}: origins differ between sources -- the runs "
                                     f"scored different windows, so one figure over both "
                                     f"would compare unlike things")
                if not np.allclose(m["y"], z["y"], equal_nan=True):
                    raise SystemExit(f"clip {i}: the truth differs between sources")
                if list(m["channels"]) != list(z["channels"]):
                    raise SystemExit(f"clip {i}: channel order differs between sources")
            else:
                meta[i] = {k: z[k] for k in ("y", "origins", "fps", "channels")
                           if k in z.files}
                for k in ("action", "object_name", "tag"):
                    if k in z.files:
                        meta[i][k] = z[k]
            for k in z.files:
                if not k.startswith(("mu_", "sigma_")):
                    continue
                head, name = k.split("_", 1)
                new = f"{pre}_{name}" if pre else name
                key = f"{head}_{new}"
                if head == "mu_"[:2] and new in seen and seen[new] != d:
                    raise SystemExit(
                        f"model name {new!r} comes from both {seen[new]} and {d}; give at "
                        f"least one source a prefix (DIR:PREFIX) so they stay distinct")
                seen.setdefault(new, d)
                per_clip.setdefault(i, {})[key] = z[k]

    os.makedirs(a.out, exist_ok=True)
    for i, arrays in sorted(per_clip.items()):
        np.savez_compressed(os.path.join(a.out, f"clip_{i}.npz"), **meta[i], **arrays)
    models = sorted({k[3:] for arrs in per_clip.values() for k in arrs if k.startswith("mu_")})
    print(f"merged {len(per_clip)} clips x {len(models)} models -> {a.out}")
    print("  " + ", ".join(models))

    # Sources need not cover the same clips -- baselines are exported for one frozen test
    # split while a cross-validated arm covers every recording -- and a clip holding only
    # some of the models draws a figure with rows missing and nothing said. Report it here,
    # where the asymmetry is created.
    full = sum(1 for arrs in per_clip.values()
               if {k[3:] for k in arrs if k.startswith("mu_")} == set(models))
    if full < len(per_clip):
        print(f"  NOTE: only {full}/{len(per_clip)} clips carry all {len(models)} models; "
              f"the rest are partial. Per source:")
        for d, pre in srcs:
            n = len(glob.glob(os.path.join(d, "clip_*.npz")))
            print(f"    {n:5d} clips  {d}" + (f"  (prefix {pre})" if pre else ""))


if __name__ == "__main__":
    raise SystemExit(main())
