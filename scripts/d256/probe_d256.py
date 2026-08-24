#!/usr/bin/env python3
"""Inventory the d256 / `Dataset256` release: counts, classes, splits, schema.

Everything in docs/d256.md is produced by this script, so the doc can be re-derived rather
than trusted. Two modes:

  --fast   (default) path walk only. Counts every clip by group/split/subject/class without
           unpickling, so it is seconds on 80k files. Class *names* need one clip per session
           directory (275 reads), which it also does.
  --full   unpickle every clip: verifies the session-dir == label_idx invariant on all of them,
           confirms the array schema is uniform, and reports value ranges. Minutes, not seconds.

Usage:
    python scripts/d256/probe_d256.py --root ~/forcevision
    python scripts/d256/probe_d256.py --root ~/forcevision --full --out docs/d256_inventory.csv
"""
from __future__ import annotations

import argparse
import collections
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src import d256  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(os.path.expanduser("~"), "forcevision"),
                    help="fetch --dest, or the Dataset256 dir itself")
    ap.add_argument("--full", action="store_true",
                    help="unpickle every clip to verify labels/schema (slower)")
    ap.add_argument("--out", default=None, help="write the per-cell counts to this CSV")
    args = ap.parse_args()

    root = d256.root_of(args.root)
    if not os.path.isdir(root):
        sys.exit(f"not a directory: {root}")
    print(f"root: {root}\n")

    cnt = d256.counts(root)
    if not cnt:
        sys.exit("no clips found -- is --root pointing at the fetch --dest?")
    total = sum(cnt.values())

    groups = sorted({k[0] for k in cnt})
    subjects = sorted({k[2] for k in cnt})
    sessions = sorted({k[3] for k in cnt})
    print(f"clips {total}   groups {len(groups)}   subjects {len(subjects)}   "
          f"classes {len(sessions)}   splits {sorted({k[1] for k in cnt})}")

    print("\n== group x split ==")
    gs = collections.Counter()
    for (g, sp, _, _), n in cnt.items():
        gs[(g, sp)] += n
    for g in groups:
        tr, va = gs[(g, "train")], gs[(g, "val")]
        pct = 100 * va / (tr + va) if tr + va else 0
        print(f"  {g:9s} train {tr:6d}   val {va:5d}   total {tr+va:6d}   val {pct:.1f}%")

    print("\n== group x subject ==")
    print("  " + " " * 9 + "".join(f"{s:>8s}" for s in subjects))
    for g in groups:
        row = collections.Counter()
        for (gg, _, su, _), n in cnt.items():
            if gg == g:
                row[su] += n
        print(f"  {g:9s}" + "".join(f"{row[s]:>8d}" for s in subjects))

    print("\n== val is which (subject, class)? ==")
    for g in groups:
        v = sorted({(k[2], k[3]) for k in cnt if k[0] == g and k[1] == "val"})
        print(f"  {g:9s} {v}")

    names = d256.label_map(root)
    print(f"\n== classes ({len(names)}) ==")
    per = collections.Counter()
    for (_, _, _, s), n in cnt.items():
        per[s] += n
    for s in sessions:
        print(f"  {s:2d}  {per[s]:6d} clips  {names.get(s, '<unknown>')}")
    if per:
        lo, hi = min(per.values()), max(per.values())
        print(f"  imbalance: {hi}/{lo} = {hi/lo:.1f}x between largest and smallest class")

    verbs, nouns = d256.ego4d_vocab(root)
    print(f"\n== ego4d vocab == verbs {len(verbs) if verbs is not None else 'MISSING'}   "
          f"nouns {len(nouns) if nouns is not None else 'MISSING'}")

    if args.out:
        with open(args.out, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["group", "split", "subject", "class", "label_text", "clips"])
            for k in sorted(cnt):
                w.writerow([k[0], k[1], k[2], k[3], names.get(k[3], ""), cnt[k]])
        print(f"\nwrote {args.out}")

    if not args.full:
        print("\n(--fast: labels/schema checked on one clip per session dir only; "
              "use --full to verify all)")
        return

    print(f"\n== --full: unpickling all {total} clips ==")
    shapes, dtypes, bad = collections.Counter(), collections.Counter(), []
    ranges = collections.defaultdict(lambda: [float("inf"), float("-inf")])
    n = 0
    for path in d256.iter_paths(root):
        try:
            c = d256.load_clip(path)          # raises if session dir != label_idx
        except ValueError as exc:
            bad.append(str(exc))
            continue
        shapes[tuple(sorted((k, v.shape) for k, v in c.signal.items()))] += 1
        dtypes[tuple(sorted((k, str(v.dtype)) for k, v in c.signal.items()))] += 1
        for k, v in c.signal.items():
            r = ranges[k]
            r[0] = min(r[0], float(v.min())); r[1] = max(r[1], float(v.max()))
        n += 1
        if n % 10000 == 0:
            print(f"    {n}/{total}", flush=True)

    print(f"\n  label_idx == session dir: {n}/{total} ok, {len(bad)} violations")
    for b in bad[:5]:
        print(f"    {b}")
    print(f"  distinct array-shape schemas: {len(shapes)}")
    print(f"  distinct dtype schemas:       {len(dtypes)}")
    if len(shapes) == 1:
        for k, v in shapes.most_common(1)[0][0]:
            print(f"    {k:22s} {v}")
    print("  observed value ranges:")
    for k in sorted(ranges):
        print(f"    {k:22s} [{ranges[k][0]:.4g}, {ranges[k][1]:.4g}]")


if __name__ == "__main__":
    main()
