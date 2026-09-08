#!/usr/bin/env python3
"""Per-split eligibility and window concentration for data/egotouch_states/.

Two questions the extractor's summary line cannot answer, both of which decide whether a
result is reportable:

1. ELIGIBILITY IS PER SPLIT, NOT CORPUS-WIDE. "69.8% clear 5 s" is the corpus number. What
   matters is how many recordings survive in the split a metric is computed on -- a test split
   that drops to a handful of recordings cannot support a per-action claim no matter how
   healthy the corpus looks. Reported at several history lengths, because the arms differ.

2. ROLLING-ORIGIN WINDOWS SCALE WITH RECORDING LENGTH. EgoTouch's lengths span 43 to 19853
   frames -- a 460x range, against ActionSense's far flatter distribution. With stride 1 an
   11-minute recording contributes ~6.6k origins and a 5-second one contributes 1, so pooling
   windows without balancing lets a few recordings own the metric. This prints what share of
   the windows the top recordings hold, which is the number that says whether clip-balancing
   is a nicety or load-bearing here.
"""
from __future__ import annotations

import argparse
import json
import os

HORIZON = 10          # steps at 10 Hz, from configs/egotouch/eval_harness.yaml
DOWNSAMPLE = 3


def origins(t_raw: int, min_history: int) -> int:
    """Rolling-origin count at stride 1 for one recording, matching eligible_recordings()."""
    t = t_raw // DOWNSAMPLE
    return max(0, t - min_history - HORIZON + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", default=os.path.join("data", "egotouch_states"))
    ap.add_argument("--histories", type=int, nargs="+", default=[20, 40, 80, 100],
                    help="min_history in frames at 10 Hz (20=2 s, 40=4 s, 80=8 s, 100=10 s)")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(os.path.join(args.states, "manifest.jsonl"))]
    with open(os.path.join(args.states, "splits.json")) as fh:
        sp = json.load(fh)
    by_idx = {r["idx"]: r for r in rows}
    splits = {k: sp[k] for k in ("train", "val", "test", "test_unseen")}

    for mh in args.histories:
        print(f"\n=== min_history={mh} frames @10Hz ({mh/10:.0f} s) + horizon {HORIZON} "
              f"=> needs T >= {(mh + HORIZON) * DOWNSAMPLE} raw frames "
              f"({(mh + HORIZON) * DOWNSAMPLE / 30:.1f} s) ===")
        print(f"{'split':<12}{'recs':>7}{'eligible':>10}{'%':>7}{'windows':>10}"
              f"{'top1%':>8}{'top10':>8}{'groups':>8}")
        for name, idxs in splits.items():
            recs = [by_idx[i] for i in idxs if i in by_idx]
            w = sorted((origins(r["T"], mh) for r in recs), reverse=True)
            elig = [x for x in w if x > 0]
            tot = sum(elig)
            top1 = sum(elig[: max(1, len(elig) // 100)]) / tot * 100 if tot else 0.0
            top10 = sum(elig[:10]) / tot * 100 if tot else 0.0
            groups = len({(r["label"].split()[0], r["label"].split()[-1])
                          for r in recs if origins(r["T"], mh) > 0})
            print(f"{name:<12}{len(recs):>7}{len(elig):>10}{100*len(elig)/max(len(recs),1):>6.1f}%"
                  f"{tot:>10}{top1:>7.1f}%{top10:>7.1f}%{groups:>8}")

    lens = sorted((r["T"] for r in rows), reverse=True)
    print(f"\nlongest recordings (raw frames @30Hz): {lens[:5]}")
    print(f"  they are {lens[0]/30:.0f} s … {lens[4]/30:.0f} s; median is {lens[len(lens)//2]/30:.1f} s")


if __name__ == "__main__":
    main()
