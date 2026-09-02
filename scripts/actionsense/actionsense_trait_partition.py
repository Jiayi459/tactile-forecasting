"""Partition ActionSense by trait class (smooth/abrupt), from the manifest labels.

The class of a recording is decided by the VERB of its manifest label -- "Slice a cucumber"
-> `slice` -> the verdict in src/actionsense/trait.py, which since 2026-09-02 matches
OpenTouch's verdict for the corresponding action. Nothing here re-derives a class: this
script joins the labels to that table and counts, so the only place a verdict lives is the
pre-registered file.

Counts are emitted to CSV rather than printed alone because a trait number transcribed by
hand is exactly the one that later fails to reproduce (SESSION_LOG 2026-09-01续5).

Windows are counted with the harness's own `origins()` at the frozen rate/history/horizon,
so "windows" here is the number of scored forecast origins, not a frame count.

    python scripts/actionsense/actionsense_trait_partition.py
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.actionsense import trait as TR                               # noqa: E402
from src.actionsense.eval_harness.baselines.base import origins       # noqa: E402
from src.actionsense.eval_harness.config import load_config           # noqa: E402
from src.actionsense.eval_harness.splits import load_splits, parse_label   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/actionsense/eval_harness.yaml")
    ap.add_argument("--csv", default="docs/actionsense/trait_partition.csv")
    a = ap.parse_args()

    cfg = load_config(a.config)
    root = cfg.abspath("states_root")
    rows = [json.loads(l) for l in open(os.path.join(root, "manifest.jsonl")) if l.strip()]

    # Which recordings the frozen harness actually scores. The `actions` filter in the config
    # is the reason a trait contrast is possible at all, so it is reported, not assumed.
    try:
        sp = load_splits(cfg)
        split_of = {i: k for k in ("train", "val", "test") for i in sp[k]}
    except Exception as exc:                                          # noqa: BLE001
        print(f"[warn] no frozen split available ({exc}); reporting the corpus only")
        split_of = {}

    unaudited = TR.unaudited(parse_label(r["label"])[0] for r in rows)
    if unaudited:
        raise SystemExit(
            f"FATAL: {sorted(unaudited)} are not in the trait table. Audit them against the "
            f"rubric in src/opentouch/trait.py and commit the verdict BEFORE counting them.")

    per = collections.defaultdict(lambda: dict(rec=0, frames=0, win=0, scored=0, scored_win=0))
    for r in rows:
        v = parse_label(r["label"])[0]
        T10 = int(r["T"]) // cfg.downsample
        w = len(origins(T10, cfg))
        d = per[v]
        d["rec"] += 1; d["frames"] += T10; d["win"] += w
        if r["idx"] in split_of:
            d["scored"] += 1; d["scored_win"] += w

    hdr = ["verb", "trait_class", "contentious", "opentouch_correspondent",
           "recordings", "frames_10hz", "windows", "scored_recordings", "scored_windows"]
    out = []
    for v in sorted(per, key=lambda k: (TR.trait_class(k), -per[k]["rec"])):
        d = per[v]
        out.append([v, TR.trait_class(v), int(TR.is_contentious(v)),
                    TR.OT_CORRESPONDENT[v], d["rec"], d["frames"], d["win"],
                    d["scored"], d["scored_win"]])

    os.makedirs(os.path.dirname(a.csv) or ".", exist_ok=True)
    with open(a.csv, "w", newline="") as f:
        w_ = csv.writer(f); w_.writerow(hdr); w_.writerows(out)

    wid = max(len(r[0]) for r in out) + 1
    print(f"corpus: {len(rows)} recordings, {len(per)} verbs; harness actions filter = "
          f"{cfg.raw['actions']} -> {len(split_of)} scored recordings\n")
    print(f"{'verb':{wid}}{'class':8}{'cont':5}{'OT action':13}{'rec':>5}{'frames':>9}"
          f"{'windows':>9}{'scoredRec':>10}{'scoredWin':>10}")
    for r in out:
        print(f"{r[0]:{wid}}{r[1]:8}{'*' if r[2] else '':5}{r[3]:13}"
              f"{r[4]:>5}{r[5]:>9}{r[6]:>9}{r[7]:>10}{r[8]:>10}")

    print()
    for cls in (TR.SMOOTH, TR.ABRUPT):
        g = [r for r in out if r[1] == cls]
        print(f"{cls:8} corpus: {sum(r[4] for r in g):3d} rec / {sum(r[6] for r in g):6d} win"
              f"   |   SCORED: {sum(r[7] for r in g):3d} rec / {sum(r[8] for r in g):6d} win"
              f"   verbs={[r[0] for r in g if r[7]] or '-'}")
    # The contrast the harness can actually compute, and whether Layer 3 can be run on it.
    sv = {r[0] for r in out if r[7]}
    both = {r[1] for r in out if r[7]}
    print(f"\nscored verbs: {sorted(sv)}; classes present among them: {sorted(both)}")
    if len(both) < 2:
        print("  => ONE class only: no trait contrast is computable on the scored corpus.")
    else:
        drop = sorted(v for v in sv if TR.is_contentious(v))
        left = sorted(v for v in sv if not TR.is_contentious(v))
        print(f"  => contrast computable. Layer-3 sensitivity drops {drop}, leaving {left or 'NOTHING'}"
              + ("" if left else " -- the sensitivity analysis cannot be run here."))
    print(f"\nwrote {a.csv}")


if __name__ == "__main__":
    raise SystemExit(main())
