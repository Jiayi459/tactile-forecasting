#!/usr/bin/env python3
"""Pull the numbers that decide whether a run is trustworthy out of selection_report.json.

Run after any sweep. Prints: the OTHER audit against the real eligible corpus (the earlier
figures were estimated from split.json task names, before the npz and length filters), and
per arm whether the retired pooled criterion would have kept a different checkpoint (Q-D).
"""
from __future__ import annotations

import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default="runs/egotouch_tactile_map/selection_report.json")
    a = ap.parse_args()
    r = json.load(open(a.report))

    print(f"config_hash: {r['config_hash']}")
    print("\nOTHER audit (recordings riding the barely-trained OTHER embedding):")
    for k, v in r["other_audit"].items():
        print(f"  {k:<12} {v}")

    print(f"\n{'arm':<28}{'epochs':>7}{'sel':>5}{'pooled':>8}{'differs':>9}{'sigma':>8}")
    n_diff = 0
    for arm, d in sorted(r["arms"].items()):
        n_diff += bool(d["selection_differs"])
        print(f"{arm:<28}{len(d['val_curves']):>7}{d['selected_epoch']:>5}"
              f"{d['selected_epoch_pooled']:>8}{str(d['selection_differs']):>9}"
              f"{d['sigma_scale']:>8.3f}")
    print(f"\n{n_diff}/{len(r['arms'])} arms would have kept a DIFFERENT checkpoint under the "
          f"retired pooled criterion.")
    if n_diff == 0:
        print("  => pooled and balanced selection agree here; the Q-D(b) change costs nothing "
              "and no ActionSense rerun is owed on selection grounds.")
    else:
        print("  => the criterion matters. These arms' pre-change numbers were selected on a "
              "criterion they were not judged by.")


if __name__ == "__main__":
    main()
