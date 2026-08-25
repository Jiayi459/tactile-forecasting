"""Aggregate CV results across runs/. Groups summary.json files by (model, scope,
protocol) and reports mean +/- std test skill vs persistence, plus per-horizon means.

    python scripts/shared/aggregate_results.py [--runs runs]
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

# Run-dir grammar: <name>_<scope>[_<category-slug>]_<protocol>_f<fold>.
# The optional category slug (per-category forecasting) has no underscores (slugified).
DIR_RE = re.compile(
    r"^(?P<name>.+?)_(?P<scope>grasp|full)(?:_(?P<category>[^_]+(?:-[^_]+)*))?"
    r"_(?P<protocol>lto|loto)_f(?P<fold>\d+)$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    args = ap.parse_args()

    groups = defaultdict(list)  # (name,scope,category,protocol) -> list of (fold, summary)
    for sj in sorted(glob.glob(os.path.join(args.runs, "*", "summary.json"))):
        m = DIR_RE.match(os.path.basename(os.path.dirname(sj)))
        if not m or not os.path.getsize(sj):
            continue
        s = json.load(open(sj))
        if "test" not in s:
            continue
        key = (m["name"], m["scope"], m["category"] or "-", m["protocol"])
        groups[key].append((int(m["fold"]), s))

    if not groups:
        print(f"No completed runs with a test split under {args.runs}/")
        return

    summary_rows = []  # (name, scope, category, protocol, mean, std, n)
    for key in sorted(groups):
        runs = sorted(groups[key])
        name, scope, category, protocol = key
        skills = [s["test"]["mean_skill"]["model"] for _, s in runs]
        folds = [f for f, _ in runs]
        cat_str = f" | cat={category}" if category != "-" else ""
        print(f"\n=== {name} | {scope}{cat_str} | {protocol} | folds {folds} (n={len(runs)}) ===")
        print(f"  mean-skill vs persistence: {np.mean(skills):+.4f} +/- {np.std(skills):.4f}  "
              f"(per-fold: {', '.join(f'{x:+.3f}' for x in skills)})")
        # per-horizon mean skill across folds
        horizons = sorted(int(h) for h in runs[0][1]["test"]["model"].keys())
        per_h = {h: np.mean([s["test"]["model"][str(h)]["skill"] for _, s in runs])
                 for h in horizons}
        print("  skill@h: " + ", ".join(f"h{h}={per_h[h]:+.3f}" for h in horizons))
        summary_rows.append((name, scope, category, protocol,
                             float(np.mean(skills)), float(np.std(skills)), len(runs)))

    # per-CATEGORY ranking (the study's headline): confirms/breaks the probe PI ordering
    cat_rows = [r for r in summary_rows if r[2] != "-"]
    if cat_rows:
        print("\n===== PER-CATEGORY RANKING (mean test skill vs persistence, high=easier) =====")
        print(f"{'category':<22}{'model':<9}{'proto':<6}{'n':>3}{'meanSkill':>11}{'std':>8}")
        print("-" * 59)
        for name, scope, category, protocol, mean, std, n in sorted(cat_rows, key=lambda r: -r[4]):
            print(f"{category:<22}{name:<9}{protocol:<6}{n:>3}{mean:>+11.4f}{std:>8.3f}")


if __name__ == "__main__":
    main()
