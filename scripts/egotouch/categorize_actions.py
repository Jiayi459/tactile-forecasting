"""Categorize EgoTouch tasks/trajectories by hand action using a task-name verb taxonomy.

Classification only (no success forecasting). Assigns each task to one action
category by scanning its name tokens left-to-right for the first known action verb
(task names follow a verb_object convention). Core grasp categories are flagged.
"""
import os
import sys
from collections import defaultdict

# Single source of truth for the taxonomy lives in the tactile_pixel package so
# train.py and this script agree. Add repo root to path for the local (torch-free) import.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.tactile_pixel.categories import (  # noqa: E402
    VERB_CATEGORY, CORE_GRASP, categorize)

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                    "datasets", "EgoTouch")
SCENES = ["Home", "Office", "Outdoor", "Retail", "Workbench"]


def main():
    rows = []  # (scene, task, category, n_traj)
    for sc in SCENES:
        sp = os.path.join(ROOT, sc)
        for t in sorted(os.listdir(sp)):
            tp = os.path.join(sp, t)
            if not os.path.isdir(tp) or t == "metadata":
                continue
            n = sum(1 for x in os.listdir(tp) if os.path.isdir(os.path.join(tp, x)))
            rows.append((sc, t, categorize(t), n))

    by_cat_tasks = defaultdict(list)
    by_cat_traj = defaultdict(int)
    for sc, t, cat, n in rows:
        by_cat_tasks[cat].append((sc, t, n))
        by_cat_traj[cat] += n

    order = sorted(by_cat_tasks, key=lambda c: (-by_cat_traj[c], c))
    tot_tasks = len(rows)
    tot_traj = sum(r[3] for r in rows)

    print(f"TOTAL: {tot_tasks} tasks, {tot_traj} trajectories\n")
    print(f"{'CATEGORY':<22} {'tasks':>6} {'traj':>6}  grasp?")
    print("-" * 50)
    for c in order:
        flag = "CORE-GRASP" if c in CORE_GRASP else ""
        print(f"{c:<22} {len(by_cat_tasks[c]):>6} {by_cat_traj[c]:>6}  {flag}")
    print("-" * 50)
    gt_tasks = sum(len(by_cat_tasks[c]) for c in CORE_GRASP if c in by_cat_tasks)
    gt_traj = sum(by_cat_traj[c] for c in CORE_GRASP if c in by_cat_traj)
    print(f"{'CORE-GRASP TOTAL':<22} {gt_tasks:>6} {gt_traj:>6}")

    # full task listing per category for review
    print("\n===== TASK ASSIGNMENTS PER CATEGORY =====")
    for c in order:
        print(f"\n[{c}]  ({len(by_cat_tasks[c])} tasks, {by_cat_traj[c]} traj)")
        for sc, t, n in by_cat_tasks[c]:
            print(f"    {sc}/{t}  ({n})")


if __name__ == "__main__":
    main()
