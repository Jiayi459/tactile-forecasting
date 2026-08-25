"""Rebuild docs/skill_comparison.md from the runs' own CSVs.

Every number in that document is read from the table the run wrote, because the alternative
-- copying figures across by hand as runs accumulate -- is how a superseded number outlives
the run that produced it. Six OpenTouch runs and two ActionSense families is already past
the point where that stays reliable.

WHICH SKILL. The driver's frame-pooled SS_vs_persistence, averaged over folds. The report
script's per-clip-equal-weight `skill` is a DIFFERENT estimator and disagrees (ar on F: 0.367
against 0.302); the two must not be mixed in one table. Frame-pooled is used here because it
is what every run wrote and what the predictability ceiling is computed over.

    python scripts/shared/build_skill_comparison.py [--out docs/skill_comparison.md]
"""
from __future__ import annotations

import argparse
import collections
import csv
import os

CH = ["F_R", "CoPx_R", "CoPy_R"]

# (label, ActionSense model name, OpenTouch model name). None means the arm has no
# counterpart there -- NOT that it scored zero.
ROWS = [("AR", "ar", "ar"),
        ("seasonal", "seasonal", "seasonal"),
        # ActionSense's probGRU predicts the FAST component against persistence-of-fast,
        # OpenTouch's predicts the RAW target under the harness. Same name, different
        # question, so the ActionSense side stays empty rather than inviting the comparison.
        ("probGRU", None, "prob_gru"),
        ("GRU-aggregate", "aggregate", "map_aggregate"),
        ("CNN (map)", "cnn", "cnn"),
        ("flatten (map)", "flatten", "flatten"),
        # the probGRU backbone reading the map: same architecture as the probGRU row above,
        # only the input differs, which is what the d1_pg run exists to isolate
        ("probGRU + CNN", None, "pg_cnn"),
        ("probGRU + flatten", None, "pg_flatten")]

# d1_map (08-22) is absent on purpose: flatten and cnn predicted arrays of zeros in it.
RUNS = [("raw", "08-17", "4-fold, location held out, uncorrected target",
         "docs/opentouch/raw/opentouch_cv4.csv"),
        ("df", "08-18", "adds dF/dt to the probGRU input",
         "docs/opentouch/df/opentouch_cv4_df.csv"),
        ("d1", "08-20", "**D1 baseline correction**, weights on val NLL",
         "docs/opentouch/d1/opentouch_cv4_d1.csv"),
        ("d1_mse", "08-21", "weights on val MSE instead",
         "docs/opentouch/d1_mse/opentouch_cv4_d1_mse.csv"),
        ("d1_map2", "08-23", "the three map encoders, `--baseline-scope shard`",
         "docs/opentouch/d1_map2/opentouch_cv4_d1_map2.csv"),
        ("d1_map3", "08-24", "`d1_map2` repeated to save checkpoints",
         "docs/opentouch/d1_map3/opentouch_cv4_d1_map3.csv"),
        ("d1_pg", "08-25", "**probGRU backbone**, three input representations",
         "docs/opentouch/d1_pg/opentouch_cv4_d1_pg.csv")]

# Per-clip-equal-weight skill and Hausdorff, from the report script rather than the driver.
# Kept separate from RUNS because the two skill conventions are different estimators, and
# mixing them in one table is the mistake this file exists to prevent.
REPORTS = [("d1", "docs/opentouch/d1/opentouch_report_d1.csv"),
           ("d1_mse", "docs/opentouch/d1_mse/opentouch_report_d1_mse.csv"),
           ("d1_map2", "docs/opentouch/d1_map2/opentouch_report_d1_map2.csv"),
           ("d1_pg", "docs/opentouch/d1_pg/opentouch_report_d1_pg.csv")]


def report_metrics(path):
    """-> {(metric, model, channel): value} from a report CSV."""
    out = {}
    if not os.path.exists(path):
        return out
    for r in csv.DictReader(open(path)):
        if r.get("scope") == "overall" and r.get("subset") == "all":
            m = r["metric"]
            out[("skill" if m.startswith("skill_vs") else m, r["model"], r["channel"])] = \
                float(r["value"])
    return out


def opentouch(path):
    a = collections.defaultdict(list)
    for r in csv.DictReader(open(path)):
        if r["horizon_step"] == "all" and r["metric"] == "SS_vs_persistence":
            a[(r["model"], r["channel"])].append(float(r["value"]))
    return {k: sum(v) / len(v) for k, v in a.items()}


def actionsense(root="docs/actionsense"):
    """The frozen harness for ar/seasonal, and the tactile_map CV at its longest history."""
    out = collections.defaultdict(list)
    p = os.path.join(root, "harness_baselines.csv")
    if os.path.exists(p):
        for r in csv.DictReader(open(p)):
            if (r["horizon_step"] == "all" and r["metric"] == "SS_vs_persistence"
                    and r["channel"] in CH):
                out[(r["model"], r["channel"])].append(float(r["value"]))
    for f in ("tactile_map_cv_results.csv", "tactile_map_cv_results_aggregate.csv"):
        p = os.path.join(root, f)
        if not os.path.exists(p):
            continue
        for r in csv.DictReader(open(p)):
            if r.get("history_s") != "10":          # its best history; see SESSION_LOG 08-20
                continue
            for c in CH:
                if r.get(f"{c}_skill"):
                    out[(r["encoder"], c)].append(float(r[f"{c}_skill"]))
    return {k: sum(v) / len(v) for k, v in out.items()}


def cell(tbl, model, ch):
    v = tbl.get((model, ch)) if model else None
    return "—" if v is None else f"{v:.3f}".replace("-", "−")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/skill_comparison.md")
    a = ap.parse_args()

    AS = actionsense()
    OT = {n: opentouch(p) for n, _, _, p in RUNS if os.path.exists(p)}
    missing = [n for n, _, _, p in RUNS if not os.path.exists(p)]
    if missing:
        print(f"note: no CSV for {', '.join(missing)} -- their columns will be empty")

    L = ["# Skill against persistence — every run, both sensors", "",
         "Skill = 1 − MSE(model)/MSE(persistence) at the full 1 s horizon, pooled over",
         "frames, averaged over folds. Right hand only: OpenTouch instruments one hand, so",
         "ActionSense's `_R` channels are the closest its two-handed target allows.", "",
         "Generated by `scripts/shared/build_skill_comparison.py` — every number is read from the",
         "run's own CSV, never transcribed. Rerun it after any new run.", "",
         "## The runs", "",
         "| run | date | what changed | arms |", "|---|---|---|---|"]
    for n, date, what, path in RUNS:
        arms = ", ".join(sorted({m for m, _ in OT.get(n, {})} - {"persistence"})) or "—"
        L.append(f"| `{n}` | {date} | {what} | {arms} |")
    L += ["",
          "`d1_map` (08-22) is **excluded**: flatten and cnn predicted arrays of zeros there.",
          "See SESSION_LOG 2026-08-22.", ""]

    for ch in CH:
        L += [f"## {ch}", "",
              "| model | ActionSense | " + " | ".join(f"`{n}`" for n, *_ in RUNS) + " |",
              "|---" * (len(RUNS) + 2) + "|"]
        for label, asn, otn in ROWS:
            cells = [cell(AS, asn, ch)] + [cell(OT.get(n, {}), otn, ch) for n, *_ in RUNS]
            L.append(f"| {label} | " + " | ".join(cells) + " |")
        L.append("")

    RM = {n: report_metrics(p) for n, p in REPORTS}
    have = [n for n in RM if any(k[0] == "hausdorff" for k in RM[n])]
    if have:
        L += ["## Hausdorff distance between forecast and truth curves", "",
              "Lower is better; `x` is the ratio to persistence. Scaled per forecast so the",
              "axes are commensurate: time spans [0,1] over the horizon, value is divided by",
              "the truth's own standard deviation there. Unlike MSE this is not pointwise, so",
              "a flat forecast through an oscillation is charged roughly its amplitude.", ""]
        for n in have:
            L += [f"### `{n}`", "",
                  "| model | " + " | ".join(CH) + " |", "|---" * (len(CH) + 1) + "|"]
            for m in sorted({k[1] for k in RM[n] if k[0] == "hausdorff"}):
                cells = []
                for c in CH:
                    v = RM[n].get(("hausdorff", m, c))
                    r = RM[n].get(("hausdorff_ratio_vs_persistence", m, c))
                    cells.append(f"{v:.3f} ({r:.2f}x)" if v is not None and r is not None
                                 else "—")
                L.append(f"| {m} | " + " | ".join(cells) + " |")
            L.append("")

    paths = {n: p for n, _, _, p in RUNS}
    if "d1_pg" in RM and "d1_map2" in RM:
        pg, m2 = opentouch(paths["d1_pg"]), opentouch(paths["d1_map2"])
        L += ["## The two skill conventions disagree on F", "",
              "Same input, same data, same folds; only the backbone differs. probGRU minus",
              "Seq2Seq under each convention:", "",
              "| input | channel | frame-pooled | per-clip | agree? |",
              "|---|---|---|---|---|"]
        # not `a`: that is the argparse namespace in this scope, and shadowing it here
        # crashed the writer at the last line with a message about a str having no .out
        for pgn, s2n, lab in (("prob_gru", "map_aggregate", "aggregate"),
                              ("pg_cnn", "cnn", "cnn"),
                              ("pg_flatten", "flatten", "flatten")):
            for c in CH:
                d1 = pg.get((pgn, c), float("nan")) - m2.get((s2n, c), float("nan"))
                d2 = (RM["d1_pg"].get(("skill", pgn, c), float("nan"))
                      - RM["d1_map2"].get(("skill", s2n, c), float("nan")))
                ok = "yes" if (d1 > 0) == (d2 > 0) else "**NO**"
                L.append(f"| {lab} | {c} | {d1:+.4f} | {d2:+.4f} | {ok} |")
        L += ["",
              "**On F the sign flips in all three arms.** The driver pools frames, so long and",
              "high-variance clips dominate it; the report weights every clip equally. probGRU",
              "is ahead where the frames are and behind on the typical clip. Any claim about",
              "which backbone is better on F must name its convention; one that does not is",
              "not supported here.", ""]

    L += open("docs/_skill_comparison_notes.md").read().rstrip().split("\n") + [""]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out, "w").write("\n".join(L) + "\n")
    print(f"wrote {a.out} ({len(RUNS)} runs x {len(ROWS)} models x {len(CH)} channels)")


if __name__ == "__main__":
    raise SystemExit(main())
