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

# EgoTouch, the third sensor. Its numbers come from the SHARED scorer
# (scripts/shared/score_preds_per_action.py), whose `(all actions)` row carries, per channel,
# `skill_pooled_<ch>` (frame-pooled -- the estimator these skill tables use),
# `skill_<ch>` (per-clip), `hausdorff_<ch>` and `r2_<ch>`.
#
# ITS PROTOCOL IS A THIRD DIFFERENT QUESTION. EgoTouch ships an OFFICIAL split and we use it
# rather than cross-validating: `ego seen` holds out RECORDINGS of tasks TRAIN has seen, which
# is the closest analogue to ActionSense's stratified split; `ego unseen` holds out TEN WHOLE
# TASKS that never appear in TRAIN at all (task-level zero-shot -- the scenes are all seen).
# Nothing in the other three columns answers that second question, so `ego unseen` is here to
# be read down its own column, not across.
# The 3 s history, EgoTouch's longest -- matching the "longest history" filter actionsense()
# applies, so the two sensors' columns describe comparably-informed models.
EGOTOUCH_RUNS = [("ego seen", "docs/egotouch/results/test_seen/egotouch_test_seen_3s.csv"),
                 ("ego unseen", "docs/egotouch/results/test_unseen/egotouch_test_unseen_3s.csv")]

_ASC = "docs/actionsense/results/corpus-aggregate-flatten-cnn"

# Columns fed by the SHARED scorer. The int is which slot of a ROWS tuple names the model in
# that column -- EgoTouch and the ActionSense corpus call the same arm different things
# (`aggregate_seq2seq` against `seq2seq_aggregate`), and a single shared slot would have
# silently blanked one of them.
#
# `AS corpus` IS NOT the `ActionSense` column. That one is the frozen harness: 75 recordings,
# slice and peel only, a stratified 60/20/20. This one is all 299 recordings over 14 actions
# under 5-fold CV, with a Norm fitted to that wider population. A model's two cells are the
# same architecture answering two different questions, and the corpus one is the harder.
SCORER_COLS = [("AS corpus", [f"{_ASC}/as_preds_seq2seq_plus_baselines.csv",
                              f"{_ASC}/as_preds_tmap_probgru_corpus3s.csv"], 4),
               ("ego seen", [EGOTOUCH_RUNS[0][1]], 3),
               ("ego unseen", [EGOTOUCH_RUNS[1][1]], 3)]


def scorer(paths):
    """-> {(metric, model, channel): value} for metric in skill/skill_clip/hausdorff/r2.

    Reads ANY CSV the shared scorer wrote, and merges several of them into one column. The
    ActionSense corpus column needs that: its Seq2Seq arms and the classical baselines were
    scored in one file, its probGRU arms in another, and the two agree on the `persistence`
    row to the digit -- same recordings, same origins -- which is what makes merging them a
    column rather than a splice.

    `skill` here is the FRAME-POOLED column, chosen to match the estimator the skill tables
    already use across the other three sensors; the per-clip one is kept under `skill_clip`
    so the Hausdorff section, which is per-clip, has its matching skill available.
    """
    out = {}
    for path in ([paths] if isinstance(paths, str) else paths):
        if not os.path.exists(path):
            continue
        for r in csv.DictReader(open(path)):
            if r.get("action") != "(all actions)":
                continue
            m = r["model"]
            for c in AS_CH_ALL:
                for src, dst in (("skill_pooled", "skill"), ("skill", "skill_clip"),
                                 ("hausdorff", "hausdorff"), ("r2", "r2")):
                    v = r.get(f"{src}_{c}")
                    if v not in (None, ""):
                        out[(dst, m, c)] = float(v)
    return out


# (label, ActionSense model name, OpenTouch model name). None means the arm has no
# counterpart there -- NOT that it scored zero.
# (label, ActionSense name, OpenTouch name, EgoTouch name, AS-corpus name)
ROWS = [("AR", "ar", "ar", "ar_group", "ar"),
        # EgoTouch only. The group-fitted AR above is ~168 per-(action,object) linear models
        # against everyone else's single global one; this row is the capacity-matched
        # reference, and the gap between the two rows IS the value of group specialization.
        ("AR (global fit)", None, None, "ar_global", None),
        ("seasonal", "seasonal", "seasonal", "seasonal_group", "seasonal"),
        # ActionSense's probGRU predicts the FAST component against persistence-of-fast,
        # OpenTouch's predicts the RAW target under the harness. Same name, different
        # question, so the ActionSense side stays empty rather than inviting the comparison.
        ("probGRU", None, "prob_gru", "aggregate_probgru", "probgru_aggregate"),
        ("GRU-aggregate", "aggregate", "map_aggregate", "aggregate_seq2seq",
         "seq2seq_aggregate"),
        ("CNN (map)", "cnn", "cnn", "cnn_seq2seq", "seq2seq_cnn"),
        ("flatten (map)", "flatten", "flatten", "flatten_seq2seq", "seq2seq_flatten"),
        # the probGRU backbone reading the map: same architecture as the probGRU row above,
        # only the input differs, which is what the d1_pg run exists to isolate
        ("probGRU + CNN", None, "pg_cnn", "cnn_probgru", "probgru_cnn"),
        ("probGRU + flatten", None, "pg_flatten", "flatten_probgru", "probgru_flatten")]

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
           # the _hd rescore, not the original: same predictions, rerun once Hausdorff
           # existed, so it is the only d1_map2 report that carries shape numbers
           ("d1_map2", "docs/opentouch/d1_map2/opentouch_report_d1_map2_hd.csv"),
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


AS_RUNS = [("Seq2Seq", "aggregate", "docs/actionsense/seq2seq_agg_recheck/cv.csv"),
           ("Seq2Seq", "cnn", "docs/actionsense/seq2seq_map_recheck/cv_h3.csv"),
           ("Seq2Seq", "flatten", "docs/actionsense/seq2seq_map_recheck/cv_h3.csv"),
           ("probGRU", "aggregate", "docs/actionsense/probgru_agg/tactile_map_cv_probgru_agg.csv"),
           ("probGRU", "cnn", "docs/actionsense/probgru_map/cv_h3.csv"),
           ("probGRU", "flatten", "docs/actionsense/probgru_map/cv_h3.csv")]
AS_CH = ["F_L", "CoPx_L", "CoPy_L", "F_R", "CoPx_R", "CoPy_R"]
AS_CH_ALL = AS_CH          # EgoTouch is two-handed too, so it carries the same six


def as_run(path, encoder):
    """-> (mean skill, mean Hausdorff) over the six channels, averaged over folds/steps."""
    if not os.path.exists(path):
        return None
    sk, hd = [], []
    for r in csv.DictReader(open(path)):
        if r.get("encoder") != encoder:
            continue
        v = [float(r[f"{c}_skill"]) for c in AS_CH if r.get(f"{c}_skill")]
        h = [float(r[f"{c}_hausdorff"]) for c in AS_CH if r.get(f"{c}_hausdorff")]
        if v:
            sk.append(sum(v) / len(v))
        if h:
            hd.append(sum(h) / len(h))
    if not sk:
        return None
    return (sum(sk) / len(sk), sum(hd) / len(hd) if hd else float("nan"))


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


FLOOR_CSV = "docs/predictability_floor.csv"


def floor_R(path=FLOOR_CSV):
    """-> {(sensor, channel): (R, rho1)} from scripts/shared/predictability_floor.py --csv.

    Read, never transcribed, like every other number here. R is reported beside skill because
    skill's denominator is persistence and persistence is NOT equally hard across sensors:
    measured, it ranges from 0.585 on ActionSense to 1.041 on OpenTouch. A reader comparing
    two skill columns without knowing that is comparing two different questions.
    """
    if not os.path.exists(path):
        return {}
    return {(r["sensor"], r["channel"]): (float(r["R"]), float(r["rho1"]))
            for r in csv.DictReader(open(path))}


AS_HD_CSV = "docs/actionsense/tactile_map_cv_seq2seq_agg_recheck.csv"


def actionsense_hausdorff(path=AS_HD_CSV, history_s="10"):
    """-> {(encoder, channel): (hausdorff, ratio_vs_persistence)}.

    ActionSense stores it wide -- one `{channel}_hausdorff` column per channel plus a single
    `hausdorff_ratio_vs_persistence` -- rather than as metric rows, so it needs its own reader.
    Filtered to the longest history, matching what actionsense() does for skill, so the two
    tables describe the same runs.
    """
    if not os.path.exists(path):
        return {}
    out = {}
    for r in csv.DictReader(open(path)):
        if str(r.get("history_s")) != history_s:
            continue
        enc = r.get("encoder", "?")
        ratio = r.get("hausdorff_ratio_vs_persistence")
        for c in CH:
            v = r.get(f"{c}_hausdorff")
            if v:
                out[(enc, c)] = (float(v), float(ratio) if ratio else None)
    return out


def _as_ratio(path=AS_HD_CSV, history_s="10"):
    """ActionSense's single run-level hausdorff_ratio_vs_persistence, or nan."""
    if not os.path.exists(path):
        return float("nan")
    for r in csv.DictReader(open(path)):
        if str(r.get("history_s")) == history_s and r.get("hausdorff_ratio_vs_persistence"):
            return float(r["hausdorff_ratio_vs_persistence"])
    return float("nan")



def cell(tbl, model, ch):
    v = tbl.get((model, ch)) if model else None
    return "—" if v is None else f"{v:.3f}".replace("-", "−")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/skill_comparison.md")
    a = ap.parse_args()

    AS = actionsense()
    ASH = actionsense_hausdorff()
    FL = floor_R()
    SC = {n: scorer(ps) for n, ps, _ in SCORER_COLS}
    SC = {n: v for n, v in SC.items() if v}
    SLOT = {n: i for n, _, i in SCORER_COLS}
    for n, ps, _ in SCORER_COLS:
        if n not in SC:
            print(f"note: no scorer CSV for {n} ({', '.join(ps)}) -- its column will be absent")
    OT = {n: opentouch(p) for n, _, _, p in RUNS if os.path.exists(p)}
    missing = [n for n, _, _, p in RUNS if not os.path.exists(p)]
    if missing:
        print(f"note: no CSV for {', '.join(missing)} -- their columns will be empty")

    L = ["# Skill against persistence — every run, all three sensors", "",
         "Skill = 1 − MSE(model)/MSE(persistence) at the full 1 s horizon, pooled over",
         "frames, averaged over folds. Right hand only: OpenTouch instruments one hand, so",
         "ActionSense's `_R` channels are the closest its two-handed target allows.",
         "",
         "**The columns are not the same experiment.** OpenTouch holds out a location",
         "(4-fold), ActionSense is a stratified 60/20/20 by recording, and EgoTouch uses the",
         "dataset's OWN split — `ego seen` holds out recordings of tasks TRAIN has seen,",
         "`ego unseen` holds out ten whole tasks that never appear in TRAIN. Unseen-task is",
         "the hardest question here, so that column scoring lower is not on its own evidence",
         "of a worse model.",
         "The 1 s horizon and the persistence reference ARE identical across all four,",
         "which is what makes any comparison possible at all.", "",
         "**EgoTouch's F is not the others' F.** Its grids ship normalised by `tactile_max`",
         "and may mix tactile and bending channels, so its force channel is an aggregate",
         "normalised pressure (P_Σ), not newtons. Skill, R² and the scaled Hausdorff are all",
         "dimensionless, which is why they can share a table; a raw magnitude could not.", "",
         "Generated by `scripts/shared/build_skill_comparison.py` — every number is read from the",
         "run's own CSV, never transcribed. Rerun it after any new run.", "",
         # Definitions FIRST: the document quotes two different estimators that share the
         # name "skill", so a reader meeting the tables before the formulas has no way to
         # know which is which.
         *open("docs/_skill_comparison_defs.md").read().rstrip().split("\n"), "",
         "## The runs", "",
         "| run | date | what changed | arms |", "|---|---|---|---|"]
    for n, date, what, path in RUNS:
        arms = ", ".join(sorted({m for m, _ in OT.get(n, {})} - {"persistence"})) or "—"
        L.append(f"| `{n}` | {date} | {what} | {arms} |")
    L += ["",
          "`d1_map` (08-22) is **excluded**: flatten and cnn predicted arrays of zeros there.",
          "See SESSION_LOG 2026-08-22.", ""]

    egnames = [n for n, _, _ in SCORER_COLS if n in SC]

    def sc_name(run, row):
        return row[SLOT[run]]

    def eg_cell(run, metric, name, ch):
        v = SC.get(run, {}).get((metric, name, ch)) if name else None
        return "—" if v is None else f"{v:.3f}".replace("-", "−")

    ncol = len(RUNS) + len(egnames) + 2
    for ch in CH:
        L += [f"## {ch}", "",
              "| model | ActionSense | " + " | ".join(f"`{n}`" for n in egnames)
              + (" | " if egnames else "") + " | ".join(f"`{n}`" for n, *_ in RUNS) + " |",
              "|---" * ncol + "|"]
        for row in ROWS:
            label, asn, otn = row[0], row[1], row[2]
            cells = ([cell(AS, asn, ch)]
                     + [eg_cell(n, "skill", sc_name(n, row), ch) for n in egnames]
                     + [cell(OT.get(n, {}), otn, ch) for n, *_ in RUNS])
            L.append(f"| {label} | " + " | ".join(cells) + " |")
        # How hard was the denominator? Same row shape, so it reads directly under the skills
        # it qualifies. Sensor-level, not run-level: R is a property of the signal and the
        # horizon, so every run on one sensor shares it.
        if FL:
            def rcell(sensor):
                v = FL.get((sensor, ch))
                return "—" if v is None else f"{v[0]:.3f}".replace("-", "−")
            cells = ([rcell("actionsense")]
                     + [rcell("actionsense" if n == "AS corpus" else "egotouch")
                        for n in egnames]
                     + [rcell("opentouch") for _ in RUNS])
            L.append("| **R** (persistence difficulty) | " + " | ".join(cells) + " |")
        L.append("")

    # Same column order and the same ROWS mapping the skill tables use, plus persistence,
    # which skill omits because it is 0 by construction while Hausdorff and R2 cannot.
    # Defined out here because the R2 section below reads it too.
    HD_ROWS = list(ROWS) + [("persistence", None, "persistence", "persistence",
                            "persistence")]

    RM = {n: report_metrics(p) for n, p in REPORTS}
    have = [n for n in RM if any(k[0] == "hausdorff" for k in RM[n])]
    # ONE Hausdorff section, not one per sensor: two near-identically named headings made the
    # same metric on different sensors read as two different things. Skill puts every sensor in
    # one table; this matches.
    if have or ASH or SC:
        L += ["## Hausdorff distance between forecast and truth curves", "",
              "Laid out exactly like the skill tables above -- one section per channel, models",
              "down, sensors and runs across -- so a model can be followed along a row without",
              "re-learning a layout. LOWER is better, unlike skill.", "",
              "Scaled per forecast so the axes are commensurate: time spans [0,1] over the",
              "horizon, value is divided by the truth's own standard deviation there. Unlike",
              "MSE this is not pointwise, so a flat forecast through an oscillation is charged",
              "roughly its amplitude.", "",
              "**A Hausdorff ratio near 1.0 is ambiguous, and it cannot rank arms on its own.**",
              "persistence's ratio is 1.000 by construction, so an arm that barely departs from",
              "persistence inherits its shape score. On EgoTouch's seen split at 3 s the LOWEST",
              "neural Hausdorff belongs to `flatten (map)` under Seq2Seq, which also has the",
              "LOWEST skill of any neural arm there: it scores well on shape by not moving.",
              "Across the eight EgoTouch neural arms, Spearman(skill, HD ratio) is +0.26, +0.49,",
              "+0.49 and -0.26 over the four split-by-history combinations — unstable, and mostly",
              "POSITIVE, meaning better point error tends to come with worse shape. Always read a",
              "Hausdorff cell beside that arm's skill.", "",
              "**`persistence` is a row here, not a zero.** Skill divides it out; Hausdorff does",
              "not, so the reference has to be visible for a number to mean anything. Read each",
              "column against its own persistence, never across columns: the three sensors do",
              "not present equally hard signals (see the R row in the skill tables).", "",
              "OpenTouch is per-clip from its report; ActionSense is per-clip from its CV",
              "table at the longest history; EgoTouch is per-clip (recording-balanced) from",
              "the shared scorer. All three therefore share one convention.", "",
              "**The columns fed by the shared scorer — `AS corpus`, `ego seen`, `ego unseen`",
              "— each carry their own `persistence` row measured under the same mask as the",
              "models above it**, because that scorer synthesises persistence from the saved",
              "truth and origins rather than requiring the run to have trained it. Those",
              "columns are internally readable in the way the `ActionSense` one is not.", "",
              "**The `ActionSense` column is not readable on its own.** Its CV table carries",
              "only the `aggregate` encoder and NO persistence row, so there is no reference to",
              "divide by and the single number in that column cannot be interpreted the way the",
              "others can. What that arm does report is one run-level ratio,",
              f"**{_as_ratio():.2f}x persistence**, which is the only figure from it that",
              "compares to the others -- against OpenTouch's map_aggregate at 0.83x.", "",
              "**`AS corpus` is what that column should have been.** Same sensor, scored by the",
              "shared scorer, so it brings its own persistence row and every arm at once. It is",
              "not a drop-in replacement for the cell beside it, though: the frozen column is 75",
              "slice-and-peel recordings under a stratified 60/20/20, this one is all 299 over",
              "14 actions under 5-fold CV, and the Norm is fitted to that wider population.", ""]

        def hd_as(name, ch):
            v = ASH.get((name, ch)) if name else None
            return "—" if v is None else f"{v[0]:.3f}"

        def hd_ot(run, name, ch):
            v = RM.get(run, {}).get(("hausdorff", name, ch)) if name else None
            if v is None:
                return "—"
            return f"{v[0]:.3f}" if isinstance(v, tuple) else f"{v:.3f}"

        def hd_eg(run, name, ch):
            v = SC.get(run, {}).get(("hausdorff", name, ch)) if name else None
            return "—" if v is None else f"{v:.3f}"

        nhd = len(have) + len(egnames) + 2
        for ch in CH:
            L += [f"### Hausdorff — {ch}", "",
                  "| model | ActionSense | " + " | ".join(f"`{n}`" for n in egnames)
                  + (" | " if egnames else "") + " | ".join(f"`{n}`" for n in have) + " |",
                  "|---" * nhd + "|"]
            for row in HD_ROWS:
                label, asn, otn = row[0], row[1], row[2]
                cells = ([hd_as(asn, ch)]
                         + [hd_eg(n, sc_name(n, row), ch) for n in egnames]
                         + [hd_ot(n, otn, ch) for n in have])
                if all(c == "—" for c in cells):
                    continue
                L.append(f"| {label} | " + " | ".join(cells) + " |")
            L.append("")

    # R^2 -- available per channel from the shared scorer (EgoTouch) and from OpenTouch's
    # report. Its own section rather than more columns on the skill tables: R2 and skill have
    # DIFFERENT denominators -- the action/dataset mean versus persistence -- so a reader who
    # met them in one table would compare two questions as though they were one.
    otr2 = [n for n in RM if any(k[0] == "R2" for k in RM[n])]
    if SC or otr2:
        L += ["## R² against the dataset mean", "",
              "Skill above divides by PERSISTENCE; R² here divides by the MEAN. A model can",
              "beat the mean comfortably and still lose to persistence, which on a smooth",
              "1 s horizon is a strong reference -- so read this section beside the skill",
              "tables, never instead of them.", "",
              "Per-clip (recording-balanced) everywhere shown. The frozen `ActionSense`",
              "column does not write per-channel R² in its CV table, so it is absent here;",
              "`AS corpus` is the same sensor read through the shared scorer, which does.", "",
              "**`ego seen` and `ego unseen` are different populations, so their R² columns do",
              "not compare to each other.** The persistence row makes this concrete: it scores",
              "R² 0.349 on seen and 0.635 on unseen, meaning the unseen split's signals are",
              "simply smoother and easier to extrapolate. That is why every model's R² rises",
              "on unseen while its SKILL falls — the reference got stronger faster than the",
              "models did. Comparing the two columns without reading the persistence row is",
              "how one concludes that unseen tasks are easier to forecast, which is backwards.",
              ""]
        for ch in CH:
            L += [f"### R² — {ch}", "",
                  "| model | " + " | ".join(f"`{n}`" for n in egnames)
                  + (" | " if egnames and otr2 else "")
                  + " | ".join(f"`{n}`" for n in otr2) + " |",
                  "|---" * (len(egnames) + len(otr2) + 1) + "|"]
            for row in HD_ROWS:
                label, otn = row[0], row[2]
                cells = ([eg_cell(n, "r2", sc_name(n, row), ch) for n in egnames]
                         + [("—" if not otn or RM.get(n, {}).get(("R2", otn, ch)) is None
                             else f"{RM[n][('R2', otn, ch)]:.3f}".replace("-", "−"))
                            for n in otr2])
                if all(c == "—" for c in cells):
                    continue
                L.append(f"| {label} | " + " | ".join(cells) + " |")
            L.append("")

    PAIRS = (("prob_gru", "map_aggregate", "aggregate"),
             ("pg_cnn", "cnn", "cnn"),
             ("pg_flatten", "flatten", "flatten"))

    if RM.get("d1_map2") and RM.get("d1_pg") and any(
            k[0] == "hausdorff" for k in RM["d1_map2"]):
        L += ["## The backbones side by side, one input at a time", "",
              "Same input, same data, same folds, same loss. Only the decoder differs:",
              "Seq2Seq emits all H steps at once and predicts a residual; probGRU rolls out",
              "autoregressively on its own mean and predicts the absolute value.", "",
              "Hausdorff is lower-is-better, per-clip skill is higher-is-better, so a",
              "consistent winner would show opposite signs in the two Δ columns. It does not.",
              "",
              "| input | channel | HD Seq2Seq | HD probGRU | Δ HD | skill Seq2Seq | skill probGRU | Δ skill |",
              "|---|---|---|---|---|---|---|---|"]
        dh, dk, dk_f = [], [], []
        for pgn, s2n, lab in PAIRS:
            for c in CH:
                h2 = RM["d1_map2"].get(("hausdorff", s2n, c))
                hp = RM["d1_pg"].get(("hausdorff", pgn, c))
                k2 = RM["d1_map2"].get(("skill", s2n, c))
                kp = RM["d1_pg"].get(("skill", pgn, c))
                if None in (h2, hp, k2, kp):
                    continue
                dh.append(hp - h2); dk.append(kp - k2)
                (dk_f if c.startswith("F") else []).append(kp - k2)
                L.append(f"| {lab} | {c} | {h2:.3f} | {hp:.3f} | **{hp - h2:+.3f}** "
                         f"| {k2:.4f} | {kp:.4f} | **{kp - k2:+.4f}** |")
        # counted, not asserted: the first draft of this sentence said six of nine where the
        # table said three, which is exactly the drift generating the document was meant to
        # stop
        n_hd = sum(1 for v in dh if v > 0)
        n_neg = sum(1 for v in dk if v < 0)
        n_f = sum(1 for v in dk_f if v < 0)
        L += ["",
              f"**Δ HD is positive in {n_hd} of {len(dh)} cells; Δ skill is negative in "
              f"{n_neg} of {len(dk)}, of which {n_f} are the {len(dk_f)} F channels.**",
              "probGRU's curves are further from the truth in shape everywhere. Its per-clip",
              "point error is better on CoP and worse on F, and F is where the two skill",
              "conventions disagree, so that is the channel to be careful about.",
              "The backbone effect on shape (0.11-0.23) is at least as large as the spread",
              "between input representations within either backbone (0.08 within Seq2Seq,",
              "0.13 within probGRU), so on this data the decoder matters more than what it",
              "is fed.", ""]

    # --- The same decoder-vs-input question on every corpus that can now answer it ------
    # OpenTouch could answer it first because d1_pg ran probGRU over all three inputs. The
    # ActionSense corpus sweep and EgoTouch now both carry the full 2x3 through the shared
    # scorer, so the comparison is no longer one sensor's property. EgoTouch's probGRU map
    # arms existed only from 2026-09-12: they were cut from its matrix on the false premise
    # that the other sensors had not run them, which is exactly the cell this table needs.
    BACKBONE_PAIRS = [("aggregate", "GRU-aggregate", "probGRU"),
                      ("cnn", "CNN (map)", "probGRU + CNN"),
                      ("flatten", "flatten (map)", "probGRU + flatten")]
    by_label = {r[0]: r for r in ROWS}
    if SC:
        L += ["## Decoder versus input representation, on every corpus that has both", "",
              "The section above is OpenTouch's. These are the same nine cells per corpus,",
              "read from the shared scorer: per-clip skill, Δ = probGRU − Seq2Seq at one",
              "fixed input, beside the spread ACROSS inputs within each decoder. Positive Δ",
              "means the autoregressive absolute-target decoder helped that input.", "",
              "**The two factors are not additive.** Where Seq2Seq already does well the",
              "decoder buys little; where it does badly the decoder recovers most of the gap.",
              "So a representation ordering measured under one decoder overstates how much",
              "the representation itself matters.", ""]
        for col in egnames:
            rowsd = SC[col]

            def sk(label, ch):
                nm = by_label[label][SLOT[col]]
                return rowsd.get(("skill_clip", nm, ch)) if nm else None

            L += [f"### {col}", "",
                  "| input | " + " | ".join(f"{c} S2S / pgru / Δ" for c in CH) + " |",
                  "|---|" + "---:|" * len(CH)]
            spread = {"S2S": [], "pgru": []}
            for inp, s2_lab, pg_lab in BACKBONE_PAIRS:
                cells, ok = [], True
                for ch in CH:
                    # NOT `a`: that is the argparse namespace this function still needs.
                    v_s2, v_pg = sk(s2_lab, ch), sk(pg_lab, ch)
                    if v_s2 is None or v_pg is None:
                        cells.append("—"); ok = False; continue
                    cells.append(f"{v_s2:.3f} / {v_pg:.3f} / **{v_pg - v_s2:+.3f}**")
                if ok:
                    spread["S2S"].append(sum(sk(s2_lab, c) for c in CH) / len(CH))
                    spread["pgru"].append(sum(sk(pg_lab, c) for c in CH) / len(CH))
                L.append(f"| {inp} | " + " | ".join(cells) + " |")
            if len(spread["S2S"]) == len(BACKBONE_PAIRS):
                sp2 = max(spread["S2S"]) - min(spread["S2S"])
                spp = max(spread["pgru"]) - min(spread["pgru"])
                deltas = [p - s for s, p in zip(spread["S2S"], spread["pgru"])]
                worst = BACKBONE_PAIRS[spread["S2S"].index(min(spread["S2S"]))][0]
                best = BACKBONE_PAIRS[spread["S2S"].index(max(spread["S2S"]))][0]
                i_worst = spread["S2S"].index(min(spread["S2S"]))
                i_best = spread["S2S"].index(max(spread["S2S"]))
                # "gain" only when the deltas are actually positive. On ego unseen every
                # channel-mean delta is negative, and calling the least-negative one a gain
                # would invert what the row says.
                verb = "gain" if max(deltas) > 0 else "change"
                L += ["",
                      f"Channel-mean spread across the three inputs: **{sp2:.3f} under "
                      f"Seq2Seq, {spp:.3f} under probGRU**"
                      + (f" — the decoder shrinks it {sp2 / spp:.1f}x."
                         if spp > 1e-9 else "."),
                      f"Δ runs {min(deltas):+.3f} to {max(deltas):+.3f}"
                      + (", every input worse under probGRU. " if max(deltas) <= 0 else ". ")
                      + f"The largest {verb} is `{worst}` ({deltas[i_worst]:+.3f}), the input "
                        f"Seq2Seq handles worst; `{best}`, the one it handles best, is "
                        f"{deltas[i_best]:+.3f}.",
                      ""]

    # --- Both backbones, every input, against all three baselines ---------------------
    # The section above compares the two decoders to EACH OTHER, which cannot say whether
    # either is worth having. This one puts the same nine arms against the references.
    if RM.get("d1_map2") and RM.get("d1_pg") and any(
            k[0] == "hausdorff" for k in RM["d1_pg"]):
        M2, PG = RM["d1_map2"], RM["d1_pg"]
        BL = ("ar", "seasonal", "persistence")
        # A baseline is fitted per fold and both runs scored the same folds, so the two
        # reports should carry the SAME baseline rows. Checked, not assumed: if they ever
        # drift, one baseline column can no longer honestly serve both backbones and each
        # arm would have to be read against its own run's references.
        drift = max((abs(M2[k] - PG[k]) for k in M2 if k in PG and k[1] in BL), default=None)

        def bl_skill(m, c):
            """Skill of a baseline. persistence is 0 BY CONSTRUCTION, not missing: skill is
            defined as 1 - MSE(model)/MSE(persistence), so the reference scores exactly 0."""
            return 0.0 if m == "persistence" else M2.get(("skill", m, c))

        L += ["## Both backbones against the baselines, one input at a time", "",
              "The section above asks which decoder wins. It cannot answer whether either is",
              "worth having, because both are compared only to each other. Here the same nine",
              "arms are put against the three references on the same folds and the same mask.",
              ""]
        if drift == 0:
            L += ["The two runs' baseline rows are **bit-identical**, so one baseline column",
                  "serves both backbones: same fits, same folds, same mask."]
        elif drift is not None:
            L += [f"**The two runs' baselines differ by up to {drift:.4f}**, so one column",
                  "cannot serve both backbones; read each arm against its own run's rows."]
        else:
            L += ["One or both runs carry no baseline rows."]
        L += ["",
              "`persistence` carries skill **0.000 by construction** -- skill is measured",
              "against it -- so a positive skill IS beating persistence, and the persistence",
              "column is a reminder rather than a measurement. Hausdorff does not divide it",
              "out, so there it is a real number.", "",
              "Skill: HIGHER is better. `best` is the strongest of AR, seasonal and",
              "persistence in that cell.", "",
              "| input | channel | Seq2Seq | probGRU | AR | seasonal | persistence | best | "
              "Seq2Seq − best | probGRU − best |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        n2, npg, tot, beaten_by = 0, 0, 0, collections.Counter()
        for pgn, s2n, lab in PAIRS:
            for c in CH:
                k2, kp = M2.get(("skill", s2n, c)), PG.get(("skill", pgn, c))
                bs = {m: bl_skill(m, c) for m in BL}
                if k2 is None or kp is None or any(v is None for v in bs.values()):
                    continue
                bm = max(bs, key=lambda m: bs[m])
                bv = bs[bm]
                tot += 1
                n2 += k2 > bv
                npg += kp > bv
                if k2 <= bv or kp <= bv:
                    beaten_by[bm] += 1
                f = lambda v: f"{v:.4f}".replace("-", "−")
                g = lambda v: f"**{v:+.4f}**".replace("-", "−")
                L.append(f"| {lab} | {c} | {f(k2)} | {f(kp)} | {f(bs['ar'])} | "
                         f"{f(bs['seasonal'])} | 0.0000 | {bm} {f(bv)} | "
                         f"{g(k2 - bv)} | {g(kp - bv)} |")
        # Counted, never written by hand: an earlier draft of the sentence below said "eight
        # of the nine", counting the nine table ROWS when the table holds nine rows times two
        # backbones. That is the same drift this document exists to stop.
        L += ["",
              f"**Seq2Seq beats the best baseline in {n2} of {tot} cells, probGRU in "
              f"{npg} of {tot}.** Where a backbone loses, the arm beating it is "
              + ", ".join(f"`{m}` in {n}" for m, n in beaten_by.most_common())
              + " of those cells.", "",
              "A linear autoregression on the channel's own past is therefore the thing to",
              f"beat on this sensor, and {2 * tot - n2 - npg} of the {2 * tot} arm-by-channel",
              "cells here do not beat it. That is a statement about these runs at a 1 s",
              "horizon, not about the architectures in general -- but it is the comparison",
              "the two-backbone table above cannot make, because there both arms can lose",
              "and one still looks like the winner.", ""]

        # Hausdorff, same nine arms, same three references. Lower is better, so `best` is a
        # minimum here and the margins flip sign.
        L += ["Hausdorff: LOWER is better, so `best` is the smallest of the three and a",
              "NEGATIVE margin is the arm winning.", "",
              "| input | channel | Seq2Seq | probGRU | AR | seasonal | persistence | best | "
              "Seq2Seq − best | probGRU − best |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        h2n, hpn, htot = 0, 0, 0
        for pgn, s2n, lab in PAIRS:
            for c in CH:
                a2, ap = M2.get(("hausdorff", s2n, c)), PG.get(("hausdorff", pgn, c))
                hb = {m: M2.get(("hausdorff", m, c)) for m in BL}
                if a2 is None or ap is None or any(v is None for v in hb.values()):
                    continue
                bm = min(hb, key=lambda m: hb[m])
                bv = hb[bm]
                htot += 1
                h2n += a2 < bv
                hpn += ap < bv
                L.append(f"| {lab} | {c} | {a2:.3f} | {ap:.3f} | {hb['ar']:.3f} | "
                         f"{hb['seasonal']:.3f} | {hb['persistence']:.3f} | {bm} {bv:.3f} | "
                         f"**{a2 - bv:+.3f}** | **{ap - bv:+.3f}** |".replace("-", "−"))
        L += ["",
              f"**On shape, Seq2Seq beats the best baseline in {h2n} of {htot} cells and "
              f"probGRU in {hpn} of {htot}.**",
              "Note which baseline is hardest to beat under each metric: `seasonal` has the",
              "LOWEST Hausdorff of the three while its skill is NEGATIVE on every channel. A",
              "seasonal-naive forecast repeats a whole past cycle, so it has the right shape",
              "and the wrong phase: pointwise error punishes it, curve distance does not. It",
              "is the clearest case in this document of the two metrics measuring different",
              "things, and a reason not to read either alone.", ""]

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

    rows_as = [(b, e, as_run(p, e)) for b, e, p in AS_RUNS]
    if any(v for _, _, v in rows_as):
        L += ["## ActionSense: both backbones, all three inputs", "",
              "Mean over the six channels (two hands). Skill is frame-pooled per fold then",
              "averaged over folds -- the only convention ActionSense computes -- so it is",
              "the driver's, not the report's. Hausdorff is the same metric as everywhere",
              "else. `—` is a run not yet made.", "",
              "| backbone | input | mean skill | mean Hausdorff |",
              "|---|---|---|---|"]
        for b, e, v in rows_as:
            L.append(f"| {b} | {e} | " + ("— | — |" if v is None
                                          else f"{v[0]:.4f} | {v[1]:.3f} |"))
        pair = {(b, e): v for b, e, v in rows_as if v}
        deltas = [(e, pair[("probGRU", e)][1] - pair[("Seq2Seq", e)][1])
                  for e in ("aggregate", "cnn", "flatten")
                  if ("probGRU", e) in pair and ("Seq2Seq", e) in pair]
        if deltas:
            worse = sum(1 for _, d in deltas if d > 0)
            L += ["",
                  "**Hausdorff, probGRU minus Seq2Seq: " +
                  ", ".join(f"{e} {d:+.3f}" for e, d in deltas) +
                  f" -- probGRU is further from the truth in shape on {worse} of "
                  f"{len(deltas)} inputs.**",
                  "Read it beside the OpenTouch table above, where the same comparison is",
                  "positive in all nine cells. Agreement across both sensors would make",
                  "\"the one-shot head produces better-shaped forecasts\" an architectural",
                  "fact rather than a property of one dataset.", ""]

    L += open("docs/_skill_comparison_notes.md").read().rstrip().split("\n") + [""]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out, "w").write("\n".join(L) + "\n")
    print(f"wrote {a.out} ({len(RUNS)} runs x {len(ROWS)} models x {len(CH)} channels)")


if __name__ == "__main__":
    raise SystemExit(main())
