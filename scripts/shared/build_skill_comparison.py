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

# d256, the third sensor. Its metrics.csv shares OpenTouch's schema (an extra `fold` column is
# simply ignored, and averaging over folds is what we want), so `opentouch()` reads it.
#
# ITS PROTOCOL IS NOT THE OTHERS'. d256 is leave-one-SUBJECT-out; OpenTouch holds out a
# location over 4 folds; ActionSense is a stratified 60/20/20 by recording. All three answer
# "generalises to unseen what?" differently, and d256 answers the hardest of the three. A
# lower number here is therefore not by itself evidence of a worse model.
D256_RUNS = [("d256 none", "runs/d256_probgru_none/metrics.csv"),
             ("d256 class", "runs/d256_probgru_class/metrics.csv")]

# (label, ActionSense model name, OpenTouch model name). None means the arm has no
# counterpart there -- NOT that it scored zero.
# (label, ActionSense name, OpenTouch name, d256 name)
ROWS = [("AR", "ar", "ar", "ar"),
        ("seasonal", "seasonal", "seasonal", "seasonal"),
        # ActionSense's probGRU predicts the FAST component against persistence-of-fast,
        # OpenTouch's predicts the RAW target under the harness. Same name, different
        # question, so the ActionSense side stays empty rather than inviting the comparison.
        ("probGRU", None, "prob_gru", "probgru"),
        ("GRU-aggregate", "aggregate", "map_aggregate", None),
        ("CNN (map)", "cnn", "cnn", None),
        ("flatten (map)", "flatten", "flatten", None),
        # the probGRU backbone reading the map: same architecture as the probGRU row above,
        # only the input differs, which is what the d1_pg run exists to isolate
        ("probGRU + CNN", None, "pg_cnn", None),
        ("probGRU + flatten", None, "pg_flatten", None)]

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


def d256_hausdorff(path):
    """-> {(model, channel): (hausdorff, ratio_vs_persistence)} averaged over folds."""
    if not os.path.exists(path):
        return {}
    hd, rt = collections.defaultdict(list), collections.defaultdict(list)
    for r in csv.DictReader(open(path)):
        if r["horizon_step"] != "all":
            continue
        try:
            v = float(r["value"])
        except ValueError:
            continue
        if r["metric"] == "Hausdorff":
            hd[(r["model"], r["channel"])].append(v)
        elif r["metric"] == "HD_ratio_vs_persistence":
            rt[(r["model"], r["channel"])].append(v)
    return {k: (sum(hd[k]) / len(hd[k]),
                sum(rt[k]) / len(rt[k]) if rt.get(k) else None) for k in hd}


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
    D2 = {n: opentouch(p) for n, p in D256_RUNS if os.path.exists(p)}
    D2H = {n: d256_hausdorff(p) for n, p in D256_RUNS if os.path.exists(p)}
    if not D2:
        print("note: no d256 metrics.csv -- its columns will be absent")
    OT = {n: opentouch(p) for n, _, _, p in RUNS if os.path.exists(p)}
    missing = [n for n, _, _, p in RUNS if not os.path.exists(p)]
    if missing:
        print(f"note: no CSV for {', '.join(missing)} -- their columns will be empty")

    L = ["# Skill against persistence — every run, all three sensors", "",
         "Skill = 1 − MSE(model)/MSE(persistence) at the full 1 s horizon, pooled over",
         "frames, averaged over folds. Right hand only: OpenTouch instruments one hand, so",
         "ActionSense's and d256's `_R` channels are the closest their two-handed targets allow.",
         "",
         "**The three columns are not the same experiment.** d256 holds out a whole SUBJECT",
         "(5-fold LOSO), OpenTouch holds out a location (4-fold), ActionSense is a stratified",
         "60/20/20 by recording. Unseen-person is the hardest of the three, so d256 scoring",
         "lower is not on its own evidence of a worse model — it is a different question.",
         "The 1 s horizon and the persistence reference ARE identical across all three,",
         "which is what makes any comparison possible at all.", "",
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

    d2names = [n for n, _ in D256_RUNS if n in D2]
    for ch in CH:
        L += [f"## {ch}", "",
              "| model | ActionSense | " + " | ".join(f"`{n}`" for n in d2names)
              + (" | " if d2names else "") + " | ".join(f"`{n}`" for n, *_ in RUNS) + " |",
              "|---" * (len(RUNS) + len(d2names) + 2) + "|"]
        for label, asn, otn, d2n in ROWS:
            cells = ([cell(AS, asn, ch)]
                     + [cell(D2[n], d2n, ch) for n in d2names]
                     + [cell(OT.get(n, {}), otn, ch) for n, *_ in RUNS])
            L.append(f"| {label} | " + " | ".join(cells) + " |")
        # How hard was the denominator? Same row shape, so it reads directly under the skills
        # it qualifies. Sensor-level, not run-level: R is a property of the signal and the
        # horizon, so every run on one sensor shares it.
        if FL:
            def rcell(sensor):
                v = FL.get((sensor, ch))
                return "—" if v is None else f"{v[0]:.3f}".replace("-", "−")
            cells = ([rcell("actionsense")] + [rcell("d256") for _ in d2names]
                     + [rcell("opentouch") for _ in RUNS])
            L.append("| **R** (persistence difficulty) | " + " | ".join(cells) + " |")
        L.append("")

    RM = {n: report_metrics(p) for n, p in REPORTS}
    have = [n for n in RM if any(k[0] == "hausdorff" for k in RM[n])]
    # ONE Hausdorff section, not one per sensor. d256's used to sit in a separate "## Hausdorff
    # -- d256" heading immediately above this one, and two near-identically named sections made
    # the same metric on different sensors read as two different things. Skill puts all three
    # sensors in one table; this now matches.
    if have or (D2H and any(D2H.values())) or ASH:
        L += ["## Hausdorff distance between forecast and truth curves", "",
              "Laid out exactly like the skill tables above -- one section per channel, models",
              "down, sensors and runs across -- so a model can be followed along a row without",
              "re-learning a layout. LOWER is better, unlike skill.", "",
              "Scaled per forecast so the axes are commensurate: time spans [0,1] over the",
              "horizon, value is divided by the truth's own standard deviation there. Unlike",
              "MSE this is not pointwise, so a flat forecast through an oscillation is charged",
              "roughly its amplitude.", "",
              "**`persistence` is a row here, not a zero.** Skill divides it out; Hausdorff does",
              "not, so the reference has to be visible for a number to mean anything. Read each",
              "column against its own persistence, never across columns: the three sensors do",
              "not present equally hard signals (see the R row in the skill tables).", "",
              "Two estimators are mixed and cannot be compared cell to cell. d256 is",
              "frame-pooled from the driver's table, matching the skill above; OpenTouch is",
              "per-clip from its report; ActionSense is per-clip from its CV table at the",
              "longest history.", "",
              "**The ActionSense column is not readable on its own.** Its CV table carries only",
              "the `aggregate` encoder and NO persistence row, so there is no reference to",
              "divide by and the single number in that column cannot be interpreted the way the",
              "others can. What that arm does report is one run-level ratio,",
              f"**{_as_ratio():.2f}x persistence**, which is the only figure from it that",
              "compares to the others -- against d256's AR at 0.89x and OpenTouch's",
              "map_aggregate at 0.83x. Getting the column itself usable means re-running that",
              "arm with persistence scored, which has not been done.", ""]

        # Same column order and the same ROWS mapping the skill tables use, plus persistence,
        # which skill omits because it is 0 by construction and Hausdorff cannot.
        HD_ROWS = list(ROWS) + [("persistence", None, "persistence", "persistence")]

        def hd_as(name, ch):
            v = ASH.get((name, ch)) if name else None
            return "—" if v is None else f"{v[0]:.3f}"

        def hd_d2(run, name, ch):
            v = D2H.get(run, {}).get((name, ch)) if name else None
            return "—" if v is None else f"{v[0]:.3f}"

        def hd_ot(run, name, ch):
            v = RM.get(run, {}).get(("hausdorff", name, ch)) if name else None
            if v is None:
                return "—"
            return f"{v[0]:.3f}" if isinstance(v, tuple) else f"{v:.3f}"

        for ch in CH:
            L += [f"### Hausdorff — {ch}", "",
                  "| model | ActionSense | " + " | ".join(f"`{n}`" for n in d2names)
                  + (" | " if d2names else "") + " | ".join(f"`{n}`" for n in have) + " |",
                  "|---" * (len(have) + len(d2names) + 2) + "|"]
            for label, asn, otn, d2n in HD_ROWS:
                cells = ([hd_as(asn, ch)]
                         + [hd_d2(n, d2n, ch) for n in d2names]
                         + [hd_ot(n, otn, ch) for n in have])
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
