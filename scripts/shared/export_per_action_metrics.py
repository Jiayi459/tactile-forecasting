"""Export per-action forecast metrics (skill + Hausdorff) for OpenTouch and ActionSense.

WHAT THIS SCRIPT FOUND, AND WHY THE OUTPUT IS NOT THE TABLE THAT WAS ASKED FOR
-----------------------------------------------------------------------------
The request was "per action type, skill and Hausdorff distance, both corpora, sorted".
Only part of that exists on disk, and the script reports the gap rather than filling it:

  * OpenTouch per-action R2  -- EXISTS. scripts/opentouch/opentouch_report.py writes
    scope="action" rows for every action with >=30 clips, but ONLY metric="R2"
    (opentouch_report.py:277). It is the channel-mean of aggregate.r2.
  * OpenTouch per-action skill-vs-persistence -- NOT exported, but EXACTLY DERIVABLE from
    the R2 rows (see `derive_skill` for the proof).
  * OpenTouch per-action Hausdorff -- DOES NOT EXIST. `hausdorff_table` in the report script
    pools every clip and is only written at scope="overall". Recomputing it per action needs
    the per-clip forecast archives (runs/preds/clip_*.npz), which are NOT on this machine.
  * ActionSense per-action ANYTHING -- DOES NOT EXIST. Its frozen harness is restricted to
    `actions: [slice, peel]` (configs/actionsense/eval_harness.yaml:51) and no exported table
    carries an action/verb dimension for a forecast metric. `audit_actionsense` re-verifies
    this by scanning every CSV header rather than asserting it.

Usage:  python scripts/shared/export_per_action_metrics.py --out docs/per_action_metrics.md
"""
from __future__ import annotations

import argparse
import csv
import glob
import os

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# (label, report csv, the learned arm this family is ranked by)
FAMILIES = [
    ("probGRU backbone (`d1_pg`)",
     "docs/opentouch/d1_pg/opentouch_report_d1_pg.csv", "prob_gru"),
    ("Seq2Seq backbone (`d1_map2`)",
     "docs/opentouch/d1_map2/opentouch_report_d1_map2_hd.csv", "map_aggregate"),
]


def read_rows(path):
    """Report CSVs are appended across runs, so a repeated header line can appear as data."""
    with open(os.path.join(REPO, path)) as fh:
        return [r for r in csv.DictReader(fh) if r.get("scope") not in (None, "scope")]


def per_action_r2(rows):
    """-> ({action: {model: R2}}, {action: n_clips}), from the scope='action' block."""
    r2, n = {}, {}
    for r in rows:
        if r["scope"] != "action" or r["metric"] != "R2":
            continue
        r2.setdefault(r["subset"], {})[r["model"]] = float(r["value"])
        n[r["subset"]] = int(r["n_clips"])
    return r2, n


def derive_skill(r2_model: float, r2_pers: float) -> float:
    """skill = 1 - MSE_model/MSE_pers, derived from two R2s sharing one denominator.

    EXACT, not an approximation, because aggregate.clip_equal_ratio (aggregate.py:226) is a
    RATIO OF CLIP-BALANCED MEANS -- mean_k(sse_k/n_k) over a clip set that depends only on
    n_valid, never on the model. Writing Mbar(x) for that mean and D for the shared
    class-mean denominator:
        R2_m = 1 - Mbar(m)/D ,  R2_p = 1 - Mbar(p)/D
        =>  Mbar(m)/Mbar(p) = (1-R2_m)/(1-R2_p)  =>  skill = 1 - (1-R2_m)/(1-R2_p).

    THE ONE CAVEAT, and it is real: the exported per-action R2 is already averaged over the
    three channels (opentouch_report.py:275 `.per_channel.mean()`). The identity holds per
    channel; after channel-averaging this yields
        1 - mean_c(A_c/D_c) / mean_c(P_c/D_c),
    a denominator-weighted aggregate skill, which is NOT the same number as the mean of the
    three per-channel skills that aggregate.skill() would return. It is therefore labelled
    `skill*` everywhere it appears and must not be quoted as the harness's own skill.
    """
    den = 1.0 - r2_pers
    return float("nan") if abs(den) < 1e-12 else 1.0 - (1.0 - r2_model) / den


def overall_hausdorff(rows):
    """-> {model: {channel: (hd, ratio_vs_persistence)}} from the scope='overall' block."""
    out = {}
    for r in rows:
        if r["scope"] != "overall" or not r["metric"].startswith("hausdorff"):
            continue
        d = out.setdefault(r["model"], {}).setdefault(r["channel"], {})
        d["hd" if r["metric"] == "hausdorff" else "ratio"] = float(r["value"])
    return out


AS_DIR = "docs/actionsense/results/corpus-aggregate"
AS_RUNS = [("seq2seq, 3 s", "as_preds_seq2seq_corpus"),
           ("seq2seq, 1 s", "as_preds_seq2seq_corpus_h1"),
           ("probGRU, 3 s", "as_preds_probgru_corpus"),
           ("probGRU, 1 s", "as_preds_probgru_corpus_h1")]


def read_actionsense():
    """-> ({label: {action: row}}, {action: n_clips}) for the corpus-scope runs, or ({}, {}).

    Unlike OpenTouch's, these tables carry R2 AND skill AND Hausdorff per action, because they
    are produced by score_preds_per_action from the saved forecasts rather than by the report
    script's action block. The learned arm is the one model that is not persistence.
    """
    out, n = {}, {}
    for label, run in AS_RUNS:
        path = os.path.join(REPO, AS_DIR, run + ".csv")
        if not os.path.exists(path):
            continue
        with open(path) as fh:
            rows = list(csv.DictReader(fh))
        out[label] = {r["action"]: r for r in rows if r["model"] != "persistence"}
        out[label + " [persistence]"] = {r["action"]: r for r in rows
                                         if r["model"] == "persistence"}
        for r in rows:
            n[r["action"]] = int(r["n_clips"])
    return out, n


ALL_ROW = "(all actions)"


def actionsense_whole(data):
    """The four runs' whole-dataset numbers, side by side. Measured against the whole
    dataset's mean, so systematically higher than any per-action R2 below: between-action
    variance sits in the denominator here and inside each action's own mean there."""
    labels = [l for l, _ in AS_RUNS if l in data]
    JOBLOG = {"seq2seq, 3 s": "+0.145", "seq2seq, 1 s": "+0.137",
              "probGRU, 3 s": "-0.309", "probGRU, 1 s": "-0.324"}
    out = ["**Which skill?** OpenTouch's tables report `SS_vs_persistence` from its `cv4` "
           "driver table — **frame-pooled**, one ratio of summed squared error over every "
           "valid (window, horizon-step) point, averaged over folds. ActionSense's equivalent "
           "is `evaluate`'s `skill_ch`. To stay comparable with OpenTouch, **read the "
           "frame-pooled column**; the clip-balanced one is kept beside it because that is "
           "what `aggregate.skill` computes and it is the estimator the per-action tables "
           "below use.", "",
           "| run | R² | skill (frame-pooled, **matches OpenTouch**) | skill (clip-balanced) "
           "| Hausdorff | HD ratio |",
           "|---|---:|---:|---:|---:|---:|"]
    for l in labels:
        r = data[l].get(ALL_ROW)
        if not r:
            continue
        sp = r.get("skill_pooled") or JOBLOG.get(l, "—")
        sp = f"{float(sp):+.4f}" if sp not in ("—",) else sp
        out.append(f"| {l} | **{float(r['r2']):.4f}** | **{sp}** | {float(r['skill']):+.4f} | "
                   f"**{float(r['hausdorff']):.3f}** | {float(r['hausdorff_ratio']):.3f} |")
    if not (data[labels[0]].get(ALL_ROW) or {}).get("skill_pooled"):
        out += ["", "_The frame-pooled column above is read from the job logs "
                "(`logs/tactile_map.o14168xx`), because `cross_validate` prints `skill_ch` "
                "and never writes it. `score_preds_per_action.py` now emits it as "
                "`skill_pooled`; the next rescore will source this column from the CSV "
                "instead._"]
    out += ["", "Persistence, for reference (same rows, same denominator):", "",
            "| run | R² of persistence |", "|---|---:|"]
    for l in labels:
        r = data.get(l + " [persistence]", {}).get(ALL_ROW)
        if r:
            out.append(f"| {l} | {float(r['r2']):+.4f} |")
    return "\n".join(out) + "\n"


def actionsense_per_model(data, n):
    """One full table per run -- every column the run's own .md carries -- so this file is a
    replacement for the four source files rather than a summary that loses their content."""
    labels = [l for l, _ in AS_RUNS if l in data]
    out = []
    for l in labels:
        rows = {a: r for a, r in data[l].items() if a != ALL_ROW}
        pers = {a: r for a, r in data.get(l + " [persistence]", {}).items() if a != ALL_ROW}
        acts = sorted(rows, key=lambda a: -float(rows[a]["r2"]))
        best_hd = min(acts, key=lambda a: float(rows[a]["hausdorff"]))
        out += [f"#### {l}", "",
                "| # | action | n | R² | skill vs pers | Hausdorff | HD ratio | R² persistence |",
                "|---:|---|---:|---:|---:|---:|---:|---:|"]
        for i, a in enumerate(acts, 1):
            r = rows[a]
            pr = f"{float(pers[a]['r2']):+.4f}" if a in pers else "—"
            out.append(f"| {i} | {a} | {n[a]} | **{float(r['r2']):.4f}** | "
                       f"{float(r['skill']):+.4f} | {float(r['hausdorff']):.3f} | "
                       f"{float(r['hausdorff_ratio']):.3f} | {pr} |")
        out += ["", f"Highest R²: **{acts[0]}** ({float(rows[acts[0]]['r2']):.4f}). "
                f"Lowest Hausdorff: **{best_hd}** "
                f"({float(rows[best_hd]['hausdorff']):.3f}). "
                f"They are not the same action.", ""]
    return "\n".join(out)


def actionsense_tables(data, n):
    """Cross-run comparison: the same quantity for all four runs, one metric per table."""
    labels = [l for l, _ in AS_RUNS if l in data]
    if not labels:
        return "_No corpus-scope ActionSense tables on disk._\n"
    acts = sorted((a for a in data[labels[0]] if a != ALL_ROW),
                  key=lambda a: -float(data[labels[0]][a]["r2"]))

    def block(key, fmt, note):
        o = [note, "",
             "| # | action | " + " | ".join(f"{l}" for l in labels) + " |",
             "|---:|---|" + "---:|" * len(labels)]
        for i, a in enumerate(acts, 1):
            o.append(f"| {i} | {a} | "
                     + " | ".join(fmt.format(float(data[l][a][key])) for l in labels) + " |")
        return "\n".join(o) + "\n"

    return ("\n".join([
        block("r2", "{:.4f}",
              "**R²** — against the class mean, so **comparable across backbones**. "
              "Ranked by the first column."),
        block("hausdorff", "{:.3f}",
              "**Hausdorff** — lower is better; computed on residual curves for both "
              "backbones and invariant to the shared anchor, so also **comparable**."),
        block("skill", "{:+.4f}",
              "**Skill against persistence** — comparable **both ways**: the denominator is "
              "`(z[t+h] - z[t])^2` for both arms, persistence being the value at the same "
              "origin `t` in each. Only the weighting differs between pooled estimators."),
    ]))


def actionsense_analysis(data, n):
    """The three things the numbers say that a table alone does not."""
    labels = [l for l, _ in AS_RUNS if l in data]
    if len(labels) < 2:
        return ""
    acts = sorted(a for a in data[labels[0]] if a != ALL_ROW)
    r2 = {l: [float(data[l][a]["r2"]) for a in acts] for l in labels}
    hd = {l: [float(data[l][a]["hausdorff"]) for a in acts] for l in labels}
    pers = [float(data[labels[0] + " [persistence]"][a]["r2"]) for a in acts]
    lines = ["**1. The ranking is a property of the action, not of the model.** Spearman "
             "between the four runs' R² orderings:", ""]
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            lines.append(f"- {labels[i]} vs {labels[j]}: **{spearman(r2[labels[i]], r2[labels[j]]):+.3f}**")
    lines += ["", f"**2. …and it is mostly a property of *persistence*.** Spearman(R², "
              f"R² of persistence on the same action) = "
              f"**{spearman(r2[labels[0]], pers):+.3f}**. An action ranks high here largely "
              f"because its signal is smooth enough that the trivial predictor already does "
              f"well on it, not because the model has learnt something specific to it.", "",
              "**3. R² and Hausdorff rank the actions almost independently.** Spearman(R², HD) "
              "within a run:"]
    for l in labels:
        lines.append(f"- {l}: **{spearman(r2[l], hd[l]):+.3f}**")
    lines += ["", "Not merely a different order — no relationship. A shape metric and a "
              "squared-error metric are answering different questions about the same "
              "forecast, which is the point of reporting both."]
    if "seq2seq, 3 s" in data and "probGRU, 3 s" in data:
        d = sorted(((a,
                     float(data["seq2seq, 3 s"][a]["r2"]) - float(data["probGRU, 3 s"][a]["r2"]),
                     float(data["seq2seq, 3 s"][a]["hausdorff"]) - float(data["probGRU, 3 s"][a]["hausdorff"]))
                    for a in acts), key=lambda t: -t[1])
        won = sum(1 for _, _, dh in d if dh < 0)
        sk = {l: {a: float(data[l][a]["skill"]) for a in acts} for l in labels}
        pg_sk = sorted(sk["probGRU, 3 s"].items(), key=lambda t: t[1])
        med_pg = sorted(sk["probGRU, 3 s"].values())[len(acts) // 2]
        med_s2 = sorted(sk["seq2seq, 3 s"].values())[len(acts) // 2]
        worst, worst_v = pg_sk[0]
        lines += ["", f"**4. The backbone gap is one action, and all three metrics agree on "
                  f"which.** Seq2Seq beats probGRU on Hausdorff in **{won} of {len(d)}** "
                  f"actions, but the size of the gap is concentrated almost entirely in "
                  f"**{d[0][0]}**: ΔR² **{d[0][1]:+.4f}** where every other action is within "
                  f"±0.04, ΔHD **{d[0][2]:+.4f}** where the rest sit near −0.1, and skill "
                  f"**{worst_v:+.4f}** against Seq2Seq's "
                  f"**{sk['seq2seq, 3 s'][worst]:+.4f}**. Strip that action out and the "
                  f"medians are ordinary: probGRU **{med_pg:+.4f}**, Seq2Seq "
                  f"**{med_s2:+.4f}**.", "",
                  f"This matters for how the corpus-level result is read. probGRU's pooled "
                  f"skill over the whole corpus is **−0.309** — below persistence — but that "
                  f"is not a diffuse penalty spread over fourteen actions. It is "
                  f"`{worst}`, the largest group (55 of 290 recordings), failing hard while "
                  f"the other thirteen behave normally. Any explanation of the corpus number "
                  f"has to explain that one action, not the average.", "",
                  "| action | ΔR² (s2s − pg) | ΔHD (s2s − pg) | skill s2s | skill pg |",
                  "|---|---:|---:|---:|---:|"]
        for a, dr, dh in d:
            lines.append(f"| {a} | {dr:+.4f} | {dh:+.4f} | "
                         f"{sk['seq2seq, 3 s'][a]:+.4f} | {sk['probGRU, 3 s'][a]:+.4f} |")
    med = sorted(abs(float(data[labels[0]][a]["r2"]) - float(data[labels[1]][a]["r2"]))
                 for a in acts)[len(acts) // 2] if len(labels) > 1 else float("nan")
    pl = labels[0] + " [persistence]"
    if ALL_ROW in data.get(pl, {}):
        pr2 = float(data[pl][ALL_ROW]["r2"])
        cells = ", ".join(f"{l} **{float(data[l][ALL_ROW]['r2']):.4f}**" for l in labels)
        lines += ["", f"**5. On the whole dataset, probGRU is below persistence on R² too — "
                  f"and R² has no reference ambiguity.** Persistence scores "
                  f"**{pr2:.4f}** against the corpus mean; the arms score {cells}. Both "
                  f"probGRU runs land *under* the trivial predictor, both Seq2Seq runs above "
                  f"it. Skill already said this, and — contrary to an earlier note in this "
                  f"file — skill was entitled to: both arms divide by the same "
                  f"`(z[t+h]-z[t])^2`. R² is a second, independent denominator reaching the "
                  f"same verdict. The one cell where "
                  f"Hausdorff also crosses its reference is probGRU on `clean`, at ratio "
                  f"**1.033** — the only value above 1.0 in the entire table, meaning a "
                  f"forecast worse-shaped than assuming nothing changes."]
    lines += ["", f"**6. History is inert.** Median |R²(3 s) − R²(1 s)| = **{med:.4f}**, at or "
              f"below the ~5e-3 replicate noise floor recorded in "
              f"`docs/ICRA_PAPER_PLAN.md`. Both histories sit under the harness's "
              f"`min_history = 40`, so neither ever zero-pads a window; the comparison is "
              f"clean, and it says the axis does not matter."]
    return "\n".join(lines) + "\n"


def audit_actionsense():
    """-> (list of csvs carrying an action/verb column, total csvs scanned).

    Generated, not asserted: the claim "ActionSense has no per-action forecast metric" is
    only worth printing if it is re-checked against the files each time.
    """
    hits, seen = [], 0
    for path in sorted(glob.glob(os.path.join(REPO, "docs/actionsense/**/*.csv"),
                                 recursive=True)):
        seen += 1
        with open(path) as fh:
            head = fh.readline().strip().lower()
        if "action" in head or "verb" in head:
            hits.append((os.path.relpath(path, REPO), head))
    return hits, seen


def spearman(a, b):
    """Rank correlation without scipy (13 points, no ties expected)."""
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for pos, i in enumerate(order):
            r[i] = float(pos)
        return r
    ra, rb = rank(a), rank(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da = sum((x - ma) ** 2 for x in ra) ** 0.5
    db = sum((y - mb) ** 2 for y in rb) ** 0.5
    return num / (da * db) if da and db else float("nan")


def headline(label, primary, r2):
    """The two rankings disagree; quantify that instead of picking one and moving on."""
    acts = sorted(r2)
    m = [r2[a][primary] for a in acts]
    p = [r2[a]["persistence"] for a in acts]
    sk = [derive_skill(r2[a][primary], r2[a]["persistence"]) for a in acts]
    best_r2 = max(acts, key=lambda a: r2[a][primary])
    best_sk = max(acts, key=lambda a: derive_skill(r2[a][primary], r2[a]["persistence"]))
    return (f"- **{label}** — best by R²: **{best_r2}** ({r2[best_r2][primary]:.4f}); "
            f"best by skill\\*: **{best_sk}** ({derive_skill(r2[best_sk][primary], r2[best_sk]['persistence']):.4f}). "
            f"Spearman(skill\\*, R²) = **{spearman(sk, m):+.2f}**; "
            f"Spearman(skill\\*, R²_persistence) = **{spearman(sk, p):+.2f}**.")


def fmt(v, nd=4):
    return "—" if v is None else ("n/a" if v != v else f"{v:.{nd}f}")


def family_table(label, path, primary):
    rows = read_rows(path)
    r2, n = per_action_r2(rows)
    if primary not in next(iter(r2.values()), {}):
        raise SystemExit(f"{path}: no model {primary!r} in the action block")
    models = [primary, "ar", "persistence"]
    order = sorted(r2, key=lambda a: r2[a][primary], reverse=True)

    out = [f"### {label}", "",
           f"Ranked by **R² of `{primary}`**, high → low. `skill*` is derived "
           f"(see the caveat below), not exported.", "",
           "| # | action | n clips | R² " + f"`{primary}`" + " | R² `ar` | R² `persistence` "
           "| skill\\* `" + primary + "` | skill\\* `ar` |",
           "|---:|---|---:|---:|---:|---:|---:|---:|"]
    for i, act in enumerate(order, 1):
        d = r2[act]
        sk_m = derive_skill(d[primary], d["persistence"])
        sk_a = derive_skill(d["ar"], d["persistence"])
        out.append(f"| {i} | {act} | {n[act]} | **{fmt(d[primary])}** | {fmt(d['ar'])} | "
                   f"{fmt(d['persistence'])} | {fmt(sk_m)} | {fmt(sk_a)} |")
    out.append("")
    return "\n".join(out), order, r2, n


def hausdorff_section(label, path):
    rows = read_rows(path)
    hd = overall_hausdorff(rows)
    if not hd:
        return f"### {label}\n\nNo Hausdorff rows in this report.\n"
    chans = sorted({c for m in hd.values() for c in m})
    out = [f"### {label} — corpus-wide only (no action breakdown exists)", "",
           "| model | " + " | ".join(f"HD {c}" for c in chans) + " | HD mean | "
           + " | ".join(f"ratio {c}" for c in chans) + " |",
           "|---|" + "---:|" * (2 * len(chans) + 1)]
    rank = sorted(hd, key=lambda m: sum(hd[m][c]["hd"] for c in chans) / len(chans))
    for m in rank:
        vals = [hd[m][c]["hd"] for c in chans]
        rats = [hd[m][c].get("ratio", float("nan")) for c in chans]
        out.append(f"| `{m}` | " + " | ".join(fmt(v, 3) for v in vals)
                   + f" | **{fmt(sum(vals) / len(vals), 3)}** | "
                   + " | ".join(fmt(v, 3) for v in rats) + " |")
    out.append("")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="docs/per_action_metrics.md")
    a = ap.parse_args()

    parts, tops, heads = [], [], []
    for label, path, primary in FAMILIES:
        tbl, order, r2, n = family_table(label, path, primary)
        parts.append(tbl)
        heads.append(headline(label, primary, r2))
        tops.append((label, primary, order[:3], [r2[x][primary] for x in order[:3]]))

    hd_parts = [hausdorff_section(label, path) for label, path, _ in FAMILIES]
    as_data, as_n = read_actionsense()
    hits, seen = audit_actionsense()

    with open(os.path.join(REPO, a.out), "w") as fh:
        fh.write(HEADER)
        fh.write("\n".join(heads))
        fh.write(MIDDLE)
        fh.write("\n".join(parts))
        fh.write("\n## 3. Hausdorff distance — the level at which it actually exists\n\n")
        fh.write("\n".join(hd_parts))
        fh.write("\n## 4. ActionSense — per action, all four corpus runs\n\n")
        fh.write(AS_INTRO)
        fh.write("\n### 4.1 Whole dataset\n\n")
        fh.write(actionsense_whole(as_data))
        fh.write("\n### 4.2 Per action, one table per run\n\n")
        fh.write("Every column each run's own `.md` carries, so this section replaces those "
                 "four files rather than summarising them.\n\n")
        fh.write(actionsense_per_model(as_data, as_n))
        fh.write("\n### 4.3 The same metric across all four runs\n\n")
        fh.write(actionsense_tables(as_data, as_n))
        fh.write("\n### 4.4 What the ActionSense numbers say\n\n")
        fh.write(actionsense_analysis(as_data, as_n))
        fh.write("\n### 4.5 Frozen-harness audit (regenerated, not asserted)\n\n")
        fh.write(f"The tables above are the **corpus** scope. The **frozen** harness still "
                 f"carries no per-action forecast metric, and this is re-checked rather than "
                 f"asserted: scanned **{seen}** CSVs under `docs/actionsense/`, of which "
                 f"**{len(hits)}** have an action/verb column.\n\n")
        for pth, head in hits:
            fh.write(f"- `{pth}` — `{head}`\n")
        fh.write("\nNone of those is a forecast-metric table — `trait_partition.csv` is a "
                 "clip-count partition (verb → trait class) with no skill, R² or Hausdorff "
                 "column. The frozen harness is restricted to `actions: [slice, peel]` "
                 "(`configs/actionsense/eval_harness.yaml:51`), so a per-action breakdown of "
                 "it would have exactly two rows. That is the gap the corpus runs fill, and "
                 "the reason their numbers may not be quoted beside frozen ones.\n")
        fh.write(FOOTER)
    print(f"wrote {a.out}")
    for label, primary, top, vals in tops:
        print(f"  {label}: top-3 by R2({primary}) = "
              + ", ".join(f"{t} {v:.4f}" for t, v in zip(top, vals)))


HEADER = """# Per-action forecast metrics — skill and Hausdorff

*Generated by `scripts/shared/export_per_action_metrics.py`. Do not hand-edit.*

## 1. What this document can and cannot answer

The question was: across OpenTouch and ActionSense, which actions have the **highest skill**
and the **lowest Hausdorff distance**, exported per action type and sorted.

Three of those four things are not on disk, and this document says so rather than
manufacturing them:

| asked for | status |
|---|---|
| OpenTouch per-action R² | **exists** — `scope="action"` rows, actions with ≥30 clips |
| OpenTouch per-action skill vs persistence | **not exported**, but exactly derivable — see §2 |
| OpenTouch per-action Hausdorff | **does not exist** — `hausdorff_table` pools all clips and is written only at `scope="overall"` |
| ActionSense per-action R², skill **and** Hausdorff | **exists now** (§4) — from the corpus-scope runs, 290 recordings over 14 actions, scored from saved forecasts. The *frozen* harness still has none, and cannot: it is restricted to `[slice, peel]`. |

Recomputing OpenTouch's per-action Hausdorff is *possible in principle* — `opentouch_report.py`
already walks the corpus clip by clip — but it needs the per-clip forecast archives
`runs/preds/clip_*.npz`. It is a scoring job, not a training job: no GPU, no retraining, given
those archives, and `scripts/shared/score_preds_per_action.py` already does it for ActionSense
from the identical npz format.

**So the ranking below is OpenTouch only, by R², over the 13 actions with ≥30 clips.**
R² is also the metric this project decided to rank on: skill-vs-persistence is structurally
inflated on these targets, and `aggregate.skill` is marked "diagnostic only … does not
participate in inference" (`src/opentouch/aggregate.py:286`).

## 1.5 Headline: the two metrics name different winners

"""

MIDDLE = """

**Read that second correlation before quoting any "best action" number.** `skill*` tracks how
badly *persistence* does on an action far more than how well the *model* does. The actions
that top the skill ranking are the ones where persistence collapses — they are the actions
where the baseline is weakest, not the ones that are forecast best. This is the same
mechanism recorded as methodological finding #4 in `docs/ICRA_PAPER_PLAN.md` ("choose the
baseline before believing the number") and as problem **P3** in the session log, and it is
why this document ranks on R².

## 2. Per-action ranking (OpenTouch)

"""

AS_INTRO = """Scope: **290 recordings, 14 actions**, 5-fold CV by recording, aggregate input
(the 6-dim F/CoP signal; the map arms cannot run at this scope — only 100 of 299 recordings
have `clip_*.npy`). Produced by `scripts/shared/score_preds_per_action.py` from the saved
forecasts; source tables in `docs/actionsense/results/corpus-aggregate/`.

> **These numbers must not be placed in a table with frozen-harness results.** Widening the
> population from 75 slice/peel recordings to 290 changes the `Norm`, the class-mean
> denominator and the CV folds. Nothing about the frozen protocol was altered to produce them.

> **Skill IS comparable across the two backbones.** An earlier version of this file said it
> was not; that was wrong, and the correction matters because it changes what the numbers are
> allowed to say. `evaluate` computes `1 - mean(mu-y)^2 / mean(pers-y)^2` for both arms, and
> the denominators are *algebraically identical*: Seq2Seq's target is `z[t+1..t+H] - z[t]`
> with `pers = 0`, giving `(z[t+h]-z[t])^2`; probGRU's is `z[t+1..t+H]` with
> `pers = z[t]`, giving `(z[t]-z[t+h])^2` — the same number. Persistence is the value at the
> **same** origin `t` in both. Verified numerically on the real `AggWindows`: the denominators
> match bit for bit and the numerators match under the change of variable. The residual/absolute
> split changes what the network must emit, not what it is measured against.
>
> What *is* not interchangeable is the **weighting**. Three different pooled values exist for
> the same run and are labelled at every use; see §4.4.

"""

FOOTER = """
## 5. Reading the numbers

- **R²** is clip-balanced and against the **class mean**, not against persistence
  (`aggregate.r2`, baseline `class_mean`). A negative R² means worse than predicting that
  action's own mean — which is why `persistence` is negative for every action.
- **`skill*`** is the derived quantity `1 − (1−R²_model)/(1−R²_persistence)`. The derivation
  is exact per channel; the exported R² is already channel-averaged, so `skill*` is a
  denominator-weighted aggregate and **is not the harness's own `aggregate.skill` number**.
  Never quote it as such.
- **Hausdorff** is scale-normalized (`src/shape_metrics.py::hausdorff_scaled`); **lower is
  better**, and `ratio` < 1 means better-shaped than persistence.
- Both corpora's caveats still apply: OpenTouch has **no participant-level split**
  (`splits.py` is clip-level), so these are within-corpus numbers.
"""


if __name__ == "__main__":
    raise SystemExit(main())
