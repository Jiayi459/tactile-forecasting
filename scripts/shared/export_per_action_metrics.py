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
    hits, seen = audit_actionsense()

    with open(os.path.join(REPO, a.out), "w") as fh:
        fh.write(HEADER)
        fh.write("\n".join(heads))
        fh.write(MIDDLE)
        fh.write("\n".join(parts))
        fh.write("\n## 3. Hausdorff distance — the level at which it actually exists\n\n")
        fh.write("\n".join(hd_parts))
        fh.write("\n## 4. ActionSense audit (regenerated, not asserted)\n\n")
        fh.write(f"Scanned **{seen}** CSVs under `docs/actionsense/`. "
                 f"Files whose header carries an action/verb column: "
                 f"**{len(hits)}**.\n\n")
        for p, head in hits:
            fh.write(f"- `{p}` — `{head}`\n")
        fh.write("\nNone of these is a forecast-metric table: `trait_partition.csv` is a "
                 "clip-count partition (verb → trait class), carrying no skill, R² or "
                 "Hausdorff column. ActionSense's frozen harness is additionally restricted "
                 "to `actions: [slice, peel]` "
                 "(`configs/actionsense/eval_harness.yaml:51`), so even a per-action "
                 "breakdown would have exactly two rows.\n")
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
| ActionSense per-action anything | **does not exist** — harness restricted to `[slice, peel]`; no table carries an action dimension (§4) |

Recomputing per-action Hausdorff is *possible in principle* — `opentouch_report.py`
already walks the corpus clip by clip — but it needs the per-clip forecast archives
`runs/preds/clip_*.npz`, and **`runs/` is empty on this machine**. It is a scoring job, not a
training job: no GPU, no retraining, given those archives.

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
