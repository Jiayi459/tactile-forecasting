"""Three-encoder skill comparison: does spatial structure beat the six moments?

Reads the per-action CSVs written by scripts/shared/score_preds_per_action.py -- one per
backbone, each holding every encoder that shared a --save-preds directory -- and draws the
comparison those tables make in numbers.

    python scripts/actionsense/plot_encoder_comparison.py \
        --csv docs/actionsense/results/corpus-aggregate-flatten-cnn/as_preds_tmap_seq2seq_corpus3s.csv \
        --csv docs/actionsense/results/corpus-aggregate-flatten-cnn/as_preds_tmap_probgru_corpus3s.csv \
        --out-dir docs/actionsense/results/corpus-aggregate-flatten-cnn

Two figures, one subplot per CSV (i.e. per backbone):
  <prefix>_by_channel.png   skill per target channel, plus the channel mean
  <prefix>_by_action.png    skill per action, sorted by how well the best arm does

WHICH SKILL EACH FIGURE USES, and why they differ. The by-channel figure reads
`skill_pooled_<ch>` -- frame-pooled, one ratio of summed squared error over every valid
(window, horizon-step) point. That is the estimator OpenTouch's cv4 table reports and the one
docs/skill_comparison.md tabulates, so it is the number to quote. The by-action figure has to
use `skill`, which is clip-balanced (each clip's ratio formed first, then averaged), because
the scorer only emits the pooled variant on the whole-dataset row. The two are not
interchangeable -- on the corpus runs they differ by 0.4 for probGRU -- so each axis says
which one it is showing rather than leaving the reader to assume.

Skill is measured against persistence, so 0 is the baseline: above it the model beats "the
signal stays where it is", below it the model is worse than doing nothing.
"""
from __future__ import annotations

import argparse
import collections
import csv
import os

ALL_ROW = "(all actions)"
PERS = "persistence"

# Categorical slots 1-3 of the reference palette, in its fixed order, unchanged. Encoders are
# an identity dimension, so the hue tracks the encoder and never its rank: `aggregate` stays
# blue whether it wins or loses, and a figure with one arm dropped does not repaint the rest.
ENC_ORDER = ["aggregate", "flatten", "cnn"]
ENC_COLOR = {"aggregate": "#2a78d6", "flatten": "#eb6834", "cnn": "#1baf7a"}
ENC_LABEL = {"aggregate": "aggregate (6 moments, no map)",
             "flatten": "flatten (1024 taxels, no spatial prior)",
             "cnn": "cnn (1024 taxels, convolutional)"}

INK, MUTED, GRID = "#1a1a19", "#5c5b54", "#d9d8d2"


def read(path: str) -> tuple[str, list[dict]]:
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit(f"{path} is empty")
    backbones = {r["model"].rsplit("_", 1)[0] for r in rows if r["model"] != PERS}
    return ("/".join(sorted(backbones)) or "?"), rows


def encoders_of(rows: list[dict]) -> list[str]:
    present = {r["model"].rsplit("_", 1)[-1] for r in rows if r["model"] != PERS}
    return [e for e in ENC_ORDER if e in present]


def channels_of(rows: list[dict]) -> list[str]:
    """Channel order as the scorer wrote it, from the skill_pooled_<ch> column names."""
    a = next((r for r in rows if r["action"] == ALL_ROW), None)
    if a is None:
        return []
    return [k[len("skill_pooled_"):] for k in a if k.startswith("skill_pooled_")]


def val(row: dict, key: str) -> float:
    v = row.get(key, "")
    return float(v) if v not in ("", None) else float("nan")


def grouped_bars(ax, groups, series, value, *, horizontal=False, label_first_group=False):
    """One bar per (group, series). `value(group, s)` -> float, NaN to skip."""
    n = len(series)
    span = 0.82
    w = span / n
    for si, s in enumerate(series):
        off = -span / 2 + w * (si + 0.5)
        pos = [i + off for i in range(len(groups))]
        vals = [value(g, s) for g in groups]
        kw = dict(color=ENC_COLOR[s], label=ENC_LABEL[s], zorder=3,
                  # a 2px surface gap between adjacent fills, so touching bars stay separable
                  linewidth=1.0, edgecolor="white")
        if horizontal:
            ax.barh(pos, vals, height=w * 0.94, **kw)
        else:
            ax.bar(pos, vals, width=w * 0.94, **kw)
        if label_first_group and vals and vals[0] == vals[0]:
            ax.annotate(f"{vals[0]:+.3f}", (pos[0], vals[0]),
                        textcoords="offset points", xytext=(0, 3 if vals[0] >= 0 else -11),
                        ha="center", fontsize=7.5, color=INK, zorder=4)


def style(ax, *, horizontal=False):
    ax.axhline(0, color=MUTED, lw=1.1, zorder=2) if not horizontal else \
        ax.axvline(0, color=MUTED, lw=1.1, zorder=2)
    ax.grid(axis="x" if horizontal else "y", color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left" if not horizontal else "bottom"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom" if not horizontal else "left"].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8.5, length=0)


def _legend_and_title(fig, ax, title):
    """Figure-level legend + suptitle, with the space they need reserved in INCHES.

    Anchoring the legend to an axes in axes-fraction coordinates put it on top of the first
    subplot's title as soon as the figure grew a second panel or a taller action list -- the
    fraction means a different number of inches in every figure this script draws.
    """
    handles, labels = ax.get_legend_handles_labels()
    band = 0.95                                   # inches for suptitle + legend
    top = 1.0 - band / fig.get_figheight()
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.legend(handles, labels, frameon=False, fontsize=8.5, labelcolor=INK, ncols=3,
               loc="upper center", bbox_to_anchor=(0.5, 1.0 - 0.42 / fig.get_figheight()))
    fig.suptitle(title, fontsize=11, color=INK, x=0.012, ha="left",
                 y=1.0 - 0.12 / fig.get_figheight())


def fig_by_channel(sources, out, prefix):
    import matplotlib.pyplot as plt
    # Small multiples of ONE measure: share the scale, or the eye compares bar lengths that
    # do not mean the same thing. A backbone that is worse everywhere should LOOK worse.
    fig, axes = plt.subplots(len(sources), 1, figsize=(9.4, 3.5 * len(sources)),
                             squeeze=False, sharey=True)
    for ax, (name, rows) in zip(axes[:, 0], sources):
        encs, chans = encoders_of(rows), channels_of(rows)
        allrow = {r["model"].rsplit("_", 1)[-1]: r for r in rows if r["action"] == ALL_ROW}
        groups = ["channel mean"] + chans

        def v(g, e):
            r = allrow.get(e)
            if r is None:
                return float("nan")
            return val(r, "skill_pooled") if g == "channel mean" else val(r, f"skill_pooled_{g}")

        grouped_bars(ax, groups, encs, v, label_first_group=True)
        style(ax)
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels(groups, fontsize=8.5, color=INK)
        # the headline group is the summary, not one channel among six
        ax.get_xticklabels()[0].set_fontweight("bold")
        ax.axvline(0.5, color=GRID, lw=0.9, ls=(0, (3, 3)), zorder=1)
        ax.set_ylabel("skill vs persistence\n(frame-pooled)", fontsize=8.5, color=MUTED)
        ax.set_title(f"{name}  ·  {allrow[encs[0]]['n_clips']} recordings",
                     fontsize=10.5, color=INK, loc="left", pad=8)
    _legend_and_title(
        fig, axes[0, 0],
        "Does spatial structure beat the six moments?   above 0 = better than persistence")
    p = os.path.join(out, f"{prefix}_by_channel.png")
    fig.savefig(p, dpi=150, facecolor="white")
    print(f"[done] {p}")


def fig_by_action(sources, out, prefix):
    import matplotlib.pyplot as plt
    # one shared action order, from the first source's best arm, so the panels stay comparable
    name0, rows0 = sources[0]
    encs0 = encoders_of(rows0)
    best = collections.defaultdict(lambda: float("-inf"))
    for r in rows0:
        if r["action"] != ALL_ROW and r["model"] != PERS:
            best[r["action"]] = max(best[r["action"]], val(r, "skill"))
    order = [a for a, _ in sorted(best.items(), key=lambda kv: kv[1])]
    if not order:
        print("[skip] no per-action rows to plot")
        return

    fig, axes = plt.subplots(1, len(sources),
                             figsize=(6.6 * len(sources), 1.9 + 0.30 * len(order)),
                             squeeze=False, sharey=True, sharex=True)
    for ax, (name, rows) in zip(axes[0], sources):
        encs = encoders_of(rows)
        by = {(r["action"], r["model"].rsplit("_", 1)[-1]): r
              for r in rows if r["model"] != PERS}

        def v(a, e):
            r = by.get((a, e))
            return val(r, "skill") if r else float("nan")

        grouped_bars(ax, order, encs, v, horizontal=True)
        style(ax, horizontal=True)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([a[:38] for a in order], fontsize=8, color=INK)
        ax.set_xlabel("skill vs persistence (clip-balanced)", fontsize=8.5, color=MUTED)
        ax.set_title(name, fontsize=10.5, color=INK, loc="left", pad=8)
    _legend_and_title(fig, axes[0][0], "Per-action skill, actions ordered by the best arm")
    p = os.path.join(out, f"{prefix}_by_action.png")
    fig.savefig(p, dpi=150, facecolor="white")
    print(f"[done] {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", action="append", required=True,
                    help="per-action CSV from score_preds_per_action.py; repeat for one "
                         "subplot per backbone")
    ap.add_argument("--out-dir", default="docs/actionsense")
    ap.add_argument("--prefix", default="encoder_comparison")
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")

    os.makedirs(a.out_dir, exist_ok=True)
    sources = [read(p) for p in a.csv]
    for name, rows in sources:
        encs = encoders_of(rows)
        print(f"  {name}: encoders={encs}  channels={len(channels_of(rows))}  "
              f"actions={len({r['action'] for r in rows if r['action'] != ALL_ROW})}")
        if len(encs) < 2:
            print(f"  WARNING: {name} holds only {encs} -- the comparison this plots needs "
                  f"several encoders in one --save-preds directory")
    fig_by_channel(sources, a.out_dir, a.prefix)
    fig_by_action(sources, a.out_dir, a.prefix)


if __name__ == "__main__":
    main()
