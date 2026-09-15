"""One recording, three inputs side by side, both backbones in every panel.

    python scripts/plot_backbones_by_input.py --dataset egotouch \
        --preds runs/egotouch_merged/egotouch_test_unseen_3s \
        --clip 76 --channel F_R --out figures/backbones_by_input_ego_unseen_clip76.png

Panel c holds the ground truth and the Seq2Seq and probGRU forecasts reading input c. The grid
(plot_clip_model_grid.py) spreads the two decoders over two rows; this puts them on one axis so
"on the same input, do they differ" is read without the eye travelling between rows.

COLOUR encodes the BACKBONE, not the input: within a panel the input is fixed and the panel title
already names it, so hue is spent where it separates something. probGRU is deep red
(#C62828) and Seq2Seq deep blue (#1F5AA6), both solid; truth is black.
The sigma bands are off by default; --bands draws them in each backbone's colour.

Drawing lives in render(), which takes plain (t, value) series, so the figure can be rebuilt from
any source of those series with exactly this styling.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "actionsense"))
from plot_clip_model_grid import (GRID, INK, LAYOUTS, MUTED, PAPER_W, PT_LABEL,  # noqa: E402
                                  PT_LEGEND, PT_TICK, PT_TITLE, TRUTH, load_clip, rolling)

# Deep red and deep blue at similar lightness, so neither forecast visually outranks the
# other and both stand clear of the black truth. The earlier pastel pair (#E29578, #83C5BE)
# receded behind the truth at 0.6 pt.
PG_COLOR, S2S_COLOR = "#C62828", "#1F5AA6"
# Thin throughout: at 7.16 in the three panels hold ~500 points each, and at 0.9-1.0 pt the
# forecasts merged into a band over the truth.
LW_TRUTH, LW_S2S, LW_PG, LW_GRID, LW_SPINE = 0.6, 0.6, 0.6, 0.3, 0.5


def render(panels, tmax, out, height=2.15, bands=False):
    """panels: one dict per input, keys `title`, `truth`, and optionally `s2s` / `pg`, each a
    tuple (t, value) or (t, value, sigma)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    def draw(ax, series, color, lw, z):
        t, v = np.asarray(series[0]), np.asarray(series[1])
        sig = series[2] if len(series) > 2 else None
        if bands and sig is not None:
            sig = np.asarray(sig)
            ax.fill_between(t, v - 2 * sig, v + 2 * sig, color=color, alpha=0.18, lw=0,
                            zorder=z - 1)
        ax.plot(t, v, "-", color=color, lw=lw, zorder=z)

    fig, axes = plt.subplots(1, len(panels), figsize=(PAPER_W, height), sharex=True,
                             sharey=True)
    for c, (ax, p) in enumerate(zip(np.atleast_1d(axes), panels)):
        draw(ax, p["truth"], TRUTH, LW_TRUTH, 2)
        if p.get("pg") is not None:
            draw(ax, p["pg"], PG_COLOR, LW_PG, 5)
        if p.get("s2s") is not None:
            draw(ax, p["s2s"], S2S_COLOR, LW_S2S, 6)
        ax.set_title(p["title"], fontsize=PT_TITLE, color=INK, loc="left", pad=2.5)
        ax.set_xlabel("time  (s)", fontsize=PT_LABEL, color=MUTED, labelpad=1.5)
        if c == 0:
            ax.set_ylabel("Total Force F", fontsize=PT_LABEL, color=MUTED, labelpad=1.5)
        ax.tick_params(colors=MUTED, labelsize=PT_TICK, length=0, pad=1.5)
        ax.grid(color=GRID, lw=LW_GRID, zorder=0)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("bottom", "left"):
            ax.spines[sp].set_color(GRID)
            ax.spines[sp].set_linewidth(LW_SPINE)
        ax.set_xlim(0, tmax)

    handles = [Line2D([], [], color=TRUTH, lw=LW_TRUTH, label="ground truth"),
               Line2D([], [], color=PG_COLOR, lw=LW_PG, label="probGRU"),
               Line2D([], [], color=S2S_COLOR, lw=LW_S2S, label="Seq2Seq")]
    fig.tight_layout(rect=(0, 0, 1, 0.915), w_pad=0.8)
    fig.legend(handles=handles, frameon=False, fontsize=PT_LEGEND, labelcolor=INK,
               ncols=len(handles), loc="upper center", bbox_to_anchor=(0.5, 0.995),
               handlelength=2.4)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=400, facecolor="white", bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    print(f"[done] {out}  ({PAPER_W}x{height} in)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(LAYOUTS))
    ap.add_argument("--preds", action="append", required=True)
    ap.add_argument("--clip", type=int, required=True)
    ap.add_argument("--channel", default="F_R")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seconds", type=float, default=None,
                    help="plot only the first N seconds (clamped to the recording)")
    ap.add_argument("--bands", action="store_true", help="draw ±2σ for both backbones")
    ap.add_argument("--height", type=float, default=2.15, help="figure height, inches")
    a = ap.parse_args()

    layout = LAYOUTS[a.dataset]
    s2s_row, pg_row = layout[0][1], layout[1][1]

    data = load_clip(a.preds, a.clip)
    chans, fps, y = data["channels"], data["fps"], data["y"]
    if a.channel not in chans:
        raise SystemExit(f"--channel {a.channel} not in {chans}")
    k = chans.index(a.channel)
    H = next(iter(data["arms"].values()))[0].shape[1]
    tt = np.arange(len(y)) / fps
    tmax = min(a.seconds, tt[-1]) if a.seconds else tt[-1]

    want = [c[0] for c in s2s_row + pg_row if c]
    missing = [m for m in want if m not in data["arms"]]
    print(f"  clip {a.clip}  action={data['action']!r}  {len(y)} frames @ {fps:g} Hz  "
          f"H={H} ({H / fps:g} s)")
    if missing:
        print(f"  MISSING (drawn without them): {missing}")

    def series(arm):
        if arm is None or arm not in data["arms"]:
            return None
        mu, sg, ors = data["arms"][arm]
        idx, val, sig = rolling(mu, sg, ors, H)
        return (idx / fps, val[:, k]) + ((sig[:, k],) if sig is not None else ())

    panels = []
    for s2s, pg in zip(s2s_row, pg_row):
        cell = s2s or pg
        panels.append(dict(title=cell[2], truth=(tt, y[:, k]),
                           s2s=series(s2s[0] if s2s else None),
                           pg=series(pg[0] if pg else None)))
    render(panels, tmax, a.out, height=a.height, bands=a.bands)


if __name__ == "__main__":
    main()
