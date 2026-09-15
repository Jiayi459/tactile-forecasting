"""One recording, three inputs side by side, both backbones and AR in every panel.

    python scripts/plot_backbones_by_input.py --dataset egotouch \
        --preds runs/egotouch_merged/egotouch_test_unseen_3s \
        --clip 76 --channel F_R --out figures/backbones_by_input_ego_unseen_clip76.pdf

Panel c holds the ground truth, Seq2Seq and probGRU reading input c, and AR. The grid
(plot_clip_model_grid.py) asks "what does each arm look like"; this asks the narrower question
the grid spreads over two rows -- on the same input, do the two decoders differ, and does either
beat AR -- by putting all three on one axis.

STYLE is the grid's, so the two figures read as one family: hue encodes the INPUT exactly as it
does there (physical state blue, flatten orange, cnn green), AR keeps its pink, truth is black.
Within a panel the input is fixed, so hue cannot separate the backbones; line style does --
Seq2Seq solid, probGRU dashed and darker. The sigma bands are off by default because two
same-hue bands on one axis merge into one; --bands draws them anyway.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "actionsense"))
from plot_clip_model_grid import (COLOR, GRID, INK, LAYOUTS, MUTED, PAPER_W,  # noqa: E402
                                  PT_LABEL, PT_LEGEND, PT_TICK, PT_TITLE, TRUTH, load_clip,
                                  rolling)

AR_COLOR = COLOR["ar"]
LEGEND_GREY = "#9a9992"


def darker(hex_color: str, f: float = 0.62) -> str:
    """The same hue at lower lightness, so the dashed probGRU line stays legible against the
    solid Seq2Seq line of its own input colour."""
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    return "#{:02x}{:02x}{:02x}".format(int(r * f), int(g * f), int(b * f))


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

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    layout = LAYOUTS[a.dataset]
    s2s_row, pg_row, base_row = layout[0][1], layout[1][1], layout[-1][1]
    ar_arm = next((c[0] for c in base_row if c and c[1] == "ar"), None)

    data = load_clip(a.preds, a.clip)
    chans, fps, y = data["channels"], data["fps"], data["y"]
    if a.channel not in chans:
        raise SystemExit(f"--channel {a.channel} not in {chans}")
    k = chans.index(a.channel)
    H = next(iter(data["arms"].values()))[0].shape[1]
    tt = np.arange(len(y)) / fps
    tmax = min(a.seconds, tt[-1]) if a.seconds else tt[-1]

    want = [c[0] for c in s2s_row + pg_row if c] + ([ar_arm] if ar_arm else [])
    missing = [m for m in want if m not in data["arms"]]
    print(f"  clip {a.clip}  action={data['action']!r}  {len(y)} frames @ {fps:g} Hz  "
          f"H={H} ({H / fps:g} s)")
    if missing:
        print(f"  MISSING (drawn without them): {missing}")

    def draw(ax, arm, color, ls, lw, z):
        mu, sg, ors = data["arms"][arm]
        idx, val, sig = rolling(mu, sg, ors, H)
        t = idx / fps
        if a.bands and sig is not None:
            ax.fill_between(t, val[:, k] - 2 * sig[:, k], val[:, k] + 2 * sig[:, k],
                            color=color, alpha=0.14, lw=0, zorder=z - 1)
        ax.plot(t, val[:, k], ls=ls, color=color, lw=lw, zorder=z)

    fig, axes = plt.subplots(1, 3, figsize=(PAPER_W, a.height), sharex=True, sharey=True)
    for c, ax in enumerate(axes):
        s2s, pg = s2s_row[c], pg_row[c]
        key, title = (s2s or pg)[1], (s2s or pg)[2]
        hue = COLOR[key]
        ax.plot(tt, y[:, k], "-", color=TRUTH, lw=0.9, zorder=2)
        if ar_arm in data["arms"]:
            draw(ax, ar_arm, AR_COLOR, "-", 0.8, 3)
        if pg and pg[0] in data["arms"]:
            draw(ax, pg[0], darker(hue), (0, (3.2, 1.6)), 0.95, 5)
        if s2s and s2s[0] in data["arms"]:
            draw(ax, s2s[0], hue, "-", 1.0, 6)
        ax.set_title(title, fontsize=PT_TITLE, color=INK, loc="left", pad=2.5)
        ax.set_xlabel("time  (s)", fontsize=PT_LABEL, color=MUTED, labelpad=1.5)
        if c == 0:
            ax.set_ylabel("Total Force F", fontsize=PT_LABEL, color=MUTED, labelpad=1.5)
        ax.tick_params(colors=MUTED, labelsize=PT_TICK, length=0, pad=1.5)
        ax.grid(color=GRID, lw=0.4, zorder=0)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("bottom", "left"):
            ax.spines[sp].set_color(GRID)
        ax.set_xlim(0, tmax)

    # Backbone entries are drawn in neutral grey: their hue changes panel to panel, and what
    # the legend has to carry is the line style that stays fixed. Not ink -- the truth is ink,
    # and a black solid "Seq2Seq" swatch was indistinguishable from the "ground truth" one.
    handles = [Line2D([], [], color=TRUTH, lw=0.9, label="ground truth"),
               Line2D([], [], color=LEGEND_GREY, lw=1.0, label="Seq2Seq"),
               Line2D([], [], color=LEGEND_GREY, lw=0.95, ls=(0, (3.2, 1.6)), label="probGRU"),
               Line2D([], [], color=AR_COLOR, lw=0.8, label="AR")]
    fig.tight_layout(rect=(0, 0, 1, 0.915), w_pad=0.8)
    fig.legend(handles=handles, frameon=False, fontsize=PT_LEGEND, labelcolor=INK,
               ncols=len(handles), loc="upper center", bbox_to_anchor=(0.5, 0.995),
               handlelength=2.4)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=400, facecolor="white", bbox_inches="tight", pad_inches=0.01)
    print(f"[done] {a.out}  ({PAPER_W}x{a.height} in)")


if __name__ == "__main__":
    main()
