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

--window/--history cut the axis down to one stretch of the recording and hold the forecasts back
over its first seconds, so the truth alone carries the history the models were given before
anything is predicted. --zoom puts such a stretch in a second row under the full-width row;
--only draws a single input large.

Drawing lives in render()/render_with_zoom(), which take plain (t, value) series, so a figure can
be rebuilt from any source of those series with exactly this styling.
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
# An enlarged panel is ~3x the width of one in the row, so it carries a heavier line.
ZOOM_LW = 1.45
# One panel of the three-panel row. A standalone enlargement drawn at this width lines up
# with the panel it came from when the two are placed one above the other.
PANEL_W = PAPER_W / 3
# Reserve a fixed ~0.18 in for the legend rather than a fixed fraction: a fraction tuned for
# the 2.15 in row left a wide gap above a taller figure. 0.18275 in is exactly the 0.915 the
# row figure was drawn with, so that figure is unchanged.
LEGEND_IN = 0.18275
# Shading for a panel split into the history the models were given and the stretch they
# predict. The split is the point of such a panel, and a band of colour says it without
# spending the title on it.
HIST_FILL, PRED_FILL = "#DEE6F2", "#FAE3D2"


def _draw(ax, series, color, lw, z, bands, tmin=None, lw_scale=1.0):
    """One series, optionally withheld before tmin -- the stretch shown as history only."""
    t, v = np.asarray(series[0]), np.asarray(series[1])
    sig = np.asarray(series[2]) if len(series) > 2 else None
    if tmin is not None:
        m = t >= tmin
        t, v = t[m], v[m]
        sig = sig[m] if sig is not None else None
    if bands and sig is not None:
        ax.fill_between(t, v - 2 * sig, v + 2 * sig, color=color, alpha=0.18, lw=0,
                        zorder=z - 1)
    ax.plot(t, v, "-", color=color, lw=lw * lw_scale, zorder=z)


def _panel(ax, p, xlim, show_ylabel, bands, lw_scale=1.0, shade=False):
    """panel dict: `title`, `truth`, optional `s2s` / `pg`, optional `forecast_from`."""
    start = p.get("forecast_from")
    if shade and start is not None:
        # Behind the grid as well as the lines: these are regions of the axis, not data.
        ax.axvspan(xlim[0], start, color=HIST_FILL, lw=0, zorder=-2)
        ax.axvspan(start, xlim[1], color=PRED_FILL, lw=0, zorder=-2)
        for lo, hi, name in ((xlim[0], start, "history"), (start, xlim[1], "prediction")):
            ax.text((lo + hi) / 2, 0.955, name, transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=PT_LABEL, color=INK, weight="bold")
    _draw(ax, p["truth"], TRUTH, LW_TRUTH, 2, False, lw_scale=lw_scale)
    if p.get("pg") is not None:
        _draw(ax, p["pg"], PG_COLOR, LW_PG, 5, bands, start, lw_scale)
    if p.get("s2s") is not None:
        _draw(ax, p["s2s"], S2S_COLOR, LW_S2S, 6, bands, start, lw_scale)
    ax.set_title(p["title"], fontsize=PT_TITLE, color=INK, loc="left", pad=2.5)
    ax.set_xlabel("time  (s)", fontsize=PT_LABEL, color=MUTED, labelpad=1.5)
    if show_ylabel:
        ax.set_ylabel("Total Force F", fontsize=PT_LABEL, color=MUTED, labelpad=1.5)
    ax.tick_params(colors=MUTED, labelsize=PT_TICK, length=0, pad=1.5)
    ax.grid(color=GRID, lw=LW_GRID, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("bottom", "left"):
        ax.spines[sp].set_color(GRID)
        ax.spines[sp].set_linewidth(LW_SPINE)
    ax.set_xlim(*xlim)


def _box(ax, t0, t1):
    """Outline on a full-range panel showing which stretch the enlargement below covers."""
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle((t0, 0.012), t1 - t0, 0.976, transform=ax.get_xaxis_transform(),
                           fill=False, ec=MUTED, lw=0.5, zorder=8))


def _legend(fig):
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=TRUTH, lw=LW_TRUTH, label="ground truth"),
               Line2D([], [], color=PG_COLOR, lw=LW_PG, label="probGRU"),
               Line2D([], [], color=S2S_COLOR, lw=LW_S2S, label="Seq2Seq")]
    fig.legend(handles=handles, frameon=False, fontsize=PT_LEGEND, labelcolor=INK,
               ncols=len(handles), loc="upper center", bbox_to_anchor=(0.5, 0.995),
               handlelength=2.4)


def _save(fig, out, height, width=PAPER_W):
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=400, facecolor="white", bbox_inches="tight", pad_inches=0.01)
    print(f"[done] {out}  ({width:.3g}x{height} in)")


def render(panels, xlim, out, height=2.15, bands=False, lw_scale=1.0, shade=False,
           width=PAPER_W, legend=True, ylim=None):
    """One row of panels on a shared axis. xlim is (t0, t1), or a bare t1."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not isinstance(xlim, (tuple, list)):
        xlim = (0, xlim)
    fig, axes = plt.subplots(1, len(panels), figsize=(width, height), sharex=True,
                             sharey=True)
    for c, (ax, p) in enumerate(zip(np.atleast_1d(axes), panels)):
        _panel(ax, p, xlim, c == 0, bands, lw_scale=lw_scale, shade=shade)
        # Separately rendered panels autoscale separately, so three of them side by side end up
        # with different y ticks and gridlines. A shared limit is what makes them comparable.
        if ylim is not None:
            ax.set_ylim(*ylim)
    # The legend keeps its size while the figure narrows, so at one panel's width it would
    # crowd the axes it explains. Panels meant to sit under the row share the row's legend.
    fig.tight_layout(rect=(0, 0, 1, 1 - (LEGEND_IN / height if legend else 0.02)), w_pad=0.8)
    if legend:
        _legend(fig)
    _save(fig, out, height, width)
    plt.close(fig)


def render_with_zoom(panels, xlim, zooms, zoom_xlim, out, height=4.0, bands=False, box=None):
    """The row above, and its panels enlarged below -- one each, or one spanning the width."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(PAPER_W, height))
    gs = fig.add_gridspec(2, len(panels), height_ratios=[1.0, 0.95])
    top = None
    for c, p in enumerate(panels):
        ax = fig.add_subplot(gs[0, c], sharey=top)
        top = top or ax
        _panel(ax, p, xlim, c == 0, bands)
        if c:
            ax.tick_params(labelleft=False)
        if box:
            _box(ax, *box)
    below = None
    for c, z in enumerate(zooms):
        ax = fig.add_subplot(gs[1, :] if len(zooms) == 1 else gs[1, c], sharey=below)
        below = below or ax
        _panel(ax, z, zoom_xlim, c == 0, bands, lw_scale=ZOOM_LW, shade=True)
        if c:
            ax.tick_params(labelleft=False)
    fig.tight_layout(rect=(0, 0, 1, 1 - LEGEND_IN / height), w_pad=0.8, h_pad=1.2)
    _legend(fig)
    _save(fig, out, height)
    plt.close(fig)


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
    ap.add_argument("--height", type=float, default=None, help="figure height, inches")
    ap.add_argument("--no-legend", action="store_true",
                    help="omit the legend, for a panel that sits beside one carrying it")
    ap.add_argument("--panel-width", action="store_true",
                    help="draw at the width of one row panel instead of the full text width")
    ap.add_argument("--only", choices=["aggregate", "flatten", "cnn"], default=None,
                    help="draw a single input as one large panel (aggregate = physical state)")
    ap.add_argument("--window", type=float, nargs=2, metavar=("T0", "T1"),
                    help="restrict the axis to this stretch of the recording")
    ap.add_argument("--history", type=float, default=0.0,
                    help="with --window: seconds at its start showing the truth alone")
    ap.add_argument("--zoom", type=float, nargs=3, metavar=("T0", "T1", "HISTORY"),
                    help="add a second row enlarging the first panel over this window")
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
        if a.only and cell[1] != a.only:
            continue
        panels.append(dict(title=cell[2], truth=(tt, y[:, k]),
                           s2s=series(s2s[0] if s2s else None),
                           pg=series(pg[0] if pg else None)))

    if a.zoom:
        t0, t1, hist = a.zoom
        zooms = [dict(p, forecast_from=t0 + hist) for p in panels]
        render_with_zoom(panels, (0, tmax), zooms, (t0, t1), a.out,
                         height=a.height or 4.0, bands=a.bands, box=(t0, t1))
    elif a.window:
        t0, t1 = a.window
        panels = [dict(p, forecast_from=t0 + a.history) for p in panels]
        render(panels, (t0, t1), a.out, height=a.height or 2.15, bands=a.bands,
               lw_scale=ZOOM_LW, shade=True, legend=not a.no_legend,
               width=PANEL_W if a.panel_width else PAPER_W)
    else:
        render(panels, (0, tmax), a.out, height=a.height or 2.15, bands=a.bands)


if __name__ == "__main__":
    main()
