"""Paper figure: what total force F and centre of pressure (c^x, c^y) mean on OpenTouch.

Panels, drawn at the paper's own full width (7 in, every label >= 6 pt):
  (a) OpenTouch's hand-shaped pressure render, with an ILLUSTRATIVE CoP marker
  (b) the matching egocentric frame -- the action being performed
  (c) a REAL D1-corrected 16x16 map from CRC: which taxels carry F at that frame, and the real CoP
  (d) the formulas with that frame's numbers substituted, and the clip's F, c^x, c^y over time

WHAT IS REAL AND WHAT IS NOT -- the caption must repeat this:
  * (a)(b) come from the user's screenshot of OpenTouch's own visualisation (figures/assets/
    opentouch_drill_{render,video}.png). The shards and their RGB video were deleted, so the real
    clip in (c)(d) CANNOT be aligned to that screenshot: same object class, not the same frame.
  * The CoP dot in (a) is illustrative. No taxel->hand map exists (SESSION_LOG.md:4414; the
    OpenTouch paper gives none), so it is placed at the brightness-weighted centroid of the render's
    own glow -- the pressure centroid OF THAT PICTURE, not a mapped sensor coordinate.
  * (c)(d) are real: exported by scripts/opentouch/export_fcop_explainer.py, which verifies its D1
    maps reproduce the cache the paper scored.

FORMULAS (code, not paraphrase): D1 per shard, src/opentouch/baseline.py:126-130,
p~ = max(p - (b + k sigma), 0), k = 1; moments src/opentouch/baseline.py:66-85 ==
src/actionsense/physical_state.py:36-56 == main.tex:104-108. x = linspace(-1,1,16) along COLUMNS,
y = linspace(-1,1,16) along ROWS; with imshow's origin="upper", col = (c^x+1)/2*15, row = (c^y+1)/2*15.
F sums ALL 256 cells -- the "169 live taxels" note in extract_opentouch.py was falsified (dead = 0
on all 26 shards, SESSION_LOG.md:4620); after D1 only cells above rest + 1 sigma are non-zero.

    python scripts/opentouch/plot_fcop_explainer.py --npz fcop_explainer/fcop_clip1234.npz
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ASSETS = os.path.join(ROOT, "figures", "assets")
plt.rcParams.update({"mathtext.fontset": "cm", "font.size": 6.5, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.5, "ytick.major.width": 0.5,
                     "xtick.labelsize": 6, "ytick.labelsize": 6})

TEAL, SEA, PEARL, PEACH, SALMON = "#006D77", "#83C5BE", "#EDF6F9", "#FFDDD2", "#E29578"
INK, MUTED = "#1a1a1a", "#555555"
PRESS = LinearSegmentedColormap.from_list("press", [PEACH, SALMON, "#9c4a3a", "#3b1f1a"])
FIG_W, FIG_H = 7.0, 3.35


def box(fig, l, b, w, h):
    """Axes placed in inches."""
    return fig.add_axes([l / FIG_W, b / FIG_H, w / FIG_W, h / FIG_H])


def glow_centroid(img: np.ndarray) -> tuple[float, float]:
    a = img.astype(float)
    w = np.where(a.min(axis=2) > 225, 0.0, a.max(axis=2))   # drop the white page
    w = np.clip(w - 25.0, 0.0, None)                          # drop the near-black glove fabric
    ys, xs = np.mgrid[0:a.shape[0], 0:a.shape[1]]
    return float((w * xs).sum() / w.sum()), float((w * ys).sum() / w.sum())


def cop_marker(ax, x, y, s=46):
    ax.scatter([x], [y], s=s, marker="o", facecolor="none", edgecolor="white", lw=2.2, zorder=6)
    ax.scatter([x], [y], s=s, marker="o", facecolor="none", edgecolor=TEAL, lw=1.1, zorder=7)
    ax.scatter([x], [y], s=6, color=TEAL, zorder=8)


def title(ax, s, x=0.0):
    ax.set_title(s, fontsize=7, loc="left", x=x, pad=3, color=INK, weight="bold")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", required=True, help="one file written by export_fcop_explainer.py")
    ap.add_argument("--out", default=os.path.join(ROOT, "figures", "fcop_explainer"))
    a = ap.parse_args()

    d = np.load(a.npz, allow_pickle=False)
    meta, tags = json.loads(str(d["meta"])), json.loads(str(d["tags"]))
    F, cx, cy, fps = d["F"], d["cx"], d["cy"], float(d["fps"])
    chosen = int(d["chosen"])
    sel = list(d["frames_idx"])
    pmap = d["d1_maps"][sel.index(chosen)]                      # (16,16) D1 map at the frame
    Fc, cxc, cyc = float(F[chosen]), float(cx[chosen]), float(cy[chosen])
    n_pos = int((pmap > 0).sum())
    # the figure's numbers must be the cache's numbers, recomputed from the map it draws
    xs, ys = np.linspace(-1, 1, 16)[None, :], np.linspace(-1, 1, 16)[:, None]
    assert abs(pmap.sum() - Fc) <= 1e-3 * max(Fc, 1.0), "map sum != cached F"
    if Fc > 0:
        assert abs((pmap * xs).sum() / pmap.sum() - cxc) < 1e-3, "map CoP-x != cached c^x"
        assert abs((pmap * ys).sum() / pmap.sum() - cyc) < 1e-3, "map CoP-y != cached c^y"

    fig = plt.figure(figsize=(FIG_W, FIG_H))

    # (a) render with illustrative CoP -----------------------------------------------------
    render = np.asarray(Image.open(os.path.join(ASSETS, "opentouch_drill_render.png")).convert("RGB"))
    ax = box(fig, 0.06, 1.62, 1.50, 1.50)
    ax.imshow(render)
    gx, gy = glow_centroid(render)
    cop_marker(ax, gx, gy, s=60)
    ax.annotate("CoP\n(illustrative)", (gx, gy), xytext=(268, 62), fontsize=6.2,
                color=TEAL, ha="center", arrowprops=dict(arrowstyle="-", color=TEAL, lw=0.7))
    ax.axis("off")
    title(ax, "(a) OpenTouch pressure render")

    # (b) the action ----------------------------------------------------------------------------
    video = np.asarray(Image.open(os.path.join(ASSETS, "opentouch_drill_video.png")).convert("RGB"))
    ax = box(fig, 0.06, 0.30, 1.50, 1.50 * video.shape[0] / video.shape[1])
    ax.imshow(video)
    ax.axis("off")
    title(ax, "(b) Egocentric view of the action")
    fig.text(0.06 / FIG_W, 0.04 / FIG_H, "(a)(b): OpenTouch visualisation,\nnot the frame in (c)(d)",
             fontsize=6, color=MUTED, va="bottom", linespacing=1.2)

    # (c) real D1 map ---------------------------------------------------------------------------
    ax = box(fig, 1.98, 0.92, 2.05, 2.05)
    shown = np.ma.masked_where(pmap <= 0, pmap)
    ax.imshow(np.zeros_like(pmap), cmap=LinearSegmentedColormap.from_list("z", [PEARL, PEARL]),
              vmin=0, vmax=1, origin="upper", extent=(-0.5, 15.5, 15.5, -0.5))
    im = ax.imshow(shown, cmap=PRESS, origin="upper", extent=(-0.5, 15.5, 15.5, -0.5))
    ax.set_xticks(np.arange(-0.5, 16, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 16, 1), minor=True)
    ax.grid(which="minor", color="white", lw=0.5)
    ax.tick_params(which="minor", length=0)
    ax.set_xticks([0, 7.5, 15], ["−1", "0", "1"])
    ax.set_yticks([0, 7.5, 15], ["−1", "0", "1"])
    ax.set_xlabel(r"$x_j$  (16 columns)", fontsize=6.5, labelpad=1)
    ax.set_ylabel(r"$y_i$  (16 rows)", fontsize=6.5, labelpad=1)
    col, row = (cxc + 1) / 2 * 15, (cyc + 1) / 2 * 15
    ax.axvline(col, color=TEAL, lw=0.6, ls=(0, (2, 2)), zorder=5)
    ax.axhline(row, color=TEAL, lw=0.6, ls=(0, (2, 2)), zorder=5)
    cop_marker(ax, col, row, s=70)
    title(ax, "(c) Real D1 map, 16×16 = 256 cells")
    cb = fig.colorbar(im, cax=box(fig, 4.08, 0.92, 0.07, 2.05))
    cb.ax.tick_params(labelsize=5.5, length=2, width=0.4)
    cb.outline.set_linewidth(0.4)
    cb.set_label(r"$\tilde p_{ij}$  (a.u.)", fontsize=6, labelpad=1)
    fig.text(1.98 / FIG_W, 0.34 / FIG_H,
             f"clip {meta['idx']}: {meta.get('action', '')} {meta.get('object_name', '')}, "
             f"frame {chosen} ({str(d['chosen_reason']).split(' (')[0]})", fontsize=6, color=MUTED)
    fig.text(1.98 / FIG_W, 0.18 / FIG_H,
             f"coloured: {n_pos} cells with $\\tilde p>0$ carry $F$;  pale: 0 after D1", fontsize=6,
             color=MUTED)

    # (d) formulas with numbers ------------------------------------------------------------------
    ax = box(fig, 4.55, 2.02, 2.40, 1.10)
    ax.axis("off")
    title(ax, "(d) How each quantity is computed")
    k = float(d["k"])
    lines = [
        (r"$\tilde p_{ij}=\max\{p_{ij}-(b_{ij}+k\,\sigma_{ij}),\,0\}$" + f",  $k={k:g}$", INK),
        ("  per-shard rest level $b$ and noise $\\sigma$ removed (D1)", MUTED),
        (r"$F=\Sigma_{i,j}\,\tilde p_{ij}$" + f"$\;=\;${Fc:,.0f}", INK),
        ("  total contact load; summed counts, not newtons", MUTED),
        (r"$c^x=\Sigma_{i,j}\,x_j\,\tilde p_{ij}\,/\,F$" + f"$\;=\;${cxc:+.2f}", INK),
        (r"$c^y=\Sigma_{i,j}\,y_i\,\tilde p_{ij}\,/\,F$" + f"$\;=\;${cyc:+.2f}", INK),
        ("  where contact concentrates, in grid units $[-1,1]$", MUTED),
    ]
    y = 0.93
    for s, c in lines:
        ax.text(0.0, y, s, fontsize=6.6 if c == INK else 6.0, color=c, va="top", transform=ax.transAxes)
        y -= 0.145 if c == INK else 0.125

    # (d, cont.) time series ---------------------------------------------------------------------
    t = np.arange(len(F)) / fps
    series = [(F, r"$F$"), (cx, r"$c^x$"), (cy, r"$c^y$")]
    for n, (v, lab) in enumerate(series):
        ax = box(fig, 4.92, 1.36 - n * 0.50, 2.00, 0.42)
        ax.plot(t, v, color=TEAL, lw=0.8)
        for tag, fr in tags.items():
            ax.axvline(fr / fps, color=MUTED, lw=0.5, ls=(0, (2, 2)))
            if n == 0:
                ax.text(fr / fps, 1.04, tag, transform=ax.get_xaxis_transform(), fontsize=5.6,
                        color=MUTED, ha="center", va="bottom")
        ax.axvline(chosen / fps, color=SALMON, lw=1.0)
        ax.set_ylabel(lab, fontsize=6.8, rotation=0, ha="right", va="center", labelpad=2)
        ax.set_xlim(t[0], t[-1])
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        if n < 2:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("time (s)", fontsize=6.5, labelpad=1)
        ax.tick_params(length=2, pad=1)

    fig.savefig(a.out + ".png", dpi=400, facecolor="white")
    fig.savefig(a.out + ".pdf", facecolor="white")
    print(f"wrote {a.out}.png/.pdf   clip {meta['idx']} frame {chosen}: F={Fc:.0f} "
          f"cx={cxc:+.3f} cy={cyc:+.3f} n_taxels>0={n_pos}")


if __name__ == "__main__":
    main()
