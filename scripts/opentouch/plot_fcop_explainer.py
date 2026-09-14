"""Paper figure: total force F and centre of pressure (CoP) on the hand -- formulas only (2026-09-14).

The hand-shaped pressure render with the action frame below it. Beside the palm, only the two
definitions from the code, no explanatory prose and no panel numbering (user rulings, 2026-09-14):

    F   = sum_{i,j} p~_ij                  arrows to the pressed spots: F sums all of them
    CoP = (1/F) sum_{i,j} p~_ij (x_j, y_i) the p~-weighted average position; the dot is labelled
                                           with its coordinates (CoP_x, CoP_y)

F IS A SUM, NOT A WEIGHTED AVERAGE (src/actionsense/physical_state.py:43 `F = p.sum()`;
src/opentouch/baseline.py moments; main.tex:106). The weighted average is the CoP.

THE MARKS ARE ILLUSTRATIVE and, with the on-canvas note removed by request, the CAPTION MUST SAY SO:
no taxel->hand map exists (SESSION_LOG.md:4414), so the F arrows point at the render's brightest glow
spots and the CoP dot sits at the render's brightness-weighted centroid. No real-clip number is
printed (the render is not a frame we hold data for). Real-data findings that belong in the text:
117/256 cells pinned at the 3072 ceiling pull CoP right; annotated peaks are not D1 F maxima
(SESSION_LOG.md 2026-09-14 §9).

    python scripts/opentouch/plot_fcop_explainer.py
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFilter

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ASSETS = os.path.join(ROOT, "figures", "assets")
plt.rcParams.update({"mathtext.fontset": "cm", "font.size": 7})

TEAL, SALMON, INK = "#006D77", "#006D77", "#1a1a1a"
FIG_W, FIG_H = 3.5, 3.20


def box(fig, l, b, w, h):
    return fig.add_axes([l / FIG_W, b / FIG_H, w / FIG_W, h / FIG_H])


def glow(img: np.ndarray) -> np.ndarray:
    """Pressure-glow weight of the render: 0 on the white page and on the black glove fabric."""
    a = img.astype(float)
    w = np.where(a.min(axis=2) > 225, 0.0, a.max(axis=2))
    return np.clip(w - 25.0, 0.0, None)


def glow_peaks(img: np.ndarray, n: int = 3, radius: int = 45) -> list[tuple[float, float]]:
    """The n brightest glow spots, at least `radius` px apart (on a blurred glow map)."""
    g = np.asarray(Image.fromarray(glow(img).astype(np.uint8)).filter(ImageFilter.GaussianBlur(6)),
                   dtype=float)
    ys, xs = np.mgrid[0:g.shape[0], 0:g.shape[1]]
    out = []
    for _ in range(n):
        r, c = np.unravel_index(np.argmax(g), g.shape)
        out.append((float(c), float(r)))
        g[(xs - c) ** 2 + (ys - r) ** 2 < radius ** 2] = 0.0
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(ROOT, "figures", "fcop_explainer"))
    a = ap.parse_args()

    render = np.asarray(Image.open(os.path.join(ASSETS, "opentouch_drill_render.png")).convert("RGB"))
    video = np.asarray(Image.open(os.path.join(ASSETS, "opentouch_drill_video.png")).convert("RGB"))
    fig = plt.figure(figsize=(FIG_W, FIG_H))

    # hand pressure + marks ------------------------------------------------------------------
    ax = box(fig, 0.04, 1.45, 1.50, 1.50)
    ax.imshow(render)
    ax.axis("off")
    ax.set_title("Hand pressure", fontsize=7.5, loc="left", pad=2, weight="bold")
    w = glow(render)
    ys, xs = np.mgrid[0:w.shape[0], 0:w.shape[1]]
    gx, gy = float((w * xs).sum() / w.sum()), float((w * ys).sum() / w.sum())

    # F: one arrow per pressed spot -- it is the SUM over all of them
    f_anchor = (1.72 / FIG_W, 2.66 / FIG_H)
    peaks = glow_peaks(render)
    for px, py in peaks:
        ax.annotate("", xy=(px, py), xycoords="data", xytext=f_anchor, textcoords="figure fraction",
                    arrowprops=dict(arrowstyle="-|>", color=SALMON, lw=0.8, mutation_scale=6,
                                    shrinkA=0, shrinkB=2), annotation_clip=False)
    fig.text(f_anchor[0] + 0.03 / FIG_W, f_anchor[1], r"$F=\sum_{i,j}\tilde p_{ij}$", fontsize=9,
             color=SALMON, va="center", ha="left")

    # CoP: dot labelled with its coordinates, leader to the formula
    ax.scatter([gx], [gy], s=46, facecolor="none", edgecolor="white", lw=2.2, zorder=6)
    ax.scatter([gx], [gy], s=46, facecolor="none", edgecolor=TEAL, lw=1.1, zorder=7)
    ax.scatter([gx], [gy], s=6, color=TEAL, zorder=8)
    ax.text(gx, gy - 22, r"$(\mathrm{CoP}_x,\ \mathrm{CoP}_y)$", fontsize=7, color=TEAL, ha="center",
            va="bottom", zorder=9,
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.92))
    c_anchor = (1.72 / FIG_W, 1.86 / FIG_H)
    ax.annotate("", xy=(gx, gy), xycoords="data", xytext=c_anchor, textcoords="figure fraction",
                arrowprops=dict(arrowstyle="-", color=TEAL, lw=0.8, shrinkA=0, shrinkB=4),
                annotation_clip=False)
    fig.text(c_anchor[0] + 0.03 / FIG_W, c_anchor[1],
             r"$\mathrm{CoP}=\frac{1}{F}\sum_{i,j}\tilde p_{ij}\,(x_j,\,y_i)$", fontsize=9,
             color=TEAL, va="center", ha="left")

    # the action -----------------------------------------------------------------------------
    h = 1.50 * video.shape[0] / video.shape[1]
    ax = box(fig, 0.04, 0.06, 1.50, h)
    ax.imshow(video)
    ax.axis("off")
    ax.set_title("Action", fontsize=7.5, loc="left", pad=2, weight="bold")

    fig.savefig(a.out + ".png", dpi=400, facecolor="white")
    fig.savefig(a.out + ".pdf", facecolor="white")
    print(f"wrote {a.out}.png/.pdf  CoP px=({gx:.0f},{gy:.0f})  F spots px={[(round(x), round(y)) for x, y in peaks]}")


if __name__ == "__main__":
    main()
