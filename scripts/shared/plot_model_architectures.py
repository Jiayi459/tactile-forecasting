"""Paper figure (`fig:model_architectures`) -- both learned forecasters on one canvas, drawn in the
visual language of the user's reference block diagram (2026-09-14): nested dashed rounded
containers in the draw.io palette, header chips INSIDE each panel (no (a)/(b)/(c) captions), a
photo of the physical sensor in a pink frame on the far left, thin right-angle wires.

    [glove photo] --tactile--> | Physical state | --downsample--> | Embedding | --h--> | Forecast | --> s_hat_{t+n}

  Physical state : s_t = [F, CoP_x, CoP_y]
  Embedding      : per-frame state -> encoder GRU -> h            (shared by both rows)
  Forecast       : TOP    ProbGRU — autoregressive  decoder GRU rolled out, mean fed back
                   BOTTOM Seq2Seq — one-shot        ONE-SHOT DECODER = two Linear heads + reshape

WHY (Embedding) IS SHARED -- `src/actionsense/tactile_map/models.py`: `build_model` hands the SAME
per-frame encoder to both backbones (models.py:133-142); ProbGRU.forward runs
`self.enc(self.frame_encoder(x))` (:114) exactly as Seq2Seq.forward runs `self.gru(self.encoder(x))`.

WHY THE Seq2Seq "decoder" IS TWO LINEAR BOXES AND NOT A GRU (ruled 2026-09-14): every Seq2Seq
implementation behind the paper holds ONE nn.GRU and no decoder module -- tactile_map/models.py:62-80,
opentouch/tactile_map.py:102-120, opentouch/gru_aggregate.py:128-142. Its decoding is
`Linear(d_h -> H*C)` then `.reshape(B, H, C)` (models.py:78-80). The two heads are grouped and
labelled "one-shot decoder", which is the correct seq2seq-sense name for them and does not assert a
recurrent decoder; main.tex:213 says the same ("two linear heads ... all H steps jointly").

NO FEEDBACK LOOP back to the glove, although the reference figure has one: that loop is closed-loop
robot control. These forecasters are open-loop; a return wire would assert a feedback path that
does not exist.

THE PHOTO: figures/assets/tactile_glove_src.png, supplied by the user (screenshot, 2026-09-14). It
looks cropped from a published figure (it carries an orange annotation arrow), so the paper MUST
credit its source / hold permission -- the user's call, flagged in SESSION_LOG.md. GLOVE_CROP keeps
the fingers and palm and drops the arrow, whose tip sits at y~300 of the 486x502 source.

NOT DRAWN (requested 2026-09-13/14): action label / nn.Embedding / e_a; preprocessing after the
downsample, and the downsample factor/rate themselves (x3, 10 Hz); forecast-origin label and scale
bars; logvar clamp; the Seq2Seq heads' shape Linear(d_h -> H*C) and the reshape label. The heads'
shapes live in the paper text (main.tex:213-215), not on the canvas.

Usage:  python scripts/shared/plot_model_architectures.py [--out figures/model_architectures]
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GLOVE_SRC = os.path.join(ROOT, "figures", "assets", "tactile_glove_src.png")
GLOVE_CROP = (40, 0, 450, 292)            # (left, top, right, bottom) px -- excludes the arrow

plt.rcParams["mathtext.fontset"] = "cm"   # serif math, as in the reference

# ------------------------------------------ "Ocean Pearl Delight" palette (user, 2026-09-14) --
TEAL, SEA, PEARL, PEACH, SALMON = "#006D77", "#83C5BE", "#EDF6F9", "#FFDDD2", "#E29578"
OUTER_FC, OUTER_EC = PEARL, SEA          # whole figure
INNER_FC, INNER_EC = SEA, TEAL           # the model
PANEL_FC, PANEL_EC = PEARL, TEAL         # Physical state, Embedding
FCAST_FC, FCAST_EC = PEACH, SALMON       # Forecast
INPUT_FC, INPUT_EC = PEACH, SALMON       # glove frame, s_t, per-frame states
GRU_FC, GRU_EC = TEAL, TEAL              # learned recurrent modules (white text)
HEAD_FC, HEAD_EC = PEARL, TEAL           # Linear heads
CHIP = TEAL                              # header chips (white text)
FEED = SALMON
INK = "#1a1a1a"
DASH = (0, (4, 2.5))

W, H = 200.0, 72.0


# -------------------------------------------------------------------------- primitives --
# Font sizes are for the 13.5 in canvas, which the paper shrinks ~0.53x to \textwidth; 12-14 pt
# here prints at ~6.5-7.5 pt (enlarged 2026-09-14 from 6-8 pt, which printed at 3-4 pt).
LW = 1.3


def rbox(ax, x, y, w, h, fc, ec, lw=LW, ls="-", r=1.2, z=1.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, ls=ls, zorder=z))


def txt(ax, x, y, s, fs=12.0, color=INK, rot=0, weight="normal", ha="center", va="center", z=5):
    ax.text(x, y, s, fontsize=fs, color=color, rotation=rot, weight=weight, ha=ha, va=va,
            zorder=z, linespacing=1.15)


def block(ax, x, y, w, h, s, fc, ec, fs=12.0, rot=0, tc=INK):
    rbox(ax, x, y, w, h, fc, ec, lw=LW, r=0.6, z=3)
    txt(ax, x + w / 2, y + h / 2, s, fs=fs, rot=rot, color=tc)


def chip(ax, cx, cy, s, fs=13.5):
    ax.text(cx, cy, s, fontsize=fs, weight="bold", color="white", ha="center", va="center", zorder=5,
            linespacing=1.05, bbox=dict(boxstyle="square,pad=0.3", fc=CHIP, ec="none"))


def wire(ax, pts, head=True, color="#222222", lw=LW, ls="-"):
    """Right-angle polyline; the arrowhead, if any, sits only on the last segment."""
    body = pts[:-1] if head else pts
    if len(body) >= 2:
        xs, ys = zip(*body)
        ax.plot(xs, ys, color=color, lw=lw, ls=ls, zorder=2.5, solid_capstyle="butt")
    if head:
        ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>", mutation_scale=13,
                                     lw=lw, ls=ls, color=color, shrinkA=0, shrinkB=0, zorder=2.6))


# ---------------------------------------------------------------------------- canvas --
def build(width=13.5):
    fig = plt.figure(figsize=(width, width * H / W))
    ax = fig.add_axes([0, 0, 1, 1])
    PY0, PY1, CHIP_Y = 4.5, 67.0, 63.2          # panel bottom / top, chip centre

    # containers, outermost first
    rbox(ax, 0.6, 0.6, 198.8, 70.8, OUTER_FC, OUTER_EC, lw=1.6, r=3.0, z=0.5)
    rbox(ax, 30.5, 2.5, 153.5, 67.0, INNER_FC, INNER_EC, lw=1.1, ls=DASH, r=2.0, z=0.8)

    # ------------------------------------------------------------------- glove photo --
    rbox(ax, 2.2, 18.5, 26.3, 33, INPUT_FC, INPUT_EC, lw=1.6, r=2.0, z=1.0)
    im = np.asarray(Image.open(GLOVE_SRC).convert("RGB").crop(GLOVE_CROP))
    ph = 24.3 * im.shape[0] / im.shape[1]
    ax.imshow(im, extent=(3.2, 27.5, 49.5 - ph, 49.5), aspect="auto", zorder=3,
              interpolation="lanczos")
    txt(ax, 15.35, 24.8, "Tactile glove", fs=14)

    wire(ax, [(28.5, 35.0), (46.0, 35.0)])
    txt(ax, 36.2, 37.2, "tactile", fs=12.5)

    # ---------------------------------------------------------------- Physical state --
    rbox(ax, 41, PY0, 19, PY1 - PY0, PANEL_FC, PANEL_EC, lw=1.1, ls=DASH, r=2.0, z=1.2)
    chip(ax, 50.5, CHIP_Y - 0.6, "Physical\nstate")
    block(ax, 46, 12, 9, 45, r"$s_t=[\,F,\ CoP_x,\ CoP_y\,]$", INPUT_FC, INPUT_EC, fs=15, rot=90)

    fxs = np.linspace(80, 102, 6)
    wire(ax, [(55, 43.0), (fxs[0] - 1.4, 43.0)])
    txt(ax, 67.4, 45.2, "downsample", fs=11.5)

    # --------------------------------------------------------------------- Embedding --
    rbox(ax, 74, PY0, 34, PY1 - PY0, PANEL_FC, PANEL_EC, lw=1.1, ls=DASH, r=2.0, z=1.2)
    chip(ax, 91, CHIP_Y, "Embedding")
    txt(ax, 91, 54.6, "per-frame physical\n" r"state $\in\mathbb{R}^{C}$", fs=12.5)
    for xb in fxs:
        rbox(ax, xb - 1.4, 39.0, 2.8, 8.0, INPUT_FC, INPUT_EC, lw=1.0, r=0.3, z=3)
    txt(ax, fxs[0], 36.0, r"$s_{t-t_{in}+1}$", fs=12.5)
    txt(ax, fxs[-1], 36.0, r"$s_t$", fs=12.5)
    wire(ax, [(91, 34.2), (91, 29.5)])
    block(ax, 78, 17.5, 26, 12, "Encoder GRU\n" r"$h\in\mathbb{R}^{d_h}$", GRU_FC, GRU_EC,
          fs=15, tc="white")

    # h bus: one trunk, a T-junction, one arrowhead per consumer (no fork stubs)
    cy, ch = 31.3, 7.5                           # decoder row
    dmid = cy + ch / 2
    smid = 17.0                                  # Seq2Seq heads' mid-height
    wire(ax, [(104, 23.5), (111, 23.5)], head=False)
    wire(ax, [(111, smid), (111, dmid)], head=False)
    txt(ax, 106.0, 25.8, "$h$", fs=15)

    # ---------------------------------------------------------------------- Forecast --
    rbox(ax, 114, PY0, 66, PY1 - PY0, FCAST_FC, FCAST_EC, lw=1.1, ls=DASH, r=2.0, z=1.2)
    chip(ax, 147, CHIP_Y, "Forecast")
    ax.plot([115, 179], [28.4, 28.4], color=SALMON, lw=1.0, ls=(0, (3, 3)), zorder=1.5)

    # top row -- ProbGRU, autoregressive
    txt(ax, 116, 57.6, "ProbGRU — autoregressive", fs=13.5, weight="bold", ha="left",
        color=TEAL)
    cxs, cw = [125.5, 142.5, 159.5], 12.0
    hw, hh, hy = 12.0, 7.3, 42.0
    wire(ax, [(111, dmid), (cxs[0] - cw / 2, dmid)])
    for k, cx in enumerate(cxs):
        block(ax, cx - cw / 2, cy, cw, ch, "Decoder\nGRU", GRU_FC, GRU_EC, fs=12.5, tc="white")
        block(ax, cx - hw / 2, hy, hw, hh, "Linear\n" r"$\mu,\ \log\sigma^2$", HEAD_FC, HEAD_EC,
              fs=11.5)
        wire(ax, [(cx, cy + ch), (cx, hy)])
        wire(ax, [(cx, hy + hh), (cx, hy + hh + 1.9)])
        txt(ax, cx, hy + hh + 4.0, fr"$\hat{{s}}_{{t+{k + 1}}}$", fs=14)
        if k < len(cxs) - 1:
            wire(ax, [(cx + cw / 2, dmid), (cxs[k + 1] - cw / 2, dmid)])
            xd = (cx + cw / 2 + cxs[k + 1] - cw / 2) / 2
            wire(ax, [(cx + hw / 2, hy + hh / 2), (xd, hy + hh / 2), (xd, 29.7),
                      (cxs[k + 1] - 3.0, 29.7), (cxs[k + 1] - 3.0, cy)],
                 color=FEED, ls=(0, (3, 2)))
    txt(ax, 171.0, 38.2, r"$\cdots$", fs=15)
    wire(ax, [(cxs[-1] + cw / 2, dmid), (182, dmid)], head=False)

    # bottom row -- Seq2Seq, one-shot: the two heads ARE its decoder (models.py:78-80)
    txt(ax, 116, 25.6, "Seq2Seq — one-shot", fs=13.5, weight="bold", ha="left", color=TEAL)
    gx, gw = 124.0, 24.0
    rbox(ax, gx, 10.3, gw, 13.2, "none", TEAL, lw=1.0, ls=(0, (3, 2)), r=1.0, z=2.8)
    block(ax, gx + 1.3, 17.6, gw - 2.6, 5.0, r"Linear,  $\mu$", HEAD_FC, HEAD_EC, fs=12.5)
    block(ax, gx + 1.3, 11.4, gw - 2.6, 5.0, r"Linear,  $\log\sigma^2$", HEAD_FC, HEAD_EC,
          fs=12.5)
    txt(ax, gx + gw / 2, 7.6, "one-shot decoder", fs=12, color="#444444")
    wire(ax, [(111, smid), (gx, smid)])
    wire(ax, [(gx + gw, smid), (182, smid)], head=False)

    # ------------------------------------------------------------------ shared output --
    omid = 26.0
    wire(ax, [(182, smid), (182, dmid)], head=False)
    wire(ax, [(182, omid), (186.2, omid)])
    rbox(ax, 186.2, omid - 6, 12, 12, PEACH, SALMON, lw=1.3, ls=(0, (3, 2)), r=0.01, z=3)
    txt(ax, 192.2, omid, r"$\hat{s}_{t+n}$", fs=17)

    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=os.path.join(ROOT, "figures", "model_architectures"),
                    help="output path WITHOUT extension; .png and .pdf are both written")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--no-pdf", action="store_true")
    a = ap.parse_args()

    stem = a.out[:-4] if a.out.lower().endswith((".png", ".pdf")) else a.out
    os.makedirs(os.path.dirname(os.path.abspath(stem)), exist_ok=True)

    fig = build()
    fig.savefig(stem + ".png", dpi=a.dpi, facecolor="white")
    print(f"wrote {stem}.png")
    if not a.no_pdf:
        fig.savefig(stem + ".pdf", facecolor="white", dpi=a.dpi)
        print(f"wrote {stem}.pdf")
    print("NOTE: the glove photo is a third-party image -- the paper must credit its source.")


if __name__ == "__main__":
    main()
