"""Schematic of the v2 probGRU forecasting model (paper Fig.-style, two panels).

Draws the model that actually lives in src/actionsense/action_dynamics.py -- NOT a generic
seq2seq cartoon. Every structural claim on the canvas was read off the code:

  (a) CAUSAL FEATURE CONSTRUCTION
      tactile frames -> physical_state (offline moments, NOT learned) -> s_t=[F,xbar,ybar]
      -> downsample x3 (30->10 Hz) -> causal Butterworth LP 0.4 Hz (`sosfilt`, forward-only)
      -> slow/fast split -> x_t in R^6 (highpass mode) and target y_t in R^3.
      Leading `warmup_sec`=5 s dropped (filter transient).  [action_dynamics.py:39-78]
      action label -> nn.Embedding(n_act, 8) -> e_a.                [action_dynamics.py:145]

  (b) PROBABILISTIC ROLLOUT
      Encoder GRU(din=6, hid=48) over the t_in history -> h.        [action_dynamics.py:146,152]
      Decoder GRU(3, hid) seeded with y_last = last OBSERVED fast target, then rolled H steps
      AUTOREGRESSIVELY: the mean is fed back as the next input (`inp = mu.unsqueeze(1)`).
                                                                    [action_dynamics.py:147,154,161]
      e_a is concatenated at the HEAD, not at the GRU input (`torch.cat([o[:,-1], e], -1)`),
      so it is drawn as a small bar entering each head box.          [action_dynamics.py:158]
      Two linear heads -> mu (3) and log-variance lv (3), lv.clamp(-6,4).  [action_dynamics.py:148-159]
      Bottom strip = the protocol: Gaussian NLL -> early-stop on VAL NLL -> post-hoc sigma
      scaling to coverage@2sigma = 0.95 -> 5-fold CV by clip.        [action_dynamics.py:176-234]

THINGS DELIBERATELY NOT DRAWN (drawing them would misrepresent the model):
  * tactile maps feeding a learned encoder -- the probGRU consumes the 6-dim state, not the map;
  * a one-shot head -- that is tactile_map/models.py and opentouch/gru_aggregate.py, not this model;
  * persistence as part of the model -- it is the baseline, so it appears only in the output plot.

The two inset curves are ILLUSTRATIVE synthetic signals (fixed seed), drawn to show the shapes
the model produces -- amplitude shrinkage toward the mean and a horizon-growing sigma band, both
of which are real properties of this NLL-trained model (docs/PROJECT_CONCLUSIONS.md 5.4-5.5).
They are not measured data and the figure says so on the canvas.

Usage:  python scripts/actionsense/plot_model_diagram.py --out docs/model_diagram.png [--pdf]
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

# ------------------------------------------------------------------ canvas + palette --
W, H = 130.0, 56.0                      # drawing units
Y0 = 3.0                                # bottom crop (the training strip used to live below)
DIV = 36.5                              # dashed divider between panels (a) and (b)
NOW = 61.5                              # the forecast origin t -- the figure's key vertical

C_OBS = "#a8cbe8"                       # observed / inputs
C_ENC = "#2d5f8a"                       # encoder (dark blue, like the reference figure)
C_DEC = "#5aa06e"                       # dynamics / decoder (green)
C_HEAD = "#dcdcdc"                      # output heads
C_PRE = "#f0f0f0"                       # offline preprocessing (not learned)
C_UNC = "#e08b2f"                       # uncertainty
C_BASE = "#909090"                      # persistence baseline
C_TXT = "#1a1a1a"

FS_BOX, FS_NOTE, FS_CAP, FS_TOP = 8.5, 7.0, 11.0, 8.0


# ------------------------------------------------------------------------- primitives --
def box(ax, x, y, w, h, label, fc, ec="#333333", fs=FS_BOX, tc=None, bold=False, round_=0.6):
    """Rounded box centred text; (x,y) is the lower-left corner."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={round_}",
                                fc=fc, ec=ec, lw=1.0, zorder=2))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fs,
            color=tc or C_TXT, zorder=3, weight="bold" if bold else "normal",
            linespacing=1.35)


def arrow(ax, p0, p1, style="-|>", color="#333333", lw=1.1, rad=0.0, ls="-", zorder=3):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=11, lw=lw,
                                 color=color, linestyle=ls, zorder=zorder,
                                 connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=0, shrinkB=0))


def note(ax, x, y, s, fs=FS_NOTE, ha="center", color="#444444", style="normal", va="center"):
    ax.text(x, y, s, ha=ha, va=va, fontsize=fs, color=color, style=style, zorder=4,
            linespacing=1.3)


def inset(fig, x, y, w, h):
    """Axes placed in drawing units (must match the main axes' cropped y-range)."""
    span = H - Y0
    return fig.add_axes([x / W, (y - Y0) / span, w / W, h / span])


def feedback(ax, xs, ys, xd, x1, y1, ybus, color=C_UNC, lw=1.2):
    """Autoregressive feedback as an explicit out-down-across-up path (not a free arc), routed
    through the gap between two decoder cells and down onto the annotation row, so the label
    below reads as its caption instead of the arrow floating between boxes.

    (xs, ys) leaves the head; xd is the drop column; the path lands on cell k+1 at (x1, y1)."""
    ls = (0, (4, 2.5))
    seg = ([xs, xd], [ys, ys]), ([xd, xd], [ys, ybus]), ([xd, x1], [ybus, ybus])
    for xx, yy in seg:
        ax.plot(xx, yy, ls=ls, lw=lw, color=color, zorder=3, solid_capstyle="butt")
    arrow(ax, (x1, ybus), (x1, y1), ls=ls, color=color, lw=lw)


# ------------------------------------------------------------------ illustrative signal --
def demo_signal(n, seed=3):
    """Smooth zero-mean 'fast component'-looking trace (schematic only)."""
    rng = np.random.default_rng(seed)
    k = np.hanning(13)
    k /= k.sum()
    s = np.convolve(rng.standard_normal(n + 24), k, mode="same")[12:12 + n]
    return s / (np.abs(s).max() + 1e-9)


# ------------------------------------------------------------------------- panel (a) --
def panel_a(ax):
    cx = 18.0

    # tactile frames: three offset squares (schematic pressure maps)
    for i, off in enumerate((0, 1.1, 2.2)):
        ax.add_patch(Rectangle((cx - 5.2 + off, 49.0 - off * 0.55), 5.4, 5.0,
                               fc="#3b2f63" if i == 2 else "#5b4f82", ec="white", lw=0.8,
                               zorder=2 + i))
    note(ax, cx + 6.6, 51.5, "tactile frames\n(T, 16, 16)", ha="left")

    arrow(ax, (cx, 48.6), (cx, 45.6))
    note(ax, cx + 1.0, 47.1, "physical_state  (offline moments, not learned)", ha="left", fs=6.5)

    box(ax, cx - 13.5, 41.0, 27, 4.6, r"$s_t=[\,F,\ \bar{x},\ \bar{y}\,]$   @ 30 Hz", C_PRE)
    arrow(ax, (cx, 41.0), (cx, 35.8))
    note(ax, cx + 1.0, 38.4, r"downsample $\times 3\ \rightarrow$ 10 Hz", ha="left")

    box(ax, cx - 16.5, 29.6, 33, 6.0,
        r"$\mathbf{x}_t=[\,F,\ x,\ y,\ v_x,\ v_y\,]\in\mathbb{R}^{5}$   (encoder input)"
        "\n" r"target  $\mathbf{y}_t=[F_f,\ x_f,\ y_f]\in\mathbb{R}^{3}$", C_OBS, fs=7.8)
    note(ax, cx, 27.4,
         "target is the FAST component: causal low-pass 0.4 Hz ($\\it{sosfilt}$), "
         "$F_f = F - \\mathrm{LP}(F)$", fs=6.4)
    note(ax, cx, 25.4, "drop leading 5 s (filter transient);  $v$ = backward difference", fs=6.4)

    # action embedding -- the small side input, deliberately NOT its own panel
    box(ax, cx - 16.5, 11.0, 20.5, 4.4, "action label\n(pour / slice / peel / …)", "#ffffff",
        fs=7.2)
    arrow(ax, (cx + 4.0, 13.2), (cx + 8.2, 13.2))
    box(ax, cx + 8.4, 11.0, 8.2, 4.4, "Emb\n$\\mathbf{e}_a\\in\\mathbb{R}^{8}$", C_ENC,
        tc="white", fs=7.2)

    ax.text(cx, 6.5, "(a) Causal feature construction", ha="center", va="center",
            fontsize=FS_CAP, color=C_TXT)


# ------------------------------------------------------------------------- panel (b) --
def panel_b(fig, ax):
    # ---- the forecast origin: the single most important line in the figure
    ax.plot([NOW, NOW], [12.5, 50.5], ls=(0, (5, 3)), lw=1.6, color="#222222", zorder=5)
    note(ax, NOW, 52.0, "forecast origin  $t$", fs=7.6, color="#222222")

    # ---- top scale bars
    arrow(ax, (40.5, 49.6), (60.8, 49.6), style="<|-|>", lw=0.9, color="#555555")
    note(ax, 50.6, 51.0, "observed history  $t_{in}$ (1–10 s; best 3 s)", fs=FS_TOP)
    arrow(ax, (62.4, 49.6), (106.5, 49.6), style="<|-|>", lw=0.9, color="#555555")
    note(ax, 84.4, 51.0, "forecast horizon  $H = 10$ steps (1 s)", fs=FS_TOP)

    # ================================================== B1: history -> encoder
    # per-frame feature vectors as small column bars
    xs = np.linspace(41.5, 59.0, 6)
    for i, xb in enumerate(xs):
        ax.add_patch(Rectangle((xb - 1.15, 41.4), 2.3, 5.2, fc=C_OBS, ec="#31688e", lw=0.8,
                               zorder=3))
    note(ax, 41.5, 39.6, r"$\mathbf{x}_{t-t_{in}+1}$", fs=7.0)
    note(ax, 50.2, 39.6, r"$\cdots$", fs=7.0)
    note(ax, 59.0, 39.6, r"$\mathbf{x}_{t}$", fs=7.0)
    note(ax, 50.2, 48.0, r"per-frame features  $\in\mathbb{R}^{5}$", fs=6.8)

    arrow(ax, (50.2, 38.4), (50.2, 34.3), lw=1.2)
    box(ax, 40.5, 28.4, 20.0, 5.6, "Encoder GRU\n$\\dim(h) = 48$", C_ENC, tc="white", fs=8.4,
        bold=True)

    # observed target trace (source of the seed and of the ground truth)
    axh = inset(fig, 40.5, 15.4, 20.0, 8.6)
    t = np.arange(60)
    sig = demo_signal(120)
    for j, (c, a) in enumerate((("#31688e", 1.0), ("#7fb3d5", 0.85), ("#b8d4e8", 0.7))):
        axh.plot(t, np.roll(sig, 7 * j)[:60] * (1 - 0.18 * j), color=c, lw=1.1, alpha=a)
    axh.set_xlim(0, 59)
    axh.axhline(0, color="#bbbbbb", lw=0.6)
    for s in axh.spines.values():
        s.set_visible(False)
    axh.set_xticks([])
    axh.set_yticks([])
    axh.patch.set_alpha(0)
    note(ax, 50.5, 24.8, r"observed fast target  $\mathbf{y}_{t-t_{in}+1:t}$", fs=6.8)

    # ================================================== B2: autoregressive decoder
    cxs = [69.0, 83.0, 97.0]
    cw, ch, cy = 12.0, 6.0, 26.4                       # decoder cell
    hw, hh, hy = 8.8, 5.4, 36.6                        # head box
    arrow(ax, (60.5, 29.4), (63.0 - 0.2, 29.4), lw=1.3)
    note(ax, 58.6, 35.4, "$h$", fs=7.8, color="#222222")

    for k, cx in enumerate(cxs):
        box(ax, cx - cw / 2, cy, cw, ch, "Decoder\nGRU", C_DEC, tc="white", fs=8.2, bold=True)
        # e_a bar + head (concat happens AT THE HEAD -- action_dynamics.py:158)
        ax.add_patch(Rectangle((cx - cw / 2 + 0.2, hy + 0.9), 1.9, 3.6, fc=C_ENC, ec="#1b3d5a",
                               lw=0.8, zorder=3))
        note(ax, cx - cw / 2 + 1.15, hy - 0.9, "$\\mathbf{e}_a$", fs=6.6, color=C_ENC)
        box(ax, cx - cw / 2 + 2.6, hy, hw, hh,
            "Linear heads\n$\\mu$,  $\\log\\sigma^{2}$", C_HEAD, fs=7.2)
        arrow(ax, (cx - cw / 2 + 2.15, hy + 2.7), (cx - cw / 2 + 2.5, hy + 2.7), lw=0.9)
        arrow(ax, (cx + 1.4, cy + ch), (cx + 1.4, hy), lw=1.1)
        arrow(ax, (cx + 1.4, hy + hh), (cx + 1.4, hy + hh + 2.4), lw=1.1)
        note(ax, cx + 1.4, hy + hh + 3.9,
             fr"$\mu_{{t+{k+1}}},\ \sigma_{{t+{k+1}}}$", fs=7.2, color="#222222")
        if k < len(cxs) - 1:
            arrow(ax, (cx + cw / 2, cy + ch / 2), (cxs[k + 1] - cw / 2, cy + ch / 2), lw=1.3)
            # autoregressive feedback: the MEAN becomes the next decoder input.
            # Routed down onto the annotation row so the label below reads as its caption.
            feedback(ax, cx - cw / 2 + 2.6 + hw, hy + hh / 2, (cx + cxs[k + 1]) / 2,
                     cxs[k + 1] - 2.6, cy, ybus=22.4)

    note(ax, 90.0, 19.8, r"autoregressive: $\hat{\mu}$ fed back as the next decoder input",
         fs=7.0, color=C_UNC)

    # seed = last observed target frame
    arrow(ax, (60.8, 19.0), (67.4, cy - 0.4), rad=-0.18, lw=1.2)
    note(ax, 63.0, 15.2, r"seed  $\hat{y}_t=\mathbf{y}_t$", fs=6.8, ha="left")

    arrow(ax, (103.2, 29.4), (107.6, 29.4), lw=1.2)
    note(ax, 105.4, 31.4, r"$\cdots$", fs=9.0)

    # ================================================== B3: output
    axo = inset(fig, 109.0, 16.6, 18.5, 15.0)
    hh_ = np.arange(1, 11)
    true = demo_signal(120, seed=3)[60:70]
    mu = true * np.linspace(0.85, 0.35, 10)            # NLL shrinkage toward the mean (real effect)
    sd = np.linspace(0.22, 0.62, 10)
    axo.fill_between(hh_, mu - 2 * sd, mu + 2 * sd, color=C_UNC, alpha=0.25, lw=0,
                     label=r"$\pm 2\sigma$")
    axo.plot(hh_, mu, color=C_UNC, lw=1.6, label=r"$\mu$")
    axo.plot(hh_, true, color="#222222", lw=1.3, label="ground truth")
    axo.axhline(demo_signal(120, seed=3)[59], color=C_BASE, lw=1.2, ls=(0, (4, 2.5)),
                label="persistence")
    axo.set_xlim(1, 10)
    for s in axo.spines.values():
        s.set_visible(False)
    axo.set_xticks([])
    axo.set_yticks([])
    axo.patch.set_alpha(0)
    axo.legend(fontsize=5.6, loc="lower left", frameon=False, handlelength=1.4,
               borderpad=0.1, labelspacing=0.25, bbox_to_anchor=(-0.03, -0.06))
    note(ax, 118.2, 33.4, r"forecast  $\hat{\mathbf{y}}_{t+1:t+H}\sim\mathcal{N}(\mu,\sigma^{2})$",
         fs=7.2, color="#222222")
    note(ax, 118.2, 14.6, "skill $=1-\\mathrm{MSE}/\\mathrm{MSE}_{pers}$\n"
                          "(report $R^2$ vs mean)", fs=6.4)

    ax.text(84.0, 6.5, "(b) Probabilistic rollout  (probGRU)", ha="center", va="center",
            fontsize=FS_CAP, color=C_TXT)


# ------------------------------------------------------------------------------- main --
def build(width=13.0):
    fig = plt.figure(figsize=(width, width * (H - Y0) / W))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(Y0, H)
    ax.axis("off")

    panel_a(ax)
    panel_b(fig, ax)

    # divider, and the two arrows that carry (a) into (b)
    ax.plot([DIV, DIV], [9.0, 51.5], ls=(0, (4, 4)), lw=1.0, color="#999999", zorder=1)
    arrow(ax, (34.7, 34.0), (41.0, 41.0), rad=-0.24, lw=1.1, color="#31688e")
    note(ax, 35.6, 40.6, r"$\mathbf{x}$", fs=7.6, color="#31688e", ha="left")
    arrow(ax, (34.7, 31.0), (40.4, 21.4), rad=0.24, lw=1.1, color="#31688e")
    note(ax, 37.0, 25.6, r"$\mathbf{y}$", fs=7.6, color="#31688e", ha="left")

    note(ax, 1.0, 54.6, "probGRU forecasting model  —  src/actionsense/action_dynamics.py",
         fs=7.4, ha="left", color="#666666")
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="docs/model_diagram.png")
    ap.add_argument("--dpi", type=int, default=220)
    ap.add_argument("--pdf", action="store_true", help="also write a vector .pdf alongside")
    a = ap.parse_args()

    fig = build()
    fig.savefig(a.out, dpi=a.dpi, facecolor="white")
    print(f"wrote {a.out}")
    print("NOTE: the two inset curves are SYNTHETIC illustrations, not measured results. The "
          "on-canvas disclaimer was removed by request (2026-08-17), so the paper caption must "
          "carry it.")
    if a.pdf:
        p = a.out.rsplit(".", 1)[0] + ".pdf"
        fig.savefig(p, facecolor="white")
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
