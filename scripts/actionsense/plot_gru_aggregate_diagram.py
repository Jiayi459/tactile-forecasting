"""Schematic of the GRU-aggregate forecaster (companion to plot_model_diagram.py, same style).

Draws the model that lives in src/actionsense/tactile_map/ with encoder="aggregate" -- the
ActionSense implementation the user selected (2026-08-25) over the two OpenTouch arms. Every
structural claim on the canvas was read off the code:

  (a) TARGET CONSTRUCTION & WINDOWING
      tactile frames -> physical_state (offline moments, NOT learned) -> the harness's RAW
      6-dim target [F, CoPx, CoPy] x {L,R}                    [eval_harness.yaml target.channels]
      -> downsample x3 (30 -> 10 Hz)                          [eval_harness.yaml rate]
      -> TRAIN-only per-channel z-score, Norm.z                [eval_harness/dataset.py:58]
      input  x = z[t-t_in+1 : t]  (left-zero-padded when the origin is early) and
      target y = z[t+1 : t+1+H] - z[t], the RESIDUAL OVER PERSISTENCE.  [data.py:174-190]
      origins = arange(min_history=40, T-H, stride=1).         [eval_harness/baselines/base.py]
      The input and the target are the SAME signal -- this arm is the neural AR.  [models.py:45]

  (b) ONE-SHOT PROBABILISTIC FORECAST
      AggEncoder: per-frame Linear(6 -> d=64) + ReLU.          [models.py:44-53]
      Encoder GRU(d -> hidden=64); only the LAST hidden state is used.   [models.py:65,72]
      TWO one-shot heads off that single state: Linear(hidden -> H*C) for mu and for logvar,
      logvar clamped to [-6,4], both reshaped to (B,H,C).      [models.py:66-76]
      The output IS the residual, and it is scored there: `_predict` returns residual-space
      values and `evaluate` uses persistence == 0.             [train.py:131,152-156]
      NOTE (verified 2026-08-25, corrects a first draft of this figure): this arm has NO
      anchor-and-de-normalize step. `z_hat = z[t] + resid` followed by `unz` exists only in
      the OpenTouch fork's predict_clip [gru_aggregate.py:225-234]; drawing it here would
      assert a code path that does not exist in src/actionsense/tactile_map/.

THE ONE THING THIS FIGURE MUST NOT LOOK LIKE: the probGRU figure. This model has NO decoder,
NO autoregressive feedback and NO seed frame -- the whole horizon comes out of a single forward
pass. The annotation sits at the same y as the probGRU figure's "autoregressive" caption, so
the two figures can be read side by side and the contrast lands.

The inset curves are ILLUSTRATIVE synthetic signals (fixed seed), not measured data; the paper
caption must say so (the on-canvas disclaimer was dropped by request on 2026-08-17).

Usage:  python scripts/actionsense/plot_gru_aggregate_diagram.py \
            --out docs/gru_aggregate_diagram.png [--pdf]
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from plot_model_diagram import (  # noqa: E402  (shared primitives -- one style, one copy)
    C_BASE, C_ENC, C_HEAD, C_OBS, C_PRE, C_TXT, C_UNC,
    FS_CAP, FS_TOP, W, H, Y0, DIV,
    arrow, box, demo_signal, note,
)

NOW = 61.5                              # forecast origin t (same column as the probGRU figure)
C_ADD = "#5aa06e"                       # the anchor/de-normalize step (green, as in the probGRU fig)


def inset(fig, x, y, w, h):
    span = H - Y0
    return fig.add_axes([x / W, (y - Y0) / span, w / W, h / span])


def grid(ax, x, y, cols, rows, cw, ch, fc, ec="#31688e"):
    """A (rows x cols) matrix of cells -- the reshaped (H, C) head output."""
    for r in range(rows):
        for c in range(cols):
            ax.add_patch(Rectangle((x + c * cw, y + r * ch), cw, ch, fc=fc, ec=ec, lw=0.35,
                                   zorder=3))


# ------------------------------------------------------------------------- panel (a) --
def panel_a(ax):
    cx = 18.0

    for i, off in enumerate((0, 1.1, 2.2)):
        ax.add_patch(Rectangle((cx - 5.2 + off, 49.0 - off * 0.55), 5.4, 5.0,
                               fc="#3b2f63" if i == 2 else "#5b4f82", ec="white", lw=0.8,
                               zorder=2 + i))
    note(ax, cx + 6.6, 51.5, "tactile frames\n(T, 2, 32, 32)", ha="left")

    arrow(ax, (cx, 48.6), (cx, 45.6))
    note(ax, cx + 1.0, 47.1, "physical_state  (offline moments, not learned)", ha="left", fs=6.5)

    box(ax, cx - 15.5, 41.0, 31, 4.6,
        r"$Y_t=[\,F,\ CoP_x,\ CoP_y\,]\times\{L,R\}\in\mathbb{R}^{6}$" "\n@ 30 Hz", C_PRE, fs=7.6)
    arrow(ax, (cx, 41.0), (cx, 37.4))
    note(ax, cx + 1.0, 39.2, r"downsample $\times 3\ \rightarrow$ 10 Hz", ha="left")

    box(ax, cx - 15.5, 32.6, 31, 4.6,
        r"TRAIN-only z-score:   $z_t=(Y_t-\mu)/\sigma$", C_PRE, fs=7.6)
    arrow(ax, (cx, 32.6), (cx, 29.2))

    box(ax, cx - 16.5, 23.0, 33, 6.2,
        r"input   $\mathbf{x}=z_{t-t_{in}+1\,:\,t}$" "\n"
        r"target  $\mathbf{y}=z_{t+1\,:\,t+H}-z_t$   (residual over persistence)",
        C_OBS, fs=7.6)

    note(ax, cx, 20.8, r"input and target are the same signal $\rightarrow$ neural AR", fs=6.4)
    note(ax, cx, 18.4, r"predicting $\mathbf{0}$ reproduces persistence exactly", fs=6.4)
    note(ax, cx, 16.0, r"origins $=\mathrm{arange}(40,\ T-H,\ 1)$;  left-zero-padded", fs=6.4)
    note(ax, cx, 13.6, "raw harness target: no high-pass, no warm-up cut", fs=6.4)

    ax.text(cx, 6.5, "(a) Target construction & windowing", ha="center", va="center",
            fontsize=FS_CAP, color=C_TXT)


# ------------------------------------------------------------------------- panel (b) --
def panel_b(fig, ax):
    ax.plot([NOW, NOW], [12.0, 50.5], ls=(0, (5, 3)), lw=1.6, color="#222222", zorder=5)
    note(ax, NOW, 52.0, "forecast origin  $t$", fs=7.6, color="#222222")

    arrow(ax, (40.5, 49.6), (60.8, 49.6), style="<|-|>", lw=0.9, color="#555555")
    note(ax, 50.6, 51.0, "observed history  $t_{in}$ (1 / 3 / 10 s)", fs=FS_TOP)
    arrow(ax, (62.4, 49.6), (100.0, 49.6), style="<|-|>", lw=0.9, color="#555555")
    note(ax, 81.2, 51.0, "forecast horizon  $H = 10$ steps (1 s)", fs=FS_TOP)

    # ================================================== B1: history -> encoder -> GRU
    xs = np.linspace(41.5, 59.0, 6)
    for xb in xs:
        ax.add_patch(Rectangle((xb - 1.15, 41.6), 2.3, 5.0, fc=C_OBS, ec="#31688e", lw=0.8,
                               zorder=3))
    note(ax, 41.5, 39.9, r"$\mathbf{x}_{t-t_{in}+1}$", fs=7.0)
    note(ax, 50.2, 39.9, r"$\cdots$", fs=7.0)
    note(ax, 59.0, 39.9, r"$\mathbf{x}_{t}$", fs=7.0)
    note(ax, 50.2, 47.9, r"per-frame aggregate F/CoP  $\in\mathbb{R}^{6}$", fs=6.8)

    arrow(ax, (50.2, 41.4), (50.2, 38.4), lw=1.2)
    box(ax, 40.5, 33.6, 20.0, 4.6, "AggEncoder (per frame)\n"
        r"Linear$(6\rightarrow d\!=\!64)$ + ReLU", C_OBS, fs=7.2)
    arrow(ax, (50.2, 33.6), (50.2, 31.0), lw=1.2)
    box(ax, 40.5, 25.0, 20.0, 6.0, "Encoder GRU\n$\\dim(h) = 64$", C_ENC, tc="white", fs=8.4,
        bold=True)
    note(ax, 50.5, 23.4, "last hidden state only  (no decoder)", fs=6.4)

    # observed aggregate trace: source of the persistence anchor z_t
    axh = inset(fig, 40.5, 14.0, 20.0, 7.6)
    sig = demo_signal(120)
    for j, (c, a) in enumerate((("#31688e", 1.0), ("#7fb3d5", 0.85), ("#b8d4e8", 0.7))):
        axh.plot(np.arange(60), np.roll(sig, 7 * j)[:60] * (1 - 0.18 * j), color=c, lw=1.1,
                 alpha=a)
    axh.set_xlim(0, 59)
    axh.axhline(0, color="#bbbbbb", lw=0.6)
    for s in axh.spines.values():
        s.set_visible(False)
    axh.set_xticks([])
    axh.set_yticks([])
    axh.patch.set_alpha(0)

    # ================================================== B2: one-shot heads
    arrow(ax, (60.5, 28.0), (63.4, 34.6), rad=-0.18, lw=1.3)
    arrow(ax, (60.5, 28.0), (63.4, 26.6), rad=0.14, lw=1.3)
    note(ax, 63.0, 30.6, "$h$", fs=7.8, color="#222222")

    box(ax, 63.6, 33.0, 19.4, 5.2,
        r"Linear$(64\rightarrow H\!\cdot\!C\!=\!60)$" "\n" r"$\mu$", C_HEAD, fs=7.2)
    box(ax, 63.6, 24.0, 19.4, 5.2,
        r"Linear$(64\rightarrow H\!\cdot\!C\!=\!60)$" "\n"
        r"$\log\sigma^{2}$,  clamp$[-6,4]$", C_HEAD, fs=7.2)

    arrow(ax, (83.2, 35.6), (85.4, 35.6), lw=1.0)
    arrow(ax, (83.2, 26.6), (85.4, 26.6), lw=1.0)
    grid(ax, 85.6, 33.9, 10, 6, 0.85, 0.6, "#f3c8a0")
    grid(ax, 85.6, 24.9, 10, 6, 0.85, 0.6, "#d7e5f0")
    note(ax, 90.0, 39.0, r"$\mu$  reshaped $(H,C)$", fs=6.4)
    note(ax, 90.0, 23.5, r"$\log\sigma^{2}$  reshaped $(H,C)$", fs=6.4)

    note(ax, 86.0, 19.8,
         "one-shot: the entire horizon in one forward pass — no decoder, no feedback",
         fs=7.0, color=C_UNC)

    # ================================================== B3: anchor + de-normalize -> output
    box(ax, 96.5, 27.4, 22.0, 5.4,
        r"residual forecast  $\hat{r}_{t+1:t+H}$" "\n" r"$(H\times C)$, normalized units",
        C_ADD, tc="white", fs=7.4)
    arrow(ax, (94.4, 35.6), (96.9, 32.2), rad=-0.14, lw=1.1)
    arrow(ax, (94.4, 27.6), (96.5, 29.2), rad=0.14, lw=1.1)

    note(ax, 92.0, 13.2,
         "trained and scored on the residual — the absolute-unit reconstruction "
         r"($\hat{z}=z_t+\hat{r}$, unz) lives only in the OpenTouch fork", fs=6.2)

    arrow(ax, (110.0, 32.8), (110.0, 35.4), lw=1.1)

    axo = inset(fig, 102.0, 35.6, 22.0, 10.8)
    hh_ = np.arange(1, 11)
    ref = demo_signal(120, seed=3)
    true = ref[60:70] - ref[59]                       # the RESIDUAL target this arm predicts
    mu = true * np.linspace(0.85, 0.35, 10)           # NLL shrinkage toward 0 (a real effect)
    sd = np.linspace(0.16, 0.52, 10)
    axo.fill_between(hh_, mu - 2 * sd, mu + 2 * sd, color=C_UNC, alpha=0.25, lw=0,
                     label=r"$\pm 2\sigma$")
    axo.plot(hh_, mu, color=C_UNC, lw=1.6, label=r"$\mu$")
    axo.plot(hh_, true, color="#222222", lw=1.3, label="ground truth")
    axo.axhline(0.0, color=C_BASE, lw=1.2, ls=(0, (4, 2.5)), label=r"persistence $\equiv 0$")
    axo.set_xlim(1, 10)
    for s in axo.spines.values():
        s.set_visible(False)
    axo.set_xticks([])
    axo.set_yticks([])
    axo.patch.set_alpha(0)
    axo.legend(fontsize=5.6, loc="lower left", frameon=False, handlelength=1.4,
               borderpad=0.1, labelspacing=0.25, bbox_to_anchor=(-0.03, -0.06))
    note(ax, 113.0, 48.2,
         r"residual forecast  $\hat{r}_{t+h}\sim\mathcal{N}(\mu,\sigma^{2})$",
         fs=7.2, color="#222222")
    note(ax, 113.5, 21.4, "Gaussian NLL;  skill $=1-\\mathrm{MSE}/\\mathrm{MSE}_{pers}$\n"
                          "persistence $\\equiv\\mathbf{0}$, so negative skill is visible", fs=6.4)

    ax.text(84.0, 6.5, "(b) One-shot probabilistic forecast  (GRU-aggregate)", ha="center",
            va="center", fontsize=FS_CAP, color=C_TXT)



# ------------------------------------------------------------------------------- main --
def build(width=13.0):
    fig = plt.figure(figsize=(width, width * (H - Y0) / W))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(Y0, H)
    ax.axis("off")

    panel_a(ax)
    panel_b(fig, ax)

    ax.plot([DIV, DIV], [9.0, 51.5], ls=(0, (4, 4)), lw=1.0, color="#999999", zorder=1)
    arrow(ax, (34.7, 27.4), (41.0, 43.0), rad=-0.22, lw=1.1, color="#31688e")
    note(ax, 35.4, 40.0, r"$\mathbf{x}$", fs=7.6, color="#31688e", ha="left")
    arrow(ax, (34.7, 24.6), (40.4, 18.6), rad=0.22, lw=1.1, color="#31688e")
    note(ax, 37.0, 20.0, r"$z$", fs=7.6, color="#31688e", ha="left")

    note(ax, 1.0, 54.6,
         "GRU-aggregate forecasting model  —  src/actionsense/tactile_map/  (encoder='aggregate')",
         fs=7.4, ha="left", color="#666666")
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="docs/gru_aggregate_diagram.png")
    ap.add_argument("--dpi", type=int, default=220)
    ap.add_argument("--pdf", action="store_true", help="also write a vector .pdf alongside")
    a = ap.parse_args()

    fig = build()
    fig.savefig(a.out, dpi=a.dpi, facecolor="white")
    print(f"wrote {a.out}")
    print("NOTE: the inset curves are SYNTHETIC illustrations, not measured results — the paper "
          "caption must carry that, as for docs/model_diagram.png.")
    if a.pdf:
        p = a.out.rsplit(".", 1)[0] + ".pdf"
        fig.savefig(p, facecolor="white")
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
