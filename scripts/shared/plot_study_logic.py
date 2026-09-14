"""Paper Fig. 1 (`fig:overview`) -- the forecasting task and the skill ratio, in one plot.

ONE axes, one force trace. Left of the origin is the input, right of it is what the model
predicts. In the predicted second the ground truth, the forecast and the copy-last baseline
are drawn together, and at every horizon step the two errors that form the skill ratio are
drawn side by side: truth-to-persistence (the term in the denominator) and truth-to-forecast
(the term in the numerator). One step's pair is labelled, and the formula on the canvas is
written in those symbols, so the reader can see which segment is divided by which.

NO NUMBERS ON THE CANVAS, deliberately (2026-09-13 ruling). Force is in arbitrary units and
carries no tick labels; no MSE, ratio or skill value is printed. A single origin's skill is
not the reported quantity anyway -- the reported S pools every origin and weights recordings
equally -- so printing one would invite exactly the misreading the caption has to prevent.

Data: `runs/as_preds_seq2seq_corpus_h1`, the 1 s-history Seq2Seq run, so the history drawn
here is the history the plotted forecast was actually made from. Total force only.

Usage:  python scripts/shared/plot_study_logic.py [--out figures/overview] [--no-pdf]
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PREDS = os.path.join(ROOT, "runs", "as_preds_seq2seq_corpus_h1")   # the 1 s-history run
ARM = "mu_seq2seq_aggregate"
CH = 3                       # F_R -- total force, right hand
FPS = 10.0
H = 10                       # 1 s horizon at 10 Hz
L = 10                       # 1 s history at 10 Hz
DEMO, DEMO_ORIGIN = 115, 103

C_TXT = "#1a1a1a"
C_GREY = "#6a6a6a"
C_TRUTH = "#141414"
C_PERS = "#8a8a8a"
C_MODEL = "#2d5f8a"
C_IN = "#e9eef3"
C_OUT = "#fbf0e8"
C_ANN = "#b5561f"


def demo():
    z = np.load(os.path.join(PREDS, f"clip_{DEMO}.npz"), allow_pickle=True)
    y = np.asarray(z["y"], dtype=np.float64)[:, CH]
    i = int(np.where(np.asarray(z["origins"]) == DEMO_ORIGIN)[0][0])
    return (y[DEMO_ORIGIN - L + 1:DEMO_ORIGIN + 1],
            y[DEMO_ORIGIN + 1:DEMO_ORIGIN + 1 + H],
            np.full(H, y[DEMO_ORIGIN]),
            np.asarray(z[ARM], dtype=np.float64)[i, :, CH],
            str(z["action"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "figures", "overview"))
    ap.add_argument("--no-pdf", action="store_true")
    a = ap.parse_args()

    hist, truth, pers, model, act = demo()
    origin_val = hist[-1]
    th = np.arange(-L + 1, 1) / FPS                  # -0.9 .. 0.0
    tp = np.arange(1, H + 1) / FPS                   # 0.1 .. 1.0

    lo = min(hist.min(), truth.min(), model.min())
    hi = max(hist.max(), truth.max(), model.max())
    rng = hi - lo
    ylim = (lo - .10 * rng, hi + .62 * rng)

    plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans",
                         "axes.edgecolor": C_GREY, "axes.linewidth": .8,
                         "text.color": C_TXT, "axes.labelcolor": C_TXT,
                         "xtick.color": C_GREY, "ytick.color": C_GREY})
    fig, ax = plt.subplots(figsize=(7.16, 3.05))
    fig.subplots_adjust(left=.058, right=.995, top=.975, bottom=.135)

    ax.axvspan(-1.016, 0, color=C_IN, lw=0, zorder=0)
    ax.axvspan(0, 1.016, color=C_OUT, lw=0, zorder=0)

    # the two errors the ratio is built from, at every horizon step
    off, off_hi = .013, .021
    for k in range(H):
        ax.plot([tp[k] - off] * 2, [truth[k], pers[k]], color=C_PERS, lw=1.5, alpha=.28,
                solid_capstyle="butt", zorder=2)
        ax.plot([tp[k] + off] * 2, [truth[k], model[k]], color=C_MODEL, lw=1.5, alpha=.26,
                solid_capstyle="butt", zorder=2)

    # the curves
    ax.plot(th, hist, color=C_TRUTH, lw=2.1, marker="o", ms=3.2, zorder=6)
    ax.plot(np.r_[0, tp], np.r_[origin_val, truth], color=C_TRUTH, lw=2.1, marker="o",
            ms=3.2, zorder=6)
    ax.plot(np.r_[0, tp], np.r_[origin_val, pers], color=C_PERS, lw=1.7,
            ls=(0, (3.0, 1.9)), zorder=4)
    ax.plot(np.r_[0, tp], np.r_[origin_val, model], color=C_MODEL, lw=2.1, zorder=5)
    ax.axvline(0, color=C_ANN, lw=1.3, zorder=7)

    # one labelled pair, where the baseline is furthest from the truth
    k = int(np.argmax(np.abs(pers - truth)))
    for dx, top, col in ((-off_hi, pers[k], C_PERS), (off_hi, model[k], C_MODEL)):
        x = tp[k] + dx
        ax.plot([x, x], [truth[k], top], color=col, lw=3.0, zorder=8,
                solid_capstyle="butt")
        for yv in (truth[k], top):                       # end caps: a measured segment
            ax.plot([x - .017, x + .017], [yv, yv], color=col, lw=1.4, zorder=8)
    ax.text(tp[k] - off_hi - .030, (pers[k] + model[k]) / 2, r"$e_{\mathrm{pers}}$",
            color=C_PERS, fontsize=11.0, ha="right", va="center", zorder=9,
            bbox=dict(boxstyle="square,pad=.10", fc=C_OUT, ec="none"))
    ax.text(tp[k] + off_hi + .030, (truth[k] + model[k]) / 2, r"$e_{\mathrm{model}}$",
            color=C_MODEL, fontsize=11.0, ha="left", va="center", zorder=9,
            bbox=dict(boxstyle="square,pad=.10", fc=C_OUT, ec="none"))

    # the formula, in the symbols marked on the curve
    # inside the predict band: x in data units so it stays tied to the band's right edge
    ax.text(.99, .88,
            r"$\mathrm{skill} = 1 - \dfrac{e_{\mathrm{model}}}{e_{\mathrm{pers}}}$",
            transform=ax.get_xaxis_transform(), fontsize=10.0, color=C_TXT, ha="right",
            va="top")

    # what each curve is, beside the curve
    ax.text(1.03, truth[-1], "ground truth", color=C_TRUTH, fontsize=9.0, va="center")
    ax.text(1.03, model[-1], "prediction", color=C_MODEL, fontsize=9.0, va="center")
    ax.text(1.03, pers[-1], "persistence", color=C_PERS, fontsize=9.0,
            va="center", linespacing=1.25)

    ax.text(-.5, .965, "input", transform=ax.get_xaxis_transform(), ha="center",
            va="top", fontsize=11.0, color=C_TXT, weight="bold")
    ax.text(.5, .965, "predict", transform=ax.get_xaxis_transform(), ha="center",
            va="top", fontsize=11.0, color=C_TXT, weight="bold")
    ax.text(.025, .06, "origin $t$", transform=ax.get_xaxis_transform(), ha="left",
            va="center", fontsize=9.5, color=C_ANN)

    ax.set_xlim(-1.03, 1.36)
    ax.set_ylim(*ylim)
    ax.set_xticks([-1.0, -.5, 0, .5, 1.0])
    ax.set_yticks([])
    ax.tick_params(labelsize=8.5, length=3, pad=2)
    ax.set_xlabel("Time(s)", fontsize=9.5, labelpad=3)
    # centre the label under the plotted second-in/second-out span, not under the axes box,
    # whose right end is empty margin holding the curve names
    xl = ax.get_xlim()
    ax.xaxis.set_label_coords((0.0 - xl[0]) / (xl[1] - xl[0]), -.105)
    ax.set_ylabel("Total Force $F$", fontsize=9.5, labelpad=6)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out + ".png", dpi=400)
    if not a.no_pdf:
        fig.savefig(a.out + ".pdf")
    print(f"wrote {a.out}.png" + ("" if a.no_pdf else f" and {a.out}.pdf"))
    print(f"clip {DEMO} ({act}) origin t={DEMO_ORIGIN}, 1 s history; labelled step h={k+1}")


if __name__ == "__main__":
    main()
