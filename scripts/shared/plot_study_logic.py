"""Paper Fig. 1 (`fig:overview`) -- the study's evaluation logic, drawn from real data.

WHAT THE FIGURE HAS TO SAY (main.tex:79-81). Temporal structure decides how strong
persistence is; only then is a forecast judged, on three axes that answer different
questions. Nothing on the canvas is synthetic: every curve, every number and the tactile
map are read off `runs/as_preds_seq2seq_corpus/clip_*.npz` and
`data/actionsense_states/clip_*.npy`.

LAYOUT
  Top row, three panels:
    (1) HOW PERSISTENT IS THE SIGNAL?  Two real F_R traces, one slowly varying and one
        rapidly changing, each annotated with its own measured
            R = E[(y_{t+H} - y_t)^2] / (2 Var y),      H = 10 @ 10 Hz
        plus a one-row strip of R over all 290 corpus recordings, so the reader sees that R
        is a continuum (.09 - 1.64) and that the two SEMANTIC classes overlap on it.
    (2) WHAT THE MODEL SEES.  A real 32x32 right-hand frame -> s_t = [F, c^x, c^y], then
        the rolling-origin window: history L, origin t, horizon H = 1 s, origins advancing
        by one sample. Tactile history only -- no future action, pose or image.
    (3) ONE FORECAST, THREE QUESTIONS.  One real 1 s horizon (clip 115, `peel`, origin
        t = 103) with truth, persistence and Seq2Seq. Two annotations carry the geometry
        that separates the axes: the model's error against persistence's at the same step
        (S), and the model point furthest from the whole truth curve (D_H).
  Bottom strip: the nested ladder R -> R^2 -> S -> D_H, one short question each, with the
  one sentence that makes the nesting binding.

DELIBERATELY NOT DRAWN
  * The traces are NOT labelled `smooth`/`abrupt`. On this corpus the semantic classes order
    R the wrong way round (smooth .720 vs abrupt .580, measured here), while the caption's
    claim is about temporal structure -- a signal property. Labelling the curves by class
    would hand the reader an ordering that Sec. Results and Limitations both deny. The strip
    shows the two classes so that overlap is visible rather than asserted.
  * Table II's corpus values (.7414 / .1276 / .830). Fig. 1 sits in the Introduction and
    defines the four quantities; quoting the results there would pre-empt them. The only
    numbers shown are the illustrated origin's own.

Usage:  python scripts/shared/plot_study_logic.py [--out figures/overview] [--no-pdf]
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from src.actionsense.trait import TRAIT_CLASS          # noqa: E402
from src.shape_metrics import hausdorff_scaled         # noqa: E402

PREDS = os.path.join(ROOT, "runs", "as_preds_seq2seq_corpus")
MAPS = os.path.join(ROOT, "data", "actionsense_states")
ARM = "mu_seq2seq_aggregate"
CH = 3                      # F_R -- the one channel all three sensors share
FPS = 10.0
H = 10                      # 1 s horizon at 10 Hz

# the illustrated recordings, chosen by measured R (see module docstring)
SLOW, FAST = 3, 112         # R = .133 (peel) and R = 1.046 (get)
DEMO, DEMO_ORIGIN = 115, 103

C_TXT = "#1a1a1a"
C_GREY = "#6a6a6a"
C_TRUTH = "#141414"
C_PERS = "#8c8c8c"
C_MODEL = "#2d5f8a"
C_BAND = "#bcd8ef"
C_ANN = "#b5561f"
C_SMOOTH = "#4e9468"
C_ABRUPT = "#c9852f"
C_OTHER = "#6a4c93"          # OpenTouch, the corpus present here only as an aggregate


# --------------------------------------------------------------------------- data --
def r_difficulty(v: np.ndarray) -> float:
    """R = E[(y_{t+H}-y_t)^2] / (2 Var y) -- main.tex eq. (3)."""
    return float(np.mean((v[H:] - v[:-H]) ** 2) / (2.0 * np.var(v)))


def load_state(idx: int) -> tuple[np.ndarray, str]:
    z = np.load(os.path.join(PREDS, f"clip_{idx}.npz"), allow_pickle=True)
    return np.asarray(z["y"], dtype=np.float64)[:, CH], str(z["action"])


def corpus_r() -> list[tuple[str, str, float]]:
    """(action, trait class, R) for every recording long enough to have a horizon."""
    out = []
    for p in sorted(glob.glob(os.path.join(PREDS, "clip_*.npz"))):
        z = np.load(p, allow_pickle=True)
        v = np.asarray(z["y"], dtype=np.float64)[:, CH]
        if len(v) <= H + 5:
            continue
        act = str(z["action"])
        out.append((act, TRAIT_CLASS[act], r_difficulty(v)))
    return out


def demo_forecast():
    """One real origin -> the three curves and the two numbers drawn beside them."""
    z = np.load(os.path.join(PREDS, f"clip_{DEMO}.npz"), allow_pickle=True)
    y = np.asarray(z["y"], dtype=np.float64)
    i = int(np.where(np.asarray(z["origins"]) == DEMO_ORIGIN)[0][0])
    truth = y[DEMO_ORIGIN + 1:DEMO_ORIGIN + 1 + H, CH]
    pers = np.full(H, y[DEMO_ORIGIN, CH])
    model = np.asarray(z[ARM], dtype=np.float64)[i, :, CH]
    skill = 1.0 - np.mean((model - truth) ** 2) / np.mean((pers - truth) ** 2)
    hm = float(hausdorff_scaled(model[None], truth[None])[0])
    hp = float(hausdorff_scaled(pers[None], truth[None])[0])
    return (truth, pers, model, float(skill), hm, hp, float(y[DEMO_ORIGIN, CH]),
            str(z["action"]))


def opentouch_floor() -> tuple[float, int]:
    """OpenTouch's F_R difficulty, read from the frozen cross-sensor table -> (R, n clips).

    NOT recomputed here, because it cannot be: this machine holds no OpenTouch state cache
    and no OpenTouch per-clip forecasts, only `docs/opentouch/**` report CSVs. What it does
    hold is `docs/predictability_floor.csv`, written by predictability_floor.py, which
    measures THE SAME R as `r_difficulty` above -- one definition, each sensor at its own
    1 s horizon. Reading the file rather than typing .045 keeps the figure tied to the
    artefact, so a regenerated table moves the figure instead of silently disagreeing.
    """
    path = os.path.join(ROOT, "docs", "predictability_floor.csv")
    with open(path) as fh:
        for row in csv.DictReader(fh):
            if row["sensor"] == "opentouch" and row["channel"] == "F_R":
                return float(row["R"]), int(row["n_recordings"])
    raise SystemExit(f"no opentouch F_R row in {path}")


def demo_map() -> np.ndarray:
    """The most-loaded real right-hand frame of the demo recording, pedestal-corrected."""
    m = np.load(os.path.join(MAPS, f"clip_{DEMO}.npy"), mmap_mode="r")
    block = np.asarray(m[::5, 1], dtype=np.float64)
    f = np.clip(block - np.percentile(block, 5, axis=0), 0, None)
    return f[int(np.argmax(f.sum(axis=(1, 2))))]


# ---------------------------------------------------------------- panel 1: R --
def panel_persistence(ax_slow, ax_fast, ax_strip, traces, rs):
    for ax, (v, r, act, lab) in zip((ax_slow, ax_fast), traces):
        t = np.arange(len(v)) / FPS
        ax.plot(t, v, color=C_TRUTH, lw=.85)
        o = int(np.argmax(np.abs(v[:len(v) - H] - v[H:])))
        ax.axvspan(o / FPS, (o + H) / FPS, color=C_BAND, alpha=.9, lw=0, zorder=0)
        ax.annotate("", xy=((o + H) / FPS, v[o + H]), xytext=(o / FPS, v[o]),
                    arrowprops=dict(arrowstyle="-|>", color=C_ANN, lw=1.1, shrinkA=0,
                                    shrinkB=0, mutation_scale=7), zorder=4)
        ax.text((o + H / 2) / FPS, max(v[o:o + H + 1]) + .09 * max(v), "1 s",
                fontsize=5.9, color=C_ANN, ha="center", va="bottom")
        ax.set_title(f"{lab}   ({act})", fontsize=6.6, color=C_GREY, pad=2, loc="left")
        ax.text(.012, .97, f"$\\mathcal{{R}}$ = {r:.2f}", transform=ax.transAxes,
                ha="left", va="top", fontsize=7.6, color=C_ANN, weight="bold",
                bbox=dict(boxstyle="round,pad=.18", fc="white", ec="none", alpha=.92))
        ax.set_xlim(0, len(v) / FPS)
        ax.set_ylim(0, max(v) * 1.45)
        ax.set_ylabel("force $F$", fontsize=6.6, labelpad=2)
        ax.set_yticks([])
        ax.tick_params(labelsize=6, length=2, pad=1)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
    ax_slow.set_xticklabels([])
    ax_fast.set_xlabel("time (s)", fontsize=6.6, labelpad=1)

    rng = np.random.default_rng(0)
    for cls, col, yc in (("smooth", C_SMOOTH, .80), ("abrupt", C_ABRUPT, .50)):
        xs = np.array([r for _, c, r in rs if c == cls])
        ax_strip.scatter(xs, yc + rng.uniform(-.095, .095, len(xs)), s=3.0, color=col,
                         alpha=.6, lw=0)
        ax_strip.text(1.72, yc, f"{cls} ({len(xs)})", fontsize=5.7, color=col,
                      va="center", ha="left", linespacing=1.1)

    # The other corpus. Only its aggregate is on disk, so only its aggregate is drawn --
    # docs/predictability_floor.csv, one definition of R measured on both sensors. Its own
    # row is empty of dots, so the label sits beside the marker rather than in the margin.
    ot_r, ot_n = opentouch_floor()
    ax_strip.plot([ot_r], [.16], marker="D", ms=4.4, color=C_OTHER)
    ax_strip.text(ot_r + .075, .16, f"OpenTouch mean, {ot_n:,} clips",
                  fontsize=5.7, color=C_OTHER, va="center", ha="left")

    ax_strip.set_xlim(0, 2.35)
    ax_strip.set_ylim(0, 1.0)
    ax_strip.set_yticks([])
    ax_strip.set_xticks([0, .5, 1.0, 1.5])
    ax_strip.tick_params(labelsize=6, length=2, pad=1)
    ax_strip.set_xlabel(r"$\mathcal{R}=\mathbb{E}[(y_{t+H}-y_t)^2]\,/\,2\,\mathrm{Var}(y)$",
                        fontsize=7.0, labelpad=2)
    ax_strip.set_title("every one of the 290 ActionSense recordings", fontsize=6.4,
                       color=C_GREY, pad=3, loc="left")
    for s in ("top", "right", "left"):
        ax_strip.spines[s].set_visible(False)


# ------------------------------------------------------- panel 2: the setup --
def panel_setup(ax, frame):
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    im = ax.inset_axes([0.005, 0.58, 0.245, 0.40])
    im.imshow(frame, cmap="inferno", interpolation="nearest",
              vmin=0, vmax=np.percentile(frame, 99))
    tot = frame.sum()
    gy, gx = np.mgrid[0:frame.shape[0], 0:frame.shape[1]]
    im.plot([(gx * frame).sum() / tot], [(gy * frame).sum() / tot], marker="+", ms=7,
            mew=1.5, color="#7fd4ff")
    im.set_xticks([]); im.set_yticks([])
    for s in im.spines.values():
        s.set_color("#9a9a9a"); s.set_linewidth(.6)
    ax.text(12.5, 100, "tactile map  $P_t$", ha="center", va="bottom", fontsize=6.8,
            color=C_TXT)

    ax.add_patch(FancyArrowPatch((28, 78), (40, 78), arrowstyle="-|>", lw=1.0,
                                 color=C_GREY, mutation_scale=8))
    ax.text(44, 82, r"$\mathbf{s}_t=[\,F,\;c^x,\;c^y\,]$", ha="left", fontsize=8.4,
            color=C_TXT)
    ax.text(44, 74, "total force and\ncentre of pressure,\none state per frame",
            ha="left", va="top", fontsize=6.1, color=C_GREY, linespacing=1.3)

    base, dy, x0, org, x1 = 40, 9.5, 4, 58, 88
    for k in range(3):
        yk = base - k * dy
        a = 1.0 if k == 0 else .34
        ax.add_patch(Rectangle((x0 + k * 3.5, yk), org - x0, 6.0, fc=C_BAND, ec="none",
                               alpha=a, zorder=2))
        ax.add_patch(Rectangle((org + k * 3.5, yk), x1 - org, 6.0, fc="none", ec=C_MODEL,
                               lw=.9, ls=(0, (2.4, 1.4)), alpha=a, zorder=2))
    ax.text((x0 + org) / 2, base + 3.0, "tactile history  $L$", ha="center", va="center",
            fontsize=6.6, color="#1d425f", zorder=3)
    ax.text((org + x1) / 2, base + 3.0, "$H\\!=\\!1$ s", ha="center", va="center",
            fontsize=6.6, color=C_MODEL, zorder=3)
    ax.plot([org, org], [base - 2 * dy, base + 8.5], color=C_ANN, lw=1.0, zorder=4)
    ax.text(org, base + 10.5, "origin $t$", ha="center", fontsize=6.8, color=C_ANN)
    ax.text(x0, 17.5, "origins step by one sample", ha="left", va="center",
            fontsize=5.8, color=C_GREY)
    ax.text(0, 0, "tactile history only\nno future action, pose or image", ha="left",
            va="bottom", fontsize=6.6, color=C_TXT, linespacing=1.35)


# ------------------------------------------ panel 3: one forecast, three questions --
def panel_forecast(ax, truth, pers, model, skill, hm, origin_val, act, r_act):
    h = np.arange(0, H + 1) / FPS
    tr = np.concatenate([[origin_val], truth])
    pe = np.concatenate([[origin_val], pers])
    mo = np.concatenate([[origin_val], model])

    ax.plot(h, pe, color=C_PERS, lw=1.4, ls=(0, (2.6, 1.6)), zorder=3,
)
    ax.plot(h, mo, color=C_MODEL, lw=1.7, zorder=4)
    ax.plot(h, tr, color=C_TRUTH, lw=1.8, zorder=5)
    ax.plot([0], [origin_val], marker="o", ms=3.6, color=C_ANN, zorder=6)
    ax.text(0, origin_val + .045 * np.ptp(tr), "origin $t$", fontsize=6.2, color=C_ANN,
            ha="left", va="bottom")

    # S: the model's error against persistence's, at the step where they differ most
    j = int(np.argmax(np.abs(pers - truth))) + 1
    ax.annotate("", xy=(h[j] - .022, tr[j]), xytext=(h[j] - .022, pe[j]),
                arrowprops=dict(arrowstyle="<->", color=C_PERS, lw=1.0, shrinkA=0,
                                shrinkB=0, mutation_scale=5), zorder=6)
    ax.annotate("", xy=(h[j] + .022, tr[j]), xytext=(h[j] + .022, mo[j]),
                arrowprops=dict(arrowstyle="<->", color=C_MODEL, lw=1.0, shrinkA=0,
                                shrinkB=0, mutation_scale=5), zorder=6)
    ax.text(h[j] + .045, (tr[j] + pe[j]) / 2, f"$S$ = {skill:+.2f}", fontsize=6.8,
            color=C_MODEL, ha="left", va="center", weight="bold")

    # D_H: the model point furthest from the whole truth curve
    sd = truth.std()
    d = np.sqrt((h[1:, None] - h[None, 1:]) ** 2 / (H / FPS) ** 2
                + ((model[:, None] - truth[None, :]) / sd) ** 2)
    wi, = np.unravel_index(np.argmax(d.min(axis=1)), (H,))
    wj = int(np.argmin(d[wi]))
    ax.plot([h[wi + 1], h[wj + 1]], [mo[wi + 1], tr[wj + 1]], color=C_ANN, lw=1.2,
            zorder=7)
    ax.plot([h[wi + 1], h[wj + 1]], [mo[wi + 1], tr[wj + 1]], marker="o", ms=2.6,
            ls="none", color=C_ANN, zorder=7)
    for yv, lab, col in ((pe[-1], "persistence", C_PERS), (mo[-1], "forecast", C_MODEL),
                         ((mo[-1] + tr[-1]) / 2, f"$D_{{\\mathrm{{H}}}}$ = {hm:.1f} sd",
                          C_ANN),
                         (tr[-1], "truth", C_TRUTH)):
        ax.text(h[-1] + .035, yv, lab, fontsize=6.6, color=col, ha="left", va="center")
    ax.set_xlim(-.03, h[-1] + .30)
    ax.set_ylim(min(tr.min(), mo.min()) - .18 * np.ptp(tr), tr.max() + .22 * np.ptp(tr))
    ax.set_xticks([0, .5, 1.0])
    ax.set_xlabel("horizon (s)", fontsize=6.6, labelpad=1)
    ax.set_ylabel("force $F$", fontsize=6.6, labelpad=2)
    ax.set_yticks([])
    ax.tick_params(labelsize=6, length=2, pad=1)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.set_title(f"one held-out origin   ($\\mathcal{{R}}$ = {r_act:.2f} for {act})",
                 fontsize=6.6, color=C_GREY, pad=2, loc="left")



# --------------------------------------------------- bottom strip: the ladder --
def ladder(ax):
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    rows = [(r"$\mathcal{R}$", "how hard is\ncopy-last?", C_ANN),
            (r"$R^2$", "is the future state\nright?", C_GREY),
            (r"$S_{\mathrm{pers}}$", "better than\ncopy-last?", C_MODEL),
            (r"$D_{\mathrm{H}}$", "is the path\nright?", C_ANN)]
    xs = [1, 25.5, 50, 74.5]
    for x, (sym, txt, col) in zip(xs, rows):
        ax.text(x, 72, sym, fontsize=10.5, color=col, weight="bold", va="center",
                ha="left")
        ax.text(x + 6.5, 72, txt, fontsize=6.9, color=C_TXT, va="center", ha="left",
                linespacing=1.25)
        if x != xs[-1]:
            ax.add_patch(FancyArrowPatch((x + 19.8, 72), (x + 23.2, 72),
                                         arrowstyle="-|>", lw=1.0, color="#a8a8a8",
                                         mutation_scale=7))
    ax.text(1, 14, "signal property", fontsize=6.2, color=C_GREY, style="italic")
    ax.text(25.5, 14, "model scores — each only readable against $\\mathcal{R}$",
            fontsize=6.2, color=C_GREY, style="italic")
    ax.plot([0.3, 20.5], [30, 30], color=C_ANN, lw=1.0, alpha=.5)
    ax.plot([24.8, 97], [30, 30], color=C_GREY, lw=1.0, alpha=.5)


# ------------------------------------------------------------------------ main --
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "figures", "overview"))
    ap.add_argument("--no-pdf", action="store_true")
    a = ap.parse_args()

    slow, act_slow = load_state(SLOW)
    fast, act_fast = load_state(FAST)
    r_slow, r_fast = r_difficulty(slow), r_difficulty(fast)
    rs = corpus_r()
    truth, pers, model, skill, hm, hp, origin_val, demo_act = demo_forecast()
    frame = demo_map()

    span = int(24 * FPS)          # one time base, so the comparison is shape not duration
    traces = ((slow[:span], r_slow, act_slow, "slowly varying"),
              (fast[:span], r_fast, act_fast, "rapidly changing"))

    plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans",
                         "axes.edgecolor": C_GREY, "axes.linewidth": .7,
                         "text.color": C_TXT, "axes.labelcolor": C_TXT,
                         "xtick.color": C_GREY, "ytick.color": C_GREY})

    fig = plt.figure(figsize=(7.16, 3.05))
    outer = fig.add_gridspec(2, 1, height_ratios=[1, .21], left=.042, right=.974,
                             top=.895, bottom=.035, hspace=.36)
    top = outer[0].subgridspec(1, 3, width_ratios=[.335, .275, .39], wspace=.25)

    g1 = top[0].subgridspec(3, 1, height_ratios=[1, 1, 1.18], hspace=.95)
    panel_persistence(fig.add_subplot(g1[0]), fig.add_subplot(g1[1]),
                      fig.add_subplot(g1[2]), traces, rs)
    panel_setup(fig.add_subplot(top[1]), frame)
    r_act = float(np.mean([r for a, _, r in rs if a == demo_act]))
    panel_forecast(fig.add_subplot(top[2]), truth, pers, model, skill, hm, origin_val,
                   demo_act, r_act)
    ladder(fig.add_subplot(outer[1]))

    for num, txt, x in (("1", "How persistent is the signal?", .042),
                        ("2", "What the model sees", .405),
                        ("3", "One forecast, three questions", .662)):
        fig.text(x, .965, num, fontsize=7.4, color="white", weight="bold", ha="center",
                 va="center", bbox=dict(boxstyle="circle,pad=.26", fc=C_TXT, ec="none"))
        fig.text(x + .019, .965, txt, fontsize=8.4, color=C_TXT, weight="bold",
                 ha="left", va="center")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out + ".png", dpi=400)
    if not a.no_pdf:
        fig.savefig(a.out + ".pdf")
    print(f"wrote {a.out}.png" + ("" if a.no_pdf else f" and {a.out}.pdf"))
    print(f"R slow (clip {SLOW}, {act_slow}) = {r_slow:.3f} | "
          f"R fast (clip {FAST}, {act_fast}) = {r_fast:.3f}")
    print(f"origin clip {DEMO} t={DEMO_ORIGIN}: skill={skill:+.3f} "
          f"D_H model={hm:.3f} pers={hp:.3f} ratio={hm / hp:.3f}")
    sm = [r for _, c, r in rs if c == "smooth"]
    ab = [r for _, c, r in rs if c == "abrupt"]
    print(f"corpus R: n={len(rs)} range {min(r for *_, r in rs):.3f}-"
          f"{max(r for *_, r in rs):.3f} | smooth {np.mean(sm):.3f} (n={len(sm)}) "
          f"| abrupt {np.mean(ab):.3f} (n={len(ab)})")
    print(f"demo action {demo_act}: mean R = {r_act:.3f}")


if __name__ == "__main__":
    main()
