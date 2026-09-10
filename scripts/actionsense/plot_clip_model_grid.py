"""One clip, nine models: a 3x3 grid of forecasts against the same ground truth.

Rows are the two neural backbones and the classical baselines; columns are the three inputs
(for the backbones) or the three baselines. Every panel draws the SAME recording and the same
channel, so the grid is a like-for-like comparison and the eye can travel across it.

    python scripts/actionsense/plot_clip_model_grid.py \
        --preds runs/as_preds_tmap_seq2seq_corpus3s \
        --preds runs/as_preds_tmap_probgru_corpus3s \
        --preds runs/as_preds_baselines \
        --clip 115 --channel F_R --out docs/actionsense/peel_model_grid.png

UNITS -- the reason every axis is labelled rather than left to the channel name.

  F_L, F_R      "total force", and it is FORCE-LIKE, not a pressure: it is the sum over all
                1024 taxels of one hand's baseline-corrected reading (physical_state.py:44,
                `F = p.sum()`). But the ActionSense gloves are conductive-thread resistive
                sensors that are never tared or calibrated (physical_state.py:62), so the
                summed reading carries no newtons -- it is proportional to force by an unknown
                constant. Values run to ~10^4. Label it as arbitrary units and do not quote it
                as a force.
  CoPx_*, CoPy_*  centre of pressure, the pressure-weighted centroid, expressed in grid
                coordinates normalised to [-1, +1] across the sensor (physical_state.py:29-33,
                "so features are sensor-size-agnostic"). DIMENSIONLESS -- not millimetres.

  time          seconds, from the npz's own `fps`.

WHAT THE FORECAST LINE IS. Following the convention of
scripts/opentouch/plot_opentouch_forecast_overlay.py: take every H-th origin and draw that
origin's whole H-step trajectory, so the line is a chain of independent one-second forecasts
rather than a single step repeated. Bands are +-2 sigma where the arm is probabilistic; the
baselines are point forecasts and have none.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# rows x columns of the grid. `None` leaves a panel empty rather than shifting the others.
ROWS = [
    ("Seq2Seq", ["seq2seq_aggregate", "seq2seq_flatten", "seq2seq_cnn"]),
    ("probGRU", ["probgru_aggregate", "probgru_flatten", "probgru_cnn"]),
    ("baselines", ["persistence", "ar", "seasonal"]),
]
COL_TITLE = [
    ["physical state", "flatten", "cnn"],
    ["physical state", "flatten", "cnn"],
    ["persistence", "linear AR", "seasonal-naive"],
]
# categorical slots of the reference palette, in its fixed order; hue tracks the input/baseline
COLOR = {
    "aggregate": "#2a78d6", "flatten": "#eb6834", "cnn": "#1baf7a",
    "persistence": "#eda100", "ar": "#e87ba4", "seasonal": "#4a3aa7",
}
INK, MUTED, GRID, TRUTH = "#1a1a19", "#5c5b54", "#d9d8d2", "#1a1a19"

UNITS = {
    "F": ("total force", "a.u."),
    "CoPx": ("centre of pressure x", "normalised, −1…+1"),
    "CoPy": ("centre of pressure y", "normalised, −1…+1"),
}


def unit_label(ch: str) -> tuple[str, str]:
    """(two-line axis label, unit) -- the quantity on one line, its unit on the next.

    On one line the CoP label is longer than a panel is tall, so matplotlib ran it across the
    neighbouring panel's label and both became unreadable.
    """
    kind = ch.split("_")[0]
    text, unit = UNITS.get(kind, (ch, ""))
    return f"{ch} — {text}\n({unit})", unit


def color_of(model: str) -> str:
    return COLOR.get(model.rsplit("_", 1)[-1], "#5c5b54")


def rolling(mu, sg, ors, H):
    """Chain every H-th origin's whole H-step trajectory -> (idx, value, sigma).

    The same convention plot_opentouch_forecast_overlay.py uses, so a panel here and a panel
    there mean the same thing.
    """
    if len(ors) == 0:
        return np.zeros(0, int), np.zeros((0, mu.shape[-1])), None
    sel = range(0, len(ors), H)
    idx = np.concatenate([np.arange(ors[j] + 1, ors[j] + 1 + H) for j in sel])
    val = np.concatenate([mu[j] for j in sel], 0)
    sig = np.concatenate([sg[j] for j in sel], 0) if sg is not None else None
    return idx, val, sig


def load_clip(dirs: list[str], clip: int) -> dict:
    """Union of every arm for one recording, checking the panels share a ground truth."""
    out, ref = {}, None
    for d in dirs:
        p = os.path.join(d, f"clip_{clip}.npz")
        if not os.path.exists(p):
            print(f"  (no clip_{clip}.npz in {d})")
            continue
        z = np.load(p, allow_pickle=True)
        if ref is None:
            ref = dict(y=z["y"], fps=float(z["fps"]),
                       channels=[str(c) for c in z["channels"]],
                       action=str(z["action"]), object_name=str(z["object_name"]))
        elif z["y"].shape != ref["y"].shape or not np.allclose(z["y"], ref["y"], equal_nan=True):
            # Panels that do not share a truth are not comparable, and the grid's whole point
            # is that they are. Say so instead of drawing nine plausible-looking curves.
            raise SystemExit(
                f"{p} holds a different ground truth for clip {clip} than the first source. "
                f"These runs were not scored on the same recording; do not put them in one "
                f"grid.")
        for k in z.files:
            if k.startswith("mu_"):
                m = k[3:]
                out[m] = (z[k], z.get(f"sigma_{m}"), z["origins"])
    if ref is None:
        raise SystemExit(f"clip_{clip}.npz not found in any of: {', '.join(dirs)}")
    ref["arms"] = out
    return ref


def check_is_test(clip: int, allow: bool = False) -> None:
    """Refuse a recording the baselines were FITTED on.

    export_baseline_forecasts.py fits persistence/AR/seasonal on the frozen TRAIN split and
    writes only TEST recordings, so a clip drawn from its output is a test recording by
    construction. Reading the grid off a train recording would show the baselines predicting
    data they were built from, and the neural panels beside them would look worse for an
    entirely procedural reason. Checked here rather than left to whoever picks the number.
    """
    import json
    from src.actionsense.eval_harness.config import load_config
    from src.actionsense.eval_harness.splits import load_splits

    try:
        sp = load_splits(load_config())
    except Exception as exc:                                   # noqa: BLE001
        print(f"  (could not read the frozen split, so not checking membership: {exc})")
        return
    where = [k for k in ("train", "val", "test") if clip in sp.get(k, [])]
    if "test" in where:
        print(f"  clip {clip} is in the frozen harness TEST split -- ok")
        return
    msg = (f"clip {clip} is in {where or ['no frozen split']}, not TEST. The baselines are "
           f"fitted on TRAIN, so on this recording they would be predicting data they were "
           f"built from. Frozen TEST recordings: {sorted(sp.get('test', []))}")
    if allow:
        print(f"  WARNING: {msg}")
    else:
        raise SystemExit(f"refusing to draw: {msg}\n  (pass --allow-non-test to override)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", action="append", required=True,
                    help="a predictions directory; repeat for each run to combine")
    ap.add_argument("--clip", type=int, required=True)
    ap.add_argument("--channel", default="F_R")
    ap.add_argument("--out", default="docs/actionsense/clip_model_grid.png")
    ap.add_argument("--seconds", type=float, default=None,
                    help="plot only the first N seconds, so the traces stay legible")
    ap.add_argument("--allow-non-test", action="store_true",
                    help="draw even if the clip is not in the frozen harness TEST split")
    a = ap.parse_args()

    check_is_test(a.clip, allow=a.allow_non_test)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = load_clip(a.preds, a.clip)
    chans, fps, y = data["channels"], data["fps"], data["y"]
    if a.channel not in chans:
        raise SystemExit(f"--channel {a.channel} not in {chans}")
    k = chans.index(a.channel)
    H = next(iter(data["arms"].values()))[0].shape[1]
    tt = np.arange(len(y)) / fps
    tmax = a.seconds if a.seconds else tt[-1]

    have = sorted(data["arms"])
    print(f"  clip {a.clip}  action={data['action']!r}  {len(y)} frames @ {fps:g} Hz  "
          f"H={H} ({H / fps:g} s)")
    print(f"  arms present: {have}")
    missing = [m for _, row in ROWS for m in row if m not in data["arms"]]
    if missing:
        print(f"  MISSING (panels left empty): {missing}")

    # A seasonal-naive that found no cycle falls back to persistence, and then two panels are
    # the same curve. That is a property of the data worth showing, but only if it is said.
    dup = ("persistence" in data["arms"] and "seasonal" in data["arms"]
           and np.allclose(data["arms"]["seasonal"][0], data["arms"]["persistence"][0]))

    ylab, ynote = unit_label(a.channel)
    fig, axes = plt.subplots(3, 3, figsize=(15.0, 9.6), sharex=True, sharey=True)
    for r, (rowname, models) in enumerate(ROWS):
        for c, m in enumerate(models):
            ax = axes[r, c]
            ax.plot(tt, y[:, k], "-", color=TRUTH, lw=1.2, label="ground truth", zorder=3)
            if m in data["arms"]:
                mu, sg, ors = data["arms"][m]
                idx, val, sig = rolling(mu, sg, ors, H)
                t = idx / fps
                col = color_of(m)
                if sig is not None:
                    ax.fill_between(t, val[:, k] - 2 * sig[:, k], val[:, k] + 2 * sig[:, k],
                                    color=col, alpha=0.20, lw=0, zorder=2,
                                    label="±2σ")
                ax.plot(t, val[:, k], "-", color=col, lw=1.3, zorder=4,
                        label=f"{H / fps:g} s forecast")
            else:
                ax.text(0.5, 0.5, "not available", transform=ax.transAxes, ha="center",
                        va="center", fontsize=10, color=MUTED)
            note = COL_TITLE[r][c]
            if m == "seasonal" and dup:
                note += "  (no cycle → ≡ persistence)"
            t_txt = f"{rowname} · {note}"
            ax.set_title(t_txt, fontsize=9.5 if len(t_txt) <= 46 else 8.5,
                         color=INK, loc="left", pad=6)
            # every panel carries both axis labels: the grid is read cell by cell, and the
            # y quantity is exactly what is easy to get wrong here
            ax.set_xlabel("time  (s)", fontsize=8.5, color=MUTED)
            ax.set_ylabel(ylab, fontsize=8.5, color=MUTED, linespacing=1.4)
            ax.tick_params(labelbottom=True, labelleft=True, colors=MUTED,
                           labelsize=8, length=0)
            ax.grid(color=GRID, lw=0.6, zorder=0)
            ax.set_axisbelow(True)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            for s in ("bottom", "left"):
                ax.spines[s].set_color(GRID)
            ax.set_xlim(0, tmax)

    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.42 / fig.get_figheight()))
    fig.legend(h, lab, frameon=False, fontsize=9, labelcolor=INK, ncols=3,
               loc="upper center", bbox_to_anchor=(0.5, 1 - 0.04 / fig.get_figheight()))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=150, facecolor="white")
    print(f"[done] {a.out}")


if __name__ == "__main__":
    main()
