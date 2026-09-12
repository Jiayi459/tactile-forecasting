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

# Rows x columns of the grid, per sensor. Each cell is (arm_name, colour_key, column title),
# or None to leave the panel empty rather than shifting its neighbours.
#
# WHY A CELL CARRIES ITS OWN COLOUR KEY. The three sensors name their arms under three mutually
# incompatible conventions, so the encoder cannot be recovered from the arm name:
#
#   ActionSense  probgru_flatten     <family>_<encoder>
#   OpenTouch    pg_cnn, prob_gru    <family>_<encoder>, but the aggregate arm is just `prob_gru`
#   EgoTouch     aggregate_seq2seq   <encoder>_<family>  -- the REVERSE order
#
# The old code took `model.rsplit("_", 1)[-1]` as the encoder. That reads "seq2seq" as an
# encoder for EgoTouch and "gru" for OpenTouch's aggregate arm, and both fall through to grey --
# the colour would have stopped tracking the input, which is the one thing the columns encode.
# Naming the key here keeps hue tied to the input across all three sensors.
LAYOUTS = {
    "actionsense": [
        ("Seq2Seq", [("seq2seq_aggregate", "aggregate", "physical state"),
                     ("seq2seq_flatten", "flatten", "flatten"),
                     ("seq2seq_cnn", "cnn", "cnn")]),
        ("probGRU", [("probgru_aggregate", "aggregate", "physical state"),
                     ("probgru_flatten", "flatten", "flatten"),
                     ("probgru_cnn", "cnn", "cnn")]),
        ("baselines", [("persistence", "persistence", "persistence"),
                       ("ar", "ar", "linear AR"),
                       ("seasonal", "seasonal", "seasonal-naive")]),
    ],
    # The only sensor that fills all nine panels: the deterministic map arms come from
    # --preds runs/preds_d1_map2 and the probabilistic ones from runs/preds_d1_pg.
    "opentouch": [
        ("map", [("map_aggregate", "aggregate", "physical state"),
                 ("flatten", "flatten", "flatten"),
                 ("cnn", "cnn", "cnn")]),
        ("probGRU", [("prob_gru", "aggregate", "physical state"),
                     ("pg_flatten", "flatten", "flatten"),
                     ("pg_cnn", "cnn", "cnn")]),
        ("baselines", [("persistence", "persistence", "persistence"),
                       ("ar", "ar", "linear AR"),
                       ("seasonal", "seasonal", "seasonal-naive")]),
    ],
    # Fills all nine since the probGRU map arms were run (24ebb3c, 5f4314a) -- they had been
    # cut on a premise that turned out to be wrong. The baselines take the `_group` scope:
    # fitted per recording group, the same per-recording basis the other two sensors use.
    "egotouch": [
        ("Seq2Seq", [("aggregate_seq2seq", "aggregate", "physical state"),
                     ("flatten_seq2seq", "flatten", "flatten"),
                     ("cnn_seq2seq", "cnn", "cnn")]),
        ("probGRU", [("aggregate_probgru", "aggregate", "physical state"),
                     ("flatten_probgru", "flatten", "flatten"),
                     ("cnn_probgru", "cnn", "cnn")]),
        ("baselines", [("persistence", "persistence", "persistence"),
                       ("ar_group", "ar", "linear AR (per group)"),
                       ("seasonal_group", "seasonal", "seasonal-naive (per group)")]),
    ],
}
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


def color_of(key: str) -> str:
    """Colour from the layout's explicit key -- see LAYOUTS on why it is not parsed from the
    arm name."""
    return COLOR.get(key, "#5c5b54")


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


# Why a prediction's mere existence proves it is held out -- one sentence per sensor, each
# pointing at the line of code that makes it true. The same argument in all three cases: the
# writer is only ever reached for data the fit never saw, so presence IS the guarantee.
HELD_OUT = {
    "actionsense": "both exporters write only recordings their fit never saw "
                   "(--scope frozen writes TEST; --scope corpus holds each out in one fold)",
    "opentouch": "one npz per TEST clip, every clip TEST in exactly one fold "
                 "(opentouch_report.py:3)",
    "egotouch": "only test_seen and test_unseen are ever written; train/val never are "
                "(egotouch/train_tactile_map.py:157)",
}


def report_heldout(arms: dict, clip: int, layout: list, dataset: str) -> None:
    """Say whether the baseline panels are held-out forecasts, from the arms themselves.

    The earlier version of this checked the clip against the frozen slice+peel split, which
    was the wrong question twice over: it rejected every corpus action even though the corpus
    baselines hold those out properly, and it would have accepted a frozen TRAIN recording had
    one ever been written. The right question is whether a baseline forecast EXISTS for this
    recording, because the writers are only reached for data the fit never saw. So a baseline
    arm being present is itself the guarantee -- and that holds for all three sensors, each for
    its own reason (HELD_OUT).
    """
    base = [cell[0] for cell in layout[-1][1] if cell and cell[0] in arms]
    if base:
        print(f"  baselines present for clip {clip}: {base} -- held out by construction, "
              f"{HELD_OUT[dataset]}")
    else:
        print(f"  no baseline arms for clip {clip}; the bottom row will be empty.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", action="append", required=True,
                    help="a predictions directory; repeat for each run to combine")
    ap.add_argument("--dataset", default="actionsense", choices=sorted(LAYOUTS),
                    help="which sensor's arm names to expect; see LAYOUTS")
    ap.add_argument("--clip", type=int, required=True)
    ap.add_argument("--channel", default="F_R")
    ap.add_argument("--out", default="docs/actionsense/clip_model_grid.png")
    ap.add_argument("--seconds", type=float, default=None,
                    help="plot only the first N seconds, so the traces stay legible")
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layout = LAYOUTS[a.dataset]
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
    missing = [c[0] for _, row in layout for c in row if c and c[0] not in data["arms"]]
    if missing:
        print(f"  MISSING (panels left empty): {missing}")
    unused = [m for m in have if m not in {c[0] for _, r in layout for c in r if c}]
    if unused:
        # An arm present in the npz but absent from the layout is usually a name the layout got
        # wrong, not a spare arm -- say it rather than silently drawing eight of nine panels.
        print(f"  in the data but not in the --dataset {a.dataset} layout: {unused}")
    report_heldout(data["arms"], a.clip, layout, a.dataset)

    # A seasonal-naive that found no cycle falls back to persistence, and then two panels are
    # the same curve. That is a property of the data worth showing, but only if it is said.
    bl = {c[1]: c[0] for c in layout[-1][1] if c}
    pers, seas = bl.get("persistence"), bl.get("seasonal")
    dup = (pers in data["arms"] and seas in data["arms"]
           and np.allclose(data["arms"][seas][0], data["arms"][pers][0]))

    ylab, ynote = unit_label(a.channel)
    fig, axes = plt.subplots(3, 3, figsize=(15.0, 9.6), sharex=True, sharey=True)
    for r, (rowname, models) in enumerate(layout):
        for c, cell in enumerate(models):
            ax = axes[r, c]
            if cell is None:
                # Never run, as opposed to run-and-missing ("not available" below): the sweep
                # has no such arm at all. Say which it is -- a blank cell alone reads as a
                # broken figure, and the two absences mean different things about the work.
                ax.set_axis_off()
                ax.text(0.5, 0.5, "not in this sweep", transform=ax.transAxes, ha="center",
                        va="center", fontsize=10, color=MUTED)
                continue
            m, ckey, coltitle = cell
            ax.plot(tt, y[:, k], "-", color=TRUTH, lw=1.2, label="ground truth", zorder=3)
            if m in data["arms"]:
                mu, sg, ors = data["arms"][m]
                idx, val, sig = rolling(mu, sg, ors, H)
                t = idx / fps
                col = color_of(ckey)
                if sig is not None:
                    ax.fill_between(t, val[:, k] - 2 * sig[:, k], val[:, k] + 2 * sig[:, k],
                                    color=col, alpha=0.20, lw=0, zorder=2,
                                    label="±2σ")
                ax.plot(t, val[:, k], "-", color=col, lw=1.3, zorder=4,
                        label=f"{H / fps:g} s forecast")
            else:
                ax.text(0.5, 0.5, "not available", transform=ax.transAxes, ha="center",
                        va="center", fontsize=10, color=MUTED)
            note = coltitle
            if m == seas and dup:
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
