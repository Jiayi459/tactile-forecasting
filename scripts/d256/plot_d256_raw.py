#!/usr/bin/env python3
"""Descriptive plots of d256's raw signal -- the two figures OpenTouch has, for two gloves.

Purely descriptive: no model, no fitting, no split. The d256 counterparts of
docs/opentouch/exploratory/{opentouch_fcop.png, opentouch_tactile_map.png}.

  --what fcop   physical state per recording: F, CoP x, CoP y against time, BOTH hands
                overlaid on each axis. Overlaid rather than given six columns because the
                question these answer is whether the two gloves move together, and that is
                unreadable across separate panels.
  --what map    filmstrip of the raw 32x32 pressure of both gloves at sampled times, CoP
                overlaid, with F(t) alongside. Needs --root (the clips) because the state
                cache stores moments, not grids; recordings are rebuilt exactly as
                extract_d256_states.py does, so what is drawn is what the model was fed.

Read the F panels for the DC pedestal. The tactile values arrive pre-scaled to ~[0,1] with a
floor well above zero and no baseline correction is applied (OQ-D2), so F is expected to sit
on a large constant. If it does, persistence is strong by construction and a low skill on F
is a statement about the target, not about the model.

Usage:
    python scripts/d256/plot_d256_raw.py --what fcop --states data/d256_states
    python scripts/d256/plot_d256_raw.py --what map --states data/d256_states --root ~/forcevision
    python scripts/d256/plot_d256_raw.py --what fcop --classes 3,10,18 --out docs/d256/x.png
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src import d256  # noqa: E402

FPS = 6.0
CHANNELS = [("F (total force, a.u.)", 0), ("CoP x  [-1,1]", 1), ("CoP y  [-1,1]", 2)]
HANDS = [("left", 0, "tab:blue"), ("right", 1, "tab:red")]


def manifest(states: str) -> list[dict]:
    with open(os.path.join(states, "manifest.jsonl")) as fh:
        return [json.loads(l) for l in fh if l.strip()]


def pick(rows, classes, per_class, subject):
    """One (or a few) recordings per class -- the longest, which are the most informative."""
    if subject:
        rows = [r for r in rows if r["subject"] == subject]
    if classes:
        want = {int(c) for c in classes.split(",")}
        rows = [r for r in rows if r["label_idx"] in want]
    by = {}
    for r in rows:
        by.setdefault(r["label_idx"], []).append(r)
    out = []
    for c in sorted(by):
        out += sorted(by[c], key=lambda r: -r["T"])[:per_class]
    return out


def _plt():
    """Import matplotlib only when something is actually drawn.

    --what report draws nothing, and it is the mode that matters most right now (it decides
    whether an F skill number is readable at all). Importing matplotlib at module scope made
    that mode fail on any environment without it, for a dependency it never uses.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_fcop(rows, states, out):
    plt = _plt()
    n = len(rows)
    fig, axes = plt.subplots(n, 3, figsize=(15, 2.1 * n), squeeze=False)
    for i, r in enumerate(rows):
        st = np.load(os.path.join(states, f"state_{r['idx']}.npy"))       # (T,2,6)
        t = np.arange(len(st)) / FPS
        for j, (title, m) in enumerate(CHANNELS):
            ax = axes[i][j]
            for hname, h, colour in HANDS:
                ax.plot(t, st[:, h, m], lw=0.7, color=colour, label=hname, alpha=0.85)
            if i == 0:
                ax.set_title(title)
            if j == 0:
                ax.set_ylabel(f"[{r['label_idx']}] {r['subject']}\n{r['label'][:26]}",
                              fontsize=7)
            if i == n - 1:
                ax.set_xlabel("time (s)")
            ax.tick_params(labelsize=6)
            if i == 0 and j == 0:
                ax.legend(fontsize=6, loc="upper right")
    fig.suptitle("d256 physical state per recording, both gloves "
                 "(blue = left, red = right; 6 Hz, no baseline correction)", y=0.999)
    fig.tight_layout()
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"  wrote {out}  ({n} recordings)")


def rebuild_grids(root, r):
    """Rebuild one recording's (T,2,32,32) grids from the clips it was folded from."""
    from scripts.d256.extract_d256_states import rebuild   # same folding, same guarantees
    cell = os.path.join(d256.root_of(root), r["group"], r["orig_split"],
                        r["subject"], str(r["session"]))
    n = len([f for f in os.listdir(cell) if f.endswith(".p")])
    for streams, first in rebuild(cell, n, verify=False):
        if first == r.get("first_clip", 0):
            return np.stack([streams["tactile-glove-left"],
                             streams["tactile-glove-right"]], axis=1)
    raise ValueError(f"recording {r['idx']}: no segment starting at clip {r.get('first_clip')}")


def plot_map(rows, states, root, out, n_frames=8):
    plt = _plt()
    n = len(rows)
    fig, axes = plt.subplots(n, n_frames + 1, figsize=(2.0 * (n_frames + 1), 2.3 * n),
                             squeeze=False)
    for i, r in enumerate(rows):
        grids = rebuild_grids(root, r)                                   # (T,2,32,32)
        st = np.load(os.path.join(states, f"state_{r['idx']}.npy"))
        T = len(grids)
        idxs = np.linspace(0, T - 1, n_frames).astype(int)
        # One colour scale per recording, so frames are comparable within a row and the
        # pedestal is visible as a uniformly warm field rather than normalised away.
        vmin, vmax = float(grids.min()), float(grids.max())
        for j, k in enumerate(idxs):
            ax = axes[i][j]
            pair = np.concatenate([grids[k, 0], np.full((32, 2), vmax), grids[k, 1]], axis=1)
            ax.imshow(pair, cmap="YlOrRd", vmin=vmin, vmax=vmax, interpolation="nearest",
                      aspect="equal")
            for hname, h, colour in HANDS:
                cx = (st[k, h, 1] + 1) / 2 * 31 + (0 if h == 0 else 34)
                cy = (st[k, h, 2] + 1) / 2 * 31
                ax.plot(cx, cy, "o", mfc="none", mec="cyan", ms=6, mew=1.4)
            ax.set_title(f"{k / FPS:.1f}s\nF={st[k, 0, 0]:.0f}/{st[k, 1, 0]:.0f}", fontsize=6)
            ax.set_xticks([]); ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(f"[{r['label_idx']}] {r['subject']}\n{r['label'][:22]}",
                              fontsize=6)
        ax = axes[i][n_frames]
        t = np.arange(T) / FPS
        for hname, h, colour in HANDS:
            ax.plot(t, st[:, h, 0], lw=0.7, color=colour, label=hname)
        for k in idxs:
            ax.axvline(k / FPS, color="0.8", lw=0.5)
        ax.set_title("F(t)", fontsize=7); ax.tick_params(labelsize=5)
        ax.set_xlabel("s", fontsize=6)
        if i == 0:
            ax.legend(fontsize=5)
    fig.suptitle("d256 raw pressure, left | right glove (32x32 each), cyan ring = CoP; "
                 "per-recording colour scale. Last column: F(t) for both hands", y=0.999)
    fig.tight_layout()
    fig.savefig(out, dpi=105, bbox_inches="tight")
    print(f"  wrote {out}  ({n} recordings x {n_frames} frames)")


def pedestal_report(rows, states) -> str:
    """How much of each channel is a constant?

    Not a decoration. OQ-D2 chose NOT to baseline-correct, so F carries whatever DC the sensor
    and the pre-scaling leave behind. If sigma/|mean| on F is a percent or two, then almost all
    of the target is a constant, persistence is near-unbeatable by construction, and a low
    skill on F says something about the target rather than about the model -- exactly the trap
    OpenTouch's D1 chased. Report it before reading any skill number.
    """
    per = {name: {h: [] for _, h, _ in HANDS} for name, _ in CHANNELS}
    for r in rows:
        st = np.load(os.path.join(states, f"state_{r['idx']}.npy"))
        for name, m in CHANNELS:
            for _, h, _ in HANDS:
                x = st[:, h, m]
                # np.ptp, not x.ptp(): the method was removed from ndarray in NumPy 2.0, and the
                # cluster env and this one are not on the same major version.
                per[name][h].append((float(x.mean()), float(x.std()), float(np.ptp(x))))
    out = [f"per-channel over {len(rows)} recordings (no baseline correction, OQ-D2):",
           f"  {'channel':22s} {'hand':>5s} {'mean':>12s} {'std':>10s} {'ptp':>10s} {'std/|mean|':>11s}"]
    for name, _ in CHANNELS:
        for hname, h, _ in HANDS:
            a = np.array(per[name][h])
            mean, std, ptp = a[:, 0].mean(), a[:, 1].mean(), a[:, 2].mean()
            rel = std / abs(mean) if abs(mean) > 1e-9 else float("nan")
            flag = "   <- almost all constant" if name.startswith("F") and rel < 0.05 else ""
            out.append(f"  {name:22s} {hname:>5s} {mean:12.4f} {std:10.4f} {ptp:10.4f} "
                       f"{rel:10.1%}{flag}")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", choices=["fcop", "map", "both", "report"], default="both",
                    help="'report' prints the pedestal table only and draws nothing -- the "
                         "table is what decides whether an F skill number is readable, and "
                         "re-rendering the whole corpus just to see it is wasteful")
    ap.add_argument("--states", default=os.path.join("data", "d256_states"))
    ap.add_argument("--root", default=os.path.join(os.path.expanduser("~"), "forcevision"),
                    help="raw clips; needed for --what map")
    ap.add_argument("--classes", default=None, help="comma list of label_idx (default: all)")
    ap.add_argument("--per-class", type=int, default=1)
    ap.add_argument("--subject", default=None)
    ap.add_argument("--outdir", default=os.path.join("docs", "d256", "exploratory"))
    ap.add_argument("--out", default=None, help="explicit output file (single --what only)")
    args = ap.parse_args()

    all_rows = manifest(args.states)
    rows = pick(all_rows, args.classes, args.per_class, args.subject)
    if not rows:
        sys.exit("no recordings selected -- check --classes/--subject")

    # The pedestal table is reported over the WHOLE corpus, not the plotted subset: it is
    # evidence about the dataset, and a handful of hand-picked recordings is not.
    report = pedestal_report(all_rows, args.states)
    header = f"corpus: {len(all_rows)} recordings"
    print(header); print(); print(report); print()

    # Write it, do not just print it. This table is what decides whether an F skill number can
    # be read at all (OQ-D2), so it has to be a committable artefact that travels with the
    # figures -- terminal scrollback does not survive a push.
    os.makedirs(args.outdir, exist_ok=True)
    txt = os.path.join(args.outdir, "d256_pedestal.txt")
    with open(txt, "w") as fh:
        fh.write(header + "\n\n" + report + "\n")
    print(f"  wrote {txt}")
    if args.what == "report":
        return
    os.makedirs(args.outdir, exist_ok=True)
    print(f"plotting {len(rows)} selected recordings")

    if args.what == "report":
        return
    if args.what in ("fcop", "both"):
        plot_fcop(rows, args.states,
                  args.out or os.path.join(args.outdir, "d256_fcop.png"))
    if args.what in ("map", "both"):
        plot_map(rows, args.states, args.root,
                 args.out or os.path.join(args.outdir, "d256_tactile_map.png"))


if __name__ == "__main__":
    main()
