"""Filmstrip of the raw 16x16 pressure map for OpenTouch clips, with CoP overlaid.

The companion to plot_opentouch_fcop.py: that one shows the three summary channels the
models actually forecast, this one shows the field they were reduced from, so the
summary can be checked against the thing it summarises. Selection is imported from that
script rather than reimplemented, so `--actions ... --n ...` picks the SAME clips and the
two figures line up; `--idx` takes the indices it prints if you want an exact set.

Descriptive only -- no model, no forecast, no R^2. Same Layer-2 constraint as the
companion script: this verifies the trait prior, it must not be used to reclassify
actions (SESSION_LOG 2026-08-12, Q3).

CoP overlay convention, taken from moments() in scripts/opentouch/extract_opentouch.py: cx weights
xs = linspace(-1,1,W) along the COLUMN axis and cy weights ys = linspace(-1,1,H) along
the ROW axis, so col = (cx+1)/2*(W-1) and row = (cy+1)/2*(H-1) under imshow's default
origin="upper". If the red dot ever sits off the pressed region, that mapping -- not the
sensor -- is what to re-check.

Note the grid has only 169 live taxels of 256; dead cells read ~0 and simply stay dark.
No baseline correction has been applied anywhere upstream (deliberate).

    python scripts/opentouch/plot_opentouch_tactile_map.py --actions "holding,picking up" --n 2
    python scripts/opentouch/plot_opentouch_tactile_map.py --idx 12,345 --frames 10
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

# Clips whose annotated event indices are known to disagree with their own signal
# (2026-08-16 audit). The labels are fine; these three indices are not, so drawing them
# would put a "peak" marker where the peak is not.
def _mismatched(path="data/opentouch_peak_mismatch.json"):
    try:
        with open(path) as f:
            return set(json.load(f)["clips"])
    except (OSError, KeyError, ValueError):
        return set()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from scripts.opentouch.plot_opentouch_fcop import load_manifest, pick  # noqa: E402


def frame_set(r, T, n):
    """n evenly spaced frames, plus the segmentation's onset/peak/post when present."""
    frames = {int(round(v)) for v in np.linspace(0, T - 1, n)}
    tags = {}
    for key, tag in (("onset_idx", "onset"), ("peak_idx", "peak"), ("post_idx", "post")):
        try:
            v = int(r.get(key))
        except (TypeError, ValueError):
            continue
        if 0 <= v < T:
            frames.add(v)
            tags[v] = tag
    return sorted(frames), tags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=os.path.expanduser("~/opentouch/cache"))
    ap.add_argument("--idx", help="comma-separated clip indices (as printed by "
                                  "plot_opentouch_fcop.py)")
    ap.add_argument("--actions", help="comma-separated actions; same selection rule as "
                                      "plot_opentouch_fcop.py")
    ap.add_argument("--n", type=int, default=2, help="clips per action")
    ap.add_argument("--frames", type=int, default=8, help="evenly spaced frames per clip")
    ap.add_argument("--out", default="docs/opentouch/exploratory/opentouch_tactile_map.png")
    args = ap.parse_args()
    if not args.idx and not args.actions:
        raise SystemExit("give --idx or --actions")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = load_manifest(args.cache)
    bad_idx = _mismatched()
    if args.idx:
        want = [int(i) for i in args.idx.split(",") if i.strip()]
        by_idx = {r["idx"]: r for r in rows}
        missing = [i for i in want if i not in by_idx]
        if missing:
            raise SystemExit(f"no manifest entry for idx {missing}")
        picked = [(by_idx[i].get("action", "?"), by_idx[i]) for i in want]
    else:
        acts = [a.strip() for a in args.actions.split(",") if a.strip()]
        picked = [(a, r) for a in acts for r in pick(rows, a, args.n)]
        if not picked:
            raise SystemExit(f"no clips for {acts}")

    # Load everything first: the frame count varies per clip (onset/peak/post may
    # coincide with an evenly spaced frame), and the last column is an F(t) curve, so
    # the grid width cannot be taken from the first clip alone.
    loaded = []
    for action, r in picked:
        path = os.path.join(args.cache, f"clip_{r['idx']}.npy")
        if not os.path.exists(path):
            raise SystemExit(
                f"{path} missing. The cache was built with --no-clips, so only the "
                f"summary channels exist; the raw maps need a re-extract without it.")
        p = np.load(path).astype(np.float32)[:, 0]      # (T,16,16), hand axis is 1
        st = np.load(os.path.join(args.cache, f"state_{r['idx']}.npy"))
        frames, tags = frame_set(r, p.shape[0], args.frames)
        if r["idx"] in bad_idx:
            # Flagged by the 2026-08-16 audit: these indices disagree with the clip's own
            # argmax(F), so the frames stay but the event captions do not.
            tags = {}
        loaded.append((action, r, p, st, frames, tags))

    ncol = max(len(f) for *_, f, _ in loaded) + 1       # +1 for the F(t) curve
    fig, axes = plt.subplots(len(loaded), ncol, squeeze=False,
                             figsize=(1.5 * ncol, 2.35 * len(loaded)))
    for row_i, (action, r, p, st, frames, tags) in enumerate(loaded):
        fps = r.get("fps_est") or 30.0
        # Per-clip scale (99.5th pct, not max) so one hot taxel cannot flatten the field.
        vmax = float(np.percentile(p, 99.5)) or float(p.max()) or 1.0
        H, W = p.shape[1], p.shape[2]
        Ft = p.reshape(len(p), -1).sum(1)         # F from the maps themselves
        for col in range(ncol - 1):
            ax = axes[row_i][col]
            ax.set_xticks([]); ax.set_yticks([])
            if col >= len(frames):
                ax.axis("off")
                continue
            f = frames[col]
            ax.imshow(p[f], cmap="magma", vmin=0, vmax=vmax, interpolation="nearest")
            F, cx, cy = st[f, 0, 0], st[f, 0, 1], st[f, 0, 2]
            if F > 0:
                ax.plot((cx + 1) / 2 * (W - 1), (cy + 1) / 2 * (H - 1),
                        "o", ms=4, mfc="none", mec="cyan", mew=1.2)
            tag = tags.get(f, "")
            # F is a SUM over taxels, so the frame of maximum F is not the visually
            # brightest one: a wide light contact outweighs a concentrated poke. Print
            # F and the loaded-cell count so the eye is not left to infer it.
            area = int((p[f] > 0.05 * vmax).sum())
            ax.set_title(f"{f / fps:.2f}s{' ' + tag if tag else ''}\n"
                         f"F={F:.0f} n={area}", fontsize=6.5,
                         color="crimson" if tag == "peak" else "black")
            if col == 0:
                flag = "\n[event idx unreliable]" if r["idx"] in bad_idx else ""
                ax.set_ylabel(f"[{r['idx']}] {action}\n"
                              f"{r.get('object_name', '') or '?'}{flag}", fontsize=7)

        # Last column: the whole F(t) curve with the sampled frames marked, so whether
        # peak_idx really sits at the maximum is read off the curve, not guessed from
        # how bright a heat map looks under a percentile colour scale.
        ax = axes[row_i][ncol - 1]
        t = np.arange(len(Ft)) / fps
        ax.plot(t, Ft, lw=0.9, color="0.3")
        for f in frames:
            ax.axvline(f / fps, color="0.75", lw=0.5, zorder=0)
        amax = int(np.argmax(Ft))
        ax.plot(amax / fps, Ft[amax], "v", ms=5, color="tab:blue")
        pk = tags and [k for k, v in tags.items() if v == "peak"]
        if pk:
            ax.axvline(pk[0] / fps, color="crimson", ls="--", lw=1.0)
        ax.set_title(f"F(t)  argmax@{amax / fps:.2f}s", fontsize=6.5)
        ax.tick_params(labelsize=6)
        ax.set_xlabel("s", fontsize=6)

    fig.suptitle("OpenTouch raw pressure (16x16), cyan ring = CoP; per-clip colour "
                 "scale.  F = summed pressure, n = cells above 5% of that scale.  "
                 "Last column: F(t), red dashed = annotated peak, blue triangle = argmax",
                 fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=2.2)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=140)
    print(f"wrote {args.out}  ({len(picked)} clips)")


if __name__ == "__main__":
    raise SystemExit(main())
