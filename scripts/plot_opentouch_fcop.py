"""Plot F and CoP against time for OpenTouch clips, grouped by action.

Descriptive only -- no model, no forecast, no R^2. This is the Layer-2 "manipulation
check" of the trait design (SESSION_LOG 2026-08-12, Q3): it shows what the physical
signal of a smooth action actually looks like next to an abrupt one.

  IT MUST NOT BE USED TO RECLASSIFY ACTIONS. The trait table in src/opentouch/trait.py
  is a semantic prior, frozen before scoring; deciding smooth vs abrupt by looking at
  the signals would make the class definition a function of the data it is later used
  to explain. Layer 2 verifies the prior, Layer 3 (contentious subset) absorbs doubt.

scripts/plot_signal_decomposition.py does the analogous plot for ActionSense but hard-
codes data/actionsense_states and the `label` manifest field; the OpenTouch cache
mirrors the file layout but names the field `action`, hence this separate entry point.

    python scripts/plot_opentouch_fcop.py --actions holding,sliding,"picking up" --n 3
    python scripts/plot_opentouch_fcop.py --actions lifting --n 5 --out docs/lifting.png
"""
from __future__ import annotations

import argparse
import json
import os

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

CHANNELS = [(0, "F (total force, a.u.)"), (1, "CoP x  [-1,1]"), (2, "CoP y  [-1,1]")]


def load_manifest(cache):
    path = os.path.join(cache, "manifest.jsonl")
    if not os.path.exists(path):
        raise SystemExit(f"no manifest at {path} (pass --cache)")
    return [json.loads(l) for l in open(path)]


def pick(rows, action, n):
    """First n clips of an action, longest first so the plots show real dynamics."""
    hits = [r for r in rows if r.get("action", "").strip().lower() == action.lower()]
    return sorted(hits, key=lambda r: -r.get("T", 0))[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=os.path.expanduser("~/opentouch/cache"))
    ap.add_argument("--actions", required=True,
                    help="comma-separated action labels, e.g. holding,sliding")
    ap.add_argument("--n", type=int, default=3, help="clips per action")
    ap.add_argument("--out", default="docs/opentouch/exploratory/opentouch_fcop.png")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = load_manifest(args.cache)
    bad_idx = _mismatched()
    actions = [a.strip() for a in args.actions.split(",") if a.strip()]
    picked = [(a, r) for a in actions for r in pick(rows, a, args.n)]
    if not picked:
        avail = sorted({r.get("action", "") for r in rows})
        raise SystemExit(f"no clips for {actions}\navailable actions: {avail}")

    fig, axes = plt.subplots(len(picked), 3, squeeze=False,
                             figsize=(13, 2.4 * len(picked)), sharex="row")
    for row_i, (action, r) in enumerate(picked):
        st = np.load(os.path.join(args.cache, f"state_{r['idx']}.npy"))
        fps = r.get("fps_est") or 30.0
        t = np.arange(st.shape[0]) / fps
        for col, (ch, name) in enumerate(CHANNELS):
            ax = axes[row_i][col]
            ax.plot(t, st[:, 0, ch], lw=0.9)
            # Clips are segmented around a pressure peak; onset/peak/post make the
            # event structure visible, which is exactly what smooth vs abrupt is about --
            # except on the clips the 2026-08-16 audit flagged, where those indices
            # disagree with the clip's own argmax(F). Drawing them there would mark a
            # "peak" where the peak is not, so they are omitted and the row says so.
            for key, style in (() if r["idx"] in bad_idx else
                               (("onset_idx", ":"), ("peak_idx", "--"), ("post_idx", ":"))):
                v = r.get(key)
                try:
                    v = int(v)
                except (TypeError, ValueError):
                    continue
                if 0 <= v < st.shape[0]:
                    ax.axvline(v / fps, color="k", ls=style, lw=0.7, alpha=0.5)
            if row_i == 0:
                ax.set_title(name, fontsize=10)
            if col == 0:
                flag = "  [event idx unreliable]" if r["idx"] in bad_idx else ""
                ax.set_ylabel(f"{action}\n{r.get('object_name', '') or '?'}{flag}",
                              fontsize=8)
            if row_i == len(picked) - 1:
                ax.set_xlabel("time (s)")
            ax.tick_params(labelsize=7)

    fig.suptitle("OpenTouch physical state per clip "
                 "(dashed = pressure peak, dotted = onset/post)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"wrote {args.out}  ({len(picked)} clips)")
    for action, r in picked:
        print(f"  idx={r['idx']:5d} {action:14s} T={r.get('T')} "
              f"fps={r.get('fps_est')} object={r.get('object_name', '')}")


if __name__ == "__main__":
    raise SystemExit(main())
