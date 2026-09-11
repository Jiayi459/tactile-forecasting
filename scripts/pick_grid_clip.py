"""Pick the recording to draw the 3x3 grid on, and stage its npz files for transfer.

The grid draws ONE recording, so only a handful of npz files ever need to leave the cluster --
not the 2893-clip directory. This scans the prediction directories in place, ranks the clips by
the criteria the ActionSense grid was chosen under, and (with --stage) copies just the winner's
files into one small directory to rsync.

THE CRITERIA, and why each one is there:

  1. every arm the layout expects is present. A clip missing arms draws empty panels, which
     wastes the comparison the grid exists to make.
  2. seasonal-naive is not bit-identical to persistence. On most recordings the autocorrelation
     finds no cycle and seasonal falls back to persistence, and then two of the nine panels are
     the same curve (SESSION_LOG 2026-09-11: 287 of 290 ActionSense recordings). Such a clip
     still plots, but it spends a panel saying nothing.
  3. duration at or above the median. The sigma fan needs several forecast origins to be
     visible as a fan rather than a smudge.

Criterion 1 is a hard filter; 2 and 3 order what survives it. Usage:

    python scripts/pick_grid_clip.py --dataset opentouch \
        --preds runs/preds_d1_map2 --preds runs/preds_d1_pg \
        --top 10 --stage runs/grid_stage_opentouch
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "actionsense"))
from plot_clip_model_grid import LAYOUTS  # noqa: E402


def clip_ids(dirs: list[str]) -> list[int]:
    """Clip numbers present in EVERY directory -- a clip missing from one cannot fill the grid."""
    sets = []
    for d in dirs:
        ids = set()
        for f in os.listdir(d):
            if f.startswith("clip_") and f.endswith(".npz"):
                ids.add(int(f[5:-4]))
        sets.append(ids)
    return sorted(set.intersection(*sets)) if sets else []


def scan(dirs: list[str], clip: int, want: set[str], bl: dict) -> dict | None:
    """Rank one clip while reading as little of it as possible.

    The hard filter only needs to know WHICH arms a file holds, and `z.files` is the zip's
    directory listing -- naming the arms costs no array reads at all. Only three arrays are
    ever decompressed: `y` for the duration, and the two baselines for the degeneracy test.
    Pulling every `mu_*` instead, as the first version did, meant reading ~7.5 GB off the
    cluster filesystem to rank OpenTouch's 2893 clips, most of it to compute nothing.
    """
    keys, n, fps, action = set(), None, None, ""
    pers, seas = bl.get("persistence"), bl.get("seasonal")
    pa = sa = None
    for d in dirs:
        p = os.path.join(d, f"clip_{clip}.npz")
        if not os.path.exists(p):
            continue
        z = np.load(p, allow_pickle=True)
        keys |= {k[3:] for k in z.files if k.startswith("mu_")}
        if n is None:
            n, fps, action = len(z["y"]), float(z["fps"]), str(z["action"])
        if pa is None and pers and f"mu_{pers}" in z.files:
            pa = z[f"mu_{pers}"]
        if sa is None and seas and f"mu_{seas}" in z.files:
            sa = z[f"mu_{seas}"]
    if n is None:
        return None
    degen = pa is not None and sa is not None and pa.shape == sa.shape and np.allclose(sa, pa)
    return dict(clip=clip, secs=n / fps, action=action, have=keys,
                missing=sorted(want - keys), seasonal_degenerate=bool(degen))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(LAYOUTS))
    ap.add_argument("--preds", action="append", required=True)
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--stage", metavar="DIR",
                    help="copy the top clip's npz from each --preds into DIR/<source>/")
    ap.add_argument("--stage-clip", type=int, help="stage this clip instead of the top one")
    a = ap.parse_args()

    layout = LAYOUTS[a.dataset]
    want = {c[0] for _, row in layout for c in row if c}
    bl = {c[1]: c[0] for c in layout[-1][1] if c}

    ids = clip_ids(a.preds)
    print(f"{len(ids)} clips present in all {len(a.preds)} directories")
    rows = [r for r in (scan(a.preds, c, want, bl) for c in ids) if r]
    if not rows:
        raise SystemExit("no readable clips")

    med = float(np.median([r["secs"] for r in rows]))
    full = [r for r in rows if not r["missing"]]
    print(f"median duration {med:.0f} s | {len(full)}/{len(rows)} clips carry all "
          f"{len(want)} arms")
    if not full:
        short = min(rows, key=lambda r: len(r["missing"]))
        print(f"  NO clip has every arm. Closest is clip {short['clip']}, missing "
              f"{short['missing']}. The grid will have empty panels whatever you pick.")
        full = rows

    # rank: seasonal that actually found a cycle first, then duration closest to 1.5x median
    # from above -- long enough for a fan, not so long the traces crush together.
    full.sort(key=lambda r: (r["seasonal_degenerate"], abs(r["secs"] - 1.5 * med)))
    print(f"\n{'clip':>7}  {'secs':>6}  {'seasonal':>9}  action")
    for r in full[:a.top]:
        print(f"{r['clip']:>7}  {r['secs']:>6.0f}  "
              f"{'≡persist' if r['seasonal_degenerate'] else 'has cycle':>9}  {r['action']}")

    if a.stage:
        pick = a.stage_clip if a.stage_clip is not None else full[0]["clip"]
        tot = 0
        for d in a.preds:
            src = os.path.join(d, f"clip_{pick}.npz")
            if not os.path.exists(src):
                continue
            dst = os.path.join(a.stage, os.path.basename(d.rstrip("/")))
            os.makedirs(dst, exist_ok=True)
            shutil.copy2(src, os.path.join(dst, f"clip_{pick}.npz"))
            tot += os.path.getsize(src)
        print(f"\n[staged] clip {pick} -> {a.stage}  ({tot / 1e6:.1f} MB total)")
        print(f"  rsync it, then plot with --clip {pick}")


if __name__ == "__main__":
    main()
