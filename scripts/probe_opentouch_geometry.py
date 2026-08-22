"""READ-ONLY probe: can OpenTouch CoP be lifted into a world frame?

Answers the three falsifiable checks from SESSION_LOG 2026-08-15续5(e). Writes NOTHING,
fits nothing, and produces no R^2 — it only reports what the data contains, so it cannot
contaminate the D1..D9 decision chain.

Two tiers, because the shards are deleted right after extraction
(stream_opentouch.sh:107, download_own_copies.sh:110) and only the cache survives:

  TIER A (--cache, FREE)   pose_<N>.npy is already on disk. Settles check (2):
      * is it (T,21,3) [per-frame] or (21,3) [one pose per clip]?  extract_opentouch.py:10
        claims the former, probe_opentouch.py:6 records the latter — they cannot both be right.
      * does T match the pressure clip's T?  Per-frame lifting needs frame-for-frame alignment.
      * does the WRIST landmark move across a clip?  A wrist pinned at ~0 means the landmarks
        are expressed in a wrist-local frame and carry no world position at all.
      * coordinate magnitude — metres/millimetres (world-ish) vs [-1,1] (normalised).

  TIER B (--shard, needs ONE re-downloaded shard, ~561 MB) settles checks (1) and (3):
      * what is the top-level `calibration` group actually made of — pressure curves only,
        or taxel GEOMETRY?  Nobody has ever opened it (SESSION_LOG:510 lists the key, not
        its contents). Geometry here is the difference between "possible" and "impossible":
        without a taxel -> hand-surface map there is no lift, full stop.
      * do `camera_poses` + `transform_slam_to_rgb` close the SLAM->world chain?

ANY of these coming back negative kills the 3-D plan; that is the point of running it first.

Usage:
    python scripts/probe_opentouch_geometry.py --cache ~/opentouch/cache
    python scripts/probe_opentouch_geometry.py --shard /path/to/office_csail_p2.hdf5
    python scripts/probe_opentouch_geometry.py --cache ~/opentouch/cache --shard <f>
"""
from __future__ import annotations

import argparse
import glob
import os
from collections import Counter

import numpy as np


# ---------------------------------------------------------------- TIER A: cache only
def tier_a(cache: str, n_detail: int) -> None:
    print("=" * 78)
    print(f"TIER A — pose_*.npy in {cache}   (no download needed)")
    print("=" * 78)

    poses = sorted(glob.glob(os.path.join(cache, "pose_*.npy")),
                   key=lambda p: int(os.path.basename(p)[5:-4]))
    clips = {int(os.path.basename(p)[5:-4]) for p in glob.glob(os.path.join(cache, "clip_*.npy"))}
    mf = os.path.join(cache, "manifest.jsonl")
    n_manifest = sum(1 for _ in open(mf)) if os.path.exists(mf) else 0

    if not poses:
        print("  NO pose_*.npy FOUND.")
        print("  -> check (2) answered NEGATIVE by absence: no hand pose in the cache at all.")
        print("     Either the shards carried none, or extract_opentouch.py's POSE_KEYS missed")
        print("     them. Per-frame world lifting is impossible without a re-extract.")
        return
    print(f"  pose files: {len(poses)}   clip files: {len(clips)}   manifest rows: {n_manifest}")

    # ---- (2a) shape: per-frame or one-per-clip? -------------------------------------
    shapes = Counter()
    for p in poses:
        shapes[np.load(p, mmap_mode="r").shape] += 1
    print("\n  --- (2a) SHAPES ---")
    for s, n in shapes.most_common(10):
        print(f"    {str(s):<20} x{n}")
    ndims = {len(s) for s in shapes}
    if ndims == {2}:
        print("    VERDICT: (21,3) — ONE pose per clip. probe_opentouch.py:6 was right,")
        print("             extract_opentouch.py:10's (T,21,3) is WRONG (docstring bug).")
        print("             => a per-frame world-space CoP TRAJECTORY is IMPOSSIBLE from the cache.")
    elif ndims == {3}:
        print("    VERDICT: (T,21,3) — per-frame. extract_opentouch.py:10 was right.")
    else:
        print(f"    VERDICT: MIXED rank {sorted(ndims)} — inspect before trusting either docstring.")

    # ---- (2b) does pose T match pressure T? -----------------------------------------
    print("\n  --- (2b) LENGTH ALIGNMENT vs clip_<N>.npy ---")
    if 3 not in ndims:
        print("    skipped (no time axis to align).")
    else:
        agree = mismatch = missing = 0
        examples = []
        for p in poses:
            idx = int(os.path.basename(p)[5:-4])
            if idx not in clips:
                missing += 1
                continue
            tp = np.load(p, mmap_mode="r").shape[0]
            tc = np.load(os.path.join(cache, f"clip_{idx}.npy"), mmap_mode="r").shape[0]
            if tp == tc:
                agree += 1
            else:
                mismatch += 1
                if len(examples) < 5:
                    examples.append((idx, tp, tc))
        print(f"    T(pose)==T(pressure): {agree}    differ: {mismatch}    no clip file: {missing}")
        for idx, tp, tc in examples:
            print(f"      e.g. clip {idx}: pose T={tp}  pressure T={tc}")
        if mismatch:
            print("    WARNING: lengths differ -> frame-for-frame alignment needs the timestamps,")
            print("             which the cache does NOT store. That is a re-extract, not a fix.")

    # ---- (2c) wrist motion + magnitude: world frame or wrist-local? ------------------
    print("\n  --- (2c) FRAME OF REFERENCE (wrist = landmark 0) ---")
    print("    A wrist pinned near 0 with ~0 range => WRIST-LOCAL frame (no world position).")
    print("    A wrist that translates over the clip => some world/anchored frame.")
    shown = 0
    all_ptp, all_absmax = [], []
    for p in poses:
        lm = np.load(p).astype(np.float64)
        if lm.ndim == 2:                      # (21,3) — one pose, no motion to measure
            all_absmax.append(np.abs(lm).max())
            if shown < n_detail:
                print(f"    {os.path.basename(p):<14} (21,3)  wrist={np.round(lm[0], 4)}  "
                      f"|coord|max={np.abs(lm).max():.4f}")
                shown += 1
            continue
        if lm.ndim != 3 or lm.shape[0] < 2:
            continue
        wrist = lm[:, 0, :]                   # (T,3)
        ptp = np.ptp(wrist, axis=0)
        all_ptp.append(ptp)
        all_absmax.append(np.abs(lm).max())
        if shown < n_detail:
            print(f"    {os.path.basename(p):<14} T={lm.shape[0]:<5} wrist[0]={np.round(wrist[0], 4)}  "
                  f"wrist ptp={np.round(ptp, 4)}  |coord|max={np.abs(lm).max():.3f}")
            shown += 1
    if all_ptp:
        P = np.asarray(all_ptp)
        print(f"\n    wrist range over {len(P)} clips — median ptp per axis: {np.round(np.median(P, 0), 5)}")
        print(f"                                        90th pct per axis: {np.round(np.percentile(P, 90, 0), 5)}")
        still = int((P.max(axis=1) < 1e-6).sum())
        print(f"    clips whose wrist NEVER moves: {still}/{len(P)}")
        if still == len(P):
            print("    VERDICT: wrist is static in every clip => WRIST-LOCAL frame.")
            print("             Landmarks give finger articulation only; NO world position.")
        else:
            print("    VERDICT: wrist translates => landmarks carry position in SOME anchored")
            print("             frame. Whether that frame is the Aria SLAM world still needs TIER B.")
    if all_absmax:
        a = np.asarray(all_absmax)
        print(f"    |coord| max: median {np.median(a):.4f}  p95 {np.percentile(a, 95):.4f}  max {a.max():.4f}")
        print("      ~<=1      -> normalised units (needs a scale before it means anything metric)")
        print("      ~0.05-2   -> metres      |   ~50-2000 -> millimetres")


# ---------------------------------------------------------------- TIER B: one shard
def _walk(g, prefix: str, depth: int, max_depth: int, max_print: int, printed: list) -> None:
    import h5py
    for k in g:
        if printed[0] >= max_print:
            return
        item = g[k]
        path = f"{prefix}/{k}"
        if isinstance(item, h5py.Group):
            print(f"    {'  ' * depth}[grp] {path}   attrs={dict(item.attrs)}")
            printed[0] += 1
            if depth < max_depth:
                _walk(item, path, depth + 1, max_depth, max_print, printed)
        else:
            val = ""
            if item.size and item.size <= 24:
                try:
                    val = f"  = {np.array2string(item[()], precision=4, threshold=24)}"
                except Exception:
                    val = ""
            print(f"    {'  ' * depth}[dst] {path}  shape={item.shape} dtype={item.dtype}"
                  f"  attrs={dict(item.attrs)}{val}")
            printed[0] += 1


def tier_b(shard: str, n_clips: int) -> None:
    import h5py
    print()
    print("=" * 78)
    print(f"TIER B — {shard}")
    print("=" * 78)
    with h5py.File(shard, "r") as f:
        print(f"  top-level keys: {list(f.keys())}")
        print(f"  file attrs    : {dict(f.attrs)}")

        # ---- (1) the calibration group — never opened before ------------------------
        print("\n  --- (1) `calibration` — GEOMETRY or just pressure curves? ---")
        if "calibration" not in f:
            print("    ABSENT in this shard.")
            print("    -> check (1) NEGATIVE: no taxel geometry anywhere => NO 3-D lift. Stop here.")
        else:
            cal = f["calibration"]
            if isinstance(cal, h5py.Dataset):
                print(f"    calibration is a DATASET shape={cal.shape} dtype={cal.dtype} "
                      f"attrs={dict(cal.attrs)}")
                if cal.size <= 64:
                    print(f"    value = {cal[()]}")
                else:
                    print(f"    first rows:\n{np.asarray(cal[:min(5, cal.shape[0])])}")
                print("    INTERPRET: a (256,2) / (16,16,2) shape smells like PER-TAXEL 2-D layout;")
                print("               (256,3)/(16,16,3) like 3-D positions -> that is the map we need.")
                print("               A 1-D curve or a couple of scalars = pressure calibration only.")
            else:
                _walk(cal, "calibration", 1, 3, 200, [0])
                print("    INTERPRET: look for anything shaped (...,3) over 256/169 taxels — that")
                print("               would be the taxel->hand-surface map. Gains/offsets/curves are not.")

        # ---- (3) SLAM -> world chain ------------------------------------------------
        print("\n  --- (3) SLAM -> world chain ---")
        if "transform_slam_to_rgb" in f:
            t = f["transform_slam_to_rgb"]
            print(f"    transform_slam_to_rgb shape={t.shape} dtype={t.dtype}")
            print(f"{np.asarray(t[()])}")
        else:
            print("    transform_slam_to_rgb ABSENT.")

        # ---- per-clip fields ---------------------------------------------------------
        print("\n  --- per-clip fields (hand_landmarks / camera_poses / timestamps) ---")
        data = f["data"]
        for cid in list(data.keys())[:n_clips]:
            g = data[cid]
            print(f"    clip {cid}:")
            for k in g:
                d = g[k]
                if isinstance(d, h5py.Group):
                    print(f"      {k:<22} [group] {list(d.keys())[:8]}")
                    continue
                print(f"      {k:<22} shape={d.shape} dtype={d.dtype}")
            if "hand_landmarks" in g:
                lm = np.asarray(g["hand_landmarks"][()], dtype=np.float64)
                print(f"      -> hand_landmarks ndim={lm.ndim} shape={lm.shape}")
                if lm.ndim == 3:
                    w = lm[:, 0, :]
                    print(f"         wrist[0]={np.round(w[0], 4)}  wrist ptp={np.round(np.ptp(w, 0), 4)}")
                else:
                    print(f"         wrist={np.round(lm[0], 4)} (single pose — no trajectory)")
                print(f"         |coord| max={np.abs(lm).max():.4f}")
            if "camera_poses" in g:
                cp = np.asarray(g["camera_poses"][()])
                print(f"      -> camera_poses shape={cp.shape} dtype={cp.dtype}")
                print(f"         first entry:\n{np.array2string(cp[0], precision=4)}")
                if cp.ndim == 3 and cp.shape[-2:] == (4, 4):
                    tr = cp[:, :3, 3]
                    print(f"         translation ptp over clip = {np.round(np.ptp(tr, 0), 4)}"
                          "   (4x4 SE(3) -> world chain is CLOSED on the camera side)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default=None, help="OpenTouch cache dir (tier A)")
    ap.add_argument("--shard", default=None, help="one .hdf5 shard (tier B)")
    ap.add_argument("--n-detail", type=int, default=8, help="tier-A per-file lines to print")
    ap.add_argument("--n-clips", type=int, default=2, help="tier-B clips to dump")
    args = ap.parse_args()
    if not args.cache and not args.shard:
        ap.error("give --cache and/or --shard")
    if args.cache:
        tier_a(os.path.expanduser(args.cache), args.n_detail)
    if args.shard:
        tier_b(os.path.expanduser(args.shard), args.n_clips)
    print("\n[done] read-only probe; nothing was written.")


if __name__ == "__main__":
    main()
