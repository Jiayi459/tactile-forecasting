#!/usr/bin/env python3
"""Extract EgoTouch's physical-state target and freeze the OFFICIAL split.

Layout in, layout out
---------------------
EgoTouch ships one directory per trajectory:

    <root>/<scene>/<task>/<timestamp>/pressure_grids.npz
        left_pressure_grid   (T, 21, 21) float, NaN where the sensor has no taxel
        right_pressure_grid  (T, 21, 21)
        tactile_max          scalar, the value the grids were normalized by (optional)

This writes `data/egotouch_states/` in the SAME layout as `data/actionsense_states/`, so the
frozen harness (`src/actionsense/eval_harness/`) consumes it unchanged:

    state_<idx>.npy   (T, 2, 6)      [F, xbar, ybar, sxx, syy, sxy] per hand, hand order (L, R)
    clip_<idx>.npy    (T, 2, 21, 21) the raw map, unless --no-clips
    manifest.jsonl    one row per trajectory
    splits.json       the OFFICIAL split, translated to recording indices

THREE THINGS DIFFER FROM ACTIONSENSE, EACH DECIDED RATHER THAN DRIFTED INTO
--------------------------------------------------------------------------
1. GRID IS 21x21, NOT 32x32. `physical_state.frame_state` is grid-agnostic -- it normalizes
   the coordinate grids to [-1, 1] -- so F/CoP/second moments are directly comparable across
   the 16 / 21 / 32-wide sensors. Only the *map* arms carry the grid in their shapes; those
   live in a separate fork (`src/egotouch/tactile_map.py`), not here.

2. NO BASELINE CORRECTION -- released-as-is (user decision 2026-09-08, OQ5). `clip_states`
   is called with `baseline_pct=None`. NaN is masked to 0.0 and nothing else is subtracted.
   Rationale: EgoTouch's grids are ALREADY normalized by `tactile_max`, and a grid may mix
   separately-normalized tactile and bending channels, so a per-taxel DC estimate would be
   removing a quantity that is not a sensor offset. d256 settled on the same no-correction
   口径 for its own reasons (2026-08-24, OQ-D2); ActionSense is the odd one out, not this.

   CONSEQUENCE FOR THE PAPER: what `frame_state` calls F is NOT newtons and NOT comparable in
   units to ActionSense's or OpenTouch's F. It is the sum of an already-normalized, possibly
   mixed-modality grid. Call it AGGREGATE NORMALIZED PRESSURE / PRESSURE MASS (P_Sigma) in the
   text. R^2 and skill are scale-free, so cross-sensor *skill* comparison stays valid; a
   cross-sensor comparison of raw F magnitudes would not be.

3. THE SPLIT IS THE AUTHORS', NOT OURS (user decision 2026-09-08, OQ3). `split.json` at the
   dataset root lists absolute .hdf5 paths under train / val / test_seen / test_unseen. The
   join key is the last three path components -- <scene>/<task>/<timestamp> -- which is exactly
   the trajectory directory. `test` in the emitted splits.json is `test_seen`; `test_unseen` is
   carried alongside as an extra key for the generalization arm (the harness ignores keys it
   does not know).

   The official split does NOT cover the release: it lists 2228 .hdf5 files, but only 1896 of
   the 1933 trajectories that ship a pressure_grids.npz appear in it. Those 37 orphans are
   DROPPED (user decision 2026-09-08, OQ-B) -- not written, not indexed, not counted in the
   corpus -- so the corpus is 1896 and the phrase "we use the official split" stays literally
   true. Folding them into train would be the easy move and would quietly make that phrase
   false. They are listed in `unassigned_dropped.json` next to the states so the loss is on
   the record and reversible, and the per-split retention (train 1458/1665, val 174/208,
   test_seen 179/208, test_unseen 85/147) is printed at the end -- test_unseen keeps only
   57.8% of its members, which is the number to quote whenever that arm is reported.

Labels: the manifest `label` is the task directory with underscores turned into spaces
(`grasp_cola` -> `"grasp cola"`). That is not cosmetic -- `eval_harness.splits.parse_label`
takes the first whitespace token as the action and the last as the object, so this one
substitution makes (action, object) grouping, `group_keys`, and the per-action scorer work
with no change to the frozen harness.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.actionsense.physical_state import clip_states  # noqa: E402
from src.tactile_pixel.categories import categorize  # noqa: E402

FPS_RAW = 30.0
GRID = 21
HANDS = 2


def traj_key(path: str) -> str:
    """'/…/Home/grasp_cola/20260412_101136_379.hdf5' -> 'Home/grasp_cola/20260412_101136_379'."""
    parts = path.replace("\\", "/").rstrip("/").split("/")
    stem = parts[-1][:-5] if parts[-1].endswith(".hdf5") else parts[-1]
    return "/".join(parts[-3:-1] + [stem])


def load_official_split(path: str) -> dict[str, set[str]]:
    with open(path) as fh:
        raw = json.load(fh)
    missing = [k for k in ("train", "val", "test_seen", "test_unseen") if k not in raw]
    if missing:
        raise SystemExit(f"split.json is missing {missing}; got keys {sorted(raw)}")
    return {k: {traj_key(p) for p in v} for k, v in raw.items()}


def load_clip(npz_path: str) -> np.ndarray | None:
    """-> (T, 2, 21, 21) float32, NaN masked to 0. None if the file is unusable."""
    with np.load(npz_path) as z:
        if "left_pressure_grid" not in z.files or "right_pressure_grid" not in z.files:
            return None
        left = np.asarray(z["left_pressure_grid"], dtype=np.float32)
        right = np.asarray(z["right_pressure_grid"], dtype=np.float32)
    if left.ndim != 3 or right.ndim != 3 or left.shape[1:] != right.shape[1:]:
        return None
    if left.shape[1:] != (GRID, GRID):
        return None
    t = min(left.shape[0], right.shape[0])          # hands are same-length in practice; be safe
    if t == 0:
        return None
    clip = np.stack([left[:t], right[:t]], axis=1)  # (T, 2, 21, 21), hand order (L, R)
    return np.nan_to_num(clip, nan=0.0, posinf=0.0, neginf=0.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=os.path.join("datasets", "EgoTouch"),
                    help="directory holding <scene>/<task>/<timestamp>/pressure_grids.npz")
    ap.add_argument("--split-json", default=None,
                    help="official split.json (default: <root>/split.json)")
    ap.add_argument("--out", default=os.path.join("data", "egotouch_states"))
    ap.add_argument("--no-clips", action="store_true",
                    help="skip clip_<idx>.npy; the aggregate arm runs without them, the map arms do not")
    ap.add_argument("--limit", type=int, default=None, help="stop after N trajectories (smoke test)")
    args = ap.parse_args()

    split_json = args.split_json or os.path.join(args.root, "split.json")
    official = load_official_split(split_json)
    os.makedirs(args.out, exist_ok=True)

    pattern = os.path.join(args.root, "*", "*", "*", "pressure_grids.npz")
    npzs = sorted(glob.glob(pattern))
    if not npzs:
        raise SystemExit(f"no pressure_grids.npz under {pattern!r} -- is --root right?")
    if args.limit:
        npzs = npzs[: args.limit]

    manifest, skipped, idx = [], [], 0
    split_of: dict[str, list[int]] = {k: [] for k in official}
    unassigned: list[str] = []

    for npz_path in npzs:
        key = traj_key(os.path.dirname(npz_path))

        # Split first, and drop before doing any work: a trajectory the official split does
        # not mention is not ours to place (OQ-B). Deciding this before load/write also keeps
        # `idx` a dense index over the corpus we actually keep.
        where = [k for k, members in official.items() if key in members]
        if len(where) > 1:
            raise SystemExit(f"{key} appears in more than one official split: {where}")
        if not where:
            unassigned.append(key)
            continue
        split = where[0]

        clip = load_clip(npz_path)
        if clip is None:
            skipped.append((key, "unreadable or wrong shape"))
            continue

        # No baseline correction: released-as-is (see module docstring, point 2).
        state = clip_states(clip, baseline_pct=None).astype(np.float32)   # (T, 2, 6)
        np.save(os.path.join(args.out, f"state_{idx}.npy"), state)
        if not args.no_clips:
            np.save(os.path.join(args.out, f"clip_{idx}.npy"), clip.astype(np.float32))

        scene, task, stamp = key.split("/")
        split_of[split].append(idx)

        manifest.append({
            "idx": idx,
            "label": task.replace("_", " "),          # -> parse_label gives (verb, object)
            "cat": categorize(task),
            "fps": FPS_RAW,
            "T": int(clip.shape[0]),
            "features": int(HANDS * 6),
            "has_clip": not args.no_clips,
            "scene": scene,
            "task": task,
            "traj": stamp,
            "split": split,
        })
        idx += 1

    with open(os.path.join(args.out, "manifest.jsonl"), "w") as fh:
        for row in manifest:
            fh.write(json.dumps(row) + "\n")

    splits = {
        "train": sorted(split_of["train"]),
        "val": sorted(split_of["val"]),
        "test": sorted(split_of["test_seen"]),        # main experiment == test_seen
        "test_unseen": sorted(split_of["test_unseen"]),
        "n": len(manifest),
        "source": "official split.json (EgoTouch release)",
        "join_key": "scene/task/timestamp",
        "dropped_unassigned": len(unassigned),
        "retention": {k: [len(split_of[k]), len(official[k])] for k in sorted(official)},
    }
    with open(os.path.join(args.out, "splits.json"), "w") as fh:
        json.dump(splits, fh, indent=2)
    with open(os.path.join(args.out, "unassigned_dropped.json"), "w") as fh:
        json.dump(sorted(unassigned), fh, indent=2)

    lens = sorted(r["T"] for r in manifest)
    need = 50 * 3     # 40 history + 10 horizon at 10 Hz == 150 raw frames at 30 Hz
    ok = sum(1 for t in lens if t >= need)
    print(f"wrote {len(manifest)} trajectories to {args.out}  (skipped {len(skipped)})")
    if lens:
        q = lambda p: lens[int(p * (len(lens) - 1))]  # noqa: E731
        print(f"  T frames @30Hz: min={lens[0]} p25={q(.25)} median={q(.5)} p75={q(.75)} max={lens[-1]}")
        print(f"  seconds:        min={lens[0]/FPS_RAW:.1f} median={q(.5)/FPS_RAW:.1f} max={lens[-1]/FPS_RAW:.1f}")
        print(f"  ELIGIBLE (T >= {need} raw frames = 5.0 s): {ok}/{len(lens)} ({100*ok/len(lens):.1f}%)")
    print(f"  split: train={len(splits['train'])} val={len(splits['val'])} "
          f"test(seen)={len(splits['test'])} test_unseen={len(splits['test_unseen'])}")
    for k in sorted(official):
        kept, listed = splits["retention"][k]
        print(f"    {k:12s} kept {kept:5d} / {listed:5d} listed  ({100*kept/max(listed,1):.1f}%)")
    if unassigned:
        print(f"  DROPPED ({len(unassigned)}): not in the official split (OQ-B). "
              f"Listed in unassigned_dropped.json.")
    for k, why in skipped[:10]:
        print(f"  skipped {k}: {why}")


if __name__ == "__main__":
    main()
