#!/usr/bin/env python3
"""Rebuild d256's continuous recordings and extract the physical-state target.

d256 ships 16-frame clips, which is far too short for a rolling-origin forecast (the
ActionSense harness wants 40 frames of history alone). But the clips are stride-1 sliding
windows -- clip `c+1` is clip `c` advanced by one frame -- so a run of consecutive clips folds
back into a continuous recording:

    base[j]      = clip_start[j]     for j = 0 … 15
    base[c + 15] = clip_c[15]        for each following clip

A cell is NOT one recording, though. Clip numbering runs `0 … N-1` unbroken, but the underlying
stream can jump: `signals1/train/S04/13` advances by one frame for every pair except 14->15,
where it breaks, giving segments of 15 and 6 clips -- 30 + 21 = 51 distinct frames, exactly the
51 the cell contains. That matches the source: ActionSense recorded several instances of an
activity per subject (15 "Clean a pan with a towel" recordings against d256's 5 cells), and
d256 concatenated their windows under one continuously-numbered directory.

So the unit of a recording is a SEGMENT, found by testing `clip_c[j+1] == clip_{c+1}[j]`, not a
cell. Getting this wrong does not merely lose data -- it splices unrelated moments into one
time series and every forecast trained on it is fitting an artefact.

This script does that, then applies `src/actionsense/physical_state.frame_state` to the two
gloves to get the same `(T, C, 6)` = `[F, xbar, ybar, sxx, syy, sxy]` per hand that the
ActionSense and OpenTouch arms are built on. Output matches `data/actionsense_states/` in
layout so the frozen harness can consume it unchanged -- plus a `subject` field, which the
ActionSense manifest lacks and which is what makes leave-one-subject-out possible here.

Reconstruction is *verified*, not trusted: every clip is checked to be an exact 16-frame
window of the recording it was folded into. A single mismatch aborts that recording rather
than writing a silently scrambled time series.

Corpus choice: `signals1` is the base timeline (stride 1, 6 Hz). `signals` and `signals2` are
the same recordings decimated by 3 and 2, so they are NOT extra data -- reproduce them with
the harness's `downsample` knob instead of extracting them separately.

No baseline correction is applied (user decision 2026-08-24, OQ-D2): the state is taken from
the raw grid, matching the口径 OpenTouch settled on. The tactile floor is well above zero, so
expect F to carry a large DC pedestal.

Usage:
    python scripts/extract_d256_states.py --root ~/forcevision --out data/d256_states
    python scripts/extract_d256_states.py --root ~/forcevision --out /tmp/x --limit 3 --aux
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import d256  # noqa: E402
from src.actionsense.physical_state import frame_state  # noqa: E402

FPS = 6.0        # signals1 = ActionSense/30Hz decimated by 4.948 +- 0.085 over 15 recordings
                 # (SESSION_LOG 2026-08-24续). Inferred from length ratios, not frame alignment.
CLIP = 16
AUX_STREAMS = ("myo-emg-left", "myo-emg-right", "myo-acc-left", "myo-acc-right",
               "joint-position", "left-hand-pose", "right-hand-pose", "gaze")


def _continues(prev, cur, key="tactile-glove-left"):
    """True iff `cur` is `prev` advanced by exactly one frame."""
    return np.array_equal(prev[key][1:], cur[key][:CLIP - 1])


def rebuild(cell_dir, n_clips, verify=True):
    """Fold a cell's clips into one recording per contiguous run. Yields (streams, first_id)."""
    seg = None
    prev = None
    start = 0
    for c in range(n_clips):
        clip = d256.load_clip(os.path.join(cell_dir, f"{c}.p")).signal
        if prev is not None and set(clip) != set(prev):
            raise ValueError(f"{cell_dir}: clip {c} has a different stream set than {c-1}")

        if prev is None or not _continues(prev, clip):
            if seg is not None:
                yield {k: np.stack(v) for k, v in seg.items()}, start
            seg = {k: [v[j] for j in range(CLIP)] for k, v in clip.items()}
            start = c
        else:
            for k in seg:
                seg[k].append(clip[k][CLIP - 1])
            if verify:
                # This clip must be exactly the segment's last 16 frames. _continues only
                # checked one stream and one frame of overlap; this checks all of both, so a
                # partial desync cannot slip through and scramble the series.
                off = c - start
                for k in seg:
                    if not np.array_equal(np.stack(seg[k][off:off + CLIP]), clip[k]):
                        raise ValueError(
                            f"{cell_dir}: clip {c} is not base[{off}:{off+CLIP}] on {k!r} -- "
                            "streams desynchronised within a segment")
        prev = clip
    if seg is not None:
        yield {k: np.stack(v) for k, v in seg.items()}, start


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(os.path.expanduser("~"), "forcevision"))
    ap.add_argument("--out", default=os.path.join("data", "d256_states"))
    ap.add_argument("--group", default="signals1",
                    help="corpus group; signals1 is the stride-1 base timeline (default)")
    ap.add_argument("--limit", type=int, default=None, help="stop after N recordings (smoke test)")
    ap.add_argument("--aux", action="store_true",
                    help="also save the non-tactile streams (EMG/acc/pose/gaze) per recording, "
                         "so a later multimodal arm needs no second pass over the corpus")
    ap.add_argument("--no-verify", action="store_true",
                    help="skip the per-clip window check (faster; only for a re-run of data "
                         "already verified once)")
    args = ap.parse_args()

    root = d256.root_of(args.root)
    os.makedirs(args.out, exist_ok=True)

    cells = {}
    for path in d256.iter_paths(root, groups=(args.group,)):
        m = d256.parse_path(path)
        cells.setdefault((m["split"], m["subject"], m["session"]), []).append(m["clip_id"])
    if not cells:
        sys.exit(f"no clips under {root}/{args.group} -- check --root/--group")

    manifest, idx, skipped = [], 0, []
    for key in sorted(cells):
        split, subject, session = key
        ids = sorted(cells[key])
        if ids != list(range(len(ids))):
            skipped.append((key, "clip ids are not contiguous 0..N-1"))
            continue
        cell_dir = os.path.join(root, args.group, split, subject, str(session))
        label_idx = session          # the session dir name IS label_idx (src/d256.py asserts it)
        label_text = d256.load_clip(os.path.join(cell_dir, "0.p")).label_text
        try:
            segments = list(rebuild(cell_dir, len(ids), verify=not args.no_verify))
        except ValueError as exc:
            skipped.append((key, str(exc)))
            continue

        for streams, first_clip in segments:
            grids = np.stack([streams["tactile-glove-left"],
                              streams["tactile-glove-right"]], axis=1)
            T = grids.shape[0]
            state = np.stack([frame_state(grids[t]) for t in range(T)])      # (T, 2, 6)

            np.save(os.path.join(args.out, f"state_{idx}.npy"), state.astype(np.float32))
            if args.aux:
                np.savez_compressed(
                    os.path.join(args.out, f"aux_{idx}.npz"),
                    **{k.replace("-", "_"): streams[k] for k in AUX_STREAMS if k in streams})
            manifest.append({
                "idx": idx, "label": label_text, "label_idx": label_idx,
                "subject": subject, "session": session, "group": args.group,
                "orig_split": split, "fps": FPS, "T": T,
                "first_clip": first_clip, "n_segments_in_cell": len(segments),
                "features": ["F", "xbar", "ybar", "sxx", "syy", "sxy"],
                "has_aux": bool(args.aux),
            })
            print(f"  [{idx:3d}] {subject} s{session:<2d} clip{first_clip:<5d} {T:5d} fr "
                  f"({T/FPS:6.1f} s)  {label_text[:40]}", flush=True)
            idx += 1
        if args.limit and idx >= args.limit:
            break

    with open(os.path.join(args.out, "manifest.jsonl"), "w") as fh:
        for row in manifest:
            fh.write(json.dumps(row) + "\n")

    T = sum(r["T"] for r in manifest)
    lens = sorted(r["T"] for r in manifest)
    print(f"\nwrote {len(manifest)} recordings from {len(cells)} cells, "
          f"{T} frames ({T/FPS/60:.1f} min) -> {args.out}")
    print(f"  subjects: {sorted({r['subject'] for r in manifest})}")
    print(f"  classes:  {len({r['label_idx'] for r in manifest})}")
    if lens:
        q = lambda p: lens[min(len(lens) - 1, int(p * len(lens)))]
        print(f"  length:   min {lens[0]}  p10 {q(.1)}  median {q(.5)}  p90 {q(.9)}  max {lens[-1]} frames")
        for need in (16, 24, 30, 40, 64):
            keep = sum(1 for L in lens if L >= need)
            org = sum(max(0, L - need + 1) for L in lens)
            print(f"    budget {need:3d} fr: {keep:4d}/{len(lens)} recordings, {org:6d} origins")
    if skipped:
        print(f"  SKIPPED {len(skipped)}:")
        for k, why in skipped[:10]:
            print(f"    {k}: {why}")


if __name__ == "__main__":
    main()
