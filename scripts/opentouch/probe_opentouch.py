"""Training-free predictability probe for the OpenTouch dataset, by action category.

OpenTouch (arXiv 2512.16842) HDF5 layout (per file), from OpenTouch-MIT/opentouch:
    data/<clip_id>/
        right_pressure        (T, 16, 16) tactile grids   <- what we use
        right_hand_landmarks  (21, 3)
        rgb_images_jpeg, timestamps
Labels: a CSV/TSV (`final_annotations`) keyed by clip id, columns include
    object_name, object_category, environment, action, grip_type.

We map each clip's free-text `action` through the SAME categorize() verb taxonomy used
for EgoTouch, so OpenTouch results sit in the same category / temporal-pattern space and
are directly comparable across datasets. Metrics come from the shared
src/tactile_pixel/predictability.py (identical math to the EgoTouch probe).

Usage (run from repo root, on CRC after `bash scripts/download_data.sh`):
    python scripts/opentouch/probe_opentouch.py --data-dir data --labels final_annotations --inspect
    python scripts/opentouch/probe_opentouch.py --data-dir data --labels final_annotations
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.tactile_pixel.categories import categorize_phrase, TEMPORAL_PATTERN  # noqa: E402
from src.tactile_pixel import predictability as P  # noqa: E402

KEY_CANDIDATES = ("key", "scene_clip", "scene::clip", "clip_id", "clip", "clip_name")


def find_hdf5(data_dir):
    files = []
    for ext in ("*.hdf5", "*.h5"):
        files += glob.glob(os.path.join(data_dir, "**", ext), recursive=True)
    return sorted(set(files))


def load_labels(path):
    """Return dict key -> row(dict). Accepts a CSV/TSV file or a dir of them."""
    import h5py  # noqa: F401  (imported lazily elsewhere; here just for parity)
    paths = []
    if os.path.isdir(path):
        for ext in ("*.csv", "*.tsv"):
            paths += glob.glob(os.path.join(path, "**", ext), recursive=True)
    elif os.path.isfile(path):
        paths = [path]
    else:  # try common fallbacks
        for cand in (path, path + ".csv", path + ".tsv"):
            if os.path.isfile(cand):
                paths = [cand]
                break
    rows, key_col = {}, None
    for p in sorted(paths):
        delim = "\t" if p.lower().endswith(".tsv") else ","
        with open(p, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=delim)
            fields = reader.fieldnames or []
            kc = next((c for c in KEY_CANDIDATES if c in fields), None)
            if kc is None:
                continue
            key_col = key_col or kc
            for row in reader:
                k = (row.get(kc) or "").strip()
                if k:
                    rows[k] = {c: (v or "").strip() for c, v in row.items()}
    return rows, key_col, paths


def clip_groups(h5file):
    """Yield (clip_id, group) for every clip in an open HDF5 file.
    Clips live under a top-level 'data' group; fall back to any group holding
    'right_pressure'."""
    import h5py
    root = h5file["data"] if "data" in h5file and isinstance(h5file["data"], h5py.Group) else h5file
    for name, obj in root.items():
        if isinstance(obj, h5py.Group) and "right_pressure" in obj:
            yield name, obj


def match_label(clip_id, scene, labels):
    for k in (clip_id, f"{scene}::{clip_id}", f"{scene}_{clip_id}", f"{scene}/{clip_id}"):
        if k in labels:
            return labels[k]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--labels", default="final_annotations")
    ap.add_argument("--min-frames", type=int, default=35)
    ap.add_argument("--inspect", action="store_true", help="dump schema + join stats, then exit")
    ap.add_argument("--out", default=os.path.join("docs", "predictability_opentouch.csv"))
    args = ap.parse_args()
    import h5py

    files = find_hdf5(args.data_dir)
    labels, key_col, label_paths = load_labels(args.labels)
    print(f"HDF5 files: {len(files)} under {args.data_dir!r}")
    print(f"Labels: {len(labels)} rows, key_col={key_col!r}, from {label_paths}")

    if args.inspect:
        if not files:
            raise SystemExit("no HDF5 files found — run scripts/download_data.sh first")
        with h5py.File(files[0], "r") as f:
            print(f"\n--- structure of {files[0]} ---")
            print("top-level keys:", list(f.keys())[:10])
            clips = list(clip_groups(f))
            print(f"clips in file: {len(clips)}; first clip id: {clips[0][0] if clips else None}")
            if clips:
                g = clips[0][1]
                print("clip datasets:", {k: getattr(g[k], "shape", None) for k in g.keys()})
                pr = g["right_pressure"][()]
                print(f"right_pressure: shape={pr.shape} dtype={pr.dtype} "
                      f"min={float(pr.min()):.1f} max={float(pr.max()):.1f}")
        # join check on first file
        scene = os.path.splitext(os.path.basename(files[0]))[0]
        with h5py.File(files[0], "r") as f:
            ids = [cid for cid, _ in clip_groups(f)]
        matched = sum(match_label(cid, scene, labels) is not None for cid in ids)
        print(f"\njoin check (file scene={scene!r}): {matched}/{len(ids)} clips matched a label")
        if labels:
            sample = next(iter(labels.values()))
            print("label columns:", list(sample.keys()))
        return

    by_action = P.new_group_dict()
    by_category = P.new_group_dict()
    by_pattern = P.new_group_dict()
    by_grip = P.new_group_dict()
    n_ok = n_seen = n_nolabel = 0
    for fp in files:
        scene = os.path.splitext(os.path.basename(fp))[0]
        with h5py.File(fp, "r") as f:
            for cid, g in clip_groups(f):
                n_seen += 1
                row = match_label(cid, scene, labels)
                if row is None:
                    n_nolabel += 1
                    continue
                pr = np.asarray(g["right_pressure"][()], dtype=np.float32)
                if pr.ndim == 3:
                    pr = pr[:, None]  # (T,16,16) -> (T,1,16,16)
                if pr.shape[0] < args.min_frames:
                    continue
                m = P.seq_metrics(pr)
                if not m:
                    continue
                action = (row.get("action") or "unknown").strip() or "unknown"
                cat = categorize_phrase(action)
                by_action[action].append(m)
                by_category[cat].append(m)
                by_pattern[TEMPORAL_PATTERN.get(cat, "Other")].append(m)
                grip = (row.get("grip_type") or "unknown").strip() or "unknown"
                by_grip[grip].append(m)
                n_ok += 1
        print(f"  {os.path.basename(fp)}: cumulative usable={n_ok}")
    print(f"\nclips seen={n_seen} unlabeled={n_nolabel} usable(T>={args.min_frames})={n_ok}\n")

    def show(title, groups, min_n=1):
        rows = P.add_predictability_index(P.aggregate(
            {g: l for g, l in groups.items() if len(l) >= min_n}))
        print(f"===== {title} (ranked by predictability_index, higher=easier) =====")
        hdr = f"{'group':<26}{'n':>5}{'persH1':>8}{'persH15':>9}{'persH30':>9}{'period':>8}{'migr15':>8}{'PI':>7}"
        print(hdr); print("-" * len(hdr))
        for g in sorted(rows, key=lambda g: -rows[g]["pi"]):
            r = rows[g]
            print(f"{g:<26}{r['n']:>5}{r['pers_nmse_h1']:>8.3f}{r['pers_nmse_h15']:>9.3f}"
                  f"{r['pers_nmse_h30']:>9.3f}{r['periodicity']:>8.3f}"
                  f"{r['contact_migration']:>8.3f}{r['pi']:>7.2f}")
        print()
        return rows

    pat_rows = show("BY TEMPORAL-PATTERN CLASS (Axis B)", by_pattern)
    cat_rows = show("BY ACTION CATEGORY (mapped via verb taxonomy)", by_category)
    act_rows = show("BY RAW OPENTOUCH ACTION", by_action, min_n=3)
    grip_rows = show("BY GRASP / GRIP TYPE", by_grip, min_n=3)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["grouping", "group", "n", *P.METRIC_KEYS, "predictability_index"])
        for gname, rows in [("temporal_pattern", pat_rows), ("action_category", cat_rows),
                            ("raw_action", act_rows), ("grip_type", grip_rows)]:
            for g, r in sorted(rows.items(), key=lambda kv: -kv[1]["pi"]):
                w.writerow([gname, g, r["n"], *[f"{r[k]:.5f}" for k in P.METRIC_KEYS],
                            f"{r['pi']:.4f}"])
    print(f"[done] wrote {args.out}")


if __name__ == "__main__":
    main()
