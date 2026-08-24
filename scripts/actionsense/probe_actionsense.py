"""Training-free predictability probe for ActionSense, by kitchen-activity category.

ActionSense (NeurIPS 2022, MIT) wearables HDF5 (per subject session), from delpreto/ActionNet:
    tactile-glove-left / tactile-glove-right / tactile_data / {data (N,H,W), time_s}
    experiment-activities / activities / {data (rows [Activity,Start/Stop,Valid,Notes]), time_s}
Continuous recording -> we segment the tactile stream into per-activity clips using the
Start/Stop label markers, resample each clip to a common 30 Hz (to match EgoTouch/OpenTouch
frame-based metrics), stack both gloves -> (T,2,H,W), and probe.

Activity labels are verb-first phrases ('Peel a cucumber', 'Pour water ...') mapped through the
shared categorize_phrase() taxonomy. Metrics from src/tactile_pixel/predictability.py.
Subjects S00-S05 wore tactile gloves; S06-S09 did not (files auto-skip: no tactile stream).

Usage (run from repo root):
    python scripts/actionsense/probe_actionsense.py --data-dir ~/actionsense
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.tactile_pixel.categories import categorize_phrase, TEMPORAL_PATTERN  # noqa: E402
from src.tactile_pixel import predictability as P  # noqa: E402
from src.actionsense import physical_state as PS  # noqa: E402

TACTILE_KEYS = ("tactile-glove-left", "tactile-glove-right")
BAD_RATINGS = ("Bad", "Maybe")


def _dec(x):
    return x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else str(x)


def activity_intervals(h5):
    """List of (label, start_s, end_s) from the experiment-activities stream."""
    grp = h5["experiment-activities"]["activities"]
    rows = [[_dec(v) for v in r] for r in grp["data"][:]]
    times = np.squeeze(np.array(grp["time_s"][:])).astype(float)
    out, open_lbl, open_t = [], None, None
    for i, row in enumerate(rows):
        label, startstop = row[0], row[1]
        rating = row[2] if len(row) > 2 else "Good"
        if rating in BAD_RATINGS:
            continue
        if startstop == "Start":
            open_lbl, open_t = label, times[i]
        elif startstop == "Stop" and open_lbl is not None:
            out.append((open_lbl, open_t, times[i]))
            open_lbl = None
    return out


def load_tactile(h5, key):
    """(data (N,H,W) float32, time_s (N,)) for one glove, or None."""
    if key not in h5 or "tactile_data" not in h5[key]:
        return None
    g = h5[key]["tactile_data"]
    d = np.asarray(g["data"][:], dtype=np.float32)
    t = np.squeeze(np.array(g["time_s"][:])).astype(float)
    if d.ndim == 2:  # (N, D) -> (N, s, s)
        s = int(round(d.shape[1] ** 0.5))
        d = d.reshape(d.shape[0], s, s)
    return d, t


def resample(X, t, target_t):
    """Linear-interpolate frames X:(n,H,W) at times t to target_t:(T,)."""
    n = X.shape[0]
    idx = np.clip(np.searchsorted(t, target_t), 1, n - 1)
    t0, t1 = t[idx - 1], t[idx]
    w = ((target_t - t0) / (t1 - t0 + 1e-12))[:, None, None]
    return X[idx - 1] * (1 - w) + X[idx] * w


def clip_for_interval(gloves, start, end, fps):
    """Stack both gloves' resampled frames for [start,end] -> (T,2,H,W) or None."""
    T = int(round((end - start) * fps))
    if T < 2:
        return None
    target_t = np.linspace(start, end, T)
    chans = []
    for gd in gloves:
        if gd is None:
            return None
        d, t = gd
        m = (t >= t[0]) & (t <= t[-1])  # guard
        if t.min() > start or t.max() < end or d.shape[0] < 2:
            return None
        chans.append(resample(d, t, target_t))
    return np.stack(chans, axis=1)  # (T, 2, H, W)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.path.join(os.path.expanduser("~"), "actionsense"))
    ap.add_argument("--target-fps", type=int, default=30)
    ap.add_argument("--min-frames", type=int, default=35)
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--jsonl", default=None,
                    help="append per-clip records here (streaming, one file at a time)")
    ap.add_argument("--report-only", action="store_true",
                    help="skip HDF5; aggregate an existing --jsonl and print/write CSV")
    ap.add_argument("--extract-states", default=None,
                    help="also save analytic physical-state trajectory per clip to this dir "
                         "(state_N.npy + manifest.jsonl) for the v1 forecaster")
    ap.add_argument("--save-clips-for", default=None,
                    help="comma-separated label substrings; save the raw resampled (T,C,H,W) "
                         "clip (float16) as clip_N.npy for matching activities (cache for local "
                         "re-processing). Requires --extract-states.")
    ap.add_argument("--out", default=os.path.join("docs", "predictability_actionsense.csv"))
    args = ap.parse_args()
    import csv
    import json

    # physical-state extraction (append across streamed files)
    sdir = args.extract_states
    clip_filters = [s.strip() for s in args.save_clips_for.split(",")] if args.save_clips_for else []
    s_manifest = None
    s_n = 0
    if sdir:
        os.makedirs(sdir, exist_ok=True)
        mpath = os.path.join(sdir, "manifest.jsonl")
        s_n = sum(1 for _ in open(mpath)) if os.path.exists(mpath) else 0
        s_manifest = open(mpath, "a")

    def report(by_activity, by_category, by_pattern):
        def show(title, groups, min_n=1):
            rows = P.add_predictability_index(P.aggregate(
                {g: l for g, l in groups.items() if len(l) >= min_n}))
            print(f"===== {title} (ranked by predictability_index, higher=easier) =====")
            hdr = (f"{'group':<48}{'n':>5}{'persH1':>8}{'persH15':>9}{'persH30':>9}"
                   f"{'period':>8}{'migr15':>8}{'PI':>7}")
            print(hdr); print("-" * len(hdr))
            for g in sorted(rows, key=lambda g: -rows[g]["pi"]):
                r = rows[g]
                print(f"{g[:47]:<48}{r['n']:>5}{r['pers_nmse_h1']:>8.3f}{r['pers_nmse_h15']:>9.3f}"
                      f"{r['pers_nmse_h30']:>9.3f}{r['periodicity']:>8.3f}"
                      f"{r['contact_migration']:>8.3f}{r['pi']:>7.2f}")
            print()
            return rows
        pat_rows = show("BY TEMPORAL-PATTERN CLASS (Axis B)", by_pattern)
        cat_rows = show("BY ACTION CATEGORY (mapped via verb taxonomy)", by_category)
        act_rows = show("BY RAW ACTIONSENSE ACTIVITY", by_activity)
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["grouping", "group", "n", *P.METRIC_KEYS, "predictability_index"])
            for gname, rows in [("temporal_pattern", pat_rows), ("action_category", cat_rows),
                                ("raw_activity", act_rows)]:
                for g, r in sorted(rows.items(), key=lambda kv: -kv[1]["pi"]):
                    w.writerow([gname, g, r["n"], *[f"{r[k]:.5f}" for k in P.METRIC_KEYS],
                                f"{r['pi']:.4f}"])
        print(f"[done] wrote {args.out}")

    if args.report_only:
        ba, bc, bp = P.new_group_dict(), P.new_group_dict(), P.new_group_dict()
        with open(args.jsonl) as fh:
            for line in fh:
                r = json.loads(line)
                ba[r["label"]].append(r["m"]); bc[r["cat"]].append(r["m"])
                bp[r["pat"]].append(r["m"])
        n = sum(len(v) for v in ba.values())
        print(f"aggregating {n} clips from {args.jsonl}\n")
        report(ba, bc, bp)
        return

    import h5py
    jf = open(args.jsonl, "a") if args.jsonl else None

    files = sorted(glob.glob(os.path.join(args.data_dir, "**", "*.hdf5"), recursive=True))
    print(f"HDF5 files: {len(files)} under {args.data_dir!r}")

    if args.inspect:
        for fp in files[:2]:
            with h5py.File(fp, "r") as h5:
                print(f"\n--- {os.path.basename(fp)} ---")
                print("top-level devices:", list(h5.keys())[:20])
                for k in TACTILE_KEYS:
                    gd = load_tactile(h5, k)
                    print(f"  {k}: {'MISSING' if gd is None else (gd[0].shape, 'Fs=%.1f' % ((gd[1].size-1)/(gd[1][-1]-gd[1][0])))}")
                ivs = activity_intervals(h5)
                print(f"  activities: {len(ivs)}; sample:", ivs[:3])
        return

    by_activity = P.new_group_dict()
    by_category = P.new_group_dict()
    by_pattern = P.new_group_dict()
    n_ok = n_iv = 0
    for fp in files:
        try:
            h5 = h5py.File(fp, "r")
        except OSError as e:
            print(f"  {os.path.basename(fp)}: CORRUPT/truncated ({e.args[0][:40]}...), skip")
            continue
        with h5:
            gloves = [load_tactile(h5, k) for k in TACTILE_KEYS]
            if all(g is None for g in gloves):
                print(f"  {os.path.basename(fp)}: no tactile stream, skip")
                continue
            try:
                ivs = activity_intervals(h5)
            except KeyError:
                print(f"  {os.path.basename(fp)}: no activity labels, skip")
                continue
            for label, start, end in ivs:
                n_iv += 1
                if label in ("None", ""):
                    continue
                clip = clip_for_interval(gloves, start, end, args.target_fps)
                if clip is None or clip.shape[0] < args.min_frames:
                    continue
                m = P.seq_metrics(clip)
                if not m:
                    continue
                cat = categorize_phrase(label)
                pat = TEMPORAL_PATTERN.get(cat, "Other")
                by_activity[label].append(m)
                by_category[cat].append(m)
                by_pattern[pat].append(m)
                if jf:
                    jf.write(json.dumps({"label": label, "cat": cat, "pat": pat, "m": m}) + "\n")
                if s_manifest is not None:
                    st = PS.clip_states(clip).astype("float32")  # (T, C, 6), baseline-corrected
                    np.save(os.path.join(sdir, f"state_{s_n}.npy"), st)
                    saved_clip = False
                    if clip_filters and any(sub in label for sub in clip_filters):
                        np.save(os.path.join(sdir, f"clip_{s_n}.npy"), clip.astype("float16"))
                        saved_clip = True
                    s_manifest.write(json.dumps({
                        "idx": s_n, "label": label, "cat": cat,
                        "fps": args.target_fps, "T": int(st.shape[0]),
                        "features": list(PS.FEATURES), "has_clip": saved_clip}) + "\n")
                    s_manifest.flush()
                    s_n += 1
                n_ok += 1
        print(f"  {os.path.basename(fp)}: cumulative usable clips={n_ok}")
    if s_manifest is not None:
        s_manifest.close()
        print(f"extracted {s_n} state trajectories -> {sdir}")
    if jf:
        jf.close()
        print(f"\nappended {n_ok} clips (of {n_iv} intervals) -> {args.jsonl}")
        return
    print(f"\nactivity intervals={n_iv}  usable(T>={args.min_frames})={n_ok}\n")
    report(by_activity, by_category, by_pattern)


if __name__ == "__main__":
    main()
