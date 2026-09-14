"""Export the REAL OpenTouch data behind the F/CoP explainer figure. Runs on CRC (CPU, seconds).

The explainer (scripts/opentouch/plot_fcop_explainer.py) must print real numbers, and the
OpenTouch caches exist only on CRC. This pulls, for every clip whose object matches --pattern
(default: drill), a few kilobytes: the D1 F/CoP time series, the raw and D1 maps at the
annotated onset/peak/post frames, and the shard's per-taxel baseline b and scale sigma.

SELF-CHECK, FATAL (CLAUDE.md rule 6 -- a wrong run must fail loudly, not look usable): the D1
maps are recomputed here from the RAW cache with src/opentouch/baseline.py, exactly as
scripts/opentouch/opentouch_apply_baseline.py wrote cache_d1, and their moments must reproduce
cache_d1/state_N.npy (F to 1e-4 relative, CoP to 1e-4 absolute). A mismatch exits 3, because it
means the maps in the figure would not be the maps the paper's numbers came from.

Every candidate is exported, so choosing the clip for the figure never needs a second CRC trip.
No drill in the manifest -> the object vocabulary is printed and the script exits 2; it never
silently substitutes another object.

    python scripts/opentouch/export_fcop_explainer.py                 # ~/opentouch/cache{,_d1}
    python scripts/opentouch/export_fcop_explainer.py --pattern "drill|screwdriver"
    python scripts/opentouch/export_fcop_explainer.py --idx 1234,1240
Then copy back:  scp <crc>:~/fcop_explainer.tar.gz .
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys
import tarfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from src.opentouch import baseline as B                      # noqa: E402

TAGS = ("onset_idx", "peak_idx", "post_idx")
F_REL_TOL, COP_ABS_TOL = 1e-4, 1e-4


def peak_mismatched() -> set[int]:
    """Clips whose annotated peak disagrees with their own signal (2026-08-16 audit)."""
    try:
        with open(os.path.join(ROOT, "data", "opentouch_peak_mismatch.json")) as f:
            return set(int(c) for c in json.load(f)["clips"])
    except (OSError, KeyError, ValueError, TypeError):
        return set()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default=os.path.expanduser("~/opentouch/cache"))
    ap.add_argument("--d1", default=os.path.expanduser("~/opentouch/cache_d1"))
    ap.add_argument("--pattern", default="drill", help="regex on object_name / object_category")
    ap.add_argument("--idx", help="comma-separated clip indices (overrides --pattern)")
    ap.add_argument("--out", default=os.path.expanduser("~/fcop_explainer"))
    a = ap.parse_args()

    rows = B.manifest(a.d1)
    if a.idx:
        want = {int(x) for x in a.idx.split(",") if x.strip()}
        cands = [r for r in rows if r["idx"] in want]
    else:
        pat = re.compile(a.pattern, re.I)
        cands = [r for r in rows
                 if pat.search(r.get("object_name", "")) or pat.search(r.get("object_category", ""))]
    print(f"export_fcop_explainer: scope=D1 d1={a.d1} raw={a.raw} "
          f"match={'idx ' + a.idx if a.idx else repr(a.pattern)} candidates={len(cands)} of {len(rows)}")
    if not cands:
        vocab = collections.Counter(r.get("object_name", "") for r in rows).most_common(80)
        print("NO MATCH. object_name vocabulary (top 80):")
        for name, n in vocab:
            print(f"  {n:4d}  {name}")
        return 2

    raw_rows = B.manifest(a.raw)
    mism = peak_mismatched()
    os.makedirs(a.out, exist_ok=True)
    est: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    table, n_ok = [], 0
    for r in sorted(cands, key=lambda x: x["idx"]):
        i, sh = r["idx"], r.get("shard", "?")
        clip_p, state_p = os.path.join(a.raw, f"clip_{i}.npy"), os.path.join(a.d1, f"state_{i}.npy")
        if not (os.path.exists(clip_p) and os.path.exists(state_p)):
            print(f"  skip {i}: missing {'clip' if not os.path.exists(clip_p) else 'state'}")
            continue
        if sh not in est:                  # same rows, same order as opentouch_apply_baseline.py
            est.update(B.shard_baselines(a.raw, [x for x in raw_rows if x.get("shard", "?") == sh]))
        base, sigma = est[sh]
        k = float(r.get("d1_k", 1.0))

        clip = np.load(clip_p)                                  # (T,1,16,16) raw counts
        corr = B.correct(clip, base, sigma, k)                  # (T,1,16,16) D1
        mom = B.moments(corr[:, 0])                             # (T,6)
        st = np.load(state_p)[:, 0, :].astype(np.float64)      # (T,6) what the paper scored
        f_rel = float(np.max(np.abs(mom[:, 0] - st[:, 0]) / np.maximum(np.abs(st[:, 0]), 1.0)))
        c_abs = float(np.max(np.abs(mom[:, 1:3] - st[:, 1:3])))
        if f_rel > F_REL_TOL or c_abs > COP_ABS_TOL:
            print(f"FATAL clip {i}: recomputed D1 moments != cache_d1 (F rel {f_rel:.2e}, "
                  f"CoP abs {c_abs:.2e}). The maps would not be the ones the paper scored.")
            return 3

        T = len(st)
        frames = {}
        for tag in TAGS:
            try:
                v = int(r.get(tag))
            except (TypeError, ValueError):
                continue
            if 0 <= v < T:
                frames[tag.replace("_idx", "")] = v
        if "peak" in frames and i not in mism:
            chosen, why = frames["peak"], "annotated peak"
        else:
            chosen, why = int(np.argmax(st[:, 0])), "argmax F (annotated peak missing or mismatched)"
        sel = sorted({*frames.values(), chosen})

        np.savez_compressed(
            os.path.join(a.out, f"fcop_clip{i}.npz"),
            meta=json.dumps(r), F=st[:, 0].astype(np.float32), cx=st[:, 1].astype(np.float32),
            cy=st[:, 2].astype(np.float32), fps=float(r.get("fps_est") or 30.0),
            tags=json.dumps(frames), frames_idx=np.asarray(sel, np.int64),
            raw_maps=clip[sel, 0].astype(np.float32), d1_maps=corr[sel, 0].astype(np.float32),
            chosen=int(chosen), chosen_reason=why, base=base.astype(np.float32),
            sigma=sigma.astype(np.float32), k=k, check_F_rel=f_rel, check_cop_abs=c_abs)
        n_pos = int((corr[chosen, 0] > 0).sum())
        table.append((i, r.get("action", ""), r.get("object_name", ""), r.get("grip_type", ""), T, sh,
                      float(st[chosen, 0]), float(st[chosen, 1]), float(st[chosen, 2]), n_pos, why))
        n_ok += 1

    tsv = os.path.join(a.out, "candidates.tsv")
    with open(tsv, "w") as f:
        f.write("idx\taction\tobject_name\tgrip_type\tT\tshard\tF_chosen\tcx_chosen\tcy_chosen\t"
                "n_taxels_pos\tchosen_frame\n")
        for t in table:
            f.write("\t".join(f"{v:.4g}" if isinstance(v, float) else str(v) for v in t) + "\n")
    print(f"{'idx':>5} {'action':<14} {'object':<28} {'T':>5} {'F@chosen':>10} {'cx':>6} {'cy':>6} {'n>0':>4}")
    for t in table:
        print(f"{t[0]:>5} {t[1][:14]:<14} {t[2][:28]:<28} {t[4]:>5} {t[6]:>10.0f} {t[7]:>+6.2f} "
              f"{t[8]:>+6.2f} {t[9]:>4}")
    tar = a.out.rstrip("/") + ".tar.gz"
    with tarfile.open(tar, "w:gz") as tf:
        tf.add(a.out, arcname=os.path.basename(a.out.rstrip("/")))
    print(f"wrote {n_ok} npz + candidates.tsv -> {tar}   (all passed the D1 self-check)")
    return 0 if n_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
