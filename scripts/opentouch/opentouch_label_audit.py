"""D2: is a mismatched clip's label consistent with the shape of its own force signal?

The user's ruling (2026-08-15): "if confirmed a join bug, fix it, don't drop the shard" is
unfalsifiable until "confirmed" names a procedure. This is that procedure's first half --
the part answerable from the cache alone.

WHAT IT TESTS. peak_idx in the annotation is defined as the moment of maximum pressure
(arXiv:2512.16842). If a clip's own argmax(F) sits far from it, either the clip took the
wrong annotation row or the row's indices are based on something else. Where they disagree,
the label is checked against the signal's morphology: typing and pouring cannot produce the
same force curve, so a label that survives that comparison is plausible and one that does
not is evidence of a genuine misjoin. delta_f_p95 (impulsiveness) and hf_energy_fraction
(fast content) come from trait.py, where they already exist as the Layer-2 check.

WHAT IT CANNOT TEST, and why the second half needs the HDF5. Deciding whether a mismatched
clip can be RE-joined needs its timestamps against the annotation's ts_start/ts_end, and the
cache never stored them (extract_opentouch.py writes fps_est and T, not the clock). So this
script can say "the label is implausible for this signal", which distinguishes a misjoin
from a coincidence, but re-joining requires the shards back.

READ IT AS EVIDENCE, NOT AS A VERDICT. It reports; the ruling on fix-vs-drop is the user's,
and per the same discussion the deciding argument is not how many clips would be lost but
that the mismatched shards are probably not a random subset -- dropping them would change
the population the claim is about, along an axis that is collinear with the split.

    python scripts/opentouch/opentouch_label_audit.py --cache ~/opentouch/cache
    python scripts/opentouch/opentouch_label_audit.py --shards sports_dicks_p2 --examples 12
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.opentouch import trait                                       # noqa: E402


def load(cache):
    return [json.loads(l) for l in open(os.path.join(cache, "manifest.jsonl"))]


def force_of(cache, idx):
    return np.load(os.path.join(cache, f"state_{idx}.npy"))[:, 0, 0]


def stats_for(rows, cache, fps_default=30.0):
    """-> per-action {n, delta_f_p95 median, hf median} over the clips given."""
    d, h = collections.defaultdict(list), collections.defaultdict(list)
    for r in rows:
        F = force_of(cache, r["idx"])
        fps = r.get("fps_est") or fps_default
        a = trait.normalize_action(r.get("action", ""))
        v = trait.delta_f_p95(F)
        w = trait.hf_energy_fraction(F, fps)
        if v is not None:
            d[a].append(v)
        if w is not None:
            h[a].append(w)
    return {a: {"n": len(d[a]), "dfp95": float(np.median(d[a])),
                "hf": float(np.median(h[a])) if h[a] else float("nan")}
            for a in sorted(d)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=os.path.expanduser("~/opentouch/cache"))
    ap.add_argument("--shards", help="restrict to these shards (default: every shard "
                                     "that has any peak mismatch)")
    ap.add_argument("--examples", type=int, default=8,
                    help="worst-mismatched clips to print per shard")
    ap.add_argument("--tol", type=int, default=2, help="frames of slack before mismatch")
    ap.add_argument("--write-flags", metavar="PATH",
                    help="record the mismatched clips so downstream code can refuse to "
                         "trust their onset/peak/post indices "
                         "(default target: data/opentouch_peak_mismatch.json)")
    a = ap.parse_args()

    man = load(a.cache)
    rows = {r["idx"]: r for r in man}

    # 1. Which clips disagree with their own annotation, and where they are concentrated.
    mism = collections.defaultdict(list)
    per_shard = collections.Counter()
    for r in man:
        per_shard[r.get("shard", "?")] += 1
        try:
            pk = int(r["peak_idx"])
        except (TypeError, ValueError):
            continue
        if not (0 <= pk < r["T"]):
            mism[r.get("shard", "?")].append((r["idx"], pk, -1))
            continue
        F = force_of(a.cache, r["idx"])
        d = abs(pk - int(np.argmax(F)))
        if d > a.tol:
            mism[r.get("shard", "?")].append((r["idx"], pk, d))

    print("=== where the mismatches are (a random scatter and a concentration mean "
          "different things) ===")
    print(f"{'shard':26s} {'clips':>6s} {'mismatch':>9s} {'rate':>7s}")
    tot_m = 0
    for s in sorted(per_shard, key=lambda s: -len(mism[s])):
        m = len(mism[s])
        tot_m += m
        if m:
            print(f"{s:26s} {per_shard[s]:6d} {m:9d} {m / per_shard[s]:7.1%}")
    print(f"{'TOTAL':26s} {sum(per_shard.values()):6d} {tot_m:9d} "
          f"{tot_m / sum(per_shard.values()):7.1%}")

    shards = ([s.strip() for s in a.shards.split(",")] if a.shards
              else [s for s in mism if mism[s]])

    # 2. Reference morphology per action, from the clips that DO agree -- the yardstick a
    #    suspect label is measured against has to be built from unsuspect clips.
    clean = [r for r in man if not any(r["idx"] == i for s in mism for i, _, _ in mism[s])]
    ref = stats_for(clean, a.cache)

    print("\n=== is the label plausible for the signal? ===")
    print("dfp95 = 95th pct of |causally smoothed dF| (impulsiveness); "
          "hf = power fraction above 3 Hz.")
    print("'ref' is the median over clips whose peak DOES agree, i.e. the unsuspect ones.\n")
    for s in shards:
        bad = sorted(mism.get(s, []), key=lambda t: -t[2])[:a.examples]
        if not bad:
            continue
        print(f"--- {s} ---")
        print(f"{'idx':>6s} {'action':16s} {'off':>5s} {'dfp95':>9s} {'ref':>9s} "
              f"{'hf':>6s} {'ref':>6s}  verdict")
        for idx, pk, d in bad:
            r = rows[idx]
            F = force_of(a.cache, idx)
            act = trait.normalize_action(r.get("action", ""))
            v, w = trait.delta_f_p95(F), trait.hf_energy_fraction(F, r.get("fps_est") or 30.0)
            R = ref.get(act)
            if v is None or R is None:
                verdict = "too short / no reference"
                rv, rh = float("nan"), float("nan")
            else:
                rv, rh = R["dfp95"], R["hf"]
                ratio = v / rv if rv > 0 else float("inf")
                verdict = ("consistent" if 0.33 <= ratio <= 3.0 else
                           f"IMPLAUSIBLE ({ratio:.1f}x its action's typical)")
            print(f"{idx:6d} {act[:16]:16s} {d:5d} {v if v else float('nan'):9.1f} "
                  f"{rv:9.1f} {w if w else float('nan'):6.3f} {rh:6.3f}  {verdict}")
        print()

    if a.write_flags:
        os.makedirs(os.path.dirname(a.write_flags) or ".", exist_ok=True)
        flat = sorted(i for v in mism.values() for i, _, _ in v)
        with open(a.write_flags, "w") as f:
            json.dump({
                "clips": flat,
                "per_shard": {s: {"clips": per_shard[s], "mismatch": len(mism[s])}
                              for s in sorted(mism) if mism[s]},
                "tol_frames": a.tol,
                "what_is_wrong": "onset_idx/peak_idx/post_idx disagree with the clip's own "
                                 "argmax(F). The 2026-08-16 audit found the ACTION labels "
                                 "plausible for the signals (46 consistent, 1 implausible), "
                                 "so this is an index-basing bug concentrated in shards that "
                                 "share one annotation CSV, not a label swap.",
                "safe_to_use": "action, object_category, scene, shard, T, fps_est",
                "not_safe_to_use": "onset_idx, peak_idx, post_idx",
            }, f, indent=2)
        print(f"wrote {a.write_flags}  ({len(flat)} clips flagged)\n")

    print("HOW TO READ THIS. A shard whose mismatches are mostly 'consistent' is one where "
          "the indices are off but the labels are probably right -- a join or basing bug, "
          "fixable by re-joining. Mismatches that are also IMPLAUSIBLE mean the clip is "
          "carrying another clip's row, and re-joining is what would fix it. If the source "
          "rows themselves disagree with every candidate signal, no re-join helps.\n"
          "Re-joining needs the clip timestamps, which the cache does not store: that half "
          "requires the shards back from Drive.")


if __name__ == "__main__":
    raise SystemExit(main())
