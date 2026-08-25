"""Score a finished OpenTouch run from its saved forecasts -- no GPU, no retraining.

The run wrote one npz per test clip (--save-preds). Every clip is TEST in exactly one fold,
so those files together cover the whole corpus once, and everything the analysis still owes
can be computed from them: overall skill, the smooth/abrupt comparison, per-action R2, the
leave-one-action-out check, and bootstrap intervals. Separating scoring from training is
what makes it cheap to add a question later instead of paying for another GPU day.

THE SPLIT IS REGENERATED, NOT GUESSED. splits.folds is deterministic given (k, seed), so
each fold's TRAIN set -- and therefore its CoP force threshold and its Norm -- can be
reconstructed exactly. The threshold has to be per fold: it is fitted on TRAIN, and pooling
the corpus to compute one would leak every fold's test data into every other fold's mask.

WHAT THE NUMBERS MEAN, AND WHAT THEY DO NOT. The DC share of F is MEASURED from the scored
predictions and reported at the end of the run, because it decides how every number above
should be read and it is not a constant: on the uncorrected target it was ~99.3% (D1
declined, 2026-08-16), so a forecaster was largely being asked
to reproduce a constant and skill reads high for reasons unrelated to dynamics. The
smooth/abrupt contrast additionally rests on two classes each dominated by one action --
picking up is 37% of abrupt, holding is 38% of smooth -- which is why per-action and
leave-one-action-out are reported next to it rather than in an appendix.

    python scripts/opentouch/opentouch_report.py --preds runs/preds
    python scripts/opentouch/opentouch_report.py --preds runs/preds --boot 2000 --out docs/report.csv
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.actionsense.eval_harness.config import load_config          # noqa: E402
from src.opentouch import aggregate, bootstrap, masking, trait       # noqa: E402
from src.opentouch import splits as SP                               # noqa: E402
from src.opentouch.baselines import origins                          # noqa: E402
from src.shape_metrics import hausdorff_scaled                       # noqa: E402
from src.opentouch.dataset import force_thresholds, load_group       # noqa: E402


def gather(cfg, preds_dir, k, seed):
    """-> (ClipStats over the whole corpus, {clip idx: action}).

    Per fold: rebuild TRAIN to get that fold's force threshold, then stack its test clips'
    truths and every model's forecast on the shared origin grid."""
    folds = SP.folds(cfg, k, seed)
    C = len(cfg.channels)
    parts, models = [], None

    for f in folds:
        thr = force_thresholds(cfg, load_group(cfg, f["train"]))
        yts, ids, mods = [], [], collections.defaultdict(list)
        for i in f["test"]:
            path = os.path.join(preds_dir, f"clip_{i}.npz")
            if not os.path.exists(path):
                continue
            z = np.load(path, allow_pickle=True)
            y, ors = z["y"], z["origins"]
            names = sorted(kk[3:] for kk in z.files if kk.startswith("mu_"))
            if models is None:
                models = names
            for t in ors:
                yts.append(y[t + 1:t + 1 + cfg.horizon])
            ids += [i] * len(ors)
            for m in names:
                mods[m].append(z[f"mu_{m}"])
        if not yts:
            continue
        ytrue = np.stack(yts)
        mask = masking.valid_mask(cfg, ytrue.reshape(-1, C), thr).reshape(ytrue.shape)
        P = {m: np.concatenate(v, 0) for m, v in mods.items()}
        parts.append(aggregate.clip_stats(ytrue, mask, np.array(ids), P, cfg.channels))

    if not parts:
        raise SystemExit(f"no clip_*.npz under {preds_dir}")

    merged = aggregate.ClipStats(
        clip_ids=np.concatenate([p.clip_ids for p in parts]),
        n_valid=np.concatenate([p.n_valid for p in parts]),
        sum_y=np.concatenate([p.sum_y for p in parts]),
        sum_y2=np.concatenate([p.sum_y2 for p in parts]),
        sse={m: np.concatenate([p.sse[m] for p in parts]) for m in parts[0].sse},
        channels=parts[0].channels)
    return merged, models


def dc_share(preds_dir, force_idx: int = 0, limit: int = 400) -> float:
    """Fraction of the force channel's mean square that is its MEAN. -> 0..1, NaN if unknown.

    MEASURED from the predictions actually being scored, rather than asserted. This line used
    to be a hardcoded "F is ~99.3% DC (D1 declined)", which was true when it was written and
    false from 2026-08-20 on: every D1 report printed it while scoring a target whose baseline
    had been removed, telling the reader to interpret the numbers as reproducing a constant
    that was no longer there.
    """
    tot_m2 = tot_ms = 0.0
    n = 0
    for path in sorted(glob.glob(os.path.join(preds_dir, "clip_*.npz")))[:limit]:
        y = np.asarray(np.load(path, allow_pickle=True)["y"], dtype=np.float64)
        if y.ndim != 2 or y.shape[1] <= force_idx or not len(y):
            continue
        f = y[:, force_idx]
        tot_m2 += float(f.mean()) ** 2 * len(f)
        tot_ms += float((f ** 2).mean()) * len(f)
        n += len(f)
    return tot_m2 / tot_ms if n and tot_ms > 0 else float("nan")


def hausdorff_table(cfg, preds_dir):
    """({model: {channel: (mean Hausdorff, ratio to persistence)}}, n_clips) -- clip-equal."""
    H, C = cfg.horizon, len(cfg.channels)
    per = collections.defaultdict(lambda: collections.defaultdict(list))
    for path in sorted(glob.glob(os.path.join(preds_dir, "clip_*.npz"))):
        z = np.load(path, allow_pickle=True)
        y, ors = np.asarray(z["y"], dtype=np.float64), np.asarray(z["origins"])
        if len(ors) == 0:
            continue
        keep = ors + H < len(y)
        ors = ors[keep]
        if not len(ors):
            continue
        true = np.stack([y[t + 1:t + 1 + H] for t in ors])          # (N,H,C)
        names = sorted(k[3:] for k in z.files if k.startswith("mu_"))
        for c in range(C):
            for m in names:
                mu = np.asarray(z[f"mu_{m}"], dtype=np.float64)[keep]
                h = hausdorff_scaled(mu[:, :, c], true[:, :, c])
                if np.isfinite(h).any():
                    per[m][cfg.channels[c]].append(float(np.nanmean(h)))
    out = {}
    n = max((len(v) for d in per.values() for v in d.values()), default=0)
    ref = {ch: float(np.mean(v)) for ch, v in per.get("persistence", {}).items()}
    for m, d in per.items():
        out[m] = {ch: (float(np.mean(v)),
                       float(np.mean(v)) / ref[ch] if ref.get(ch) else float("nan"))
                  for ch, v in d.items()}
    return out, n


def action_of(cfg, ids):
    from src.opentouch.dataset import eligible_clips
    m = {r["idx"]: trait.normalize_action(r.get("action", "")) for r in eligible_clips(cfg, ())}
    return np.array([m.get(int(i), "") for i in ids])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/opentouch/eval_harness.yaml")
    ap.add_argument("--preds", default="runs/preds")
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--boot", type=int, default=2000, help="bootstrap resamples")
    ap.add_argument("--out", default="docs/opentouch/raw/opentouch_report.csv")
    a = ap.parse_args()

    cfg = load_config(a.config)
    st, models = gather(cfg, a.preds, a.folds, a.seed)
    chans = list(cfg.channels)
    ref = "persistence"
    rows = []

    print(f"clips scored: {len(st.clip_ids)} | models: {', '.join(models)}")
    print(f"\n=== overall: R2 vs the class mean, per channel ===")
    print(f"{'model':16s} " + " ".join(f"{c:>12s}" for c in chans))
    R = {m: aggregate.r2(st, m) for m in models}
    for m in models:
        print(f"{m:16s} " + " ".join(f"{v:12.4f}" for v in R[m].per_channel))
        for ci, c in enumerate(chans):
            rows.append(dict(scope="overall", subset="all", model=m, channel=c,
                             metric="R2", value=float(R[m].per_channel[ci]),
                             n_clips=len(st.clip_ids)))

    print(f"\n=== overall: skill vs {ref} (>0 better) ===")
    print(f"{'model':16s} " + " ".join(f"{c:>12s}" for c in chans))
    for m in models:
        if m == ref:
            continue
        s = aggregate.skill(st, m, ref)
        print(f"{m:16s} " + " ".join(f"{v:12.4f}" for v in s))
        for ci, c in enumerate(chans):
            rows.append(dict(scope="overall", subset="all", model=m, channel=c,
                             metric=f"skill_vs_{ref}", value=float(s[ci]),
                             n_clips=len(st.clip_ids)))

    # ---- G2: one model per arm, fit once, separated only here (user ruling 2026-08-15) --
    acts = action_of(cfg, st.clip_ids)
    cls = np.array([trait.trait_class(x) if x else "" for x in acts])
    rows_by = {c: np.flatnonzero(cls == c) for c in (trait.SMOOTH, trait.ABRUPT)}
    cont = np.array([trait.is_contentious(x) if x else False for x in acts])
    hd, n_hd = hausdorff_table(cfg, a.preds)
    if hd:
        print("\n=== Hausdorff distance between forecast and truth curves "
              "(lower better; x = ratio to persistence) ===")
        print(f"{'model':16s} " + " ".join(f"{ch:>18s}" for ch in cfg.channels))
        for m in sorted(hd):
            print(f"{m:16s} " + " ".join(
                f"{hd[m][ch][0]:9.4f} ({hd[m][ch][1]:5.2f}x)" if ch in hd[m]
                else " " * 18 for ch in cfg.channels))
        print("Scaled per forecast: time spans [0,1] over the horizon, value is divided by "
              "the truth's own standard deviation there. Unlike MSE this is not pointwise, "
              "so a flat forecast through an oscillation is charged roughly its amplitude.")
        for m in sorted(hd):
            for ch, (v, r) in hd[m].items():
                rows.append(dict(scope="overall", subset="all", model=m, channel=ch,
                                 metric="hausdorff", value=v, n_clips=n_hd))
                rows.append(dict(scope="overall", subset="all", model=m, channel=ch,
                                 metric="hausdorff_ratio_vs_persistence", value=r,
                                 n_clips=n_hd))

    print(f"\n=== G2: smooth ({len(rows_by[trait.SMOOTH])} clips) vs "
          f"abrupt ({len(rows_by[trait.ABRUPT])}) ===")
    print(f"{'model':16s} {'class':8s} " + " ".join(f"{c:>12s}" for c in chans))
    for m in models:
        for c in (trait.SMOOTH, trait.ABRUPT):
            r = aggregate.r2(st, m, rows=rows_by[c])
            print(f"{m:16s} {c:8s} " + " ".join(f"{v:12.4f}" for v in r.per_channel))
            for ci, ch in enumerate(chans):
                rows.append(dict(scope="trait", subset=c, model=m, channel=ch,
                                 metric="R2", value=float(r.per_channel[ci]),
                                 n_clips=len(rows_by[c])))

    print(f"\n=== G2: dR2 = R2(smooth) - R2(abrupt), with {a.boot}x bootstrap CI ===")
    print("Two independent resamples, one per class: the classes are disjoint sets of "
          "clips, so there is nothing to pair.")
    print(f"{'model':16s} " + " ".join(f"{c:>22s}" for c in chans))
    for m in models:
        d = aggregate.delta_r2(aggregate.r2(st, m, rows=rows_by[trait.SMOOTH]),
                               aggregate.r2(st, m, rows=rows_by[trait.ABRUPT]))
        def stat(ia, ib, _m=m):
            return aggregate.delta_r2(
                aggregate.r2(st, _m, rows=rows_by[trait.SMOOTH][ia]),
                aggregate.r2(st, _m, rows=rows_by[trait.ABRUPT][ib]))
        bs = bootstrap.bootstrap_two_sample(stat, len(rows_by[trait.SMOOTH]),
                                            len(rows_by[trait.ABRUPT]), b=a.boot, seed=0)
        cells = [f"{d[ci]:7.4f} [{bs.lo[ci]:6.3f},{bs.hi[ci]:6.3f}]" for ci in range(len(chans))]
        print(f"{m:16s} " + " ".join(cells))
        for ci, ch in enumerate(chans):
            rows.append(dict(scope="trait", subset="delta", model=m, channel=ch,
                             metric="dR2", value=float(d[ci]),
                             ci_lo=float(bs.lo[ci]), ci_hi=float(bs.hi[ci]),
                             n_clips=len(st.clip_ids)))

    print(f"\n=== G2 sensitivity: dropping the contentious subset "
          f"({int(cont.sum())} clips) ===")
    keep = {c: np.array([i for i in rows_by[c] if not cont[i]]) for c in rows_by}
    n_ab = sum(1 for i in rows_by[trait.ABRUPT] if cont[i])
    frac = n_ab / max(int(cont.sum()), 1)
    print(f"smooth {len(keep[trait.SMOOTH])} | abrupt {len(keep[trait.ABRUPT])} | "
          f"{frac:.0%} of the dropped clips are abrupt -- the two arms are NOT trimmed "
          f"equally, so this pair of tables is not a symmetric robustness check and the "
          f"difference between them is mostly a change in what abrupt contains")
    for m in models:
        d = aggregate.delta_r2(aggregate.r2(st, m, rows=keep[trait.SMOOTH]),
                               aggregate.r2(st, m, rows=keep[trait.ABRUPT]))
        print(f"{m:16s} " + " ".join(f"{v:12.4f}" for v in d))
        for ci, ch in enumerate(chans):
            rows.append(dict(scope="trait", subset="delta_no_contentious", model=m,
                             channel=ch, metric="dR2", value=float(d[ci]),
                             n_clips=len(keep[trait.SMOOTH]) + len(keep[trait.ABRUPT])))

    # ---- construct validity: is the contrast just two actions? -------------------------
    print("\n=== per action (>=30 clips), R2 of each model ===")
    cnt = collections.Counter(acts[acts != ""])
    big = [x for x, n in cnt.most_common() if n >= 30]
    print(f"{'action':18s} {'n':>5s} {'class':8s} " + " ".join(f"{m[:11]:>11s}" for m in models))
    for act in big:
        sel = np.flatnonzero(acts == act)
        vals = [aggregate.r2(st, m, rows=sel).per_channel.mean() for m in models]
        print(f"{act[:18]:18s} {len(sel):5d} {trait.trait_class(act):8s} "
              + " ".join(f"{v:11.4f}" for v in vals))
        for m, v in zip(models, vals):
            rows.append(dict(scope="action", subset=act, model=m, channel="mean",
                             metric="R2", value=float(v), n_clips=len(sel)))

    print("\n=== leave-one-action-out: dR2 with that action removed from its class ===")
    print("If dropping one action flips or erases the contrast, the contrast was that "
          "action, not the trait.")
    for m in models:
        base = aggregate.delta_r2(aggregate.r2(st, m, rows=rows_by[trait.SMOOTH]),
                                  aggregate.r2(st, m, rows=rows_by[trait.ABRUPT])).mean()
        line = [f"{m:16s} full {base:7.4f}"]
        for act in ("picking up", "holding"):
            sm = np.array([i for i in rows_by[trait.SMOOTH] if acts[i] != act])
            ab = np.array([i for i in rows_by[trait.ABRUPT] if acts[i] != act])
            if len(sm) == 0 or len(ab) == 0:
                continue
            v = aggregate.delta_r2(aggregate.r2(st, m, rows=sm),
                                   aggregate.r2(st, m, rows=ab)).mean()
            line.append(f"| without {act}: {v:7.4f}")
            rows.append(dict(scope="loao", subset=f"drop {act}", model=m, channel="mean",
                             metric="dR2", value=float(v), n_clips=len(sm) + len(ab)))
        print("  " + " ".join(line))

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    cols = ["scope", "subset", "model", "channel", "metric", "value", "ci_lo", "ci_hi",
            "n_clips"]
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {a.out} ({len(rows)} rows)")
    dc = dc_share(a.preds)
    if np.isfinite(dc) and dc > 0.9:
        print(f"F is {dc:.1%} DC in the scored target, so read every number above as 'how "
              f"well the constant plus its drift was reproduced'.")
    elif np.isfinite(dc):
        print(f"F is {dc:.1%} DC in the scored target -- the baseline has been removed, so "
              f"these numbers are about the dynamics, not about reproducing a constant. "
              f"They are NOT comparable in absolute terms with runs on the uncorrected "
              f"target; skill against persistence is.")
    else:
        print("Could not measure the DC share of F from these predictions, so the usual "
              "caveat cannot be stated either way -- check the cache before reading the "
              "numbers as being about dynamics.")


if __name__ == "__main__":
    raise SystemExit(main())
