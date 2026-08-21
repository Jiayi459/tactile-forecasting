"""Is skill ~0.4 the models falling short, or the data running out?

After D1 the corrected F oscillates fast, and every model -- AR, probGRU, persistence,
seasonal -- forecasts something close to a flat local mean (2026-08-20 forecast figures).
Two readings fit that: the fast component is real dynamics nobody has modelled yet, or it
is measurement noise, which is unpredictable by construction and makes "predict the mean"
close to optimal. They call for opposite next steps, so this measures which.

THE ARGUMENT. Write the signal as x = s + e, a smooth part plus white noise of variance
v_e. Then:

  * lag-1 autocorrelation. A smooth s has r_s(1) close to 1, so the drop from r(0)=1 to
    r(1) is essentially the noise share: v_e ~ (1 - r(1)) * var(x). White noise gives
    r(1) ~ 0; a signal with real fast structure keeps r(1) high.

  * A CEILING ON SKILL, which is the part that matters. Persistence's error at horizon h is
    (s_{t+h} - s_t) + e_{t+h} - e_t, so its MSE is D_h + 2*v_e. An oracle that knew s
    exactly still cannot beat v_e. So

        skill_max(h) = 1 - v_e / MSE_persistence(h)

    and every term on the right is measurable. If the signal were PURE white noise, D_h = 0
    and the ceiling is exactly 1 - v_e/(2*v_e) = 0.5 -- no forecaster of any kind can do
    better than 0.5 against persistence on pure noise.

So an observed skill sitting just under the computed ceiling means the models are near the
limit of what the data supports, and the honest next move is to report that limit rather
than to train something bigger.

THE ASSUMPTION, stated because it is doing real work: v_e ~ (1 - r(1)) * var(x) treats the
smooth part as perfectly correlated at one frame. If s itself decorrelates within a frame
the noise is overestimated and the ceiling comes out too low, so this is a CONSERVATIVE
bound on how much room the models still have. A second estimate from the lag-2
autocorrelation is printed alongside as a cross-check: for white noise plus a locally
linear s, v_e also equals (r(1) - r(2))/(1 - r(1)) corrected... rather than rely on that
algebra, the script simply reports r(1), r(2) and r(3) so the shape is visible.

    python scripts/opentouch_predictability_ceiling.py --cache ~/opentouch/cache_d1 \
        --compare-cache ~/opentouch/cache --csv docs/opentouch_cv4_d1.csv
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os

import numpy as np


def clips(cache, limit):
    rows = [json.loads(l) for l in open(os.path.join(cache, "manifest.jsonl")) if l.strip()]
    out = []
    for r in rows:
        p = os.path.join(cache, f"state_{r['idx']}.npy")
        if os.path.exists(p):
            out.append((r["idx"], np.load(p)[:, 0, :3].astype(np.float64)))
        if len(out) >= limit:
            break
    return out


def autocorr(x, max_lag):
    """r(k) for k=0..max_lag on the mean-removed clip; NaN when the clip is too short."""
    x = x - x.mean()
    v = float((x * x).mean())
    if v <= 0 or len(x) <= max_lag + 1:
        return np.full(max_lag + 1, np.nan)
    return np.array([1.0] + [float((x[:-k] * x[k:]).mean()) / v
                             for k in range(1, max_lag + 1)])


def measure(cache, limit, max_lag, chans):
    data = clips(cache, limit)
    if not data:
        raise SystemExit(f"no state_*.npy under {cache}")
    R = {c: [] for c in range(len(chans))}
    var = {c: [] for c in range(len(chans))}
    diffs = collections.defaultdict(list)          # (channel, h) -> var(x_{t+h}-x_t)
    for _, s in data:
        for c in range(len(chans)):
            x = s[:, c]
            r = autocorr(x, max_lag)
            if np.isfinite(r).all():
                R[c].append(r)
                var[c].append(float(x.var()))
                for h in (1, 5, 10, 20, 30):
                    if len(x) > h:
                        diffs[(c, h)].append(float(((x[h:] - x[:-h]) ** 2).mean()))
    return data, R, var, diffs


def sharpness(preds_dir, chans, max_lag):
    """Mean predicted sigma per channel and horizon -> is the band as narrow as it could be?

    Coverage alone does not answer that. A model emitting the global mean with the global
    standard deviation also covers ~95%, so a wide, well-covering band is not evidence that
    anything was learned. What matters is sharpness GIVEN calibration: how close the
    predicted sigma sits to the irreducible noise floor sqrt(v_e) measured above. At the
    floor the band cannot be narrowed by any model; far above it, the model is uncertain
    where it need not be, and that gap is trainable.
    """
    import glob
    # Sums and counts, not a mean per clip: clips too short to yield any origin give an
    # EMPTY slice, whose mean is NaN, and one NaN in a list poisons the aggregate. That is
    # what turned the whole 2026-08-21 sharpness table into "nan (nanx)". Accumulating lets
    # an empty clip contribute nothing, and weights each clip by how many origins it has
    # rather than counting a 2-origin clip equally with a 200-origin one.
    tot = collections.defaultdict(float)
    cnt = collections.defaultdict(int)
    # sigma against the model's OWN realised error at the same origins. The noise floor is
    # an ORACLE bound -- it assumes the smooth component's value at t+h is known -- so a
    # real model's sigma exceeds it by however much it does not know where the envelope is
    # going, and sigma/floor therefore measures ignorance, not miscalibration. sigma/RMSE is
    # the calibration question, and it is computed here from one file so it cannot mix the
    # two MSE conventions in this repo (the ceiling's median-over-clips against the driver's
    # frame-pooled skill), which are not interconvertible.
    sse = collections.defaultdict(float)
    scn = collections.defaultdict(int)
    for f in sorted(glob.glob(os.path.join(preds_dir, "clip_*.npz"))):
        z = np.load(f, allow_pickle=True)
        for k in z.files:
            if not k.startswith("sigma_"):
                continue
            sg = np.asarray(z[k], dtype=float)          # (n_origins, H, C)
            if sg.ndim != 3 or sg.shape[0] == 0:
                continue
            mdl = k[6:]
            mu = np.asarray(z.get(f"mu_{mdl}"), dtype=float)
            y, ors = np.asarray(z["y"], dtype=float), np.asarray(z["origins"])
            for c in range(min(sg.shape[-1], len(chans))):
                for h in (1, 5, 10, 20, 30):
                    if h > sg.shape[1]:
                        continue
                    col = sg[:, h - 1, c]
                    good = np.isfinite(col)
                    if good.any():
                        tot[(mdl, c, h)] += float(col[good].sum())
                        cnt[(mdl, c, h)] += int(good.sum())
                    if mu is None or mu.shape != sg.shape:
                        continue
                    tgt = ors + h                        # mu[:, h-1] predicts frame o+h
                    ok = tgt < len(y)
                    if ok.any():
                        e = y[tgt[ok], c] - mu[ok, h - 1, c]
                        e = e[np.isfinite(e)]
                        sse[(mdl, c, h)] += float((e ** 2).sum())
                        scn[(mdl, c, h)] += int(e.size)
    out = collections.defaultdict(dict)
    for key, n in cnt.items():
        mdl, c, h = key
        rmse = np.sqrt(sse[key] / scn[key]) if scn.get(key) else float("nan")
        out[(mdl, c)][h] = (tot[key] / n, rmse)
    return out


def observed_skill(csv_path):
    """{(model, channel, h): skill} from a driver metric table, if one is given."""
    out = collections.defaultdict(list)
    try:
        for r in csv.DictReader(open(csv_path)):
            if r["metric"] == "SS_vs_persistence" and r["model"] != "persistence":
                h = r["horizon_step"]
                out[(r["model"], r["channel"], h)].append(float(r["value"]))
    except (OSError, KeyError):
        return {}
    return {k: sum(v) / len(v) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=os.path.expanduser("~/opentouch/cache_d1"))
    ap.add_argument("--compare-cache", help="e.g. the uncorrected cache, for contrast")
    ap.add_argument("--csv", help="a driver metric table, to print observed skill beside "
                                  "the ceiling")
    ap.add_argument("--preds", help="a saved prediction dir, to compare the model's own "
                                    "sigma against the measured noise floor (sharpness)")
    ap.add_argument("--limit", type=int, default=600, help="clips to measure")
    ap.add_argument("--max-lag", type=int, default=30)
    a = ap.parse_args()
    chans = ["F_R", "CoPx_R", "CoPy_R"]

    for tag, cache in [("D1", a.cache)] + ([("raw", a.compare_cache)]
                                           if a.compare_cache else []):
        _, R, var, diffs = measure(cache, a.limit, a.max_lag, chans)
        print(f"\n=== {tag}: {cache} ===")
        print(f"{'channel':9s} {'r(1)':>7s} {'r(2)':>7s} {'r(3)':>7s} {'r(10)':>7s} "
              f"{'noise share':>12s} {'first r<0.2':>12s}")
        noise = {}
        for c, ch in enumerate(chans):
            if not R[c]:
                continue
            r = np.median(np.stack(R[c]), axis=0)
            below = np.flatnonzero(r < 0.2)
            noise[c] = (1.0 - r[1]) * float(np.median(var[c]))
            print(f"{ch:9s} {r[1]:7.3f} {r[2]:7.3f} {r[3]:7.3f} {r[10]:7.3f} "
                  f"{1 - r[1]:12.1%} {int(below[0]) if below.size else -1:12d}")

        print(f"\n{'channel':9s} {'h':>3s} {'MSE_persist':>13s} {'noise var':>12s} "
              f"{'skill ceiling':>14s}   observed")
        obs = observed_skill(a.csv) if a.csv else {}
        for c, ch in enumerate(chans):
            if c not in noise:
                continue
            for h in (1, 5, 10, 20, 30):
                d = diffs.get((c, h))
                if not d:
                    continue
                mse_p = float(np.median(d))
                ceil = 1.0 - noise[c] / mse_p if mse_p > 0 else float("nan")
                got = "  ".join(f"{m} {obs[(m, ch, str(h))]:.3f}"
                                for m in ("ar", "prob_gru")
                                if (m, ch, str(h)) in obs)
                print(f"{ch:9s} {h:3d} {mse_p:13.4g} {noise[c]:12.4g} {ceil:14.3f}   {got}")

    if a.preds:
        sh = sharpness(a.preds, chans, a.max_lag)
        if sh:
            print(f"\n=== sharpness: predicted sigma vs the noise floor ({a.preds}) ===")
            print("sigma, then (sigma/floor) and [sigma/own RMSE]. The first is how far "
                  "above an ORACLE's irreducible error the model sits -- ignorance about "
                  "the envelope, not a defect. The second is calibration: ~1.0 means the "
                  "band matches the errors the model actually makes.")
            print(f"{'model':10s} {'channel':9s} {'floor':>10s} " +
                  " ".join(f"{'h=' + str(h):>22s}" for h in (1, 5, 10, 20, 30)))
            for (mdl, c), per_h in sorted(sh.items()):
                if c not in noise:
                    continue
                fl = float(np.sqrt(noise[c]))
                cells = []
                for h in (1, 5, 10, 20, 30):
                    v = per_h.get(h)
                    if not v:
                        cells.append(" " * 22); continue
                    sg, rm = v
                    cells.append(f"{sg:7.4g} ({sg / fl:.2f}x)"
                                 f"[{sg / rm:.2f}]" if np.isfinite(rm)
                                 else f"{sg:7.4g} ({sg / fl:.2f}x)      ")
                print(f"{mdl:10s} {chans[c]:9s} {fl:10.4g} " + " ".join(cells))
            print("(x) near 1.0 would mean the band is as narrow as ANY model could be; "
                  "above 1.0 is how much the model does not know about where the smooth "
                  "component is heading, and that gap is the same one the skill table "
                  "shows at long horizons. [.] near 1.0 means honest: the band matches the "
                  "errors actually made. Well above 1.0 is over-dispersion that training "
                  "can attack; below 1.0 is overconfidence and should show as coverage "
                  "under nominal.")

    print("\nHOW TO READ IT. Pure white noise has r(1) ~ 0 and a ceiling of exactly 0.500: "
          "no forecaster can beat persistence by more than that on noise. An observed skill "
          "close to the ceiling means the models are near the data's limit and the finding "
          "to report is the limit; an observed skill far below it means there is structure "
          "left to capture. The ceiling is conservative -- it assumes the smooth part is "
          "perfectly correlated frame to frame, which overstates the noise.")


if __name__ == "__main__":
    raise SystemExit(main())
