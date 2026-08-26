"""Score d256's baselines over the LOSO folds, and score an external model on the same origins.

Reuses the ActionSense harness end to end -- baselines, masking, metrics, the rolling-origin
batcher -- because d256's target is the same 6-dim both-hands vector. Nothing about
persistence, seasonal or AR is dataset-specific; what is specific is the protocol around them,
which is all this file adds:

  * FIVE FOLDS, NOT ONE SPLIT. Each subject is held out in turn, so every baseline is fitted
    five times and every number is a mean over folds with the spread reported. A single-split
    number would hide that fold 2 (S03) tests on 15 classes while the others test on 20.
  * FIT ON TRAIN, SELECT ON VAL, TOUCH TEST ONCE. Norm, force thresholds, AR coefficients and
    seasonal periods all come from that fold's TRAIN; AR order is chosen on that fold's VAL.
    Recomputing Norm per fold matters: a global Norm fitted over all subjects would leak the
    held-out subject's scale into its own test scores.

`score_external` exists so the probGRU arm is scored on EXACTLY the origins the baselines were
scored on -- same mask, same metrics, same folds. A model evaluated on its own windows is not
comparable to a baseline evaluated on the harness's.
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from src.actionsense.eval_harness import masking, metrics
from src.actionsense.eval_harness.baselines import base as BL
from src.actionsense.eval_harness.baselines.ar import AR
from src.actionsense.eval_harness.baselines.persistence import Persistence
from src.actionsense.eval_harness.baselines.seasonal import SeasonalNaive
from src.actionsense.eval_harness.config import Config, load_config
from src.shape_metrics import hausdorff_scaled

from .dataset import Norm, force_thresholds, group_keys, load_group
from . import splits as S

MODELS = ("persistence", "seasonal", "ar")
CLASSES = {"persistence": Persistence, "seasonal": SeasonalNaive, "ar": AR}
DEFAULT_CONFIG = os.path.join("configs", "d256", "eval_harness.yaml")


def _hand(channel: str) -> str:
    return "left" if channel.endswith("_L") else "right"


def _result(ytrue, yhat, mask) -> dict:
    return {
        "ch_mse": metrics.masked_channel_mse(ytrue, yhat, mask),
        "ch_mae": metrics.masked_channel_mae(ytrue, yhat, mask),
        "hz_mse": metrics.masked_horizon_mse(ytrue, yhat, mask),
        "hz_mae": metrics.masked_horizon_mae(ytrue, yhat, mask),
        "hausdorff": _hausdorff_per_channel(ytrue, yhat, mask),
        "n": mask.reshape(-1, mask.shape[-1]).sum(0),
    }


def _hausdorff_per_channel(ytrue, yhat, mask) -> np.ndarray:
    """Mean scaled Hausdorff per channel, on the same masked forecasts as the MSE.

    Beside MSE because MSE is pointwise: a flat forecast through the middle of an
    oscillation scores far better than its shape deserves, which is what every arm in this
    project has done (SESSION_LOG 2026-08-20). Hausdorff charges a flat line roughly the
    amplitude it failed to follow.

    A forecast is dropped when its horizon is entirely masked or its truth is constant over
    the horizon (hausdorff_scaled returns NaN there -- no shape to compare, and calling it a
    perfect match would flatter every model).
    """
    C = ytrue.shape[-1]
    out = np.full(C, np.nan)
    for c in range(C):
        keep = mask[:, :, c].all(axis=1)
        if not keep.any():
            continue
        hd = hausdorff_scaled(yhat[keep, :, c], ytrue[keep, :, c])
        if np.isfinite(hd).any():
            out[c] = float(np.nanmean(hd))
    return out


def fold_context(cfg: Config, fold: dict):
    """Everything a fold's scoring needs, all TRAIN-derived. Shared by the baselines and by
    any external model, so both are normalized and masked identically."""
    train = load_group(cfg, fold["train"])
    val = load_group(cfg, fold["val"])
    test = load_group(cfg, fold["test"])
    tr = fold["train"]
    return {
        "train": train, "val": val, "test": test,
        "gtr": group_keys(cfg, tr, tr),
        "gva": group_keys(cfg, fold["val"], tr),
        "gte": group_keys(cfg, fold["test"], tr),
        "norm": Norm.from_train(train),
        "thr": force_thresholds(cfg, train),
    }


def run_fold(cfg: Config, fold: dict, ctx: dict | None = None):
    """Fit/select/score every baseline on one fold. Returns (results, mask, extras)."""
    ctx = ctx or fold_context(cfg, fold)
    C = len(cfg.channels)
    results, mask, extras = {}, None, {}
    for name in MODELS:
        bl = CLASSES[name](cfg, ctx["norm"])
        bl.fit(ctx["train"], ctx["gtr"])
        bl.select(ctx["val"], ctx["gva"], cfg.horizon)
        ytrue, yhat = BL.predict_series(bl, ctx["test"], ctx["gte"], cfg)
        if mask is None:
            mask = masking.valid_mask(cfg, ytrue.reshape(-1, C), ctx["thr"]).reshape(ytrue.shape)
        results[name] = _result(ytrue, yhat, mask)
        if name == "seasonal":
            extras["seasonal_periods"] = dict(bl.periods)
        if name == "ar":
            extras["ar_orders"] = dict(bl.order)
    return results, mask, extras


def score_external(cfg: Config, fold: dict, name: str, preds: dict[int, np.ndarray],
                   ctx: dict | None = None) -> dict:
    """Score a model's own forecasts on the harness's origins.

    `preds[idx]` must be (n_origins(idx), H, 6) in RAW units, ordered by
    `baselines.base.origins(T, cfg)` for that recording -- i.e. produced by iterating the same
    origins in the same order. The ordering is asserted by shape, so a model that used its own
    window sampler fails loudly here instead of being silently compared on different data.
    """
    ctx = ctx or fold_context(cfg, fold)
    C, H = len(cfg.channels), cfg.horizon
    yts, yhs = [], []
    for i, Y in sorted(ctx["test"].items()):
        orig = BL.origins(len(Y), cfg)
        if i not in preds:
            raise KeyError(f"{name}: no forecasts for test recording {i}")
        P = np.asarray(preds[i])
        if P.shape != (len(orig), H, C):
            raise ValueError(
                f"{name}: recording {i} gave {P.shape}, harness expects "
                f"{(len(orig), H, C)} -- the model did not use the harness origins")
        for k, t in enumerate(orig):
            yts.append(Y[t + 1:t + 1 + H])
            yhs.append(P[k])
    ytrue, yhat = np.stack(yts), np.stack(yhs)
    mask = masking.valid_mask(cfg, ytrue.reshape(-1, C), ctx["thr"]).reshape(ytrue.shape)
    return _result(ytrue, yhat, mask)


def build_rows(cfg: Config, per_fold: list[dict]) -> list[dict]:
    """Tidy long rows, one per (fold, model, channel, horizon_step, metric)."""
    rows = []
    H, chans = cfg.horizon, cfg.channels
    for fold_i, (fold, results) in enumerate(per_fold):
        names = list(results)

        def emit(model, ci, step, metric, value, n):
            rows.append({"fold": fold_i, "held_out": fold["held_out"], "model": model,
                         "channel": chans[ci], "hand": _hand(chans[ci]),
                         "horizon_step": str(step), "metric": metric,
                         "value": float(value), "n_frames": int(n),
                         "config_hash": cfg.config_hash})

        for m in names:
            R = results[m]
            for ci in range(len(chans)):
                n = R["n"][ci]
                for h in range(H):
                    emit(m, ci, h + 1, "MSE", R["hz_mse"][h, ci], n)
                    emit(m, ci, h + 1, "MAE", R["hz_mae"][h, ci], n)
                    for b in names:
                        emit(m, ci, h + 1, f"SS_vs_{b}",
                             metrics.skill(R["hz_mse"][h, ci], results[b]["hz_mse"][h, ci]), n)
                emit(m, ci, "all", "MSE", R["ch_mse"][ci], n)
                emit(m, ci, "all", "MAE", R["ch_mae"][ci], n)
                emit(m, ci, "all", "Hausdorff", R["hausdorff"][ci], n)
                for b in names:
                    emit(m, ci, "all", f"SS_vs_{b}",
                         metrics.skill(R["ch_mse"][ci], results[b]["ch_mse"][ci]), n)
                    # Ratio, not difference: Hausdorff is dimensionless but its absolute
                    # scale is arbitrary, so only comparisons between models carry meaning.
                    ref = results[b]["hausdorff"][ci]
                    emit(m, ci, "all", f"HD_ratio_vs_{b}",
                         R["hausdorff"][ci] / ref if ref and np.isfinite(ref) else np.nan, n)
    return rows


def summarize(cfg: Config, rows: list[dict], ref: str = "persistence") -> str:
    """Headline: mean skill vs `ref` across folds, per model and channel. Must be > 0."""
    import collections
    agg = collections.defaultdict(list)
    for r in rows:
        if r["horizon_step"] == "all" and r["metric"] == f"SS_vs_{ref}":
            agg[(r["model"], r["channel"])].append(r["value"])
    out = [f"mean skill vs {ref} over folds (>0 means better than {ref})",
           f"  {'model':12s} " + " ".join(f"{c:>12s}" for c in cfg.channels)]
    for m in sorted({k[0] for k in agg}):
        if m == ref:
            continue
        cells = []
        for c in cfg.channels:
            v = agg.get((m, c), [])
            cells.append(f"{np.mean(v):+.3f}±{np.std(v):.2f}" if v else f"{'-':>12s}")
        out.append(f"  {m:12s} " + " ".join(f"{s:>12s}" for s in cells))
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--out", default=None, help="override paths.out_csv")
    ap.add_argument("--folds", default=None, help="comma list, e.g. 0,1 (default: all)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    fs = S.folds(cfg)
    S.save(cfg, fs)
    print(S.summarize(cfg, fs))
    if args.folds:
        keep = {int(x) for x in args.folds.split(",")}
        fs = [f for f in fs if f["fold"] in keep]

    per_fold, extras = [], {}
    for f in fs:
        print(f"\n[fold {f['fold']}] held out {f['held_out']} ...", flush=True)
        results, _, ex = run_fold(cfg, f)
        per_fold.append((f, results))
        extras[f["fold"]] = ex
        for m in MODELS:
            print(f"    {m:12s} MSE/channel " +
                  " ".join(f"{v:9.4g}" for v in results[m]["ch_mse"]))

    rows = build_rows(cfg, per_fold)
    out_csv = args.out or cfg.abspath("out_csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    import pandas as pd
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\nwrote {len(rows)} rows -> {out_csv}  (config_hash {cfg.config_hash})")
    print()
    print(summarize(cfg, rows))


if __name__ == "__main__":
    main()
