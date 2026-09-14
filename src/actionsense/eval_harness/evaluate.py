"""Frozen evaluation entry point.

Protocol (constraint 2, no leakage): load the frozen split -> fit Norm + force thresholds on
TRAIN -> each baseline fit(TRAIN) then select hyperparameters on VAL -> forecast TEST (touched
once) -> score with the frozen metrics + CoP mask. Skill scores are computed on IDENTICAL masked
frame sets (the mask depends only on target-frame force, not the model). Output is a tidy long
table (CSV + parquet) stamped with the config hash; determinism is asserted (two runs identical).

Standard prediction format for scoring EXTERNAL models (e.g. the GRU) against this harness:
a dict {recording_idx: yhat} where yhat has shape (n_origins, H, 6) and n_origins/order match
baselines.origins(len(Y), cfg) for that recording (target-time indexed, h = 1..H). See
score_external() and the README section "Evaluating a new model".

    python -m src.actionsense.eval_harness.evaluate
    python -m src.actionsense.eval_harness.evaluate --model-preds preds.npz --model-name gru
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from . import baselines as BL
from . import masking, metrics
from .config import load_config, Config
from .dataset import Norm, force_thresholds, group_keys, load_group
from .splits import load_splits

MODELS = ["persistence", "seasonal", "ar"]
CLASSES = {"persistence": BL.Persistence, "seasonal": BL.SeasonalNaive, "ar": BL.AR}


def _hand(channel: str) -> str:
    return "L" if channel.endswith("_L") else "R"


def _result(ytrue, yhat, mask) -> dict:
    return {
        "hz_mse": metrics.masked_horizon_mse(ytrue, yhat, mask),
        "hz_mae": metrics.masked_horizon_mae(ytrue, yhat, mask),
        "ch_mse": metrics.masked_channel_mse(ytrue, yhat, mask),
        "ch_mae": metrics.masked_channel_mae(ytrue, yhat, mask),
        "n": mask.reshape(-1, 6).sum(0).astype(int),
    }


def fit_and_forecast(cfg: Config, splits: dict):
    """Fit/select every baseline; return (results, mask, extras). results[name] has masked
    metric arrays; extras carries the seasonal periods."""
    from src.calibration import prepare_fold
    cfg = prepare_fold(cfg, splits)
    train, val, test = (load_group(cfg, splits[k]) for k in ("train", "val", "test"))
    gtr = group_keys(cfg, splits["train"]); gva = group_keys(cfg, splits["val"])
    gte = group_keys(cfg, splits["test"])
    norm = Norm.from_train(train)
    thr = force_thresholds(cfg, train)

    results, mask, extras = {}, None, {}
    for name in MODELS:
        bl = CLASSES[name](cfg, norm)
        bl.fit(train, gtr)
        bl.select(val, gva, cfg.horizon)
        ytrue, yhat = BL.predict_series(bl, test, gte, cfg)
        if mask is None:
            mask = masking.valid_mask(cfg, ytrue.reshape(-1, 6), thr).reshape(ytrue.shape)
        results[name] = _result(ytrue, yhat, mask)
        if name == "seasonal":
            extras["seasonal_periods"] = dict(bl.periods)
            extras["ar_orders"] = None
        if name == "ar":
            extras["ar_orders"] = dict(bl.order)
    return results, norm, extras


def build_rows(cfg: Config, results: dict) -> list[dict]:
    """Tidy long rows: [model, channel, hand, horizon_step, metric, value, n_frames, config_hash]."""
    rows = []
    H, chans = cfg.horizon, cfg.channels

    def emit(model, ci, step, metric, value, n):
        rows.append({"model": model, "channel": chans[ci], "hand": _hand(chans[ci]),
                     "horizon_step": str(step), "metric": metric, "value": float(value),
                     "n_frames": int(n), "config_hash": cfg.config_hash})

    for m in MODELS:
        R = results[m]
        for ci in range(6):
            n = R["n"][ci]
            for h in range(H):
                emit(m, ci, h + 1, "MSE", R["hz_mse"][h, ci], n)
                emit(m, ci, h + 1, "MAE", R["hz_mae"][h, ci], n)
                for b in MODELS:
                    ss = metrics.skill(R["hz_mse"][h, ci], results[b]["hz_mse"][h, ci])
                    emit(m, ci, h + 1, f"SS_vs_{b}", ss, n)
            emit(m, ci, "all", "MSE", R["ch_mse"][ci], n)
            emit(m, ci, "all", "MAE", R["ch_mae"][ci], n)
            for b in MODELS:
                ss = metrics.skill(R["ch_mse"][ci], results[b]["ch_mse"][ci])
                emit(m, ci, "all", f"SS_vs_{b}", ss, n)
    return rows


def score_external(cfg: Config, splits: dict, name: str, preds: dict[int, np.ndarray],
                   ref_results: dict, norm: Norm) -> dict:
    """Score an external model's predictions against the frozen baselines. `preds[idx]` is
    (n_origins, H, 6) aligned to baselines.origins(len(Y), cfg). Returns a results-style dict and
    appends rows via build_rows when merged into `ref_results`."""
    from src.calibration import prepare_fold
    cfg = prepare_fold(cfg, splits)
    test = load_group(cfg, splits["test"])
    thr = force_thresholds(cfg, load_group(cfg, splits["train"]))
    yts, yhs = [], []
    for i, Y in sorted(test.items()):
        ors = BL.origins(len(Y), cfg)
        if i not in preds or preds[i].shape[0] != len(ors):
            raise ValueError(f"preds[{i}] must have shape ({len(ors)}, {cfg.horizon}, 6)")
        for j, t in enumerate(ors):
            yts.append(Y[t + 1:t + 1 + cfg.horizon]); yhs.append(preds[i][j])
    ytrue, yhat = np.stack(yts), np.stack(yhs)
    mask = masking.valid_mask(cfg, ytrue.reshape(-1, 6), thr).reshape(ytrue.shape)
    return _result(ytrue, yhat, mask)


def write_table(rows: list[dict], out_csv: str) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    df.to_parquet(os.path.splitext(out_csv)[0] + ".parquet", index=False)
    return df


def _fingerprint(results: dict) -> np.ndarray:
    return np.concatenate([results[m]["hz_mse"].ravel() for m in MODELS]
                          + [results[m]["ch_mse"].ravel() for m in MODELS])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--rebuild-splits", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--model-preds", default=None, help=".npz of {idx: (n_origins,H,6)} to score")
    ap.add_argument("--model-name", default="model")
    args = ap.parse_args()
    cfg = load_config(args.config) if args.config else load_config()
    splits = load_splits(cfg, rebuild=args.rebuild_splits)
    from src.calibration import prepare_fold
    cfg = prepare_fold(cfg, splits)
    print(f"config_hash={cfg.config_hash}  fps={cfg.fps:.0f}  horizon={cfg.horizon} steps  "
          f"fit_scope={cfg.fit_scope}  split(train {len(splits['train'])}/val {len(splits['val'])}"
          f"/test {len(splits['test'])})")

    results, norm, extras = fit_and_forecast(cfg, splits)
    assert np.array_equal(_fingerprint(results), _fingerprint(fit_and_forecast(cfg, splits)[0])), \
        "NON-DETERMINISTIC: two runs differ"
    print("determinism check: PASS (two runs identical)")

    print("\nseasonal periods (frames) per group:")
    for g, T in sorted(extras["seasonal_periods"].items()):
        print(f"  {g:20} T={T}" + ("  (fallback->persistence)" if not T else f"  ({T/cfg.fps:.2f}s)"))
    print("AR order per group:", {g: p for g, p in sorted(extras["ar_orders"].items())})

    if args.model_preds:
        from src.calibration import load_external_predictions
        preds = load_external_predictions(args.model_preds, cfg, splits)
        results[args.model_name] = score_external(cfg, splits, args.model_name, preds, results, norm)
        MODELS.append(args.model_name)

    rows = build_rows(cfg, results)
    out = args.out or cfg.abspath("out_csv")
    df = write_table(rows, out)

    # sidecar: estimated fit parameters per group (seasonal period + AR order), for inspection
    fit_rows = [{"group": g, "seasonal_period_frames": extras["seasonal_periods"].get(g),
                 "seasonal_period_s": (None if not extras["seasonal_periods"].get(g)
                                       else round(extras["seasonal_periods"][g] / cfg.fps, 3)),
                 "ar_order": extras["ar_orders"].get(g), "config_hash": cfg.config_hash}
                for g in sorted(extras["ar_orders"])]
    pd.DataFrame(fit_rows).to_csv(os.path.splitext(out)[0] + "_fitparams.csv", index=False)

    # console summary: full-horizon skill vs persistence (computed straight from results)
    print(f"\n{'model':12}{'nRMSE':>8}   full-horizon skill vs persistence (per channel)")
    ref = results["persistence"]["ch_mse"]
    for m in MODELS:
        r = results[m]
        nrmse = metrics.normalized_rmse(r["ch_mse"], norm.std)
        ss = metrics.skill(r["ch_mse"], ref)
        sk = "  ".join(f"{c[:6]}:{v:+.2f}" for c, v in zip(cfg.channels, ss))
        print(f"{m:12}{nrmse:>8.3f}   {sk}")
    print(f"\n[done] {out}  (+ .parquet)  rows={len(df)}")


if __name__ == "__main__":
    main()
