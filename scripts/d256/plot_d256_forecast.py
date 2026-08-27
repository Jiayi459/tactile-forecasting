#!/usr/bin/env python3
"""History, truth, and each model's 1 s forecast for d256 -- one figure per channel.

The d256 counterpart of docs/opentouch/raw/opentouch_forecast_*.png. Answers the question MSE
cannot: not "how far off", but "what shape did it draw". A flat line through a swinging signal
scores respectably on MSE and is obvious here, which is why this exists alongside the table
(SESSION_LOG 2026-08-20, and the Hausdorff column added for the same reason).

Forecasts are RE-COMPUTED from the saved checkpoint rather than dumped during training: the
checkpoint carries its hp and config_hash, the fold is rebuilt deterministically from the same
config, and Norm/FeatNorm are refit from that fold's TRAIN exactly as training did. So nothing
here can silently disagree with what was scored -- and it costs no forecast storage.

Usage:
    python scripts/d256/plot_d256_forecast.py --run runs/d256_probgru_none
    python scripts/d256/plot_d256_forecast.py --run runs/d256_probgru_none --fold 2 --n 4
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.actionsense.eval_harness.baselines.base import origins, predict_series  # noqa: E402
from src.actionsense.eval_harness.config import load_config  # noqa: E402
from src.d256 import dataset as D  # noqa: E402
from src.d256 import evaluate as E  # noqa: E402
from src.d256 import prob_gru as PG  # noqa: E402
from src.d256 import splits as S  # noqa: E402


def _window(y, fps, seconds, start):
    """Which slice of the recording to draw.

    Default picks the most active stretch rather than the first N seconds: a recording often
    opens with the hand at rest, and a panel of flat line says nothing about any model.
    Activity = rolling standard deviation over the window length.
    """
    T = len(y)
    if not seconds or seconds <= 0 or T / fps <= seconds:
        return 0.0, T / fps
    w = int(seconds * fps)
    if start is not None:
        lo = int(np.clip(start * fps, 0, max(0, T - w)))
        return lo / fps, (lo + w) / fps
    best, best_v = 0, -1.0
    for lo in range(0, T - w + 1, max(1, w // 4)):
        v = float(np.std(y[lo:lo + w]))
        if v > best_v:
            best, best_v = lo, v
    return best / fps, (best + w) / fps


def load_model(run: str, fold: int, cfg):
    import torch
    path = os.path.join(run, f"fold{fold}.pt")
    if not os.path.exists(path):
        sys.exit(f"no checkpoint at {path} -- has the run finished?")
    ck = torch.load(path, map_location="cpu")
    if ck.get("config_hash") != cfg.config_hash:
        sys.exit(f"checkpoint was trained under config_hash {ck.get('config_hash')} but the "
                 f"config now hashes to {cfg.config_hash}; the protocol changed since.")
    return ck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/d256_probgru_none")
    ap.add_argument("--config", default=os.path.join("configs", "d256", "eval_harness.yaml"))
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--n", type=int, default=3, help="test recordings to draw (columns)")
    ap.add_argument("--seconds", type=float, default=20.0,
                    help="span to draw per panel. OpenTouch's clips are ~3.5 s so it drew them "
                         "whole; d256 recordings run to 178 s, and at 6 Hz a whole one is too "
                         "dense to read the shape off -- which is the only thing this figure "
                         "is for. 0 draws everything.")
    ap.add_argument("--start", type=float, default=None,
                    help="window start in seconds (default: the most active stretch)")
    ap.add_argument("--outdir", default=os.path.join("docs", "d256", "forecast"))
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch

    cfg = load_config(args.config)
    fold = next(f for f in S.folds(cfg) if f["fold"] == args.fold)
    ck = load_model(args.run, args.fold, cfg)
    hp = {**PG.DEFAULT_HP, **ck["hp"]}

    # Refit exactly as training did: the fold's TRAIN only.
    train = D.load_group(cfg, fold["train"])
    norm = D.Norm.from_train(train)
    fnorm = PG.FeatNorm.from_train(cfg, fold["train"], hp["features"] == "raw+df")
    aids, n_act = PG.action_ids(cfg, hp["arm"], fold["train"] + fold["val"] + fold["test"])
    model = PG.ProbGRU(PG.feature_dim(hp), n_act, hp["hidden"], PG.N_OUT, hp["dropout"])
    model.load_state_dict(ck["state_dict"]); model.eval()

    rows = {r["idx"]: r for r in D.eligible_recordings(cfg)}
    picks = sorted(fold["test"], key=lambda i: -rows[i]["T"])[:args.n]
    preds = PG.forecast(model, cfg, picks, hp, norm, fnorm, aids, device="cpu")

    # The classical arms, fit on the same fold, so the drawn curves are the scored ones.
    ctx = E.fold_context(cfg, fold)
    classical = {}
    for name in E.MODELS:
        bl = E.CLASSES[name](cfg, ctx["norm"])
        bl.fit(ctx["train"], ctx["gtr"]); bl.select(ctx["val"], ctx["gva"], cfg.horizon)
        classical[name] = bl

    os.makedirs(args.outdir, exist_ok=True)
    H, fps = cfg.horizon, cfg.fps

    # Layout copied from docs/opentouch/raw/opentouch_forecast_*.png deliberately: ROWS ARE
    # MODELS, columns are recordings. Overlaying every model in one panel -- the first version
    # here -- hides exactly what these plots exist to show, which is the SHAPE each model
    # draws. A flat line and a lagging line look alike when four curves share an axis.
    MODEL_ROWS = [("ar", "tab:blue"), ("persistence", "0.35"),
                  ("probgru", "tab:red"), ("seasonal", "tab:green")]
    UNITS = {"F": "total force (a.u.)", "CoPx": "CoP x  [-1,1]", "CoPy": "CoP y  [-1,1]"}

    for ci, ch in enumerate(cfg.channels):
        nrow, ncol = len(MODEL_ROWS), len(picks)
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.3 * ncol, 2.5 * nrow),
                                 squeeze=False, sharex="col")
        for cj, i in enumerate(picks):
            Y = D.load_target(cfg, i)
            t = np.arange(len(Y)) / fps
            orig = origins(len(Y), cfg)
            lo_t, hi_t = _window(Y[:, ci], fps, args.seconds, args.start)
            # Non-overlapping origins, so successive forecast segments tile the axis instead
            # of painting over each other -- the same reading the OpenTouch figure gives.
            keep = orig[::H]
            for ri, (mname, colour) in enumerate(MODEL_ROWS):
                ax = axes[ri][cj]
                ax.plot(t, Y[:, ci], color="black", lw=1.0, zorder=5,
                        label="real" if (ri == 0 and cj == 0) else None)
                # STITCHED, not one plot call per origin. OpenTouch concatenates the
                # non-overlapping forecasts into a single line
                # (plot_opentouch_forecast_overlay.py: "stitched into a continuous line",
                # `idx = np.concatenate([arange(o+1, o+1+H) for o in sel])`). Drawing each
                # 6-frame block separately -- the first version here -- broke the curve into
                # disconnected pieces and made a tracking model look like a stuttering one.
                # The step between blocks stays visible, because it is real: each block
                # restarts from a new origin.
                ft, yv = [], []
                for o in keep:
                    ft.append((np.arange(H) + o + 1) / fps)
                    if mname == "probgru":
                        yv.append(preds[i][list(orig).index(o), :, ci])
                    else:
                        yv.append(classical[mname].predict(Y[:o + 1], H, ctx["gte"][i])[:, ci])
                if ft:
                    ax.plot(np.concatenate(ft), np.concatenate(yv), color=colour, lw=1.5,
                            alpha=0.9, zorder=4,
                            label=f"{mname} {H/fps:.0f} s forecast" if cj == 0 else None)
                ax.set_xlim(lo_t, hi_t)
                ax.grid(alpha=0.25, lw=0.4)
                ax.tick_params(labelsize=7)
                if cj == 0:
                    ax.set_ylabel(f"{mname}\n{UNITS.get(ch.rsplit('_', 1)[0], ch)}", fontsize=8)
                if ri == 0:
                    ax.set_title(f"[{rows[i]['label_idx']}] {rows[i]['subject']} — "
                                 f"{rows[i]['label'][:26]}", fontsize=8)
                if ri == nrow - 1:
                    ax.set_xlabel("time (s)")
                if cj == 0:
                    ax.legend(fontsize=6, loc="upper right")
        fig.suptitle(f"d256 {ch}: real vs rolling {H/fps:.0f} s forecast (one model per row) — "
                     f"fold {args.fold}, test {fold['held_out']}, arm '{hp['arm']}'", y=0.998)
        fig.tight_layout()
        out = os.path.join(args.outdir, f"d256_forecast_{ch}.png")
        fig.savefig(out, dpi=110, bbox_inches="tight"); plt.close(fig)
        print(f"  wrote {out}")


if __name__ == "__main__":
    main()
