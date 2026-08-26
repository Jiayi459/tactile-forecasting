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
    ap.add_argument("--n", type=int, default=5, help="test recordings to draw")
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
    style = {"persistence": ("0.55", "--"), "seasonal": ("tab:green", ":"),
             "ar": ("tab:orange", "-."), "probgru": ("tab:red", "-")}

    for ci, ch in enumerate(cfg.channels):
        fig, axes = plt.subplots(len(picks), 1, figsize=(13, 2.4 * len(picks)), squeeze=False)
        for r, i in enumerate(picks):
            ax = axes[r][0]
            Y = D.load_target(cfg, i)
            t = np.arange(len(Y)) / fps
            ax.plot(t, Y[:, ci], color="0.15", lw=1.0, label="truth", zorder=5)
            orig = origins(len(Y), cfg)
            # A handful of origins, spread out: drawing all of them would black out the axis.
            for k in np.linspace(0, len(orig) - 1, min(6, len(orig))).astype(int):
                o = orig[k]
                ft = (np.arange(H) + o + 1) / fps
                ax.plot(ft, preds[i][k, :, ci], color=style["probgru"][0],
                        ls=style["probgru"][1], lw=1.6, alpha=0.9,
                        label="probGRU" if (r == 0 and k == 0) else None, zorder=4)
                for name in E.MODELS:
                    yh = classical[name].predict(Y[:o + 1], H, ctx["gte"][i])
                    c, ls = style[name]
                    ax.plot(ft, yh[:, ci], color=c, ls=ls, lw=1.1, alpha=0.75,
                            label=name if (r == 0 and k == 0) else None, zorder=3)
                ax.axvline(o / fps, color="0.85", lw=0.5, zorder=1)
            ax.set_ylabel(f"[{rows[i]['label_idx']}] {rows[i]['subject']}\n"
                          f"{rows[i]['label'][:24]}", fontsize=7)
            ax.tick_params(labelsize=7)
            if r == 0:
                ax.legend(fontsize=7, ncol=5, loc="upper right")
            if r == len(picks) - 1:
                ax.set_xlabel("time (s)")
        fig.suptitle(f"d256 {ch} -- {H}-step ({H/fps:.1f} s) forecasts, fold {args.fold} "
                     f"(test {fold['held_out']}), arm '{hp['arm']}'", y=0.999)
        fig.tight_layout()
        out = os.path.join(args.outdir, f"d256_forecast_{ch}.png")
        fig.savefig(out, dpi=110, bbox_inches="tight"); plt.close(fig)
        print(f"  wrote {out}")


if __name__ == "__main__":
    main()
