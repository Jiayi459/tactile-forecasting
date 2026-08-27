#!/usr/bin/env python3
"""READ-ONLY: is probGRU's deficit its ANCHOR rather than its model?

THE OBSERVATION. In docs/d256/forecast_none/d256_forecast_F_L.png the probGRU segments sit
systematically ABOVE the truth, while persistence -- which by construction starts at the last
observed value -- tracks it. probGRU's skill is worst exactly on F (-0.353 on F_L) and near
zero on CoP, and F is the channel whose absolute level differs most between people.

THE MECHANISM THIS TESTS. probGRU predicts the ABSOLUTE value: its mu head is a free Linear,
trained on targets z-scored by a Norm fitted on that fold's TRAIN subjects. Under
leave-one-SUBJECT-out the test person's force level is not the training people's, so the
head regresses toward a level that is simply wrong for them. Persistence and AR never can:
they are anchored to the observed history in raw units. The repo already names this
distinction for OpenTouch -- docs/_skill_comparison_notes.md, "part of what looks like detail
is the anchor, not the model" -- but there the split is by location, not by person, so the
level shift is smaller.

WHAT IT REPORTS, per fold and channel:
  bias        mean(pred - truth). Zero if the model is not systematically displaced.
  level shift mean(test) - mean(train). If bias tracks this, the anchor is the culprit.
  skill       as scored.
  skill_deb   skill after subtracting the fold's own mean bias from every forecast.

`skill_deb` is the decisive number, and it is NOT a proposed fix -- subtracting a bias computed
on test is cheating. It is an upper bound on what removing the anchor error could buy. If it
lands near AR, the architecture is fine and the anchoring is wrong; if it stays negative, the
model has a deeper problem and re-anchoring would be wasted work.

    python scripts/d256/probe_d256_anchor.py --run runs/d256_probgru_none
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.actionsense.eval_harness import masking, metrics  # noqa: E402
from src.actionsense.eval_harness.baselines.base import origins  # noqa: E402
from src.actionsense.eval_harness.config import load_config  # noqa: E402
from src.d256 import dataset as D  # noqa: E402
from src.d256 import evaluate as E  # noqa: E402
from src.d256 import prob_gru as PG  # noqa: E402
from src.d256 import splits as S  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/d256_probgru_none")
    ap.add_argument("--config", default=os.path.join("configs", "d256", "eval_harness.yaml"))
    args = ap.parse_args()

    import torch
    cfg = load_config(args.config)
    C, H = len(cfg.channels), cfg.horizon
    print(f"{'fold':>4} {'ch':>8s} {'bias':>9s} {'level shift':>12s} {'skill':>8s} "
          f"{'skill_deb':>10s}")
    rows = []
    for fold in S.folds(cfg):
        ck_path = os.path.join(args.run, f"fold{fold['fold']}.pt")
        if not os.path.exists(ck_path):
            continue
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        hp = {**PG.DEFAULT_HP, **ck["hp"]}
        ctx = E.fold_context(cfg, fold)
        fnorm = PG.FeatNorm.from_train(cfg, fold["train"], hp["features"] == "raw+df")
        aids, n_act = PG.action_ids(cfg, hp["arm"],
                                    fold["train"] + fold["val"] + fold["test"])
        model = PG.ProbGRU(PG.feature_dim(hp), n_act, hp["hidden"], PG.N_OUT, hp["dropout"])
        model.load_state_dict(ck["state_dict"]); model.eval()
        preds = PG.forecast(model, cfg, fold["test"], hp, ctx["norm"], fnorm, aids,
                            device="cpu")

        yts, yhs = [], []
        for i, Y in sorted(ctx["test"].items()):
            for k, t in enumerate(origins(len(Y), cfg)):
                yts.append(Y[t + 1:t + 1 + H]); yhs.append(preds[i][k])
        yt, yh = np.stack(yts), np.stack(yhs)
        mask = masking.valid_mask(cfg, yt.reshape(-1, C), ctx["thr"]).reshape(yt.shape)

        tr_mean = np.concatenate(list(ctx["train"].values()), 0).mean(0)
        te_mean = np.concatenate(list(ctx["test"].values()), 0).mean(0)

        base_mse = metrics.masked_channel_mse(yt, np.repeat(
            np.stack([Y[t:t + 1] for i, Y in sorted(ctx["test"].items())
                      for t in origins(len(Y), cfg)]), H, axis=1), mask)
        mse = metrics.masked_channel_mse(yt, yh, mask)
        bias = np.array([float(np.nanmean((yh - yt)[:, :, c][mask[:, :, c]]))
                         for c in range(C)])
        mse_deb = metrics.masked_channel_mse(yt, yh - bias[None, None, :], mask)

        for c, ch in enumerate(cfg.channels):
            sk = 1 - mse[c] / base_mse[c]
            skd = 1 - mse_deb[c] / base_mse[c]
            shift = te_mean[c] - tr_mean[c]
            print(f"{fold['fold']:>4} {ch:>8s} {bias[c]:9.3f} {shift:12.3f} "
                  f"{sk:8.3f} {skd:10.3f}")
            rows.append((fold["fold"], ch, bias[c], shift, sk, skd))

    if not rows:
        sys.exit(f"no checkpoints under {args.run}")
    import collections
    by = collections.defaultdict(list)
    for _, ch, b, sh, sk, skd in rows:
        by[ch].append((b, sh, sk, skd))
    print(f"\n{'channel':>8s} {'|bias|':>8s} {'skill':>8s} {'skill_deb':>10s} {'gain':>8s}")
    for ch in cfg.channels:
        a = np.array(by[ch])
        print(f"{ch:>8s} {np.abs(a[:,0]).mean():8.3f} {a[:,2].mean():8.3f} "
              f"{a[:,3].mean():10.3f} {a[:,3].mean()-a[:,2].mean():8.3f}")
    a = np.array([r[2:] for r in rows], dtype=float)
    r = np.corrcoef(a[:, 0], a[:, 1])[0, 1]
    print(f"\ncorr(bias, level shift) = {r:+.3f} over {len(rows)} fold-channels")
    print("  strongly positive => the model is reproducing TRAIN's level on a test person")
    print("     whose level differs, i.e. the anchor, not the dynamics.")


if __name__ == "__main__":
    main()
