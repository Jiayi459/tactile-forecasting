"""Visualize TEST-SET prediction outcomes (thin CLI over action_dynamics).

Holds out a test set, trains one model (via the library) on the rest, then on ALL test-set
forecast windows plots, per predicted variable (F_fast force, x_fast/y_fast center-of-pressure):
  * predicted-mean vs true SCATTER with the ideal y=x line (accuracy across the whole test set)
  * the skill vs persistence and the +/-2 sigma band coverage (calibration)

    python scripts/plot_test_results.py --actions Slice,Peel --t-in 30 --t-out 10
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.actionsense import action_dynamics as AD  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/actionsense_states")
    ap.add_argument("--actions", default="Slice,Peel")
    ap.add_argument("--downsample", type=int, default=3)
    ap.add_argument("--cut", type=float, default=0.4)
    ap.add_argument("--input-mode", default="highpass", help="raw | highpass")
    ap.add_argument("--hand", default="active", help="left | right | active")
    ap.add_argument("--t-in", type=int, default=30)         # 3 s past (best from the sweep)
    ap.add_argument("--t-out", type=int, default=10)        # 1 s future
    ap.add_argument("--hidden", type=int, default=48)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="docs/actionsense/test_results.png")
    args = ap.parse_args()
    import torch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    subs = [s.strip() for s in args.actions.split(",")]
    data = AD.load_pooled(args.root, subs, args.downsample, args.cut,
                          input_mode=args.input_mode, hand=args.hand)
    tr_ids, te_ids = AD.split_train_test(len(data), seed=args.seed)
    train = [data[i] for i in tr_ids]; test = [data[i] for i in te_ids]
    print(f"train {len(train)} / test {len(test)} clips; t_in={args.t_in} t_out={args.t_out}")

    model, norm = AD.train(train, len(subs), args.t_in, args.t_out,
                           hidden=args.hidden, epochs=args.epochs, seed=0)

    # forecast every TEST window, collect (true, mean, sd) for all 3 vars over all horizon steps
    X, A, Yin, Y, _ = AD.windows(test, args.t_in, args.t_out, 2)
    with torch.no_grad():
        mu, lv = model(torch.tensor(norm.nx(X)), torch.tensor(A),
                       torch.tensor(norm.ny(Yin)[:, -1]), args.t_out)
    mu = norm.dy(mu.numpy()); sd = np.sqrt(np.exp(lv.numpy())) * norm.sd_y
    pers = np.repeat(Yin[:, -1:], args.t_out, 1)             # persistence-of-fast

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for j, name in enumerate(AD.TARGETS):
        yt = Y[:, :, j].ravel(); yp = mu[:, :, j].ravel(); ys = sd[:, :, j].ravel()
        skill = 1 - np.mean((yp - yt) ** 2) / (np.mean((pers[:, :, j].ravel() - yt) ** 2) + 1e-12)
        cov = float((np.abs(yt - yp) <= 2 * ys).mean())
        lim = np.percentile(np.abs(np.concatenate([yt, yp])), 99)
        ax = axes[j]
        ax.plot([-lim, lim], [-lim, lim], "r-", lw=1, label="ideal (y=x)")
        ax.scatter(yt, yp, s=3, alpha=0.15, color="C0")
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
        ax.set_xlabel(f"true {name}"); ax.set_ylabel(f"predicted {name}")
        ax.set_title(f"{name}   skill={skill:+.2f}   coverage@2sd={cov:.2f}", fontsize=10)
        ax.legend(loc="upper left", fontsize=8)
    fig.suptitle(f"Test-set predicted vs true — {args.actions}  ({len(test)} held-out clips, "
                 f"{len(X)} windows, {args.t_out/ (30/args.downsample):.0f}s forecast)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=120)
    print(f"[done] {args.out}")


if __name__ == "__main__":
    main()
