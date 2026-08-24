"""Train the F/CoP-input probGRU while logging train/val/test loss per epoch -> overfitting check.

The library `action_dynamics.train` doesn't record a loss history, so this script reuses its data +
model (load_pooled, Norm, windows, ProbGRU, the Gaussian-NLL objective) and runs its OWN loop that
evaluates the loss on train / val / test splits every epoch. Splits are by CLIP (no window leakage);
norm is fit on TRAIN only. Writes docs/actionsense/fcop_loss_curve.png.

    python scripts/actionsense/plot_fcop_loss_curve.py --input-mode raw --hand right --history 3
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.actionsense import action_dynamics as AD  # noqa: E402


def losses(model, X, A, Yin, Y, norm, t_out):
    """(mean Gaussian NLL, mean MSE) on a window set. NLL = training objective (mean+variance);
    MSE = mean-only (drives the skill metric). Comparing them separates mean- vs variance-overfit."""
    if len(X) == 0:
        return np.nan, np.nan
    with torch.no_grad():
        xt = torch.tensor(norm.nx(X)); at = torch.tensor(A)
        yl = torch.tensor(norm.ny(Yin)[:, -1]); yt = torch.tensor(norm.ny(Y))
        mu, lv = model(xt, at, yl, t_out)
        nll = float((0.5 * (lv + (yt - mu) ** 2 * torch.exp(-lv))).mean())
        mse = float(((yt - mu) ** 2).mean())
        return nll, mse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/actionsense_states")
    ap.add_argument("--actions", default="Slice,Peel")
    ap.add_argument("--input-mode", default="raw")
    ap.add_argument("--hand", default="right")
    ap.add_argument("--history", type=float, default=3.0)
    ap.add_argument("--future-sec", type=float, default=1.0)
    ap.add_argument("--downsample", type=int, default=3)
    ap.add_argument("--cut", type=float, default=0.4)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--hidden", type=int, default=48)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="docs/actionsense/fcop_loss_curve.png")
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    subs = [s.strip() for s in args.actions.split(",")]
    fps = 30.0 / args.downsample
    t_in, t_out = int(round(args.history * fps)), int(round(args.future_sec * fps))
    data = AD.load_pooled(args.root, subs, args.downsample, args.cut,
                          input_mode=args.input_mode, hand=args.hand)
    # split CLIPS 70/15/15
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(data)); n = len(data)
    ntr, nva = int(0.70 * n), int(0.15 * n)
    tr = [data[i] for i in order[:ntr]]
    va = [data[i] for i in order[ntr:ntr + nva]]
    te = [data[i] for i in order[ntr + nva:]]
    norm = AD.Norm.from_clips(tr)                                   # TRAIN only
    W = lambda c: AD.windows(c, t_in, t_out, 2)                     # noqa: E731
    Xtr, Atr, Yintr, Ytr, _ = W(tr); Xva, Ava, Yinva, Yva, _ = W(va); Xte, Ate, Yinte, Yte, _ = W(te)
    print(f"{args.input_mode}/{args.hand} hist={args.history}s | clips tr/va/te={len(tr)}/{len(va)}/{len(te)}"
          f"  windows={len(Xtr)}/{len(Xva)}/{len(Xte)}")

    torch.manual_seed(args.seed)
    model = AD.ProbGRU(Xtr.shape[-1], len(subs), args.hidden)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    xt = torch.tensor(norm.nx(Xtr)); at = torch.tensor(Atr)
    yl = torch.tensor(norm.ny(Yintr)[:, -1]); yt = torch.tensor(norm.ny(Ytr))
    H = {m: {"train": [], "val": [], "test": []} for m in ("nll", "mse")}
    splits = {"train": (Xtr, Atr, Yintr, Ytr), "val": (Xva, Ava, Yinva, Yva), "test": (Xte, Ate, Yinte, Yte)}
    for ep in range(args.epochs):
        model.train(); perm = torch.randperm(len(xt))
        for i in range(0, len(xt), 64):
            b = perm[i:i + 64]; opt.zero_grad()
            mu, lv = model(xt[b], at[b], yl[b], t_out)
            (0.5 * (lv + (yt[b] - mu) ** 2 * torch.exp(-lv)).mean()).backward(); opt.step()
        model.eval()
        for s, (Xs, As, Yins, Ys) in splits.items():
            n_, m_ = losses(model, Xs, As, Yins, Ys, norm, t_out)
            H["nll"][s].append(n_); H["mse"][s].append(m_)

    be_nll = int(np.nanargmin(H["nll"]["val"])); be_mse = int(np.nanargmin(H["mse"]["val"]))
    print(f"min val NLL @epoch {be_nll + 1} ({H['nll']['val'][be_nll]:.3f}); "
          f"min val MSE @epoch {be_mse + 1} ({H['mse']['val'][be_mse]:.3f})")
    print(f"final NLL tr/va/te = {H['nll']['train'][-1]:.3f}/{H['nll']['val'][-1]:.3f}/{H['nll']['test'][-1]:.3f}")
    print(f"final MSE tr/va/te = {H['mse']['train'][-1]:.3f}/{H['mse']['val'][-1]:.3f}/{H['mse']['test'][-1]:.3f}")

    ep = np.arange(1, args.epochs + 1)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    for ax, metric, be, ylab in [(axes[0], "nll", be_nll, "Gaussian NLL (mean+variance objective)"),
                                 (axes[1], "mse", be_mse, "MSE of the MEAN (drives skill)")]:
        for k, c in [("train", "C0"), ("val", "C1"), ("test", "C2")]:
            ax.plot(ep, H[metric][k], color=c, lw=2, label=f"{k}")
        te_es, te_fin = H[metric]["test"][be], H[metric]["test"][-1]      # deployed test loss
        ax.axvline(be + 1, color="0.5", ls=":", lw=1)
        ax.plot(be + 1, te_es, "o", color="k", ms=8, zorder=5,
                label=f"AFTER early-stop (deploy ep{be + 1}): test={te_es:.3f}")
        ax.plot(args.epochs, te_fin, "X", color="C3", ms=10, zorder=5,
                label=f"BEFORE (deploy final ep{args.epochs}): test={te_fin:.3f}")
        ax.annotate("", xy=(args.epochs, te_es), xytext=(args.epochs, te_fin),
                    arrowprops=dict(arrowstyle="<-", color="C3", lw=1.5))
        ax.text(args.epochs * 0.78, (te_es + te_fin) / 2,
                f"early stopping\nrecovers {te_fin - te_es:+.3f}", color="C3", fontsize=9, va="center")
        ax.set_xlabel("epoch"); ax.set_ylabel(ylab); ax.legend(fontsize=8); ax.grid(alpha=.3)
        ax.set_title(metric.upper())
    fig.suptitle(f"F/CoP probGRU: BEFORE vs AFTER early stopping  ({args.input_mode}/{args.hand}, "
                 f"{args.history:.0f}s hist) — same curve, different DEPLOYED checkpoint", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95]); os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=120); print(f"[done] {args.out}")


if __name__ == "__main__":
    main()
