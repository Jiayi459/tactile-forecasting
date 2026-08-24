"""Train/val/test loss vs epoch for the tactile-map model (flatten + cnn) -> overfitting check.

Same idea as scripts/actionsense/plot_fcop_loss_curve.py but for the map->F/CoP probGRU. Splits recordings
70/15/15 by CLIP (norm on TRAIN only), trains each encoder while logging Gaussian NLL + MSE on
train/val/test every epoch. Loss is evaluated on a capped random subset per split for speed. Writes
docs/actionsense/tactile_map_loss_curve.png (rows = flatten/cnn, cols = NLL/MSE).

    python scripts/actionsense/plot_tactile_map_loss_curve.py --history 3 --epochs 60
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.actionsense.eval_harness.config import load_config          # noqa: E402
from src.actionsense.eval_harness.dataset import Norm                # noqa: E402
from src.actionsense.tactile_map import data as D                    # noqa: E402
from src.actionsense.tactile_map import train as T                   # noqa: E402
from src.actionsense.tactile_map.models import build_model           # noqa: E402


def capped(ds, cap, seed=0):
    if len(ds) <= cap:
        return ds
    idx = np.random.default_rng(seed).permutation(len(ds))[:cap]
    return Subset(ds, idx.tolist())


@torch.no_grad()
def losses(model, ds, dev, batch=128):
    model.eval(); snll = smse = n = 0.0
    for x, y in DataLoader(ds, batch_size=batch):
        x, y = x.to(dev), y.to(dev)
        mu, lv = model(x)
        snll += float((0.5 * (lv + (y - mu) ** 2 * torch.exp(-lv))).sum())
        smse += float(((y - mu) ** 2).sum()); n += y.numel()
    return snll / n, smse / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tm-config", default="configs/actionsense/tactile_map.yaml")
    ap.add_argument("--history", type=float, default=3.0)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--cap", type=int, default=2500)
    ap.add_argument("--out", default="docs/actionsense/tactile_map_loss_curve.png")
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = load_config()
    tmc = yaml.safe_load(open(args.tm_config)); tm = {**tmc["preprocess"], **tmc["model"], **tmc["optim"]}
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t_in = int(round(args.history * cfg.fps))
    recs = T.recordings(cfg, require_maps=True)
    rng = np.random.default_rng(0); order = rng.permutation(len(recs)); n = len(recs)
    ntr, nva = int(0.70 * n), int(0.15 * n)
    tr = [recs[i] for i in order[:ntr]]; va = [recs[i] for i in order[ntr:ntr + nva]]
    te = [recs[i] for i in order[ntr + nva:]]

    maps_tr, tgts_tr = D.load_raw(cfg, tr, tm["baseline_frames"])
    mnorm = D.MapNorm.from_train(maps_tr, tm["alpha"]); tnorm = Norm.from_train(tgts_tr)
    ds_tr = D.MapWindows(D.normalize(maps_tr, mnorm), {i: tnorm.z(t) for i, t in tgts_tr.items()}, cfg, t_in)
    ds_va = T._dataset(cfg, tm, t_in, va, mnorm, tnorm)
    ds_te = T._dataset(cfg, tm, t_in, te, mnorm, tnorm)
    ev = {"train": capped(ds_tr, args.cap), "val": capped(ds_va, args.cap), "test": capped(ds_te, args.cap)}
    print(f"clips tr/va/te={len(tr)}/{len(va)}/{len(te)}  windows train={len(ds_tr)} (eval cap {args.cap})")

    H = {}
    for enc in tmc["sweep"]["encoders"]:
        torch.manual_seed(0)
        model = build_model(enc, cfg.horizon, tm["d"], tm["hidden"]).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=tm["lr"])
        tl = DataLoader(ds_tr, batch_size=tm["batch"], shuffle=True)
        H[enc] = {m: {s: [] for s in ev} for m in ("nll", "mse")}
        for ep in range(args.epochs):
            model.train()
            for x, y in tl:
                x, y = x.to(dev), y.to(dev); opt.zero_grad()
                mu, lv = model(x)
                (0.5 * (lv + (y - mu) ** 2 * torch.exp(-lv)).mean()).backward(); opt.step()
            for s, d in ev.items():
                nll, mse = losses(model, d, dev)
                H[enc]["nll"][s].append(nll); H[enc]["mse"][s].append(mse)
        be = int(np.argmin(H[enc]["nll"]["val"]))
        print(f"  {enc}: min-val NLL @epoch {be + 1}; final NLL tr/va/te = "
              f"{H[enc]['nll']['train'][-1]:.3f}/{H[enc]['nll']['val'][-1]:.3f}/{H[enc]['nll']['test'][-1]:.3f}")

    encs = list(H); ep = np.arange(1, args.epochs + 1)
    fig, axes = plt.subplots(len(encs), 2, figsize=(14, 5.2 * len(encs)), squeeze=False)
    for ri, enc in enumerate(encs):
        for ci, metric in enumerate(["nll", "mse"]):
            ax = axes[ri][ci]; be = int(np.argmin(H[enc]["val"] if False else H[enc][metric]["val"]))
            for k, c in [("train", "C0"), ("val", "C1"), ("test", "C2")]:
                ax.plot(ep, H[enc][metric][k], color=c, lw=2, label=k)
            ax.axvline(be + 1, color="0.6", ls=":", lw=1, label=f"min-val ep {be + 1}")
            ax.set_title(f"{enc} — {metric.upper()}"); ax.set_xlabel("epoch")
            ax.set_ylabel("Gaussian NLL" if metric == "nll" else "MSE of the mean")
            ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.suptitle(f"Tactile-map probGRU loss vs epoch ({args.history:.0f}s history) — overfitting check "
                 f"(train-only norm; split by clip)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97]); os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=120); print(f"[done] {args.out}")


if __name__ == "__main__":
    main()
