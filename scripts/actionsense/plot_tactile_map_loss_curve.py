"""Train/val/test loss vs epoch for the tactile-map / aggregate forecaster -> overfitting check.

Same idea as scripts/actionsense/plot_fcop_loss_curve.py: split recordings 70/15/15 by CLIP
(norm fitted on TRAIN only), train each encoder while logging Gaussian NLL + MSE on all three
splits every epoch, and mark the min-val epoch. Loss is evaluated on a capped random subset per
split for speed. This is a DIAGNOSTIC on one split, deliberately not the 5-fold CV protocol --
the CV keeps no per-epoch history, so a curve has to come from a separate, cheaper run.

THREE DEFECTS FIXED 2026-09-05, two of which predate this session and made the script
unrunnable rather than merely limited:

  1. FOUR-ITEM UNPACK. MapWindows/AggWindows return (x, aid, last, y) -- four items, always,
     deliberately (data.py:151). This script still did `for x, y in DataLoader(...)` and died
     with "too many values to unpack" before the first epoch. Every forward now goes through
     train.py::_call, the one place that knows which backbone needs aid and last.
  2. THE AGGREGATE ARM WAS FED MAPS. The encoder loop walked [aggregate, flatten, cnn] while
     sharing a single MapWindows dataset, so AggEncoder -- which expects (B, t_in, C) -- was
     handed (B, t_in, 2, 32, 32). The aggregate arm could never have produced a curve. The
     dataset is now built per encoder, mirroring cross_validate.
  3. ONE BACKBONE, ONE POPULATION. build_model was called without `backbone`, so only Seq2Seq
     could be drawn, and the recording list was hardcoded to the frozen slice+peel split.
     --backbone and --scope now exist, matching train_tactile_map.py.

    python scripts/actionsense/plot_tactile_map_loss_curve.py \
        --encoders aggregate --backbone probgru --scope corpus --history 1 --epochs 60
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
from src.actionsense.eval_harness.dataset import Norm, load_target   # noqa: E402
from src.actionsense.tactile_map import data as D                    # noqa: E402
from src.actionsense.tactile_map import train as T                   # noqa: E402
from src.actionsense.tactile_map.models import build_model           # noqa: E402


def capped(ds, cap, seed=0):
    if len(ds) <= cap:
        return ds
    idx = np.random.default_rng(seed).permutation(len(ds))[:cap]
    return Subset(ds, idx.tolist())


@torch.no_grad()
def losses(model, ds, dev, H, batch=128):
    """-> (mean Gaussian NLL, mean MSE of the mean) over the whole subset."""
    model.eval()
    snll = smse = n = 0.0
    for x, aid, last, y in DataLoader(ds, batch_size=batch):
        x, aid, last, y = x.to(dev), aid.to(dev), last.to(dev), y.to(dev)
        mu, lv = T._call(model, x, aid, last, H)
        snll += float((0.5 * (lv + (y - mu) ** 2 * torch.exp(-lv))).sum())
        smse += float(((y - mu) ** 2).sum())
        n += y.numel()
    return snll / n, smse / n


def datasets_for(cfg, tm, t_in, tr, va, te, encoder, kw):
    """(train, val, test) datasets for ONE encoder. Mirrors cross_validate's branch: the
    aggregate arm reads the 6-dim F/CoP signal, the map arms read the pressure grid."""
    if encoder == "aggregate":
        tnorm = Norm.from_train({i: load_target(cfg, i) for i in tr})
        def mk(ids):
            return D.AggWindows({i: tnorm.z(load_target(cfg, i)) for i in ids}, cfg, t_in, **kw)
        return mk(tr), mk(va), mk(te)
    maps_tr, tgts_tr = D.load_raw(cfg, tr, tm["baseline_frames"])
    mnorm = D.MapNorm.from_train(maps_tr, tm["alpha"])
    tnorm = Norm.from_train(tgts_tr)
    ds_tr = D.MapWindows(D.normalize(maps_tr, mnorm),
                         {i: tnorm.z(t) for i, t in tgts_tr.items()}, cfg, t_in, **kw)
    return (ds_tr,
            T._dataset(cfg, tm, t_in, va, mnorm, tnorm, **kw),
            T._dataset(cfg, tm, t_in, te, mnorm, tnorm, **kw))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tm-config", default="configs/actionsense/tactile_map.yaml")
    ap.add_argument("--encoders", default=None, help="comma list; default = the config's sweep")
    ap.add_argument("--backbone", default="seq2seq", choices=["seq2seq", "probgru"])
    ap.add_argument("--scope", default="frozen", choices=["frozen", "corpus"])
    ap.add_argument("--history", type=float, default=3.0)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--cap", type=int, default=2500)
    ap.add_argument("--out", default=None,
                    help="default names itself after backbone/scope/history so the four runs "
                         "of a sweep cannot overwrite one another")
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = load_config()
    tmc = yaml.safe_load(open(args.tm_config))
    tm = {**tmc["preprocess"], **tmc["model"], **tmc["optim"]}
    encoders = args.encoders.split(",") if args.encoders else tmc["sweep"]["encoders"]
    out = args.out or (f"docs/actionsense/loss_curve_{args.backbone}_{args.scope}"
                       f"_h{args.history:g}s.png")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t_in = int(round(args.history * cfg.fps))

    need_maps = any(e != "aggregate" for e in encoders)
    recs = (T.corpus_recordings(cfg, require_maps=need_maps) if args.scope == "corpus"
            else T.recordings(cfg, require_maps=True))
    rng = np.random.default_rng(0)
    order = rng.permutation(len(recs))
    n = len(recs)
    ntr, nva = int(0.70 * n), int(0.15 * n)
    tr = [recs[i] for i in order[:ntr]]
    va = [recs[i] for i in order[ntr:ntr + nva]]
    te = [recs[i] for i in order[ntr + nva:]]

    # probGRU predicts the ABSOLUTE target and carries an action embedding whose vocabulary is
    # built from TRAIN only; Seq2Seq keeps the residual target and ignores the action id.
    # Same rule as cross_validate -- stated in one place there, obeyed here.
    pg = args.backbone == "probgru"
    verbs = D.verbs_of(cfg, recs)
    vocab, by_idx = D.action_vocab(verbs, tr)
    kw = dict(aids={i: D.aid_of(vocab, by_idx, i) for i in recs}, residual=not pg)
    tm = {**tm, "n_act": len(vocab)}
    print(f"scope={args.scope} backbone={args.backbone} t_in={t_in} "
          f"({args.history:g}s)  clips tr/va/te={len(tr)}/{len(va)}/{len(te)}  "
          f"encoders={encoders}  vocab={len(vocab)}")

    H = {}
    for enc in encoders:
        ds_tr, ds_va, ds_te = datasets_for(cfg, tm, t_in, tr, va, te, enc, kw)
        ev = {"train": capped(ds_tr, args.cap), "val": capped(ds_va, args.cap),
              "test": capped(ds_te, args.cap)}
        print(f"  [{enc}] windows train={len(ds_tr)} val={len(ds_va)} test={len(ds_te)} "
              f"(eval cap {args.cap})", flush=True)
        torch.manual_seed(0)
        model = build_model(enc, cfg.horizon, tm["d"], tm["hidden"],
                            backbone=args.backbone, n_act=tm["n_act"],
                            n_out=len(cfg.channels)).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=tm["lr"])
        tl = DataLoader(ds_tr, batch_size=tm["batch"], shuffle=True)
        H[enc] = {m: {s: [] for s in ev} for m in ("nll", "mse")}
        for ep in range(args.epochs):
            model.train()
            for x, aid, last, y in tl:
                x, aid, last, y = x.to(dev), aid.to(dev), last.to(dev), y.to(dev)
                opt.zero_grad()
                mu, lv = T._call(model, x, aid, last, cfg.horizon)
                (0.5 * (lv + (y - mu) ** 2 * torch.exp(-lv)).mean()).backward()
                opt.step()
            for s, d in ev.items():
                nll, mse = losses(model, d, dev, cfg.horizon)
                H[enc]["nll"][s].append(nll)
                H[enc]["mse"][s].append(mse)
        be = int(np.argmin(H[enc]["nll"]["val"]))
        print(f"  {enc}: min-val NLL @epoch {be + 1}; final NLL tr/va/te = "
              f"{H[enc]['nll']['train'][-1]:.3f}/{H[enc]['nll']['val'][-1]:.3f}/"
              f"{H[enc]['nll']['test'][-1]:.3f}", flush=True)

    encs = list(H)
    ep = np.arange(1, args.epochs + 1)
    fig, axes = plt.subplots(len(encs), 2, figsize=(14, 5.2 * len(encs)), squeeze=False)
    for ri, enc in enumerate(encs):
        for ci, metric in enumerate(["nll", "mse"]):
            ax = axes[ri][ci]
            be = int(np.argmin(H[enc][metric]["val"]))
            for k, c in [("train", "C0"), ("val", "C1"), ("test", "C2")]:
                ax.plot(ep, H[enc][metric][k], color=c, lw=2, label=k)
            ax.axvline(be + 1, color="0.6", ls=":", lw=1, label=f"min-val ep {be + 1}")
            ax.set_title(f"{enc} — {metric.upper()}")
            ax.set_xlabel("epoch")
            ax.set_ylabel("Gaussian NLL" if metric == "nll" else "MSE of the mean")
            ax.legend(fontsize=8)
            ax.grid(alpha=.3)
    fig.suptitle(f"{args.backbone} loss vs epoch — {args.history:g}s history, scope={args.scope} "
                 f"— overfitting check (train-only norm; split by recording)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    raise SystemExit(main())
