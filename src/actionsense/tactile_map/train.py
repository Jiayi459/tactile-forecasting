"""Train + cross-validate the tactile-map -> F/CoP forecaster (probabilistic, residual).

Mirrors the F/CoP probGRU protocol (src/actionsense/action_dynamics.py): a probabilistic head
(mean + log-variance) trained with Gaussian NLL, 5-fold CV by recording, sigma calibration on a
VAL subset held out from TRAIN, and skill-vs-persistence + coverage reported per channel & step.
Target is the RESIDUAL over persistence, so persistence == predicting residual 0.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..eval_harness.config import Config
from ..eval_harness.dataset import Norm, load_target
from ..eval_harness.splits import load_splits
from . import data as D
from ...shape_metrics import hausdorff_scaled
from .models import build_model

EPS = 1e-12


def recordings(cfg: Config, require_maps: bool = True) -> list[int]:
    """All Slice/Peel recordings in the frozen split (optionally only those with local maps)."""
    sp = load_splits(cfg)
    allrec = sorted(sp["train"] + sp["val"] + sp["test"])
    return D.available_idxs(cfg, allrec) if require_maps else allrec


def _dataset(cfg, tm, t_in, idxs, mnorm, tnorm):
    maps, tgts = D.load_raw(cfg, idxs, tm["baseline_frames"])
    return D.MapWindows(D.normalize(maps, mnorm), {i: tnorm.z(t) for i, t in tgts.items()}, cfg, t_in)


def _materialize(ds):
    """Stack a whole (small) dataset into (X, Y) tensors once -- avoids per-epoch DataLoader overhead."""
    items = [ds[k] for k in range(len(ds))]
    return torch.stack([a for a, _ in items]), torch.stack([b for _, b in items])


def train_model(train_ds, val_ds, cfg: Config, encoder: str, tm: dict, seed: int = 0,
                materialize: bool = False):
    """Train one probabilistic model (Gaussian NLL); keep best-VAL-NLL weights. `materialize`=True
    stacks the data into tensors once (fast for the tiny aggregate model on CPU); the map path keeps
    the lazy DataLoader (a 10 s map window set would be tens of GB)."""
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(encoder, cfg.horizon, tm["d"], tm["hidden"]).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=tm["lr"]); bs = tm["batch"]
    if materialize:
        Xtr, Ytr = _materialize(train_ds); Xtr, Ytr = Xtr.to(dev), Ytr.to(dev)
        Xva = Yva = None
        if len(val_ds):
            Xva, Yva = _materialize(val_ds); Xva, Yva = Xva.to(dev), Yva.to(dev)
    else:
        tl = DataLoader(train_ds, batch_size=bs, shuffle=True)
    best, best_state = np.inf, None
    for _ in range(tm["epochs"]):
        model.train()
        if materialize:
            perm = torch.randperm(len(Xtr))
            for i in range(0, len(Xtr), bs):
                b = perm[i:i + bs]; mu, lv = model(Xtr[b])
                loss = 0.5 * (lv + (Ytr[b] - mu) ** 2 * torch.exp(-lv)).mean()
                opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                if Xva is not None:
                    mu, lv = model(Xva)
                    v = float((0.5 * (lv + (Yva - mu) ** 2 * torch.exp(-lv))).mean())
                else:
                    v = float(loss.item())
        else:
            for x, y in tl:
                x, y = x.to(dev), y.to(dev); mu, lv = model(x)
                loss = 0.5 * (lv + (y - mu) ** 2 * torch.exp(-lv)).mean()      # Gaussian NLL on residual
                opt.zero_grad(); loss.backward(); opt.step()
            v = _val_nll(model, val_ds, dev) if len(val_ds) else float(loss.item())
        if v < best:
            best, best_state = v, {k: t.cpu().clone() for k, t in model.state_dict().items()}
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    return model


@torch.no_grad()
def _val_nll(model, ds, dev):
    model.eval(); s = n = 0.0
    for x, y in DataLoader(ds, batch_size=128):
        x, y = x.to(dev), y.to(dev)
        mu, lv = model(x)
        s += float((0.5 * (lv + (y - mu) ** 2 * torch.exp(-lv))).sum()); n += y.numel()
    return s / max(n, 1)


@torch.no_grad()
def _predict(model, ds, batch=128):
    """-> (mu, sd, resid_true), each (N,H,6) in normalized RESIDUAL space."""
    dev = next(model.parameters()).device
    mus, sds, ys = [], [], []
    for x, y in DataLoader(ds, batch_size=batch):
        mu, lv = model(x.to(dev))
        mus.append(mu.cpu().numpy()); sds.append(np.exp(0.5 * lv.cpu().numpy())); ys.append(y.numpy())
    if not mus:
        z = np.zeros((0, ds.H, 6)); return z, z, z
    return np.concatenate(mus), np.concatenate(sds), np.concatenate(ys)


def evaluate(model, ds, sigma_scale=1.0):
    """skill vs persistence (per channel, per step) + coverage@2sd + Hausdorff.

    Persistence predicts residual 0. -> dict, not a tuple: this used to return three values
    and adding a fourth to a positional return is the shape that crashed a whole OpenTouch
    job on 2026-08-19, after it had finished training.

    Hausdorff is computed on the RESIDUAL curves, which is exact rather than approximate:
    the metric is invariant to adding the same constant to both sets, and absolute equals
    residual plus the last observed value, the same constant for prediction and truth alike.
    It answers what MSE cannot -- a flat forecast through an oscillation is charged its
    amplitude, where MSE rewards it for sitting in the middle.
    """
    mu, sd, resid = _predict(model, ds)
    em = (mu - resid) ** 2; ep = resid ** 2
    out = {"skill_ch": 1 - em.mean((0, 1)) / (ep.mean((0, 1)) + EPS),
           "skill_step": 1 - em.mean(0) / (ep.mean(0) + EPS),
           "coverage": float((np.abs(resid - mu) <= 2 * sigma_scale * sd).mean())}
    C = mu.shape[-1] if len(mu) else 0
    hd, hd_p = np.full(C, np.nan), np.full(C, np.nan)
    for c in range(C):
        h = hausdorff_scaled(mu[:, :, c], resid[:, :, c])
        p0 = hausdorff_scaled(np.zeros_like(mu[:, :, c]), resid[:, :, c])
        if np.isfinite(h).any():
            hd[c], hd_p[c] = np.nanmean(h), np.nanmean(p0)
    out["hausdorff_ch"] = hd
    out["hausdorff_ratio_ch"] = hd / hd_p       # vs persistence, as the skill columns are
    return out


def calibrate_sigma(model, ds, target=0.95):
    mu, sd, resid = _predict(model, ds)
    if len(mu) == 0:
        return 1.0
    return float(np.percentile(np.abs(resid - mu) / (sd + 1e-9), 100 * target) / 2.0)


def cross_validate(cfg: Config, tm: dict, encoder: str, t_in: int, recs: list[int],
                   folds: int = 5, seed: int = 0):
    """5-fold CV by recording. Norms + model fit on TRAIN; sigma calibrated on a VAL subset of TRAIN;
    skill + coverage measured on the held-out TEST fold. -> (sk_ch (folds,6), sk_step (folds,H,6),
    cov_raw, cov_cal)."""
    rng = np.random.default_rng(seed)
    fold_of = rng.integers(0, folds, size=len(recs))
    skc, sks, cr, cc, hdc, hdr = [], [], [], [], [], []
    for f in range(folds):
        te = [recs[i] for i in range(len(recs)) if fold_of[i] == f]
        tr = [recs[i] for i in range(len(recs)) if fold_of[i] != f]
        if len(te) < 1 or len(tr) < 4:
            continue
        r2 = np.random.default_rng(seed * 100 + f)
        idx = r2.permutation(len(tr)); nv = max(2, len(tr) // 6)
        val, trn = [tr[i] for i in idx[:nv]], [tr[i] for i in idx[nv:]]
        if encoder == "aggregate":                          # neural AR on the aggregate 6-dim F/CoP
            tnorm = Norm.from_train({i: load_target(cfg, i) for i in trn})
            mk = lambda ids: D.AggWindows({i: tnorm.z(load_target(cfg, i)) for i in ids}, cfg, t_in)  # noqa: E731
            train_ds, val_ds, test_ds = mk(trn), mk(val), mk(te)
        else:                                               # map input (flatten / cnn)
            maps_tr, tgts_tr = D.load_raw(cfg, trn, tm["baseline_frames"])
            mnorm = D.MapNorm.from_train(maps_tr, tm["alpha"]); tnorm = Norm.from_train(tgts_tr)
            train_ds = D.MapWindows(D.normalize(maps_tr, mnorm),
                                    {i: tnorm.z(t) for i, t in tgts_tr.items()}, cfg, t_in)
            val_ds = _dataset(cfg, tm, t_in, val, mnorm, tnorm)
            test_ds = _dataset(cfg, tm, t_in, te, mnorm, tnorm)
        model = train_model(train_ds, val_ds, cfg, encoder, tm, seed=seed,
                            materialize=(encoder == "aggregate"))
        s = calibrate_sigma(model, val_ds)
        ev = evaluate(model, test_ds, sigma_scale=1.0)
        sk_ch, sk_step, c_raw = ev["skill_ch"], ev["skill_step"], ev["coverage"]
        hd_ch, hd_ratio = ev["hausdorff_ch"], ev["hausdorff_ratio_ch"]
        c_cal = evaluate(model, test_ds, sigma_scale=s)["coverage"]
        skc.append(sk_ch); sks.append(sk_step); cr.append(c_raw); cc.append(c_cal)
        hdc.append(hd_ch); hdr.append(hd_ratio)
    return {"skill_ch": np.array(skc), "skill_step": np.array(sks),
            "coverage_raw": float(np.mean(cr)), "coverage_cal": float(np.mean(cc)),
            "hausdorff_ch": np.array(hdc), "hausdorff_ratio_ch": np.array(hdr)}
