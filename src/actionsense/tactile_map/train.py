"""Train + cross-validate the tactile-map -> F/CoP forecaster (probabilistic, residual).

Mirrors the F/CoP probGRU protocol (src/actionsense/action_dynamics.py): a probabilistic head
(mean + log-variance) trained with Gaussian NLL, 5-fold CV by recording, sigma calibration on a
VAL subset held out from TRAIN, and skill-vs-persistence + coverage reported per channel & step.
Target is the RESIDUAL over persistence, so persistence == predicting residual 0.
"""
from __future__ import annotations

import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..eval_harness.config import Config
from ..eval_harness.dataset import Norm, load_target
from ..eval_harness.splits import load_splits
from ..eval_harness.weighting import recording_weights, weighted_percentile
from . import data as D
from ...shape_metrics import hausdorff_scaled
from .models import ProbGRU, build_model

EPS = 1e-12


def recordings(cfg: Config, require_maps: bool = True) -> list[int]:
    """All Slice/Peel recordings in the frozen split (optionally only those with local maps)."""
    sp = load_splits(cfg)
    allrec = sorted(sp["train"] + sp["val"] + sp["test"])
    return D.available_idxs(cfg, allrec) if require_maps else allrec


def corpus_recordings(cfg: Config, require_maps: bool = False) -> list[int]:
    """EVERY recording in the manifest that has a state file -- the full 299-recording corpus,
    not the frozen slice+peel split `recordings()` returns.

    EXPLORATORY ONLY, and the distinction is not cosmetic. The frozen harness scores
    `actions: [slice, peel]` (configs/actionsense/eval_harness.yaml:51) = 75 recordings, and
    every number in docs/actionsense/ was produced on that population with a Norm fitted to it.
    Widening the population changes the normalization, the class mean, and the CV folds, so
    results from this function must never be placed in the same table as harness numbers.
    Nothing here mutates the config: `cross_validate` takes `recs` directly and never reads
    cfg.raw["actions"] (only splits.py does), so the frozen protocol is untouched.
    """
    import json
    root = cfg.abspath("states_root")
    idxs = []
    with open(os.path.join(root, "manifest.jsonl")) as fh:
        for line in fh:
            if line.strip():
                i = int(json.loads(line)["idx"])
                if os.path.exists(os.path.join(root, f"state_{i}.npy")):
                    idxs.append(i)
    idxs = sorted(idxs)
    return D.available_idxs(cfg, idxs) if require_maps else idxs


def _dataset(cfg, tm, t_in, idxs, mnorm, tnorm, aids=None, residual=True):
    maps, tgts = D.load_raw(cfg, idxs, tm["baseline_frames"])
    return D.MapWindows(D.normalize(maps, mnorm), {i: tnorm.z(t) for i, t in tgts.items()},
                        cfg, t_in, aids=aids, residual=residual)


def _call(model, x, aid, last, H):
    """The ONE place the two backbones differ at call time.

    Seq2Seq reads the window alone; ProbGRU also needs the action id and the last observed
    value to seed its decoder. Keeping the difference to this function is what lets the
    training loop, the validation pass and prediction stay single-copy.
    """
    if isinstance(model, ProbGRU):
        return model(x, aid, last, H)
    return model(x)


def _materialize(ds):
    """Stack a whole (small) dataset into (X, Y) tensors once -- avoids per-epoch DataLoader overhead."""
    items = [ds[k] for k in range(len(ds))]
    return tuple(torch.stack([it[j] for it in items]) for j in range(4))


def train_model(train_ds, val_ds, cfg: Config, encoder: str, tm: dict, seed: int = 0,
                materialize: bool = False):
    """Train one probabilistic model (Gaussian NLL); keep best-VAL-NLL weights. `materialize`=True
    stacks the data into tensors once (fast for the tiny aggregate model on CPU); the map path keeps
    the lazy DataLoader (a 10 s map window set would be tens of GB)."""
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(encoder, cfg.horizon, tm["d"], tm["hidden"],
                        backbone=tm.get("backbone", "seq2seq"),
                        n_act=int(tm.get("n_act", 1)),
                        n_out=len(cfg.channels),
                        in_shape=tm.get("in_shape")).to(dev)   # None -> ActionSense (2,32,32)
    opt = torch.optim.Adam(model.parameters(), lr=tm["lr"]); bs = tm["batch"]
    if materialize:
        Xtr, Atr, Ltr, Ytr = (t.to(dev) for t in _materialize(train_ds))
        Xva = None
        if len(val_ds):
            Xva, Ava, Lva, Yva = (t.to(dev) for t in _materialize(val_ds))
    else:
        tl = DataLoader(train_ds, batch_size=bs, shuffle=True)
    # Checkpoint selection口径. clip_balanced is the unified default (2026-09-08, Q-D(b));
    # "pooled" reproduces the pre-2026-09-08 behaviour for a controlled comparison.
    criterion = tm.get("val_criterion", "clip_balanced")
    best, best_state, best_epoch, curves = np.inf, None, -1, []
    # A long arm that prints nothing is indistinguishable from a hung one -- the reason
    # cross_validate reports per fold. `log_every` is opt-in so ActionSense's per-fold output
    # is unchanged; the EgoTouch config sets it, because one of its arms is a single 60-epoch
    # run over ~494k windows with no fold boundary to report at.
    log_every = int(tm.get("log_every", 0))
    _t0 = time.time()
    for _ep in range(tm["epochs"]):
        model.train()
        if materialize:
            perm = torch.randperm(len(Xtr))
            for i in range(0, len(Xtr), bs):
                b = perm[i:i + bs]
                mu, lv = _call(model, Xtr[b], Atr[b], Ltr[b], cfg.horizon)
                loss = 0.5 * (lv + (Ytr[b] - mu) ** 2 * torch.exp(-lv)).mean()
                opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                if Xva is not None:
                    mu, lv = _call(model, Xva, Ava, Lva, cfg.horizon)
                    nll = 0.5 * (lv + (Yva - mu) ** 2 * torch.exp(-lv))
                    per = nll.flatten(1).mean(1).cpu().numpy()
                    v_pool, v_bal = float(per.mean()), _balanced(per, _rec_ids(val_ds))
                else:
                    v_pool = v_bal = float(loss.item())
        else:
            for x, aid, last, y in tl:
                x, aid, last, y = x.to(dev), aid.to(dev), last.to(dev), y.to(dev)
                mu, lv = _call(model, x, aid, last, cfg.horizon)
                loss = 0.5 * (lv + (y - mu) ** 2 * torch.exp(-lv)).mean()      # Gaussian NLL
                opt.zero_grad(); loss.backward(); opt.step()
            v_pool, v_bal = (_val_nll(model, val_ds, dev, cfg.horizon) if len(val_ds)
                             else (float(loss.item()), float(loss.item())))
        v = v_bal if criterion == "clip_balanced" else v_pool
        curves.append((v_pool, v_bal))
        if log_every and ((_ep + 1) % log_every == 0 or _ep == 0):
            el = time.time() - _t0
            print(f"      epoch {_ep + 1}/{tm['epochs']} | val NLL {v:.4f} "
                  f"(best {min(best, v):.4f}) | {el:.0f}s elapsed, "
                  f"~{el / (_ep + 1) * (tm['epochs'] - _ep - 1):.0f}s left", flush=True)
        if v < best:
            best, best_state = v, {k: t.cpu().clone() for k, t in model.state_dict().items()}
            best_epoch = len(curves) - 1
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    # Recorded, not printed: comparing `selected_epoch` with `selected_epoch_pooled` answers
    # "would the old criterion have kept a different checkpoint?" from ONE run. `_val_nll` is
    # evaluated under no_grad and feeds nothing but selection, so the weight trajectory is
    # identical either way -- the two criteria are read off the same run, not two runs.
    model.val_curves = curves
    model.val_criterion = criterion
    model.selected_epoch = best_epoch
    model.selected_epoch_pooled = int(np.argmin([p for p, _ in curves])) if curves else -1
    model.selection_differs = bool(curves) and model.selected_epoch != model.selected_epoch_pooled
    return model


def _rec_ids(ds) -> np.ndarray:
    """Recording idx per sample, in dataset order. Both window datasets expose `.index`."""
    return np.array([i for i, _ in ds.index], dtype=np.int64)


def _balanced(per_sample: np.ndarray, rec: np.ndarray) -> float:
    """Mean over RECORDINGS of each recording's mean NLL, vs the window-pooled mean.

    Rolling-origin window counts scale with recording length, so a split's longest recordings
    own a share of the checkpoint-selection signal far out of proportion to their number: on
    EgoTouch val, ten of ~124 recordings hold 39-44% of the windows, and on the ActionSense
    frozen split ten of 15 hold 84%. Balancing per recording makes selection agree with the way
    the reported metrics are already aggregated (score_preds_per_action's clip_balanced_mean),
    rather than letting two different aggregations decide the model and then judge it.
    """
    if len(per_sample) == 0:
        return float("nan")
    return float(np.mean([per_sample[rec == r].mean() for r in np.unique(rec)]))


@torch.no_grad()
def _val_nll(model, ds, dev, H) -> tuple[float, float]:
    """-> (window_pooled, clip_balanced) mean NLL per element. Both, always, so one run can
    answer whether the two criteria would have selected different epochs."""
    model.eval(); per = []
    for x, aid, last, y in DataLoader(ds, batch_size=128):
        x, aid, last, y = x.to(dev), aid.to(dev), last.to(dev), y.to(dev)
        mu, lv = _call(model, x, aid, last, H)
        nll = 0.5 * (lv + (y - mu) ** 2 * torch.exp(-lv))
        per.append(nll.flatten(1).mean(1).cpu().numpy())
    if not per:
        return float("nan"), float("nan")
    per = np.concatenate(per)
    return float(per.mean()), _balanced(per, _rec_ids(ds))


@torch.no_grad()
def _predict(model, ds, batch=128):
    """-> (mu, sd, y_true, persistence), each (N,H,C) in the dataset's OWN target space.

    `persistence` is returned rather than assumed: in residual space it is zeros, in absolute
    space it is the last observed value repeated across the horizon. Hard-coding zeros here
    would silently score the probGRU arm against the wrong reference.
    """
    dev = next(model.parameters()).device
    H = ds.H
    mus, sds, ys, ps = [], [], [], []
    for x, aid, last, y in DataLoader(ds, batch_size=batch):
        mu, lv = _call(model, x.to(dev), aid.to(dev), last.to(dev), H)
        mus.append(mu.cpu().numpy()); sds.append(np.exp(0.5 * lv.cpu().numpy()))
        ys.append(y.numpy())
        ln = last.numpy()
        ps.append(np.zeros_like(ys[-1]) if ds.residual
                  else np.repeat(ln[:, None, :], H, axis=1))
    if not mus:
        z = np.zeros((0, H, 1)); return z, z, z, z
    return (np.concatenate(mus), np.concatenate(sds),
            np.concatenate(ys), np.concatenate(ps))


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
    mu, sd, y, pers = _predict(model, ds)
    em = (mu - y) ** 2; ep = (pers - y) ** 2
    out = {"skill_ch": 1 - em.mean((0, 1)) / (ep.mean((0, 1)) + EPS),
           "skill_step": 1 - em.mean(0) / (ep.mean(0) + EPS),
           "coverage": float((np.abs(y - mu) <= 2 * sigma_scale * sd).mean())}
    C = mu.shape[-1] if len(mu) else 0
    hd, hd_p = np.full(C, np.nan), np.full(C, np.nan)
    for c in range(C):
        h = hausdorff_scaled(mu[:, :, c], y[:, :, c])
        p0 = hausdorff_scaled(pers[:, :, c], y[:, :, c])
        if np.isfinite(h).any():
            hd[c], hd_p[c] = np.nanmean(h), np.nanmean(p0)
    out["hausdorff_ch"] = hd
    out["hausdorff_ratio_ch"] = hd / hd_p       # vs persistence, as the skill columns are
    return out


def calibrate_sigma(model, ds, target=0.95):
    """Sigma scale s.t. the RECORDING-BALANCED |z| distribution hits `target` coverage.

    The percentile is weighted by 1/n_windows(recording) (2026-09-09 doctrine): pooled, a val
    split's longest recordings would set the calibration for everyone -- the same failure the
    checkpoint-selection fix closed. DataLoader without shuffle preserves ds.index order, so
    the per-window weights line up with _predict's output rows.
    """
    mu, sd, y, _ = _predict(model, ds)
    if len(mu) == 0:
        return 1.0
    r = np.abs(y - mu) / (sd + 1e-9)                              # (N,H,C)
    w = np.repeat(recording_weights(_rec_ids(ds)), r[0].size)     # window weight -> each element
    return float(weighted_percentile(r.ravel(), w, 100 * target) / 2.0)


@torch.no_grad()
def _per_recording(model, ds, tnorm, H):
    """-> {idx: (y_raw (T,C), origins, mu_raw (n,H,C), sigma_raw (n,H,C))}.

    The CV concatenates every test recording's origins into one array, which is all a metric
    needs and useless for a figure: clip identity is gone. This splits them back out, in RAW
    units, in the layout scripts/opentouch/plot_opentouch_forecast_overlay.py already reads --
    so ONE overlay plotter serves both sensors rather than a second being written here.
    """
    mu, sd, _, _ = _predict(model, ds)
    if not len(mu):
        return {}
    src = ds.tgts if hasattr(ds, "tgts") else ds.sig       # z-normed ABSOLUTE target
    absolute = not ds.residual
    idxs = [i for i, _ in ds.index]
    out, at = {}, 0
    for i in sorted(set(idxs)):
        ors = np.asarray([t for j, t in ds.index if j == i])
        sl = slice(at, at + len(ors)); at += len(ors)
        z = np.asarray(src[i], dtype=np.float64)
        # A residual model predicts the delta from the value at the origin, so the anchor is
        # z[origin] -- taken from the series, not reconstructed from the persistence array,
        # which is all zeros in that space and carries no anchor at all.
        m = np.asarray(mu[sl], dtype=np.float64)
        if not absolute:
            m = m + z[ors][:, None, :]
        out[i] = (tnorm.unz(z), ors, tnorm.unz(m),
                  np.asarray(sd[sl], dtype=np.float64) * np.asarray(tnorm.std))
    return out


def corpus_tag(cfg: Config) -> str:
    """'data/egotouch_states' -> 'egotouch'. The npz's `tag` drives the figure caption, so
    hardcoding it made every corpus that reused this writer claim to be ActionSense."""
    return os.path.basename(cfg.abspath("states_root").rstrip("/")).split("_")[0]


def save_predictions(store: dict, cfg: Config, out_dir: str, verbs: dict,
                     objects: dict | None = None):
    """Write one clip_<idx>.npz per recording, in the OpenTouch overlay format.

    MERGES into an existing clip_<idx>.npz rather than replacing it. The sweep calls this once
    per encoder, all into one --save-preds directory, and `np.savez_compressed` truncates the
    file it opens -- so a flatten,cnn,aggregate run used to leave only `mu_cnn` behind, the two
    earlier arms silently gone, and the per-action scorer would compare one model against
    persistence while appearing to compare three.

    Merging is only safe while the arms describe the SAME recordings, so `y` and `origins` from
    the existing file must match what is being written. They will not match across scopes or
    histories (a different t_in shifts every window origin), and there the merge raises instead
    of interleaving two populations into one npz that nothing downstream could tell apart.
    """
    import json
    os.makedirs(out_dir, exist_ok=True)
    idxs = sorted({i for d in store.values() for i in d})
    n_merged = 0
    for i in idxs:
        first = next(d[i] for d in store.values() if i in d)
        y, origins = first[0], first[1]
        keep = {}
        path = os.path.join(out_dir, f"clip_{i}.npz")
        if os.path.exists(path):
            with np.load(path, allow_pickle=False) as z:
                if z["y"].shape != y.shape or not np.allclose(z["y"], y, equal_nan=True) \
                        or z["origins"].shape != origins.shape \
                        or not np.array_equal(z["origins"], origins):
                    raise ValueError(
                        f"{path} holds a different recording than the one being saved "
                        f"(y {z['y'].shape} vs {y.shape}, origins {z['origins'].shape} vs "
                        f"{origins.shape}). That means this --save-preds directory is already "
                        f"holding another scope or history; point --save-preds at a fresh "
                        f"directory rather than mixing two populations into one npz.")
                keep = {k: z[k] for k in z.files
                        if k.startswith(("mu_", "sigma_"))}
                n_merged += bool(keep)
        arms = {f"mu_{m}": d[i][2] for m, d in store.items() if i in d}
        arms |= {f"sigma_{m}": d[i][3] for m, d in store.items() if i in d}
        np.savez_compressed(
            path,
            y=y, origins=origins, fps=cfg.fps,
            action=verbs.get(i, ""), object_name=(objects or {}).get(i, ""),
            channels=np.array(cfg.channels), tag=corpus_tag(cfg),
            **{**keep, **arms})            # a re-run of the same arm replaces its own keys
    have = sorted({k[3:] for k in np.load(os.path.join(out_dir, f"clip_{idxs[0]}.npz")).files
                   if k.startswith("mu_")}) if idxs else []
    print(f"  saved {len(idxs)} recordings of forecasts -> {out_dir}"
          f"{f' (merged into {n_merged} existing)' if n_merged else ''}; arms now: {have}")
    return len(idxs)


def cross_validate(cfg: Config, tm: dict, encoder: str, t_in: int, recs: list[int],
                   folds: int = 5, seed: int = 0, save_preds: str | None = None):
    """5-fold CV by recording. Norms + model fit on TRAIN; sigma calibrated on a VAL subset of TRAIN;
    skill + coverage measured on the held-out TEST fold. -> (sk_ch (folds,6), sk_step (folds,H,6),
    cov_raw, cov_cal)."""
    rng = np.random.default_rng(seed)
    fold_of = rng.integers(0, folds, size=len(recs))
    skc, sks, cr, cc, hdc, hdr = [], [], [], [], [], []
    preds = {}                       # recording idx -> forecasts, filled only if requested
    # Per-fold progress, flushed. Without it the log says nothing between the header and the
    # summary line of a whole encoder x history combination, so a slow run and a hung one
    # look identical -- the reason src/opentouch/prob_gru.py prints per epoch.
    t0 = time.time()
    for f in range(folds):
        te = [recs[i] for i in range(len(recs)) if fold_of[i] == f]
        tr = [recs[i] for i in range(len(recs)) if fold_of[i] != f]
        if len(te) < 1 or len(tr) < 4:
            continue
        r2 = np.random.default_rng(seed * 100 + f)
        idx = r2.permutation(len(tr)); nv = max(2, len(tr) // 6)
        val, trn = [tr[i] for i in idx[:nv]], [tr[i] for i in idx[nv:]]

        # The probGRU backbone predicts the ABSOLUTE target and carries an action embedding
        # whose vocabulary is built from THIS FOLD's TRAIN only. Seq2Seq keeps the residual
        # target it has always had, and its every-item action id is simply ignored.
        pg = tm.get("backbone", "seq2seq") == "probgru"
        verbs = D.verbs_of(cfg, recs)
        vocab, by_idx = D.action_vocab(verbs, trn)
        tm = {**tm, "n_act": len(vocab)}
        aids = {i: D.aid_of(vocab, by_idx, i) for i in recs}
        kw = dict(aids=aids, residual=not pg)

        if encoder == "aggregate":                          # neural AR on the aggregate 6-dim F/CoP
            tnorm = Norm.from_train({i: load_target(cfg, i) for i in trn})
            mk = lambda ids: D.AggWindows({i: tnorm.z(load_target(cfg, i)) for i in ids},  # noqa: E731
                                          cfg, t_in, **kw)
            train_ds, val_ds, test_ds = mk(trn), mk(val), mk(te)
        else:                                               # map input (flatten / cnn)
            maps_tr, tgts_tr = D.load_raw(cfg, trn, tm["baseline_frames"])
            mnorm = D.MapNorm.from_train(maps_tr, tm["alpha"]); tnorm = Norm.from_train(tgts_tr)
            train_ds = D.MapWindows(D.normalize(maps_tr, mnorm),
                                    {i: tnorm.z(t) for i, t in tgts_tr.items()}, cfg, t_in,
                                    **kw)
            val_ds = _dataset(cfg, tm, t_in, val, mnorm, tnorm, **kw)
            test_ds = _dataset(cfg, tm, t_in, te, mnorm, tnorm, **kw)
        model = train_model(train_ds, val_ds, cfg, encoder, tm, seed=seed,
                            materialize=(encoder == "aggregate"))
        s = calibrate_sigma(model, val_ds)
        ev = evaluate(model, test_ds, sigma_scale=1.0)
        sk_ch, sk_step, c_raw = ev["skill_ch"], ev["skill_step"], ev["coverage"]
        hd_ch, hd_ratio = ev["hausdorff_ch"], ev["hausdorff_ratio_ch"]
        c_cal = evaluate(model, test_ds, sigma_scale=s)["coverage"]
        skc.append(sk_ch); sks.append(sk_step); cr.append(c_raw); cc.append(c_cal)
        hdc.append(hd_ch); hdr.append(hd_ratio)
        if save_preds:
            preds.update(_per_recording(model, test_ds, tnorm, cfg.horizon))
        el = time.time() - t0
        done = len(skc)
        print(f"    [{encoder} t_in={t_in}] fold {done}/{folds} | "
              f"train {len(train_ds)} test {len(test_ds)} windows | "
              f"meanSkill {float(np.mean(sk_ch)):+.3f} | "
              f"{el:.0f}s elapsed, ~{el / done * (folds - done):.0f}s left", flush=True)
    return {"skill_ch": np.array(skc), "skill_step": np.array(sks),
            "coverage_raw": float(np.mean(cr)), "coverage_cal": float(np.mean(cc)),
            "hausdorff_ch": np.array(hdc), "hausdorff_ratio_ch": np.array(hdr),
            "preds": preds}
