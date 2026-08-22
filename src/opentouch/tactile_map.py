"""Tactile-map -> F/CoP forecasters for OpenTouch: flatten vs CNN vs aggregate.

Mirrors src/actionsense/tactile_map/ (models.py + data.py + train.py, compacted into one
module). Copied verbatim: the three per-frame encoders, an IDENTICAL GRU and one-shot
PROBABILISTIC head behind all of them so the encoder is the only variable, the
residual-over-persistence target, the Gaussian NLL with logvar clamped to [-6,4], log1p
amplitude compression, the global TRAIN map scale, harness-aligned origins with causal
left zero-padding, and d=64 / hidden=64 / lr=3e-3 / batch=64.

THREE THINGS DIFFER, EACH DECIDED RATHER THAN DRIFTED INTO (user rulings, 2026-08-19):

  1. GRID IS 1x16x16, NOT 2x32x32. Forced: OpenTouch instruments one hand on a 16x16 grid,
     so FLAT is 256 rather than 2048 and the CNN's two stride-2 convs walk 16->8->4. Same
     class of change as the 6->3 channel fix the other forks made.

  2. THE BASELINE IS D1's PER-TAXEL MEDIAN, NOT ActionSense's mean of the first N frames.
     ActionSense can use the opening frames because its recordings start out of contact;
     OpenTouch clips are segmented AROUND A PRESSURE PEAK (median 2.80 s), so the first
     second is 36% of the clip and is often already in contact -- extract_opentouch.py's
     docstring warned about exactly this. The median was validated on this corpus instead:
     duty_cycle came out below 0.5 for every taxel in all 26 shards, which is the condition
     under which a taxel's median IS its resting level (SESSION_LOG 2026-08-16).
     CAVEAT worth stating wherever these numbers appear: the median is pooled over a
     SHARD, so for a held-out location it is estimated from that location's own frames --
     transductive in the inputs, never in the targets. --baseline-scope train (or trainval)
     restricts it, at the cost of having NO estimate at all for a location that is entirely
     held out -- and under location-held-out CV that is every test location, so those scopes
     drop the whole test set. That is why "shard" is the default rather than a loosening:
     see scope_ids(). Until 2026-08-22 this paragraph described a flag that did not exist,
     while base_scope="shard" quietly meant TRAIN+VAL, and the resulting empty test set was
     filled with zeros and scored.

  3. THE HISTORY SWEEP IS {1,2,3} s, NOT {1,3,10}. OQ-H (2026-08-11): a 10 s history leaves
     90 of 2958 clips, and even 3 s already leaves ~75% of clips with no unpadded window.

The aggregate arm here is PROBABILISTIC, which overturns OQ-G for this family: ActionSense's
three encoders share one probabilistic head, and that is the whole point of the comparison.
src/opentouch/gru_aggregate.py is left alone as the pre-registered deterministic arm.
"""
from __future__ import annotations

import collections
import json
import os

import numpy as np
import torch
import torch.nn as nn

from src.actionsense.eval_harness.config import Config
from .baselines import origins
from .dataset import Norm, load_target
from .gru_aggregate import configure_determinism

IN_CH, GRID = 1, 16
FLAT = IN_CH * GRID * GRID                      # 256 (ActionSense: 2*32*32 = 2048)
DEFAULT_HP = {"d": 64, "hidden": 64, "epochs": 60, "lr": 0.003, "batch": 64, "seed": 0,
              "alpha": 10.0, "log_train_every": 5}


# ------------------------------------------------------------------------------ models --
class FlattenEncoder(nn.Module):
    """Flatten each frame -> linear -> embedding (no spatial structure exploited)."""

    def __init__(self, d: int):
        super().__init__()
        self.proj = nn.Sequential(nn.Flatten(), nn.Linear(FLAT, d), nn.ReLU())

    def forward(self, x):                        # (B,t_in,1,16,16) -> (B,t_in,d)
        B, T = x.shape[:2]
        return self.proj(x.reshape(B * T, IN_CH, GRID, GRID)).reshape(B, T, -1)


class CNNEncoder(nn.Module):
    """Small conv stack per frame -> embedding (exploits spatial structure)."""

    def __init__(self, d: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(IN_CH, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),    # 16->8
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU(),    # 8->4
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(32, d), nn.ReLU())

    def forward(self, x):
        B, T = x.shape[:2]
        return self.conv(x.reshape(B * T, IN_CH, GRID, GRID)).reshape(B, T, -1)


class AggEncoder(nn.Module):
    """Per-frame projection of the aggregate F/CoP -> embedding. The neural counterpart of
    the linear AR baseline: same target, input is the aggregate history rather than the map."""

    def __init__(self, d: int, n_in: int = 3):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(n_in, d), nn.ReLU())

    def forward(self, x):
        return self.proj(x)


class Seq2Seq(nn.Module):
    """encoder -> GRU -> one-shot PROBABILISTIC head -> (mu, lv), each (B,H,C).

    Predicts the RESIDUAL against the last observed value, as a Gaussian per (step, channel).
    At worst it predicts 0 and matches persistence; its job is the delta."""

    def __init__(self, encoder: nn.Module, d: int, hidden: int, horizon: int, n_out: int):
        super().__init__()
        self.encoder = encoder
        self.gru = nn.GRU(d, hidden, batch_first=True)
        self.mu = nn.Linear(hidden, horizon * n_out)
        self.lv = nn.Linear(hidden, horizon * n_out)
        self.H, self.n_out = horizon, n_out

    def forward(self, x):
        _, h = self.gru(self.encoder(x))
        last = h[-1]
        return (self.mu(last).reshape(-1, self.H, self.n_out),
                self.lv(last).clamp(-6, 4).reshape(-1, self.H, self.n_out))


def build_model(encoder: str, cfg: Config, hp: dict) -> Seq2Seq:
    C = len(cfg.channels)
    enc = {"flatten": FlattenEncoder, "cnn": CNNEncoder,
           "aggregate": lambda d: AggEncoder(d, C)}[encoder](hp["d"])
    return Seq2Seq(enc, hp["d"], hp["hidden"], cfg.horizon, C)


# -------------------------------------------------------------------------------- data --
def compress(x, alpha):
    """log1p amplitude compression, normalized so compress(1/alpha) ~ O(1). No train stats."""
    return np.log1p(alpha * np.clip(x, 0.0, None)) / np.log1p(alpha)


def shard_of(cfg: Config) -> dict[int, str]:
    root = cfg.abspath("states_root")
    return {r["idx"]: r.get("shard", "?") for r in
            (json.loads(l) for l in open(os.path.join(root, "manifest.jsonl")))}


def taxel_baselines(cfg: Config, idxs: list[int], max_frames: int = 20000,
                    stride: int = 3) -> dict[str, np.ndarray]:
    """Per shard, the per-taxel median over that shard's frames -- D1's estimator.

    Subsampled the same way the D1 report measured it; a median needs the values, not all
    of them."""
    root = cfg.abspath("states_root")
    by_shard = collections.defaultdict(list)
    for i, sh in shard_of(cfg).items():
        if i in set(idxs):
            by_shard[sh].append(i)
    out = {}
    missing = 0
    for sh, ids in by_shard.items():
        frames, n = [], 0
        for i in sorted(ids):
            p = os.path.join(root, f"clip_{i}.npy")
            if not os.path.exists(p):
                missing += 1
                continue
            a = np.load(p).astype(np.float32).reshape(-1, GRID * GRID)[::stride]
            frames.append(a); n += len(a)
            if n >= max_frames:
                break
        if frames:
            out[sh] = np.median(np.concatenate(frames, 0)[:max_frames], axis=0)
    if not out:
        # Skipping a missing clip is right; skipping ALL of them and returning {} is not.
        # build_inputs then yields an empty dict, both map arms train on nothing, and the
        # run still emits a full metric table -- which is what happened on 2026-08-22:
        # flatten and cnn came back identical to four decimals with sigma exactly 0, because
        # neither had seen a single frame. A run that reports numbers from no data is worse
        # than one that crashes.
        raise FileNotFoundError(
            f"no clip_*.npy under {root} ({missing} expected files absent), so the map arms "
            f"have no input. The aggregate arm reads state_*.npy and would still run, which "
            f"is why this fails loudly instead of quietly. If this cache holds corrected "
            f"states only, link the maps in beside them: "
            f"ln -s ~/opentouch/cache/clip_*.npy {root}/")
    return out


def scope_ids(cfg: Config, train_ids: list[int], val_ids: list[int], scope: str) -> list[int]:
    """Which clips the per-taxel baseline may be estimated from.

    "shard" is the scope this module's docstring has always described and, until 2026-08-22,
    the only one that works under location-held-out CV. A taxel's resting level is a property
    of the hardware, so pooling a shard's own frames to find it is transductive in the INPUTS
    and never in the targets -- the argument recorded on 2026-08-16.

    "trainval" was what `base_scope="shard"` actually did, despite the name, and "train" is
    stricter still. Both leave a wholly held-out location with NO baseline, at which point
    every one of its clips is dropped. That is not a conservative choice, it is a broken one:
    the 2026-08-22 map run reported a full metric table for flatten and cnn whose predictions
    were arrays of zeros, because every test clip had been dropped this way and filled in.
    """
    if scope == "train":
        return sorted(set(train_ids))
    if scope == "trainval":
        return sorted(set(train_ids) | set(val_ids))
    if scope == "shard":
        return sorted(shard_of(cfg))
    raise ValueError(f"baseline scope must be shard/trainval/train, got {scope!r}")


def load_map(cfg: Config, idx: int, base: np.ndarray) -> np.ndarray:
    """clip_<idx>.npy -> (T,1,16,16) float32, baseline removed and clipped at zero."""
    p = os.path.join(cfg.abspath("states_root"), f"clip_{idx}.npy")
    m = np.load(p).astype(np.float32)[:: cfg.downsample].reshape(-1, GRID * GRID)
    return np.clip(m - base[None, :], 0.0, None).reshape(-1, IN_CH, GRID, GRID)


class MapNorm:
    """Global scalar normalization of the compressed map (same scaling for every taxel)."""

    def __init__(self, mean, std, alpha):
        self.mean, self.std, self.alpha = mean, std, alpha

    @staticmethod
    def from_train(maps: dict[int, np.ndarray], alpha: float) -> "MapNorm":
        v = np.concatenate([compress(m, alpha).reshape(-1) for m in maps.values()])
        return MapNorm(float(v.mean()), float(v.std() + 1e-6), alpha)

    def apply(self, m):
        return ((compress(m, self.alpha) - self.mean) / self.std).astype(np.float32)


def windows(cfg: Config, ids: list[int], t_in: int, inputs: dict[int, np.ndarray],
            norm: Norm):
    """(X, Y) over the harness origins. Y is the RESIDUAL against the last observed value;
    early origins are left zero-padded, so a prediction exists at every origin."""
    H = cfg.horizon
    Xs, Ys = [], []
    for i in ids:
        M = inputs[i]
        z = norm.z(np.asarray(load_target(cfg, i), dtype=np.float64)).astype(np.float32)
        n = min(len(M), len(z))
        M, z = M[:n], z[:n]
        for t in origins(n, cfg):
            w = M[max(t - t_in + 1, 0): t + 1]
            if w.shape[0] < t_in:
                w = np.concatenate([np.zeros((t_in - w.shape[0],) + M.shape[1:],
                                             np.float32), w], 0)
            Xs.append(w); Ys.append(z[t + 1: t + 1 + H] - z[t])
    if not Xs:
        return torch.zeros(0), torch.zeros(0)
    return torch.from_numpy(np.stack(Xs)), torch.from_numpy(np.stack(Ys).astype(np.float32))


def build_inputs(cfg: Config, encoder: str, ids: list[int], base_ids: list[int],
                 norm: Norm, alpha: float, mnorm: "MapNorm | None" = None):
    """-> (inputs per clip, MapNorm or None). `base_ids` is what the baseline and the map
    scale may be estimated from (TRAIN, or every clip of the shard -- see the module docstring)."""
    if encoder == "aggregate":
        return {i: norm.z(np.asarray(load_target(cfg, i), dtype=np.float64)).astype(np.float32)
                for i in ids}, None
    bases = taxel_baselines(cfg, base_ids)
    sh = shard_of(cfg)
    raw = {i: load_map(cfg, i, bases[sh[i]]) for i in ids if sh[i] in bases}
    if not raw:
        raise FileNotFoundError(
            f"every one of the {len(ids)} clips was dropped for want of a shard baseline; "
            f"the {encoder!r} arm cannot train on an empty input set")
    if mnorm is None:
        mnorm = MapNorm.from_train(raw, alpha)
    return {i: mnorm.apply(m) for i, m in raw.items()}, mnorm


# ---------------------------------------------------------------------------- training --
def nll(mu, lv, y):
    return 0.5 * (lv + (y - mu) ** 2 * torch.exp(-lv)).mean()


@torch.no_grad()
def _scores(m, X, Y, batch, dev):
    m.eval()
    tot_nll = tot_se = n = cnt = 0.0
    for i in range(0, len(X), batch):
        yb = Y[i:i + batch].to(dev)
        mu, lv = m(X[i:i + batch].to(dev))
        tot_nll += float(nll(mu, lv, yb)) * len(mu); n += len(mu)
        tot_se += float(((mu - yb) ** 2).sum()); cnt += yb.numel()
    return tot_nll / max(n, 1.0), tot_se / max(cnt, 1.0)


def train(cfg: Config, encoder: str, train_ids: list[int], val_ids: list[int], t_in: int,
          hp: dict | None = None, norm: Norm | None = None, device: str | None = None,
          base_scope: str = "shard", verbose: bool = True):
    """Fit on TRAIN, keep the lowest-VAL-NLL weights. -> (model, norm, mnorm, history)."""
    hp = {**DEFAULT_HP, **(hp or {})}
    gen = configure_determinism(int(hp["seed"]))
    if norm is None:
        norm = Norm.from_train({i: load_target(cfg, i) for i in train_ids})
    base_ids = scope_ids(cfg, train_ids, val_ids, base_scope)
    tr_in, mnorm = build_inputs(cfg, encoder, train_ids, base_ids, norm, hp["alpha"])
    va_in, _ = build_inputs(cfg, encoder, val_ids, base_ids, norm, hp["alpha"], mnorm)

    Xtr, Ytr = windows(cfg, [i for i in train_ids if i in tr_in], t_in, tr_in, norm)
    Xva, Yva = windows(cfg, [i for i in val_ids if i in va_in], t_in, va_in, norm)
    if len(Xtr) == 0:
        raise ValueError("no TRAIN windows")

    dev = torch.device(device) if device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    m = build_model(encoder, cfg, hp).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=hp["lr"])
    bs, n_ep = int(hp["batch"]), int(hp["epochs"])
    if verbose:
        print(f"    [{encoder} t_in={t_in}] windows train {len(Xtr)} val {len(Xva)} | "
              f"device {dev}", flush=True)
    best, best_state = np.inf, None
    hist = {"train_nll": [], "val_nll": [], "val_mse": []}
    for ep in range(n_ep):
        m.train()
        perm = torch.randperm(len(Xtr), generator=gen)
        for i in range(0, len(Xtr), bs):
            b = perm[i:i + bs]
            mu, lv = m(Xtr[b].to(dev))
            loss = nll(mu, lv, Ytr[b].to(dev))
            opt.zero_grad(); loss.backward(); opt.step()
        want = (ep % int(hp["log_train_every"]) == 0) or (ep == n_ep - 1)
        tr = _scores(m, Xtr, Ytr, bs, dev)[0] if want else float("nan")
        va, va_mse = (_scores(m, Xva, Yva, bs, dev) if len(Xva)
                      else ((tr if want else float("nan")), float("nan")))
        hist["train_nll"].append(tr); hist["val_nll"].append(va)
        hist["val_mse"].append(va_mse)
        if va < best:
            best = va
            best_state = {k: v.detach().clone() for k, v in m.state_dict().items()}
        if verbose:
            trs = f"{tr:.5f}" if np.isfinite(tr) else "  --   "
            print(f"    [{encoder} t_in={t_in}] epoch {ep + 1}/{n_ep} train {trs} "
                  f"val {va:.5f} mse {va_mse:.5f}", flush=True)
    if best_state:
        m.load_state_dict(best_state)
    m.eval()
    hist.update(best_val_nll=float(best), encoder=encoder, t_in=int(t_in),
                device=str(dev), base_scope=base_scope)
    return m, norm, mnorm, hist


@torch.no_grad()
def predict_with_sigma(model, cfg: Config, encoder: str, norm: Norm, mnorm,
                       test_ids: list[int], t_in: int, base_ids: list[int]):
    """({idx: mu}, {idx: sigma}) in RAW units.

    The head predicts a residual in z-space, so the anchor is a constant shift and cancels:
    sigma_raw = exp(lv/2) * norm.std, exactly as in prob_gru."""
    model.eval()
    inputs, _ = build_inputs(cfg, encoder, test_ids, base_ids, norm,
                             mnorm.alpha if mnorm else DEFAULT_HP["alpha"], mnorm)
    dev = next(model.parameters()).device
    mus, sds = {}, {}
    C = len(cfg.channels)
    for i in test_ids:
        Y = np.asarray(load_target(cfg, i), dtype=np.float64)
        z = norm.z(Y)
        ors = origins(len(z), cfg)
        if len(ors) == 0:
            # A clip too short for a single origin genuinely has no forecast to make; the
            # empty arrays are the right answer and score_external expects them.
            mus[i] = np.zeros((0, cfg.horizon, C))
            sds[i] = np.zeros((0, cfg.horizon, C))
            continue
        if i not in inputs:
            # Zeros here are not an answer, they are a fabrication, and they scored as one:
            # on 2026-08-22 every test clip landed in this branch (their shards had no
            # baseline under the trainval scope) and flatten and cnn returned complete
            # metric tables built entirely from zero arrays, identical to four decimals.
            raise RuntimeError(
                f"clip {i} has {len(ors)} origins but no input: its shard has no taxel "
                f"baseline under the current scope. Under location-held-out CV a held-out "
                f"shard is absent from TRAIN+VAL by construction -- use baseline scope "
                f"'shard', which estimates the resting level from that shard's own frames "
                f"(transductive in inputs, never in targets).")
        X, _ = windows(cfg, [i], t_in, {i: inputs[i]}, norm)
        mu, lv = model(X.to(dev))
        resid = mu.cpu().numpy().astype(np.float64)
        mus[i] = norm.unz(z[ors][:, None, :] + resid)
        sds[i] = np.exp(0.5 * lv.cpu().numpy().astype(np.float64)) * np.asarray(norm.std)
    return mus, sds


@torch.no_grad()
def predict(model, cfg: Config, encoder: str, norm: Norm, mnorm, test_ids: list[int],
            t_in: int, base_ids: list[int]) -> dict[int, np.ndarray]:
    """{clip idx: (n_origins,H,C) RAW-unit forecasts}, for evaluate.score_external.

    Undoes both conventions: add the residual back onto the last observed normalized value,
    then de-normalize -- the same two steps gru_aggregate.predict_clip takes."""
    model.eval()
    inputs, _ = build_inputs(cfg, encoder, test_ids, base_ids, norm, mnorm.alpha if mnorm
                             else DEFAULT_HP["alpha"], mnorm)
    dev = next(model.parameters()).device
    out = {}
    for i in test_ids:
        Y = np.asarray(load_target(cfg, i), dtype=np.float64)
        z = norm.z(Y)
        ors = origins(len(z), cfg)
        if i not in inputs or len(ors) == 0:
            out[i] = np.zeros((len(ors), cfg.horizon, len(cfg.channels)))
            continue
        X, _ = windows(cfg, [i], t_in, {i: inputs[i]}, norm)
        mu, _ = model(X.to(dev))
        resid = mu.cpu().numpy().astype(np.float64)
        out[i] = norm.unz(z[ors][:, None, :] + resid)
    return out
