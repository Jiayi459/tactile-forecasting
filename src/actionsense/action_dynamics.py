"""v2 physical-state action-dynamics forecaster — the LIBRARY (import, don't run).

Single source of truth for the model + data + training + forecasting used by the thin CLIs
scripts/actionsense/train_action_dynamics.py and scripts/actionsense/plot_action_forecast.py. See docs/TACTILE_FORECAST_PLAN.md.

Idea: split each physical-state signal (force, center-of-pressure) into a slow (grip/postural)
and a fast (stroke/pour) component; forecast the FAST component with a probabilistic GRU
(mean + variance). Baseline = persistence-of-fast.

CAUSALITY: the slow/fast split uses a CAUSAL forward-only filter (sosfilt) and velocities use a
CAUSAL backward difference, so no feature at time t depends on any future sample (required for a
forecasting task — filtfilt would leak the future). The causal filter has a startup transient, so
the first `warmup_sec` seconds of every clip are dropped (in both training and evaluation).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from scipy.signal import butter, sosfilt

FEATS_HIGHPASS = ("F_fast", "x_fast", "y_fast", "F_slow", "vx", "vy")   # decomposed input (6)
FEATS_RAW = ("F", "x", "y", "vx", "vy")                                 # raw input, no split (5)
TARGETS = ("F_fast", "x_fast", "y_fast")                                # output = fast (3)
HANDS = {"left": 0, "right": 1}                                         # state channel per hand


def feats_for(input_mode):
    return FEATS_HIGHPASS if input_mode == "highpass" else FEATS_RAW


# --------------------------------------------------------------------------- #
# Signal processing + features  (all CAUSAL)
# --------------------------------------------------------------------------- #
def slow_fast(sig, fps, cut):
    """CAUSAL low/high-pass split: slow = forward-only Butterworth low-pass (sosfilt);
    fast = sig - slow. Output at t uses only samples <= t (no future leakage)."""
    sos = butter(2, cut / (fps / 2.0), "low", output="sos")
    slow = sosfilt(sos, sig, axis=0)
    return slow, sig - slow


def _causal_diff(sig, fps):
    """Backward difference velocity (causal): v[t] = (sig[t]-sig[t-1])*fps, v[0]=0."""
    v = np.zeros_like(sig)
    v[1:] = np.diff(sig) * fps
    return v


def build_features(state, fps_raw, cut, ds, input_mode="highpass", hand="active", warmup_sec=5.0):
    """state (T,C,6) -> (feat (T',D), target (T',3)) for one hand.

    hand: 'left' | 'right' | 'active' (active = highest mean force).
    input_mode: 'highpass' -> [F_fast,x_fast,y_fast,F_slow,vx,vy]; 'raw' -> [F,x,y,vx,vy].
    target: always fast [F_fast,x_fast,y_fast].
    warmup_sec: drop the leading filter-transient (both train & eval)."""
    state = state[::ds]
    fps = fps_raw / ds
    if hand == "active":
        h = int(np.argmax(state[:, :, 0].mean(0)))
    else:
        h = HANDS[hand]
    F, x, y = state[:, h, 0], state[:, h, 1], state[:, h, 2]
    Fs, Ff = slow_fast(F, fps, cut)
    _, xf = slow_fast(x, fps, cut)
    _, yf = slow_fast(y, fps, cut)
    vx, vy = _causal_diff(x, fps), _causal_diff(y, fps)
    target = np.stack([Ff, xf, yf], 1).astype(np.float32)
    if input_mode == "highpass":
        feat = np.stack([Ff, xf, yf, Fs, vx, vy], 1).astype(np.float32)
    else:  # raw
        feat = np.stack([F, x, y, vx, vy], 1).astype(np.float32)
    w = int(round(warmup_sec * fps))
    return feat[w:], target[w:]


def load_pooled(root, action_subs, ds, cut, input_mode="highpass", hand="active",
                warmup_sec=5.0, fps_default=30.0, min_len=20):
    """Read the state dataset -> list of (feat, target, action_id) for one hand + input_mode.
    Matches a clip to an action if its label STARTS WITH the action string."""
    rows = [json.loads(l) for l in open(os.path.join(root, "manifest.jsonl"))]
    data = []
    for r in rows:
        aid = next((i for i, s in enumerate(action_subs)
                    if r["label"].lower().startswith(s.lower())), None)
        if aid is None:
            continue
        st = np.load(os.path.join(root, f"state_{r['idx']}.npy"))
        feat, targ = build_features(st, r.get("fps", fps_default), cut, ds,
                                    input_mode=input_mode, hand=hand, warmup_sec=warmup_sec)
        if feat.shape[0] >= min_len:
            data.append((feat, targ, aid))
    return data


def windows(clips, t_in, t_out, stride):
    """Sliding (past t_in -> future t_out) windows within each clip. Input frames [s, s+t_in)
    are STRICTLY before target frames [s+t_in, s+t_in+t_out) — no cross-clip leakage."""
    Xs, As, Yin, Ys, gid = [], [], [], [], []
    for gi, (feat, targ, aid) in enumerate(clips):
        win = t_in + t_out
        for s in range(0, feat.shape[0] - win + 1, stride):
            Xs.append(feat[s:s + t_in]); Yin.append(targ[s:s + t_in])
            Ys.append(targ[s + t_in:s + win]); As.append(aid); gid.append(gi)
    if not Xs:
        d = clips[0][0].shape[1] if clips else len(FEATS_HIGHPASS)
        return (np.zeros((0, t_in, d), np.float32), np.zeros(0, int),
                np.zeros((0, t_in, 3), np.float32), np.zeros((0, t_out, 3), np.float32), np.zeros(0, int))
    return np.stack(Xs), np.array(As), np.stack(Yin), np.stack(Ys), np.array(gid)


def split_train_test(n, frac=0.25, seed=1, force_test=()):
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    n_test = max(2, int(round(frac * n)))
    test = set(order[:n_test].tolist()) | set(force_test)
    return [i for i in range(n) if i not in test], sorted(test)


# --------------------------------------------------------------------------- #
# Normalization + model
# --------------------------------------------------------------------------- #
@dataclass
class Norm:
    mu_x: np.ndarray; sd_x: np.ndarray; mu_y: np.ndarray; sd_y: np.ndarray

    @classmethod
    def from_clips(cls, clips):
        """Z-score stats from the GIVEN clips only (pass train clips -> no test leakage)."""
        X = np.concatenate([f for f, _, _ in clips]); Y = np.concatenate([t for _, t, _ in clips])
        return cls(X.mean(0), X.std(0) + 1e-6, Y.mean(0), Y.std(0) + 1e-6)

    def nx(self, x): return ((x - self.mu_x) / self.sd_x).astype(np.float32)
    def ny(self, y): return ((y - self.mu_y) / self.sd_y).astype(np.float32)
    def dy(self, y): return y * self.sd_y + self.mu_y


class ProbGRU(nn.Module):
    def __init__(self, din, n_act, hid):
        super().__init__()
        self.emb = nn.Embedding(n_act, 8)
        self.enc = nn.GRU(din, hid, batch_first=True)
        self.dec = nn.GRU(3, hid, batch_first=True)
        self.mu = nn.Linear(hid + 8, 3)
        self.lv = nn.Linear(hid + 8, 3)

    def forward(self, x, aid, y_last, t_out):
        _, h = self.enc(x)
        e = self.emb(aid)
        inp = y_last.unsqueeze(1)
        mus, lvs = [], []
        for _ in range(t_out):
            o, h = self.dec(inp, h)
            oc = torch.cat([o[:, -1], e], -1)
            mu = self.mu(oc); lv = self.lv(oc).clamp(-6, 4)
            mus.append(mu); lvs.append(lv)
            inp = mu.unsqueeze(1)
        return torch.stack(mus, 1), torch.stack(lvs, 1)


# --------------------------------------------------------------------------- #
# Train / evaluate / forecast
# --------------------------------------------------------------------------- #
def _nll(m, X, A, Yin, Y, norm, t_out):
    """Mean Gaussian NLL on a window set (for early-stopping model selection)."""
    with torch.no_grad():
        mu, lv = m(torch.tensor(norm.nx(X)), torch.tensor(A),
                   torch.tensor(norm.ny(Yin)[:, -1]), t_out)
        return float((0.5 * (lv + (torch.tensor(norm.ny(Y)) - mu) ** 2 * torch.exp(-lv))).mean())


def train(clips, n_act, t_in, t_out, norm=None, hidden=48, epochs=80, lr=3e-3, seed=0,
          val_clips=None):
    """Train the probGRU. If `val_clips` is given, EARLY-STOP: keep the weights with the lowest
    VAL NLL over training (the loss curve overfits badly after ~epoch 10 without this)."""
    norm = norm or Norm.from_clips(clips)
    X, A, Yin, Y, _ = windows(clips, t_in, t_out, 2)
    xt = torch.tensor(norm.nx(X)); at = torch.tensor(A)
    yl = torch.tensor(norm.ny(Yin)[:, -1]); yt = torch.tensor(norm.ny(Y))
    Xv = windows(val_clips, t_in, t_out, 2) if val_clips else None
    torch.manual_seed(seed)
    m = ProbGRU(X.shape[-1], n_act, hidden)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    best, best_state = float("inf"), None
    for _ in range(epochs):
        m.train(); perm = torch.randperm(len(xt))
        for i in range(0, len(xt), 64):
            b = perm[i:i + 64]; opt.zero_grad()
            mu, lv = m(xt[b], at[b], yl[b], t_out)
            (0.5 * (lv + (yt[b] - mu) ** 2 * torch.exp(-lv)).mean()).backward()
            opt.step()
        if Xv is not None and len(Xv[0]):                       # early-stopping check on VAL
            m.eval()
            v = _nll(m, Xv[0], Xv[1], Xv[2], Xv[3], norm, t_out)
            if v < best:
                best, best_state = v, {k: t.clone() for k, t in m.state_dict().items()}
    if best_state is not None:
        m.load_state_dict(best_state)
    return m, norm


def _predict(model, norm, clips, t_in, t_out):
    X, A, Yin, Y, _ = windows(clips, t_in, t_out, 2)
    model.eval()
    with torch.no_grad():
        mu, lv = model(torch.tensor(norm.nx(X)), torch.tensor(A),
                       torch.tensor(norm.ny(Yin)[:, -1]), t_out)
    mu = norm.dy(mu.numpy()); sd = np.sqrt(np.exp(lv.numpy())) * norm.sd_y
    pers = np.repeat(Yin[:, -1:], t_out, 1)
    return mu, sd, Y, pers


def calibrate_sigma(model, norm, val_clips, t_in, t_out, target=0.95):
    """Post-hoc scalar sigma-scale s so that coverage@2sd ~= `target` on a VALIDATION set.
    Neural nets underestimate sigma (overconfident); scaling sd by s makes the +/-2sd band honest.
    s = percentile(|Y-mu|/sd, 100*target) / 2  ->  2*s*sd then contains `target` of the residuals."""
    mu, sd, Y, _ = _predict(model, norm, val_clips, t_in, t_out)
    r = np.abs(Y - mu) / (sd + 1e-9)
    return float(np.percentile(r, 100 * target) / 2.0)


def evaluate(model, norm, clips, t_in, t_out, sigma_scale=1.0):
    """Return (skill_per_target (3,), skill_per_step (t_out,3), mean_skill, coverage@2sd).
    Skill = 1 - MSE_model/MSE_persistence (scale-free). coverage uses the calibrated sigma_scale."""
    mu, sd, Y, pers = _predict(model, norm, clips, t_in, t_out)
    em = (mu - Y) ** 2; ep = (pers - Y) ** 2                # (N, t_out, 3)
    sk_target = 1 - em.mean((0, 1)) / (ep.mean((0, 1)) + 1e-12)     # per channel
    sk_step = 1 - em.mean(0) / (ep.mean(0) + 1e-12)                 # per (step, channel)
    cov = float((np.abs(Y - mu) <= 2 * sigma_scale * sd).mean())
    return sk_target, sk_step, float(sk_target.mean()), cov


def forecast_clip(model, norm, clip, t_in, t_out, target=0, sigma_scale=1.0):
    """Honest multi-step forecast on one clip: at non-overlapping anchors, seed once with the
    true last frame then roll t_out steps on the model's OWN predictions. sigma_scale calibrates
    the predicted std for the +/-2sd bands."""
    feat, targ, aid = clip
    k = target
    fn = norm.nx(feat)
    model.eval()
    seg = []
    with torch.no_grad():
        for a in range(t_in, feat.shape[0] - t_out, t_out):
            x = torch.tensor(fn[a - t_in:a][None])
            yl = torch.tensor(norm.ny(targ[a - 1])[None])
            mu, lv = model(x, torch.tensor([aid]), yl, t_out)
            t = np.arange(a, a + t_out)
            seg.append((t, norm.dy(mu[0].numpy())[:, k],
                        np.sqrt(np.exp(lv[0, :, k].numpy())) * norm.sd_y[k] * sigma_scale,
                        np.full(t_out, targ[a - 1, k])))
    ts = np.concatenate([s[0] for s in seg]); mus = np.concatenate([s[1] for s in seg])
    sds = np.concatenate([s[2] for s in seg]); pers = np.concatenate([s[3] for s in seg])
    trues = targ[ts, k]
    sk = 1 - np.mean((mus - trues) ** 2) / (np.mean((pers - trues) ** 2) + 1e-12)
    return dict(segments=seg, ts=ts, mu=mus, sd=sds, pers=pers, true=trues, skill=float(sk))


# --------------------------------------------------------------------------- #
# Checkpoints
# --------------------------------------------------------------------------- #
def save(path, model, norm, meta):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({"model": model.state_dict(),
                "norm": (norm.mu_x, norm.sd_x, norm.mu_y, norm.sd_y),
                "meta": meta}, path)


def load(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    meta = ck["meta"]
    m = ProbGRU(meta["din"], meta["n_act"], meta["hidden"])
    m.load_state_dict(ck["model"]); m.eval()
    return m, Norm(*ck["norm"]), meta
