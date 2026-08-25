"""probGRU for d256 -- the ActionSense probabilistic GRU, architecture and loss verbatim.

The point of this arm is a number comparable across three sensors. So the model is NOT
redesigned for d256: `ProbGRU` below is the same action embedding (8) -> encoder GRU ->
autoregressive decoder GRU seeded with the last observed target -> mean/log-variance heads over
[decoder state ; action embedding], logvar clamped to [-6, 4], Gaussian NLL, early stopping on
VAL NLL, hidden 48 / epochs 80 / lr 3e-3 / batch 64. See src/opentouch/prob_gru.py for the same
transplant onto OpenTouch.

WHAT DIFFERS, AND WHY (each forced, not preferred)

1. SIX CHANNELS, TWO HANDS. d256 instruments both gloves, like ActionSense and unlike
   OpenTouch. Target is [F_L, CoPx_L, CoPy_L, F_R, CoPx_R, CoPy_R]; inputs append a causal
   velocity for each hand's CoP, giving 10 features. ActionSense differences CoP and not force;
   that asymmetry is kept, with `features="raw+df"` available as the same ablation OpenTouch
   added (dF/dt for both hands -> 12).

2. fps = 6, MEASURED. `causal_velocity` scales by fps, so this is not cosmetic. d256 carries no
   timestamps; the rate was recovered by matching recording lengths against ActionSense's known
   30 Hz (4.948 +- 0.085 over 15 unambiguous pairings -> decimation by 5). SESSION_LOG
   2026-08-24续.

3. TWO ARMS OVER THE ACTION EMBEDDING -- this is the experiment (OQ-D5, user decision
   2026-08-24), not an implementation detail:
     arm "none"  -> every recording gets id 0. The embedding degenerates to a constant, so the
                    model forecasts tactile from tactile history alone. This is the number
                    comparable to the OpenTouch and ActionSense arms.
     arm "class" -> id = label_idx, i.e. the model is told which of the 20 activities is being
                    performed. Skill will very likely be higher, but it answers a DIFFERENT
                    question: predictability GIVEN the activity. It must never be reported
                    beside the other two arms without that qualifier.
   The difference between them is the quantity of interest: what knowing the activity is worth.
   Both arms are otherwise byte-identical, so the difference is attributable.

4. NO WARMUP CUT, and WINDOWS COME FROM THE HARNESS. ActionSense drops 5 s because its causal
   filter has a startup transient; there is no filter here. Origins come from
   `baselines.base.origins`, so the model trains on exactly the windows `evaluate.score_external`
   scores it on.

5. NORMALIZATION IS PER FOLD. Targets use the fold's TRAIN-fitted `Norm` -- the one the
   baselines use, so both are on one scale. Inputs get their own TRAIN-fitted z-score because
   they include velocities, which `Norm` does not cover (mirrors ActionSense's nx/ny).
   Refitting per fold is required, not tidiness: a Norm fitted across all subjects would carry
   the held-out subject's scale into its own test scores.
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn

from src.actionsense.eval_harness.baselines.base import origins
from src.actionsense.eval_harness.config import Config

from .dataset import Norm, load_target, rows_by_idx

DEFAULT_HP = {
    "hidden": 48, "epochs": 80, "lr": 0.003, "batch": 64, "seed": 0,
    "t_in": 24,          # = eval.min_history: the encoder sees exactly the history the
                         # harness guarantees every origin has.
    "log_train_every": 5,
    "features": "raw",   # "raw" = ActionSense's inputs verbatim; "raw+df" adds dF/dt per hand
    "arm": "none",       # "none" | "class"  -- see point 3 above
    "dropout": 0.0,
    "patience": 15,
}
N_OUT = 6
COP_COLS = [1, 2, 4, 5]
F_COLS = [0, 3]


def pick_device(spec: str | None = None) -> torch.device:
    if spec:
        return torch.device(spec)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def configure_determinism(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def causal_velocity(sig: np.ndarray, fps: float) -> np.ndarray:
    """v[t] = (sig[t] - sig[t-1]) * fps, v[0] = 0. Backward difference, so no feature at time
    t sees a sample after t."""
    v = np.zeros_like(sig)
    v[1:] = np.diff(sig, axis=0) * fps
    return v


def features(Y: np.ndarray, fps: float, with_df: bool = False) -> np.ndarray:
    """(T,6) raw target -> (T,10) [6 raw channels, vx_L, vy_L, vx_R, vy_R], or (T,12) with df."""
    Y = np.asarray(Y, dtype=np.float64)
    cols = [Y, causal_velocity(Y[:, COP_COLS], fps)]
    if with_df:
        cols.append(causal_velocity(Y[:, F_COLS], fps))
    return np.concatenate(cols, axis=1)


def feature_dim(hp: dict) -> int:
    return 12 if hp.get("features") == "raw+df" else 10


class FeatNorm:
    """TRAIN-fitted z-score over the input features (targets use the harness `Norm`)."""

    def __init__(self, mean, std, with_df):
        self.mean, self.std, self.with_df = mean, std, with_df

    @staticmethod
    def from_train(cfg: Config, ids: list[int], with_df: bool) -> "FeatNorm":
        allf = np.concatenate([features(load_target(cfg, i), cfg.fps, with_df) for i in ids], 0)
        std = allf.std(0)
        std[std < 1e-8] = 1.0
        return FeatNorm(allf.mean(0), std, with_df)

    def z(self, x):
        return (x - self.mean) / self.std


def action_ids(cfg: Config, arm: str, ids: list[int]) -> tuple[dict[int, int], int]:
    """Recording idx -> embedding id, plus the vocabulary size.

    "none" collapses everything to one id, which is what makes this arm's skill comparable to
    the other two datasets. "class" uses label_idx directly -- no rare-class merging is needed
    (unlike OpenTouch's long-tailed action field) because all 20 classes are present in every
    fold's TRAIN, asserted by tests/test_d256_splits.py.
    """
    if arm == "none":
        return {i: 0 for i in ids}, 1
    if arm != "class":
        raise ValueError(f"unknown arm {arm!r}; expected 'none' or 'class'")
    rows = rows_by_idx(cfg)
    return {i: int(rows[i]["label_idx"]) for i in ids}, 20


def window_set(cfg: Config, ids: list[int], hp: dict, norm: Norm, fnorm: FeatNorm,
               aids: dict[int, int]):
    """Harness-aligned windows -> (X (N,t_in,D), A (N,), Ylast (N,6), Y (N,H,6)), normalized.

    One item per rolling origin, in the order `evaluate.score_external` will ask for them.
    `t_in` defaults to eval.min_history and origins start there, so no window needs padding;
    a shorter t_in simply looks back less far.
    """
    H, t_in = cfg.horizon, hp["t_in"]
    Xs, As, YL, Ys = [], [], [], []
    for i in ids:
        Y = load_target(cfg, i)
        f = fnorm.z(features(Y, cfg.fps, fnorm.with_df)).astype(np.float32)
        z = norm.z(np.asarray(Y, dtype=np.float64)).astype(np.float32)
        for t in origins(len(z), cfg):
            w = f[max(t - t_in + 1, 0): t + 1]
            if w.shape[0] < t_in:
                w = np.concatenate([np.zeros((t_in - w.shape[0], f.shape[1]), np.float32), w], 0)
            Xs.append(w); As.append(aids[i]); YL.append(z[t]); Ys.append(z[t + 1: t + 1 + H])
    if not Xs:
        D = feature_dim(hp)
        return (torch.zeros(0, t_in, D), torch.zeros(0, dtype=torch.long),
                torch.zeros(0, N_OUT), torch.zeros(0, H, N_OUT))
    return (torch.from_numpy(np.stack(Xs)), torch.tensor(As, dtype=torch.long),
            torch.from_numpy(np.stack(YL)), torch.from_numpy(np.stack(Ys)))


class ProbGRU(nn.Module):
    """Verbatim from src/actionsense/action_dynamics.py; only n_out is read from the data."""

    def __init__(self, din: int, n_act: int, hid: int, n_out: int = N_OUT, dropout: float = 0.0):
        super().__init__()
        self.emb = nn.Embedding(n_act, 8)
        self.enc = nn.GRU(din, hid, batch_first=True)
        self.dec = nn.GRU(n_out, hid, batch_first=True)
        self.mu = nn.Linear(hid + 8, n_out)
        self.lv = nn.Linear(hid + 8, n_out)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, aid, y_last, t_out):
        _, h = self.enc(x)
        e = self.emb(aid)
        inp = y_last.unsqueeze(1)
        mus, lvs = [], []
        for _ in range(t_out):
            o, h = self.dec(inp, h)
            oc = self.drop(torch.cat([o[:, -1], e], -1))
            mu = self.mu(oc)
            lv = self.lv(oc).clamp(-6, 4)
            mus.append(mu); lvs.append(lv)
            inp = mu.unsqueeze(1)               # autoregressive: feed the mean back
        return torch.stack(mus, 1), torch.stack(lvs, 1)


def nll(mu: torch.Tensor, lv: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Gaussian NLL up to a constant -- ActionSense's loss verbatim."""
    return (0.5 * (lv + (y - mu) ** 2 * torch.exp(-lv))).mean()


@torch.no_grad()
def _score(model, X, A, YL, Y, H, batch, dev) -> tuple[float, float]:
    model.eval()
    tot_nll, tot_mse, n = 0.0, 0.0, 0
    for k in range(0, len(X), batch):
        xb, ab = X[k:k + batch].to(dev), A[k:k + batch].to(dev)
        yl, yb = YL[k:k + batch].to(dev), Y[k:k + batch].to(dev)
        mu, lv = model(xb, ab, yl, H)
        b = len(xb)
        tot_nll += float(nll(mu, lv, yb)) * b
        tot_mse += float(((mu - yb) ** 2).mean()) * b
        n += b
    return (tot_nll / max(n, 1)), (tot_mse / max(n, 1))


def train(cfg: Config, fold: dict, hp: dict | None = None, device=None, verbose=True):
    """Train one arm on one LOSO fold. Returns (model, norm, fnorm, aids, history)."""
    hp = {**DEFAULT_HP, **(hp or {})}
    configure_determinism(hp["seed"])
    dev = pick_device(device)
    H = cfg.horizon
    with_df = hp["features"] == "raw+df"

    tr_ids, va_ids = fold["train"], fold["val"]
    norm = Norm.from_train({i: load_target(cfg, i) for i in tr_ids})
    fnorm = FeatNorm.from_train(cfg, tr_ids, with_df)
    aids, n_act = action_ids(cfg, hp["arm"], tr_ids + va_ids + fold["test"])

    Xtr, Atr, Ltr, Ytr = window_set(cfg, tr_ids, hp, norm, fnorm, aids)
    Xva, Ava, Lva, Yva = window_set(cfg, va_ids, hp, norm, fnorm, aids)
    if len(Xtr) == 0 or len(Xva) == 0:
        raise ValueError(f"fold {fold['fold']}: empty train/val windows "
                         f"({len(Xtr)}/{len(Xva)}) -- check eval.min_history")

    model = ProbGRU(feature_dim(hp), n_act, hp["hidden"], N_OUT, hp["dropout"]).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=hp["lr"])
    g = torch.Generator().manual_seed(hp["seed"])

    best, best_state, best_ep, since = float("inf"), None, -1, 0
    hist = {"epoch": [], "train_nll": [], "val_nll": [], "val_mse": []}
    t0 = time.time()
    for ep in range(1, hp["epochs"] + 1):
        model.train()
        perm = torch.randperm(len(Xtr), generator=g)
        for k in range(0, len(perm), hp["batch"]):
            j = perm[k:k + hp["batch"]]
            xb, ab = Xtr[j].to(dev), Atr[j].to(dev)
            yl, yb = Ltr[j].to(dev), Ytr[j].to(dev)
            opt.zero_grad()
            mu, lv = model(xb, ab, yl, H)
            loss = nll(mu, lv, yb)
            loss.backward()
            opt.step()

        v_nll, v_mse = _score(model, Xva, Ava, Lva, Yva, H, hp["batch"], dev)
        t_nll = float("nan")
        if ep % hp["log_train_every"] == 0 or ep == 1:
            t_nll, _ = _score(model, Xtr, Atr, Ltr, Ytr, H, hp["batch"], dev)
        hist["epoch"].append(ep); hist["train_nll"].append(t_nll)
        hist["val_nll"].append(v_nll); hist["val_mse"].append(v_mse)

        # Early stopping reads VAL NLL only -- the original ActionSense note is that this loss
        # overfits badly after ~epoch 10, and OpenTouch saw it from epoch 2.
        if v_nll < best - 1e-6:
            best, best_ep, since = v_nll, ep, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            since += 1
            if since >= hp["patience"]:
                if verbose:
                    print(f"    early stop at epoch {ep} (best {best_ep}, val NLL {best:.4f})")
                break
        if verbose and (ep % hp["log_train_every"] == 0 or ep == 1):
            print(f"    ep {ep:3d}  train NLL {t_nll:8.4f}  val NLL {v_nll:8.4f}  "
                  f"val MSE {v_mse:8.4f}", flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)
    hist["best_val_nll"] = best
    hist["best_epoch"] = best_ep
    hist["seconds"] = time.time() - t0
    hist["n_train_windows"] = int(len(Xtr))
    hist["n_val_windows"] = int(len(Xva))
    return model, norm, fnorm, aids, hist


@torch.no_grad()
def forecast(model, cfg: Config, ids: list[int], hp: dict, norm: Norm, fnorm: FeatNorm,
             aids: dict[int, int], device=None) -> dict[int, np.ndarray]:
    """Per-recording forecasts in RAW units, one row per harness origin, in harness order.

    Returned shape per recording is (n_origins, H, 6) -- exactly what
    `evaluate.score_external` asserts, so a mismatch fails there rather than being silently
    scored against different windows.
    """
    hp = {**DEFAULT_HP, **(hp or {})}
    dev = pick_device(device)
    model.eval()
    out = {}
    for i in ids:
        X, A, L, _ = window_set(cfg, [i], hp, norm, fnorm, aids)
        if len(X) == 0:
            continue
        mus = []
        for k in range(0, len(X), hp["batch"]):
            mu, _ = model(X[k:k + hp["batch"]].to(dev), A[k:k + hp["batch"]].to(dev),
                          L[k:k + hp["batch"]].to(dev), cfg.horizon)
            mus.append(mu.cpu().numpy())
        out[i] = norm.unz(np.concatenate(mus, 0).astype(np.float64))
    return out
