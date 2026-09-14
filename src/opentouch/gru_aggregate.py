"""GRU-aggregate for OpenTouch: DETERMINISTIC point forecaster on the aggregate F/CoP signal.

SUPERSEDED 2026-08-19: OQ-G, which made this arm deterministic and deleted its Gaussian
head, was overturned globally at the user's direction -- every arm now emits a variance.
The probabilistic version of exactly this model is `aggregate` in src/opentouch/tactile_map.py
(same AggEncoder, same GRU, with the mu/logvar head restored), and that is what runs.

This file is NOT edited into the new behaviour: its git history is the pre-registration
record for the deterministic arm, and rewriting it would erase what was committed before
any number existed. It stays as the artifact, importable and tested; new work uses the
tactile_map family so that flatten, cnn and aggregate differ only in the encoder.


FORKED from the `aggregate` branch of src/actionsense/tactile_map/ (Q6, decided 2026-08-12:
"单独 fork deterministic GRU-aggregate;CNN-map / flatten-map 继续暂缓").

WHY A FORK EXISTS AT ALL. In ActionSense, "GRU-aggregate" is not a standalone script: it is the
`AggWindows` + `AggEncoder` branch living inside the tactile_map package, a sibling of the
CNN-map and flatten-map encoders that read the raw 16x16/32x32 pressure grid. The decision to
defer the map branch was made because OpenTouch does not need the raw map -- but GRU-aggregate
never reads the map either; it consumes the aggregate signal that is already in the cache as
state_N.npy. It was only swept up in the deferral because it is physically in the same files.
Extracting just this path keeps G1's central question answerable ("does AR > GRU > persistence
replicate on a second sensor?") without unblocking the map models. CNN-map / flatten-map remain
deferred and are deliberately absent here.

THREE DELIBERATE DIFFERENCES FROM THE ACTIONSENSE ORIGINAL
  1. DETERMINISTIC, not probabilistic (OQ-G, 2026-08-11: "GRU-aggregate 用点预测"). One head
     (the mean), MSE loss. The original's Gaussian NLL head + logvar + sigma calibration are
     gone -- not disabled, absent -- because a probabilistic head that is never scored
     probabilistically is unfalsifiable decoration, and MSE is the loss that matches how the
     harness actually scores (metrics.py).
  2. CHANNEL COUNT FROM DATA (3 here, 6 there). The original hardcodes 6 in the zero-pad; that
     is the same class of bug the other OpenTouch forks fixed.
  3. HARNESS SPLITS, NOT 5-FOLD CV. The original cross-validates by recording. This fork takes
     explicit train/val/test id lists and follows the frozen protocol: fit on TRAIN, select on
     VAL, touch TEST once. It therefore needs no splits.py to be importable or unit-testable --
     the caller supplies ids -- which is why it can be written while splits.py is still blocked.

CONVENTIONS KEPT IDENTICAL (so numbers stay comparable to ActionSense's four-way table)
  * Input  = normalized aggregate history, left-zero-padded for early origins, so a prediction
    exists at EVERY harness origin and `evaluate.score_external` alignment holds exactly.
  * Target = RESIDUAL OVER PERSISTENCE in normalized space: y = z(Y[t+1..t+H]) - z(Y[t]).
    Predicting 0 reproduces persistence, so the model's job is precisely the delta and a
    negative skill is immediately visible rather than hidden behind a large shared DC term.
  * Origins come from baselines.origins(), the same function the classical baselines use.

DETERMINISM (Q5's "seed 冻结" applied to training, not just to the bootstrap). `configure_determinism`
seeds torch, forces deterministic kernels, and shuffling draws from an explicit torch.Generator
rather than global RNG state, so two runs of the same commit with the same seed produce bitwise
identical weights on CPU. On CUDA, cuDNN's RNN kernels are not guaranteed deterministic even in
this mode; a GPU run must record that caveat instead of claiming bitwise reproducibility.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from src.actionsense.eval_harness.config import Config
from .baselines import origins
from .dataset import Norm, load_target

DEFAULT_HP = {"d": 64, "hidden": 64, "epochs": 60, "lr": 0.003, "batch": 64, "seed": 0}


def configure_determinism(seed: int) -> torch.Generator:
    """Seed every source of nondeterminism we control; return the generator to use for
    shuffling. Call once per training run, before the model is built."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    g = torch.Generator()
    g.manual_seed(seed)
    return g


# ------------------------------------------------------------------------------- data --
class AggWindows(Dataset):
    """Rolling-origin windows of the normalized aggregate signal.

    `sig_n[idx]` is (T,C) ALREADY normalized. Item k -> (X (t_in,C), Y (H,C)) where X is the
    causal history ending at the origin (left-zero-padded when the origin is early) and Y is the
    residual-over-persistence future. C is read from the data, not hardcoded (difference 2)."""

    def __init__(self, sig_n: dict[int, np.ndarray], cfg: Config, t_in: int):
        self.sig, self.t_in, self.H = sig_n, int(t_in), cfg.horizon
        self.index = [(i, int(t)) for i in sorted(sig_n) for t in origins(len(sig_n[i]), cfg)]

    def __len__(self) -> int:
        return len(self.index)

    def window(self, i: int, t: int) -> np.ndarray:
        S = self.sig[i]
        w = S[max(t - self.t_in + 1, 0): t + 1]
        if w.shape[0] < self.t_in:
            pad = np.zeros((self.t_in - w.shape[0], S.shape[1]), dtype=np.float32)
            w = np.concatenate([pad, w], 0)
        return w.astype(np.float32)

    def __getitem__(self, k: int):
        i, t = self.index[k]
        y = self.sig[i][t + 1: t + 1 + self.H] - self.sig[i][t]        # residual over persistence
        return torch.from_numpy(self.window(i, t)), torch.from_numpy(y.astype(np.float32))


def normalized_signals(cfg: Config, idxs: list[int], norm: Norm) -> dict[int, np.ndarray]:
    return {i: norm.z(load_target(cfg, i)) for i in idxs}


# ------------------------------------------------------------------------------ model --
class AggEncoder(nn.Module):
    """Per-frame projection of the aggregate C-dim F/CoP into the embedding space."""

    def __init__(self, d: int, n_in: int):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(n_in, d), nn.ReLU())

    def forward(self, x):                                   # (B,t_in,C) -> (B,t_in,d)
        return self.proj(x)


class Seq2SeqPoint(nn.Module):
    """encoder -> GRU over t_in frames -> one-shot POINT head -> (B,H,C) residual forecast.

    The deterministic counterpart of the ActionSense Seq2Seq: a single `head`, no logvar."""

    def __init__(self, encoder: nn.Module, d: int, hidden: int, horizon: int, n_out: int):
        super().__init__()
        self.encoder = encoder
        self.gru = nn.GRU(d, hidden, batch_first=True)
        self.head = nn.Linear(hidden, horizon * n_out)
        self.H, self.n_out = horizon, n_out

    def forward(self, x):
        _, h = self.gru(self.encoder(x))                    # h: (1,B,hidden)
        return self.head(h[-1]).reshape(-1, self.H, self.n_out)


def build_model(cfg: Config, hp: dict) -> Seq2SeqPoint:
    C = len(cfg.channels)
    return Seq2SeqPoint(AggEncoder(hp["d"], C), hp["d"], hp["hidden"], cfg.horizon, C)


# ---------------------------------------------------------------------------- training --
def _stack(ds: AggWindows) -> tuple[torch.Tensor, torch.Tensor]:
    """Materialize a whole dataset once. The aggregate model's inputs are (t_in, 3) -- tiny --
    so this is far cheaper than per-epoch DataLoader overhead, and it removes worker-order
    nondeterminism. (The map branch could not do this; it is deferred anyway.)"""
    items = [ds[k] for k in range(len(ds))]
    if not items:
        return torch.zeros(0), torch.zeros(0)
    return torch.stack([a for a, _ in items]), torch.stack([b for _, b in items])


@torch.no_grad()
def _mse(model: Seq2SeqPoint, X: torch.Tensor, Y: torch.Tensor, batch: int) -> float:
    model.eval()
    tot = n = 0.0
    for i in range(0, len(X), batch):
        pred = model(X[i:i + batch])
        tot += float(((pred - Y[i:i + batch]) ** 2).sum()); n += Y[i:i + batch].numel()
    return tot / max(n, 1.0)


def train(cfg: Config, train_ids: list[int], val_ids: list[int], t_in: int,
          hp: dict | None = None, norm: Norm | None = None
          ) -> tuple[Seq2SeqPoint, Norm, dict]:
    """Fit on TRAIN, keep the best-VAL-MSE weights. TEST is never touched here.

    `norm` MUST be the TRAIN-fitted Norm (pass the harness's own so the GRU and the classical
    baselines share one normalization); it is fitted from `train_ids` if not supplied.
    -> (model, norm, history) where history records per-epoch train/val MSE for the log.
    """
    from src.calibration import validate_training
    validate_training(cfg, train_ids, val_ids)
    hp = {**DEFAULT_HP, **(hp or {})}
    gen = configure_determinism(int(hp["seed"]))
    if norm is None:
        norm = Norm.from_train({i: load_target(cfg, i) for i in train_ids})

    Xtr, Ytr = _stack(AggWindows(normalized_signals(cfg, train_ids, norm), cfg, t_in))
    Xva, Yva = _stack(AggWindows(normalized_signals(cfg, val_ids, norm), cfg, t_in))
    if len(Xtr) == 0:
        raise ValueError("no TRAIN origins: clips too short for this history/horizon")

    model = build_model(cfg, hp)
    opt = torch.optim.Adam(model.parameters(), lr=hp["lr"])
    bs = int(hp["batch"])
    best, best_state, history = np.inf, None, {"train_mse": [], "val_mse": []}
    for _ in range(int(hp["epochs"])):
        model.train()
        perm = torch.randperm(len(Xtr), generator=gen)       # explicit generator: reproducible
        for i in range(0, len(Xtr), bs):
            b = perm[i:i + bs]
            loss = ((model(Xtr[b]) - Ytr[b]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        tr = _mse(model, Xtr, Ytr, bs)
        va = _mse(model, Xva, Yva, bs) if len(Xva) else tr   # no VAL -> select on TRAIN, logged
        history["train_mse"].append(tr); history["val_mse"].append(va)
        if va < best:
            best, best_state = va, {k: v.detach().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    history["best_val_mse"] = float(best)
    history["selected_on"] = "val" if len(Xva) else "train"
    return model, norm, history


def select_history(cfg: Config, train_ids: list[int], val_ids: list[int],
                   histories_s=(1.0, 2.0, 3.0), hp: dict | None = None):
    """Pick the input history length on VAL only (OQ-H: sweep {1,2,3} s; 0.5 s is the harness
    min_history floor, not a sweep point). -> (best_t_in, {t_in: best_val_mse})."""
    scores = {}
    for s in histories_s:
        t_in = max(1, int(round(s * cfg.fps)))
        _, _, hist = train(cfg, train_ids, val_ids, t_in, hp)
        scores[t_in] = hist["best_val_mse"]
    return min(scores, key=scores.get), scores


# -------------------------------------------------------------------------- prediction --
@torch.no_grad()
def predict_clip(model: Seq2SeqPoint, cfg: Config, norm: Norm, Y: np.ndarray,
                 t_in: int) -> np.ndarray:
    """One clip's RAW-unit forecasts, (n_origins, H, C), ordered exactly like
    baselines.origins(len(Y), cfg) -- the format `evaluate.score_external` requires.

    Undoes both conventions in one step: add the residual back onto the last observed
    normalized value (persistence anchor), then de-normalize."""
    model.eval()
    z = norm.z(np.asarray(Y, dtype=np.float64))
    ors = origins(len(z), cfg)
    C = len(cfg.channels)
    if len(ors) == 0:
        return np.zeros((0, cfg.horizon, C), dtype=np.float64)
    ds = AggWindows({0: z}, cfg, t_in)
    X = torch.from_numpy(np.stack([ds.window(0, int(t)) for t in ors]))
    resid = model(X).numpy()                                  # (n_origins,H,C) normalized delta
    z_hat = z[ors][:, None, :] + resid                        # anchor on the last observation
    return norm.unz(z_hat)


def predict(model: Seq2SeqPoint, cfg: Config, norm: Norm, test_ids: list[int],
            t_in: int) -> dict[int, np.ndarray]:
    """{clip_idx: (n_origins,H,C) raw-unit forecasts} for evaluate.score_external."""
    return {i: predict_clip(model, cfg, norm, load_target(cfg, i), t_in) for i in test_ids}


@dataclass(frozen=True)
class RunRecord:
    """What has to be logged next to a GRU-aggregate number for it to be reproducible."""
    t_in: int
    hp: dict
    history: dict
    torch_version: str
    numpy_version: str
    deterministic: bool

    @staticmethod
    def of(t_in: int, hp: dict, history: dict) -> "RunRecord":
        return RunRecord(t_in=int(t_in), hp=dict(hp), history=dict(history),
                         torch_version=torch.__version__, numpy_version=np.__version__,
                         deterministic=not torch.cuda.is_available())
