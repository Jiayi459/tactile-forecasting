"""probGRU for OpenTouch — the ActionSense probabilistic GRU, architecture and loss verbatim.

WHY THIS EXISTS ALONGSIDE gru_aggregate.py. OQ-G (2026-08-11) made GRU-aggregate a
deterministic point forecaster and deleted the Gaussian head. The user overturned that on
2026-08-13: train OpenTouch with the SAME probGRU as ActionSense -- same architecture, same
loss. gru_aggregate.py is left untouched (it is the pre-registered deterministic arm); this
is a second, separate arm.

WHAT IS COPIED VERBATIM FROM src/actionsense/action_dynamics.py
  * ProbGRU: action embedding (8) -> encoder GRU -> AUTOREGRESSIVE decoder GRU seeded with the
    last observed target, mean and log-variance heads over [decoder state ; action embedding],
    logvar clamped to [-6, 4], the predicted mean fed back as the next decoder input.
  * The loss: 0.5 * (lv + (y - mu)^2 * exp(-lv)), i.e. Gaussian NLL up to a constant.
  * Early stopping on VAL NLL (the original notes the loss overfits badly after ~epoch 10).
  * Its hyperparameters: hidden 48, epochs 80, lr 3e-3, batch 64 (NOT gru_aggregate's 64/60 --
    those came from the tactile_map aggregate branch, a different model).

WHAT DIFFERS, AND WHY (each one is forced, not a preference)
  1. TARGET = RAW [F, CoPx, CoPy] (user decision 2026-08-13), not ActionSense's FAST. The
     frozen harness scores RAW, so this arm can be compared with persistence/seasonal/AR on
     one scale; a FAST-target model is not harness-scorable (SESSION_LOG 2026-08-06 note).
     Consequence to keep in mind when reading any number from it: F currently carries a large
     DC offset (D1 open), so much of the target is a constant.
  2. INPUT = all-frequency history [F, CoPx, CoPy, vx, vy] -- ActionSense's `raw` input mode,
     with the same CAUSAL backward-difference velocities. No low-pass anywhere in this file.
  3. NO WARMUP CUT. ActionSense drops the first 5 s of every clip solely because its causal
     filter has a startup transient. There is no filter here, and OpenTouch's median clip is
     2.80 s (84 frames), so a 5 s cut would empty the corpus rather than clean it.
  4. WINDOWS COME FROM THE HARNESS, not from ActionSense's stride-2 sampler: origins() defines
     both training and scoring windows, so what the model is trained on and what
     evaluate.score_external() reads are the same rolling origins. Stride is a sampling
     detail, not part of the architecture or the loss.
  5. TARGETS ARE NORMALIZED BY THE HARNESS'S TRAIN-FITTED Norm (the one the baselines use), so
     the GRU and the classical baselines share one normalization. Input features get their own
     TRAIN-fitted z-score because they include velocities, which the harness Norm does not
     cover -- this mirrors ActionSense's separate nx/ny normalizers.

ACTION VOCABULARY. The embedding needs an id per clip. OpenTouch's action field is long-tailed
(~50 values), so ids are built FROM TRAIN ONLY: any action with fewer than
`baselines.min_group_size` TRAIN clips collapses into "other" (id 0), which is also where an
action unseen in TRAIN lands at test time. The threshold is the same one the AR baselines use
to merge rare object categories, so the two arms treat rarity the same way.
"""
from __future__ import annotations

import collections
import time

import numpy as np
import torch
import torch.nn as nn

from src.actionsense.eval_harness.config import Config
from .baselines import origins
from .dataset import Norm, eligible_clips, load_target
from .tactile_map import GRID, CNNEncoder, FlattenEncoder
from .tactile_map import build_inputs as build_map_inputs
from .tactile_map import scope_ids
from .gru_aggregate import configure_determinism

DEFAULT_HP = {"hidden": 48, "epochs": 80, "lr": 0.003, "batch": 64, "seed": 0,
              # Train NLL is a curve for the log, not a selection signal -- early stopping
              # reads VAL only -- so it does not need a full extra pass over TRAIN every
              # epoch. Evaluating it every 5th epoch keeps the curve readable and drops
              # roughly a fifth of the wall clock (an epoch is one fwd+bwd pass over TRAIN
              # plus one fwd over TRAIN plus one fwd over VAL; this removes 4/5 of the
              # middle term). Set to 1 to restore the old behaviour.
              "log_train_every": 5,
              # "raw" = ActionSense's five inputs verbatim; "raw+df" adds dF/dt as an
              # ablation. Recorded in the checkpoint, so a forecast can never be replayed
              # under the wrong feature set.
              "features": "raw",
              # Both OFF by default, so the arm stays input- and architecture-identical to
              # ActionSense unless a run says otherwise. They exist because the 2026-08-17
              # curves overfit from epoch 2 -- five times sooner than ActionSense's own note
              # -- and the first thing to try is the one with a mechanism behind it
              # (features="raw+df" removes the per-location DC level the model can memorise),
              # not a pile of regularisers applied at once.
              "weight_decay": 0.0, "dropout": 0.0,
              # WHICH VAL CURVE PICKS THE WEIGHTS. The 2026-08-20 D1 curves showed the two
              # disagreeing on real data, not just in principle: fold0's VAL MSE was still
              # falling at epoch 8 while its VAL NLL had been climbing since epoch 1, and
              # the min-NLL and min-MSE epochs landed in different places in every fold. So
              # "nll" hands the harness -- which scores point error and nothing else --
              # weights chosen by a criterion it does not measure.
              #
              # The default stays "nll" so every number already in the log remains
              # reproducible; "mse" is the one to run when the report is the point. Both
              # state dicts are kept either way (a 48-unit GRU is a few hundred KB, so
              # keeping the loser costs nothing) and both epochs are recorded, which is what
              # makes the disagreement visible instead of merely possible.
              "select_on": "nll",
              # WHICH INPUT REPRESENTATION FEEDS THE SAME BACKBONE. "raw" is ActionSense's
              # five aggregate channels, verbatim; "flatten" and "cnn" hand the 16x16 map to
              # the per-frame encoders from tactile_map.py and feed their embeddings to this
              # file's encoder GRU instead.
              #
              # The tactile_map family already varies the encoder, but behind ITS OWN
              # backbone -- one-shot head, residual target, no action embedding -- so
              # comparing one of its arms against probGRU moves the input and the
              # architecture together. Here the architecture, the loss, the autoregressive
              # decoder and the action embedding are all held fixed and the input is the only
              # thing that changes, which is the comparison the map runs were for.
              "input": "raw", "d": 64, "alpha": 10.0}
OTHER = 0          # reserved embedding id: rare-in-TRAIN or unseen-at-TEST actions


def pick_device(spec: str | None = None) -> torch.device:
    """'cuda' when a GPU is actually present, else CPU. Note the determinism caveat that
    gru_aggregate.configure_determinism documents: cuDNN's RNN kernels are not guaranteed
    deterministic even under deterministic algorithms, so a CUDA run cannot claim bitwise
    reproducibility the way a CPU run can -- it must record that it ran on GPU."""
    if spec:
        return torch.device(spec)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------------------------- data --
def causal_velocity(sig: np.ndarray, fps: float) -> np.ndarray:
    """v[t] = (sig[t] - sig[t-1]) * fps, v[0] = 0 -- ActionSense's _causal_diff. Backward,
    so no feature at time t sees a sample after t."""
    v = np.zeros_like(sig)
    v[1:] = np.diff(sig, axis=0) * fps
    return v


def features(Y: np.ndarray, fps: float, with_df: bool = False) -> np.ndarray:
    """(T,3) raw [F,CoPx,CoPy] -> (T,5) [F, CoPx, CoPy, vx, vy], ActionSense's `raw` mode.

    WITH_DF APPENDS dF/dt, AS AN ABLATION. ActionSense differences CoP and not force --
    FEATS_RAW is ("F","x","y","vx","vy") -- and no reason for the asymmetry is recorded;
    in its highpass mode force is already split into F_slow and F_fast, so the raw mode
    reads like an omission rather than a decision.

    It matters more here than it did there. With D1 declined the force channel is ~99.3%
    constant, and a difference is the one view of F that carries no DC at all: it removes
    the per-clip and per-session resting level that the model would otherwise have to
    cancel inside its hidden state first. The decoder is already seeded with the last
    observed target, so the LEVEL is supplied there; handing the encoder the RATE separates
    the two cleanly. Appended last so the first three columns stay the raw channels.
    """
    Y = np.asarray(Y, dtype=np.float64)
    v = causal_velocity(Y[:, 1:3], fps)
    cols = [Y, v]
    if with_df:
        cols.append(causal_velocity(Y[:, 0:1], fps))
    return np.concatenate(cols, axis=1)


class FeatNorm:
    """TRAIN-only z-score for the input features (the harness Norm covers 3 channels).

    Carries `with_df` so the feature layout travels with its normalizer: window_set and
    predict read it off this object instead of taking a parallel flag that could drift out
    of step with the statistics it was fitted against."""

    def __init__(self, mean, std, with_df: bool = False):
        self.mean, self.std, self.with_df = mean, std, with_df

    @staticmethod
    def from_train(cfg: Config, ids: list[int], with_df: bool = False) -> "FeatNorm":
        allx = np.concatenate(
            [features(load_target(cfg, i), cfg.fps, with_df) for i in ids], 0)
        sd = allx.std(0)
        sd[sd < 1e-8] = 1.0
        return FeatNorm(allx.mean(0), sd, with_df)

    def z(self, x):
        return (x - self.mean) / self.std


def action_vocab(cfg: Config, train_ids: list[int]) -> tuple[dict[str, int], dict[int, str]]:
    """-> (action -> embedding id, clip idx -> action). Built from TRAIN only: letting VAL/TEST
    actions define the vocabulary would leak the split into the model's input space."""
    by_idx = {r["idx"]: (r.get("action") or "").strip().lower()
              for r in eligible_clips(cfg, actions=())}
    counts = collections.Counter(by_idx.get(i, "") for i in train_ids)
    min_n = cfg.raw["baselines"].get("min_group_size", 1)
    vocab = {"other": OTHER}
    for a, n in sorted(counts.items()):
        if a and n >= min_n:
            vocab[a] = len(vocab)
    return vocab, by_idx


def _aid(vocab, by_idx, i: int) -> int:
    return vocab.get(by_idx.get(i, ""), OTHER)


def map_inputs(cfg: Config, hp: dict, ids: list[int], base_ids: list[int], norm: Norm,
               mnorm=None):
    """Normalized maps for `ids`, or (None, None) for the raw-aggregate arm.

    Delegates to tactile_map so the baseline subtraction, the log1p compression and the map
    scaling are the SAME code the map family uses -- a second implementation of D1's
    per-taxel median is the last thing this needs."""
    if str(hp.get("input", "raw")) == "raw":
        return None, None
    return build_map_inputs(cfg, "flatten", ids, base_ids, norm, float(hp["alpha"]), mnorm)


def frame_encoder_for(hp: dict) -> "nn.Module | None":
    """The per-frame encoder named by hp["input"], or None for the verbatim raw path."""
    kind = str(hp.get("input", "raw"))
    if kind == "raw":
        return None
    if kind not in ("flatten", "cnn"):
        raise ValueError(f"input must be raw/flatten/cnn, got {kind!r}")
    return (FlattenEncoder if kind == "flatten" else CNNEncoder)(int(hp["d"]))


def window_set(cfg: Config, ids: list[int], t_in: int, norm: Norm, fnorm: FeatNorm,
               vocab: dict[str, int], by_idx: dict[int, str],
               maps: dict[int, np.ndarray] | None = None):
    """Harness-aligned windows -> (X, A (N,), Ylast (N,3), Y (N,H,3)) normalized.

    X is (N,t_in,5) from the aggregate features, or (N,t_in,1,16,16) when `maps` supplies
    already-normalized tactile maps -- the target, the origins, the padding and the action id
    are identical either way, so the input is the only thing that differs.

    One item per rolling origin, exactly the origins score_external() will ask about. Early
    origins are LEFT-zero-padded rather than dropped, matching gru_aggregate."""
    H = cfg.horizon
    Xs, As, YL, Ys = [], [], [], []
    for i in ids:
        Y = load_target(cfg, i)
        if maps is None:
            f = fnorm.z(features(Y, cfg.fps, fnorm.with_df)).astype(np.float32)
        else:
            if i not in maps:
                continue
            f = np.asarray(maps[i], dtype=np.float32)
        z = norm.z(np.asarray(Y, dtype=np.float64)).astype(np.float32)
        a = _aid(vocab, by_idx, i)
        for t in origins(len(z), cfg):
            w = f[max(t - t_in + 1, 0): t + 1]
            if w.shape[0] < t_in:
                pad = np.zeros((t_in - w.shape[0],) + f.shape[1:], np.float32)
                w = np.concatenate([pad, w], 0)
            Xs.append(w); As.append(a); YL.append(z[t]); Ys.append(z[t + 1: t + 1 + H])
    if not Xs:
        shape = (0, t_in) + ((5,) if maps is None else (1, GRID, GRID))
        return (torch.zeros(*shape), torch.zeros(0, dtype=torch.long),
                torch.zeros(0, 3), torch.zeros(0, H, 3))
    return (torch.from_numpy(np.stack(Xs)), torch.tensor(As, dtype=torch.long),
            torch.from_numpy(np.stack(YL)), torch.from_numpy(np.stack(Ys)))


# ------------------------------------------------------------------------------ model --
class ProbGRU(nn.Module):
    """Verbatim from src/actionsense/action_dynamics.py (only `n_out` is read from the data
    instead of hardcoded 3 -- the same channel-count fix the other OpenTouch forks made)."""

    def __init__(self, din: int, n_act: int, hid: int, n_out: int = 3, dropout: float = 0.0,
                 frame_encoder: nn.Module | None = None):
        super().__init__()
        # Applied per frame before the encoder GRU, so a map arm differs from the raw arm in
        # this module ALONE. None keeps the verbatim ActionSense path.
        self.frame_encoder = frame_encoder
        self.emb = nn.Embedding(n_act, 8)
        self.enc = nn.GRU(din, hid, batch_first=True)
        self.dec = nn.GRU(n_out, hid, batch_first=True)
        self.mu = nn.Linear(hid + 8, n_out)
        self.lv = nn.Linear(hid + 8, n_out)
        # Identity at p=0, so the verbatim architecture is what runs unless asked otherwise.
        self.drop = nn.Dropout(dropout)

    def forward(self, x, aid, y_last, t_out):
        if self.frame_encoder is not None:
            x = self.frame_encoder(x)            # (B,t_in,1,16,16) -> (B,t_in,d)
        _, h = self.enc(x)
        e = self.emb(aid)
        inp = y_last.unsqueeze(1)
        mus, lvs = [], []
        for _ in range(t_out):
            o, h = self.dec(inp, h)
            oc = self.drop(torch.cat([o[:, -1], e], -1))
            mu = self.mu(oc); lv = self.lv(oc).clamp(-6, 4)
            mus.append(mu); lvs.append(lv)
            inp = mu.unsqueeze(1)                       # autoregressive: feed the mean back
        return torch.stack(mus, 1), torch.stack(lvs, 1)


def nll(mu: torch.Tensor, lv: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Gaussian NLL up to a constant -- ActionSense's loss, unchanged."""
    return 0.5 * (lv + (y - mu) ** 2 * torch.exp(-lv)).mean()


@torch.no_grad()
def _val_scores(m, X, A, YL, Y, H, batch, dev) -> tuple[float, float]:
    """(mean NLL, mean squared error of the MEAN).

    Both, because they can move in opposite directions and the difference is diagnostic.
    The Gaussian NLL contains (y-mu)^2 * exp(-lv): a model can lower it on TRAIN by
    shrinking sigma where it happens to be right, and that same shrinkage makes the term
    explode wherever it is wrong on VAL. So a VAL NLL that climbs while VAL MSE stays flat
    is not the mean getting worse -- it is the variance head getting overconfident, which
    calls for a different fix and, more urgently, means early stopping on NLL is selecting
    weights by a criterion the harness does not score (it scores point error only)."""
    m.eval()
    tot_nll = tot_se = n = cnt = 0.0
    for i in range(0, len(X), batch):
        yb = Y[i:i + batch].to(dev)
        mu, lv = m(X[i:i + batch].to(dev), A[i:i + batch].to(dev),
                   YL[i:i + batch].to(dev), H)
        tot_nll += float(nll(mu, lv, yb)) * len(mu); n += len(mu)
        tot_se += float(((mu - yb) ** 2).sum()); cnt += yb.numel()
    return tot_nll / max(n, 1.0), tot_se / max(cnt, 1.0)


# ---------------------------------------------------------------------------- training --
def train(cfg: Config, train_ids: list[int], val_ids: list[int], t_in: int,
          hp: dict | None = None, norm: Norm | None = None, verbose: bool = True,
          device: str | None = None, base_scope: str = "shard"):
    """Fit on TRAIN, keep the weights that minimise the VAL curve named by hp["select_on"]
    (NLL by default, MSE when the harness metric is what matters). TEST is never touched.

    `verbose` prints the window count up front and one line per epoch, flushed. The
    autoregressive decoder runs `horizon` sequential GRU steps per batch, so a CPU run is
    slow enough that a silent loop is indistinguishable from a hung one.

    -> (model, norm, fnorm, vocab, by_idx, history)"""
    hp = {**DEFAULT_HP, **(hp or {})}
    gen = configure_determinism(int(hp["seed"]))
    if norm is None:
        norm = Norm.from_train({i: load_target(cfg, i) for i in train_ids})
    fnorm = FeatNorm.from_train(cfg, train_ids, "df" in str(hp.get("features", "raw")))
    vocab, by_idx = action_vocab(cfg, train_ids)
    # The map scale is fitted on TRAIN and reused for VAL, exactly as FeatNorm is; the taxel
    # baseline may see the whole shard, an input property (see tactile_map.scope_ids).
    base_ids = scope_ids(cfg, train_ids, val_ids, base_scope)
    mtr, mnorm = map_inputs(cfg, hp, train_ids, base_ids, norm)
    mva, _ = map_inputs(cfg, hp, val_ids, base_ids, norm, mnorm)

    Xtr, Atr, Ltr, Ytr = window_set(cfg, train_ids, t_in, norm, fnorm, vocab, by_idx, mtr)
    Xva, Ava, Lva, Yva = window_set(cfg, val_ids, t_in, norm, fnorm, vocab, by_idx, mva)
    if len(Xtr) == 0:
        raise ValueError("no TRAIN origins: clips too short for this history/horizon")

    H, bs = cfg.horizon, int(hp["batch"])
    n_ep = int(hp["epochs"])
    dev = pick_device(device)
    if verbose:
        print(f"    [{hp.get('input', 'raw')} t_in={t_in}] windows: "
              f"train {len(Xtr)} val {len(Xva)} | "
              f"batches/epoch {-(-len(Xtr) // bs)} | vocab {len(vocab)} | device {dev}",
              flush=True)
    fe = frame_encoder_for(hp)
    din = int(hp["d"]) if fe is not None else int(Xtr.shape[-1])
    m = ProbGRU(din, len(vocab), int(hp["hidden"]), n_out=len(cfg.channels),
                dropout=float(hp["dropout"]), frame_encoder=fe).to(dev)
    # The map scaling is part of what the model needs to be replayed, and there is nowhere
    # else to put it without changing this function's return arity -- which is the shape that
    # cost a whole job on 2026-08-19.
    m.input_norm = mnorm
    opt = torch.optim.Adam(m.parameters(), lr=hp["lr"],
                           weight_decay=float(hp["weight_decay"]))
    sel = str(hp.get("select_on", "nll")).lower()
    if sel not in ("nll", "mse"):
        raise ValueError(f"select_on must be 'nll' or 'mse', got {sel!r}")
    # Both are tracked whichever one selects, so the checkpoint can always answer "and what
    # would the other criterion have picked?" without a second training run.
    best_nll, best_mse = np.inf, np.inf
    state_nll = state_mse = None
    history = {"train_nll": [], "val_nll": [], "val_mse": []}
    t0 = time.time()
    for ep in range(n_ep):
        m.train()
        perm = torch.randperm(len(Xtr), generator=gen)
        for i in range(0, len(Xtr), bs):
            b = perm[i:i + bs]
            mu, lv = m(Xtr[b].to(dev), Atr[b].to(dev), Ltr[b].to(dev), H)
            loss = nll(mu, lv, Ytr[b].to(dev))
            opt.zero_grad(); loss.backward(); opt.step()
        every = max(1, int(hp["log_train_every"]))
        want_tr = (ep % every == 0) or (ep == n_ep - 1)
        tr = (_val_scores(m, Xtr, Atr, Ltr, Ytr, H, bs, dev)[0] if want_tr
              else float("nan"))
        if len(Xva):
            va, va_mse = _val_scores(m, Xva, Ava, Lva, Yva, H, bs, dev)
        else:
            va, va_mse = (tr if want_tr else float("nan")), float("nan")
        history["train_nll"].append(tr); history["val_nll"].append(va)
        history["val_mse"].append(va_mse)
        improved = (va_mse < best_mse) if sel == "mse" else (va < best_nll)
        if verbose:
            el = time.time() - t0
            trs = f"{tr:.5f}" if np.isfinite(tr) else "  --   "
            print(f"    [t_in={t_in}] epoch {ep + 1}/{n_ep} train {trs} val {va:.5f} "
                  f"mse {va_mse:.5f}"
                  f"{'  *best' if improved else ''} | {el:.0f}s elapsed, "
                  f"~{el / (ep + 1) * (n_ep - ep - 1):.0f}s left", flush=True)
        snap = None
        if va < best_nll:
            best_nll = va
            snap = state_nll = {k: v.detach().clone() for k, v in m.state_dict().items()}
        if np.isfinite(va_mse) and va_mse < best_mse:
            best_mse = va_mse
            state_mse = snap or {k: v.detach().clone() for k, v in m.state_dict().items()}
    # Fall back rather than fail when MSE selection was asked for but VAL never produced a
    # finite one (no VAL windows at all): the run still yields a model, and history records
    # which criterion actually chose it.
    chosen = state_mse if (sel == "mse" and state_mse is not None) else state_nll
    if sel == "mse" and state_mse is None:
        sel = "nll (mse requested, but VAL MSE was never finite)"
    if chosen is not None:
        m.load_state_dict(chosen)
    m.eval()
    history["selected_on_metric"] = sel
    history["best_val_nll"] = float(best_nll)
    vm = np.asarray(history["val_mse"], dtype=float)
    if np.isfinite(vm).any():
        history["best_val_mse"] = float(np.nanmin(vm))
        history["best_val_mse_epoch"] = int(np.nanargmin(vm)) + 1
        history["best_val_nll_epoch"] = int(np.nanargmin(
            np.asarray(history["val_nll"], dtype=float))) + 1
    history["selected_on"] = "val" if len(Xva) else "train"
    history["n_actions"] = len(vocab)
    history["device"] = str(dev)
    history["features"] = str(hp.get("features", "raw"))
    history["n_features"] = int(Xtr.shape[-1])
    history["input"] = str(hp.get("input", "raw"))
    history["baseline_scope"] = base_scope
    history["weight_decay"] = float(hp["weight_decay"])
    history["dropout"] = float(hp["dropout"])
    return m, norm, fnorm, vocab, by_idx, history


def select_history(cfg: Config, train_ids: list[int], val_ids: list[int],
                   histories_s=(1.0, 2.0, 3.0), hp: dict | None = None,
                   device: str | None = None, keep: bool = False):
    """Input history chosen on VAL only. -> (best_t_in, {t_in: score}, kept).

    Scored by the SAME criterion that picks the weights (hp["select_on"]), because choosing
    the input length by one curve and the weights by another would be two different
    definitions of "best" inside one run.

    THE ARITY IS FIXED. `kept` is {} unless keep=True, rather than the return being two
    values or three depending on a flag -- that shape crashed the 2026-08-19 diagnostic run
    after it had finished the whole sweep, because the caller unpacked three while a run
    without --save-preds produced two. A return whose length depends on an argument is a
    trap for exactly this.

    WHY `keep` EXISTS. The sweep trains one model per history length and, until now, threw
    every one of them away, returning a single scalar each; the chosen length was then
    trained a second time from scratch. That made a figure like ActionSense's
    plot_forecast_overlay -- whose rows ARE the history lengths -- impossible to draw
    without paying for the training twice. With keep=True the trained models come back, so
    their forecasts cost one prediction pass instead of one training run."""
    hp = {**DEFAULT_HP, **(hp or {})}      # `hp` is optional, so read it only once merged
    scores, kept = {}, {}
    for s in histories_s:
        t_in = max(1, int(round(s * cfg.fps)))
        print(f"  sweep: history {s} s -> t_in={t_in} frames", flush=True)
        out = train(cfg, train_ids, val_ids, t_in, hp, device=device)
        key = "best_val_mse" if str(hp["select_on"]).lower() == "mse" else "best_val_nll"
        scores[t_in] = out[-1].get(key, out[-1]["best_val_nll"])
        if keep:
            kept[t_in] = out
    return min(scores, key=scores.get), scores, kept


# -------------------------------------------------------------------------- prediction --
def _test_maps(model, cfg: Config, hp: dict, ids: list[int], norm: Norm,
               base_ids: list[int] | None):
    """Maps for TEST clips, scaled by the MapNorm the model was trained with.

    Refitting the scale on TEST would be a leak; `model.input_norm` carries the TRAIN-fitted
    one, which is why train() attaches it."""
    if str(hp.get("input", "raw")) == "raw":
        return None
    mn = getattr(model, "input_norm", None)
    if mn is None:
        raise RuntimeError("model has no input_norm: it was not trained on maps")
    return build_map_inputs(cfg, "flatten", ids,
                            base_ids if base_ids is not None else ids,
                            norm, mn.alpha, mn)[0]


@torch.no_grad()
def predict_clip(model, cfg: Config, norm: Norm, fnorm: FeatNorm, vocab, by_idx,
                 i: int, t_in: int, maps: dict | None = None) -> np.ndarray:
    """One clip's RAW-unit mean forecasts (n_origins, H, C), ordered like origins() -- the
    format evaluate.score_external() requires. The variance head is trained (it is in the
    loss) but not returned here: the frozen harness scores point error only."""
    model.eval()
    X, A, L, _ = window_set(cfg, [i], t_in, norm, fnorm, vocab, by_idx, maps)
    if len(X) == 0:
        return np.zeros((0, cfg.horizon, len(cfg.channels)), dtype=np.float64)
    dev = next(model.parameters()).device
    mu, _ = model(X.to(dev), A.to(dev), L.to(dev), cfg.horizon)
    return norm.unz(mu.cpu().numpy().astype(np.float64))


def predict(model, cfg: Config, norm: Norm, fnorm: FeatNorm, vocab, by_idx,
            test_ids: list[int], t_in: int, hp: dict | None = None,
            base_ids: list[int] | None = None) -> dict[int, np.ndarray]:
    maps = _test_maps(model, cfg, hp or {}, test_ids, norm, base_ids)
    return {i: predict_clip(model, cfg, norm, fnorm, vocab, by_idx, i, t_in,
                            None if maps is None else {i: maps[i]} if i in maps else {})
            for i in test_ids}


@torch.no_grad()
def predict_with_sigma(model, cfg: Config, norm: Norm, fnorm: FeatNorm, vocab, by_idx,
                       i: int, t_in: int,
                       maps: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(mu, sigma) in RAW units, both (n_origins, H, C).

    The harness scores the mean only, so predict() drops the variance head; a forecast
    plot is the one place it should be visible, since a probabilistic model whose spread
    is never shown is indistinguishable from a point model. z-scoring is per channel and
    linear, so sigma_raw = exp(lv/2) * norm.std -- the mean shift cancels.
    """
    model.eval()
    X, A, L, _ = window_set(cfg, [i], t_in, norm, fnorm, vocab, by_idx, maps)
    C = len(cfg.channels)
    if len(X) == 0:
        z = np.zeros((0, cfg.horizon, C))
        return z, z
    dev = next(model.parameters()).device
    mu, lv = model(X.to(dev), A.to(dev), L.to(dev), cfg.horizon)
    mu = mu.cpu().numpy().astype(np.float64)
    sd = np.exp(0.5 * lv.cpu().numpy().astype(np.float64)) * np.asarray(norm.std)
    return norm.unz(mu), sd
