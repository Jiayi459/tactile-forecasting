"""Tactile-MAP input pipeline for F/CoP forecasting.

Per recording, the raw pressure map clip_<idx>.npy (T, 2, 32, 32) is turned into model input:
  1. downsample [::ds]                      (10 Hz; matches the harness target)
  2. causal per-taxel baseline (first N)    base = clip[:N].mean(0); x = clip - base; clip>=0
  3. log1p amplitude compression            (fixed; tames heavy-tailed peaks)
  4. global TRAIN scale (one mean/std over ALL taxels/hands/frames -> same scaling every taxel)

The TARGET is the harness's 6-dim F/CoP (eval_harness.dataset.load_target), z-normed per channel
on TRAIN (eval_harness.dataset.Norm). Windows/origins/split all come from the harness so exported
predictions align 1:1 with evaluate.py --model-preds.

CAUSALITY: the baseline uses only the first N frames (past); windows use only frames <= origin t.
Windows are sliced lazily (a 10 s history over all clips would be tens of GB if materialized).
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from ..eval_harness.config import Config
from ..eval_harness.dataset import Norm, load_target
from ..eval_harness.baselines.base import origins
from ..eval_harness.splits import parse_label

H_MOMENTS = 3          # F, CoP-x, CoP-y per hand (target has 6 = 2 hands x 3)
OTHER = 0              # reserved embedding id: rare-in-TRAIN or unseen-at-TEST verbs


def verbs_of(cfg: Config, idxs) -> dict[int, str]:
    """recording idx -> verb, from the manifest label ("Slice a cucumber" -> "slice")."""
    import json
    root = cfg.abspath("states_root")
    want = set(idxs)
    with open(os.path.join(root, "manifest.jsonl")) as f:
        rows = [json.loads(l) for l in f if l.strip()]
    return {r["idx"]: parse_label(r["label"])[0] for r in rows if r["idx"] in want}


def action_vocab(verbs: dict[int, str], train_idxs, min_count: int = 3):
    """-> (vocab, by_idx), built from TRAIN ONLY.

    A verb too rare in TRAIN, and any verb unseen there at test time, collapses into `other`
    (id 0). Building the vocabulary on anything but TRAIN would leak the test set's label
    distribution into the embedding table, which is the same discipline
    src/opentouch/prob_gru.py applies.
    """
    import collections
    n = collections.Counter(verbs[i] for i in train_idxs if i in verbs)
    keep = sorted(v for v, c in n.items() if c >= min_count)
    vocab = {"other": OTHER, **{v: k + 1 for k, v in enumerate(keep)}}
    return vocab, {i: verbs.get(i, "other") for i in verbs}


def aid_of(vocab: dict[str, int], by_idx: dict[int, str], i: int) -> int:
    return vocab.get(by_idx.get(i, "other"), OTHER)


def clip_path(cfg: Config, idx: int) -> str:
    return os.path.join(cfg.abspath("states_root"), f"clip_{idx}.npy")


def available_idxs(cfg: Config, idxs: list[int]) -> list[int]:
    """Subset of idxs whose raw map clip_<idx>.npy exists locally (for pre-restream smoke runs)."""
    return [i for i in idxs if os.path.exists(clip_path(cfg, i))]


def load_map(cfg: Config, idx: int, baseline_frames: int) -> np.ndarray:
    """clip_<idx>.npy (T,2,32,32) -> (T',2,32,32) float32: downsample + causal first-N baseline."""
    clip = np.load(clip_path(cfg, idx)).astype(np.float32)[:: cfg.downsample]   # (T',2,32,32)
    n = min(baseline_frames, len(clip))
    base = clip[:n].mean(0, keepdims=True)                                       # per-taxel, past-only
    return np.clip(clip - base, 0.0, None)


def compress(x: np.ndarray, alpha: float) -> np.ndarray:
    """log1p amplitude compression, normalized so compress(1/alpha)~O(1). Fixed (no train stats)."""
    return np.log1p(alpha * np.clip(x, 0.0, None)) / np.log1p(alpha)


@dataclass(frozen=True)
class MapNorm:
    """Global scalar normalization of the compressed map (same scaling for every taxel)."""
    mean: float
    std: float
    alpha: float

    @staticmethod
    def from_train(train_maps: dict[int, np.ndarray], alpha: float) -> "MapNorm":
        vals = np.concatenate([compress(m, alpha).reshape(-1) for m in train_maps.values()])
        return MapNorm(float(vals.mean()), float(vals.std() + 1e-6), alpha)

    def apply(self, m: np.ndarray) -> np.ndarray:
        return ((compress(m, self.alpha) - self.mean) / self.std).astype(np.float32)


def load_raw(cfg: Config, idxs: list[int], baseline_frames: int
             ) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """-> (maps: idx->(T',2,32,32) baseline-corrected raw, targets: idx->(T',6))."""
    maps = {i: load_map(cfg, i, baseline_frames) for i in idxs}
    tgts = {i: load_target(cfg, i) for i in idxs}
    # guard: map and target must share the time axis (both downsampled the same way)
    for i in idxs:
        n = min(len(maps[i]), len(tgts[i]))
        maps[i], tgts[i] = maps[i][:n], tgts[i][:n]
    return maps, tgts


def normalize(maps: dict[int, np.ndarray], mnorm: MapNorm) -> dict[int, np.ndarray]:
    return {i: mnorm.apply(m) for i, m in maps.items()}


class MapWindows(Dataset):
    """Lazy rolling-origin windows aligned to the harness origins.

    Returns (X (t_in,2,32,32) normalized map history, Y (H,6) normalized target future). For
    origins with < t_in frames of history, the window is LEFT-padded with zeros (post-baseline
    "no contact") -> a prediction exists at EVERY harness origin (score_external alignment)."""

    def __init__(self, maps_n: dict[int, np.ndarray], tgts_n: dict[int, np.ndarray],
                 cfg: Config, t_in: int, aids: dict[int, int] | None = None,
                 residual: bool = True):
        self.maps, self.tgts, self.t_in, self.H = maps_n, tgts_n, t_in, cfg.horizon
        self.aids, self.residual = aids or {}, residual
        self.index = [(i, int(t)) for i in sorted(maps_n) for t in origins(len(maps_n[i]), cfg)]

    def __len__(self):
        return len(self.index)

    def _window(self, i: int, t: int) -> np.ndarray:
        M = self.maps[i]
        win = M[max(t - self.t_in + 1, 0): t + 1]                # (<=t_in, 2,32,32)
        if win.shape[0] < self.t_in:                             # causal left-pad with zeros
            pad = np.zeros((self.t_in - win.shape[0],) + M.shape[1:], np.float32)
            win = np.concatenate([pad, win], 0)
        return win

    def __getitem__(self, k: int):
        i, t = self.index[k]
        x = self._window(i, t)
        last = self.tgts[i][t]
        # residual=True: the CHANGE from the last observed value. At worst the model predicts
        # 0 and matches persistence; its job is the delta. residual=False: the absolute
        # target, which is what the probGRU backbone predicts on both sensors.
        fut = self.tgts[i][t + 1: t + 1 + self.H]
        y = fut - last if self.residual else fut
        # FOUR items always, whichever backbone consumes them. An item whose length depends
        # on a flag is the shape that crashed an OpenTouch job after it had finished training
        # (SESSION_LOG 2026-08-19); Seq2Seq simply ignores the middle two.
        return (torch.from_numpy(x),
                torch.tensor(self.aids.get(i, OTHER), dtype=torch.long),
                torch.from_numpy(last.astype(np.float32)),
                torch.from_numpy(y.astype(np.float32)))


class AggWindows(Dataset):
    """Aggregate-F/CoP input: past 6-dim history -> residual future 6-dim (same signal, autoregressive
    -- the neural AR). `sig_n` maps idx -> (T,6) already-normalized F/CoP. Left-pads early origins with
    zeros; residual-over-persistence target, identical convention to MapWindows."""

    def __init__(self, sig_n: dict[int, np.ndarray], cfg: Config, t_in: int,
                 aids: dict[int, int] | None = None, residual: bool = True):
        self.sig, self.t_in, self.H = sig_n, t_in, cfg.horizon
        self.aids, self.residual = aids or {}, residual
        self.index = [(i, int(t)) for i in sorted(sig_n) for t in origins(len(sig_n[i]), cfg)]

    def __len__(self):
        return len(self.index)

    def _window(self, i: int, t: int) -> np.ndarray:
        S = self.sig[i]
        w = S[max(t - self.t_in + 1, 0): t + 1]
        if w.shape[0] < self.t_in:
            w = np.concatenate([np.zeros((self.t_in - w.shape[0], 6), np.float32), w], 0)
        return w

    def __getitem__(self, k: int):
        i, t = self.index[k]
        x = self._window(i, t)
        last = self.sig[i][t]
        fut = self.sig[i][t + 1: t + 1 + self.H]
        y = fut - last if self.residual else fut
        return (torch.from_numpy(x.astype(np.float32)),
                torch.tensor(self.aids.get(i, OTHER), dtype=torch.long),
                torch.from_numpy(last.astype(np.float32)),
                torch.from_numpy(y.astype(np.float32)))


def recording_windows(map_n: np.ndarray, cfg: Config, t_in: int) -> tuple[np.ndarray, np.ndarray]:
    """For export: all (n_origins, t_in, 2,32,32) windows of one recording + the origin indices."""
    ors = origins(len(map_n), cfg)
    ds_ = MapWindows({0: map_n}, {0: np.zeros((len(map_n), 6), np.float32)}, cfg, t_in)
    X = np.stack([ds_._window(0, int(t)) for t in ors]) if len(ors) else np.zeros((0, t_in, 2, 32, 32), np.float32)
    return X, ors
