"""Load the RAW 6-dim both-hands target + global TRAIN normalization stats.

Target per recording: (T', 6) = [F_L, CoPx_L, CoPy_L, F_R, CoPx_R, CoPy_R], where the 6
columns are the physical moments straight from state_N.npy (shape (T, 2, 6): 2 hands,
[F,CoPx,CoPy,sxx,syy,sxy]). We take moments 0..2 of each hand. Downsample by cfg.downsample
to the effective rate. NO high-pass, NO warmup cut (this is the raw signal).

NORMALIZATION (constraint 3): global, dataset-level, TRAIN-split-derived per-channel
(mean, std). Never per-frame / per-window. Fit on TRAIN recordings only; applied verbatim
to VAL/TEST. AR fitting uses the normalized signal; metrics are reported in raw units.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np

from .config import Config
from .splits import parse_label

HANDS = 2
MOMENTS_PER_HAND = 3          # F, CoPx, CoPy (drop the shear moments sxx,syy,sxy)


def group_keys(cfg: Config, idxs: list[int]) -> dict[int, str]:
    """Map each recording idx -> its fit group. 'group' scope => 'action-object'
    (e.g. 'slice-cucumber'); 'global' scope => 'ALL'. Subject is unavailable (no manifest
    field), so activity x object is the finest grouping we can form (see SESSION_LOG Step 0)."""
    if cfg.fit_scope == "global":
        return {i: "ALL" for i in idxs}
    root = cfg.abspath("states_root")
    lab = {}
    with open(os.path.join(root, "manifest.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            lab[r["idx"]] = r["label"]
    a_o = {i: parse_label(lab[i]) for i in idxs}
    return {i: f"{a}-{o}" for i, (a, o) in a_o.items()}


def load_target(cfg: Config, idx: int) -> np.ndarray:
    """One recording -> (T', 6) raw target at the effective rate."""
    from src.calibration import require_prepared
    require_prepared(cfg)
    root = cfg.abspath("states_root")
    st = np.load(os.path.join(root, f"state_{idx}.npy"))     # (T, 2, 6)
    st = st[:: cfg.downsample]                                # -> effective rate
    left = st[:, 0, :MOMENTS_PER_HAND]
    right = st[:, 1, :MOMENTS_PER_HAND]
    return np.concatenate([left, right], axis=1).astype(np.float64)   # (T', 6)


def load_group(cfg: Config, idxs: list[int]) -> dict[int, np.ndarray]:
    return {i: load_target(cfg, i) for i in idxs}


@dataclass(frozen=True)
class Norm:
    """Global per-channel z-score from TRAIN only."""
    mean: np.ndarray   # (6,)
    std: np.ndarray    # (6,)

    @staticmethod
    def from_train(train: dict[int, np.ndarray]) -> "Norm":
        allx = np.concatenate(list(train.values()), axis=0)   # (sum T, 6)
        mean = allx.mean(0)
        std = allx.std(0)
        std[std < 1e-8] = 1.0
        return Norm(mean=mean, std=std)

    def z(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def unz(self, z: np.ndarray) -> np.ndarray:
        return z * self.std + self.mean


def force_thresholds(cfg: Config, train: dict[int, np.ndarray]) -> np.ndarray:
    """Per-hand low-contact threshold = TRAIN `percentile`th pct of that hand's raw total
    force. Returns array indexed like the force channels (cfg.force_idx order)."""
    pct = cfg.raw["mask"]["percentile"]
    allx = np.concatenate(list(train.values()), axis=0)       # (sum T, 6)
    return np.array([np.percentile(allx[:, fi], pct) for fi in cfg.force_idx])
