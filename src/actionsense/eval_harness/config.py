"""Config loading + hashing for the frozen harness.

The YAML at configs/actionsense/eval_harness.yaml is the SINGLE source of truth. `config_hash` is a
sha256 over the exact file bytes; it is stamped into the results table so any row is
traceable to the config that produced it. Changing the config changes the hash.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Any

import yaml

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DEFAULT_CONFIG = os.path.join(REPO_ROOT, "configs", "actionsense", "eval_harness.yaml")


@dataclass(frozen=True)
class Config:
    raw: dict[str, Any]
    path: str
    config_hash: str

    # -- convenience accessors (derived; never a second source of truth) --
    @property
    def fps(self) -> float:
        return self.raw["rate"]["fps_raw"] / self.raw["rate"]["downsample"]

    @property
    def downsample(self) -> int:
        return int(self.raw["rate"]["downsample"])

    @property
    def horizon(self) -> int:
        """Horizon in STEPS (1 s * fps). At 10 Hz => 10."""
        return int(round(self.raw["rate"]["horizon_s"] * self.fps))

    @property
    def origin_horizon(self) -> int:
        """Steps of future an origin must have -- ORIGIN ELIGIBILITY only, never the forecast
        length. Defaults to `horizon`, so every existing config is unchanged. A horizon ablation
        sets `eval.origin_horizon_s` to its longest horizon, which keeps every horizon on the
        same origin set: the forecast length is then the only variable (SESSION_LOG 2026-09-14)."""
        s = self.raw.get("eval", {}).get("origin_horizon_s")
        if s is None:
            return self.horizon
        steps = int(round(float(s) * self.fps))
        if steps < self.horizon:
            raise ValueError(f"eval.origin_horizon_s={s} is {steps} steps, shorter than the "
                             f"{self.horizon}-step horizon: origins would lack a full target")
        return steps

    @property
    def force_idx(self) -> list[int]:
        return list(self.raw["target"]["force_idx"])

    @property
    def cop_idx(self) -> list[int]:
        return list(self.raw["target"]["cop_idx"])

    @property
    def channels(self) -> list[str]:
        return list(self.raw["target"]["channels"])

    @property
    def fit_scope(self) -> str:
        return self.raw["baselines"]["fit_scope"]

    @property
    def seasonal_range(self) -> tuple[int, int]:
        """Seasonal period search range in FRAMES (from the config's seconds range)."""
        b = self.raw["baselines"]
        return (max(2, int(round(b["seasonal_period_min_s"] * self.fps))),
                int(round(b["seasonal_period_max_s"] * self.fps)))

    def abspath(self, key: str) -> str:
        p = self.raw["paths"][key]
        return p if os.path.isabs(p) else os.path.join(REPO_ROOT, p)


def load_config(path: str = DEFAULT_CONFIG) -> Config:
    with open(path, "rb") as f:
        data = f.read()
    h = hashlib.sha256(data).hexdigest()[:16]
    return Config(raw=yaml.safe_load(data), path=path, config_hash=h)
