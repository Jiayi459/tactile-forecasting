"""Load d256's RAW 6-dim both-hands target + manifest-driven grouping.

d256's cache is shape-identical to ActionSense's -- `state_N.npy` is `(T, 2, 6)`, two hands x
`[F, CoPx, CoPy, sxx, syy, sxy]` -- because `scripts/d256/extract_d256_states.py` feeds the two
32x32 gloves through the same `physical_state.frame_state`. So `load_target`, `Norm` and
`force_thresholds` are REUSED verbatim from the ActionSense harness rather than forked; they
contain no ActionSense-specific logic. Only the two things that genuinely differ live here:

  * `group_keys` reads `label_idx` straight from the manifest. ActionSense has to parse
    'Slice a cucumber' into (action, object) because its manifest carries only a label string;
    d256's session directory name IS the class, so the field is already there.
  * `eligible_recordings` filters on the rebuilt recordings, which are SEGMENTS, not cells --
    see the extractor's docstring for why a cell can hold several.

Target per recording: `(T', 6)` = `[F_L, CoPx_L, CoPy_L, F_R, CoPx_R, CoPy_R]`, downsampled by
`cfg.downsample`. NO high-pass, NO warmup cut, NO baseline correction (OQ-D2).
"""
from __future__ import annotations

import collections
import json
import os

from src.actionsense.eval_harness.config import Config
from src.actionsense.eval_harness.dataset import (  # noqa: F401  (re-exported)
    HANDS, MOMENTS_PER_HAND, Norm, force_thresholds, load_group, load_target,
)


def manifest(cfg: Config) -> list[dict]:
    root = cfg.abspath("states_root")
    path = os.path.join(root, "manifest.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found -- run scripts/d256/extract_d256_states.py first")
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def rows_by_idx(cfg: Config) -> dict[int, dict]:
    return {r["idx"]: r for r in manifest(cfg)}


def group_keys(cfg: Config, idxs: list[int],
               train_idxs: list[int] | None = None) -> dict[int, str]:
    """Map recording idx -> AR fit group.

    'global' scope => one group. Otherwise `cfg.fit_scope` names a manifest field directly
    (normally `label_idx`). Rare groups are pooled into "other" exactly as OpenTouch does:
    a group with fewer than `min_group_size` recordings cannot support a stable AR fit.

    Counting is TRAIN-relative when `train_idxs` is given, and that is the case that matters
    here. Under LOSO a whole subject is held out, so a class can be plentiful corpus-wide and
    still be thin -- or absent -- in one fold's TRAIN. Counting corpus-wide would then hand AR
    a group it never fitted (the KeyError OpenTouch hit on 2026-08-13). The catch-all must
    itself be fittable, so if every TRAIN group clears the threshold the smallest are pooled
    into "other" until it holds `min_group_size` recordings.
    """
    if cfg.fit_scope == "global":
        return {i: "ALL" for i in idxs}
    field = cfg.fit_scope
    min_size = cfg.raw["baselines"].get("min_group_size", 1)
    rows = rows_by_idx(cfg)
    key = lambda i: str(rows[i].get(field, "")).strip()   # noqa: E731

    demoted: set[str] = set()
    if train_idxs is None:
        counts = collections.Counter(str(r.get(field, "")).strip() for r in manifest(cfg))
    else:
        counts = collections.Counter(key(i) for i in train_idxs)
        pooled = sum(n for v, n in counts.items() if n < min_size)
        for v, n in sorted(((v, n) for v, n in counts.items() if n >= min_size),
                           key=lambda vn: (vn[1], vn[0])):
            if pooled >= min_size:
                break
            demoted.add(v)
            pooled += n

    out = {}
    for i in idxs:
        v = key(i)
        if not v or counts.get(v, 0) < min_size or v in demoted:
            out[i] = "other"
        else:
            out[i] = v
    return out


def missing_groups(train_groups: dict[int, str], other_groups: dict[int, str]) -> set[str]:
    """Groups in val/test that TRAIN never fitted. Empty = safe. splits.py asserts on this so
    the failure surfaces at split construction rather than as a KeyError inside baselines/ar.py.
    """
    return set(other_groups.values()) - set(train_groups.values())


def eligible_recordings(cfg: Config) -> list[dict]:
    """Manifest rows long enough to yield >= 1 forecast origin at the configured rate.

    A FILTER only -- partitioning is splits.py. `T` in the manifest is the rebuilt SEGMENT
    length; a cell that held several recordings contributes one row per segment.
    """
    ds = cfg.downsample
    need = cfg.raw["eval"]["min_history"] + cfg.horizon
    return [r for r in manifest(cfg) if (r["T"] // ds) >= need]


def budget_table(cfg: Config, candidates=(12, 18, 24, 30, 40)) -> str:
    """How many recordings and origins survive each min_history choice.

    `eval.min_history` is provisional in the frozen config: it is set to 4 s to match the
    physical history ActionSense uses, but whether that leaves a usable corpus depends on the
    segment-length distribution, which is not known until the extractor has run. Print this
    before trusting the setting.
    """
    ds, H = cfg.downsample, cfg.horizon
    lens = sorted((r["T"] // ds) for r in manifest(cfg))
    if not lens:
        return "  (no recordings)"
    out = [f"  {len(lens)} recordings, T/ds: min {lens[0]}  median {lens[len(lens)//2]}  "
           f"max {lens[-1]}   horizon {H}",
           f"  {'min_history':>12s} {'need':>5s} {'kept':>12s} {'origins':>9s}"]
    for mh in candidates:
        need = mh + H
        kept = [L for L in lens if L >= need]
        origins = sum(L - need + 1 for L in kept)
        flag = "  <- config" if mh == cfg.raw["eval"]["min_history"] else ""
        out.append(f"  {mh:12d} {need:5d} {len(kept):6d}/{len(lens):<5d} {origins:9d}{flag}")
    return "\n".join(out)
