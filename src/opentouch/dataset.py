"""Load the RAW 3-dim right-hand-only target + manifest-driven grouping/filtering.

This is a NEW file, not a fork: OpenTouch's cache layout (state_N.npy is (T,1,6), one hand)
and manifest schema (structured JSON fields action/object_category/scene/environment/grip_type,
not ActionSense's free-text "Slice a cucumber" label string) are different enough from
ActionSense's that writing this directly is clearer than branching a shared function.
`Norm` and `force_thresholds` ARE reused unchanged from ActionSense (see imports below) --
they contain no hardcoded hand/channel count and no ActionSense-specific logic.

Target per clip: (T', 3) = [F_R, CoPx_R, CoPy_R], moments 0..2 of the single hand axis in
state_N.npy (shape (T, 1, 6): 1 hand, [F,CoPx,CoPy,sxx,syy,sxy]). Downsample by cfg.downsample
(currently 1: native 30 Hz, not upsampled).
"""
from __future__ import annotations

import collections
import json
import os

import numpy as np

from src.actionsense.eval_harness.config import Config
from src.actionsense.eval_harness.dataset import Norm, force_thresholds  # noqa: F401  (re-exported)

MOMENTS_PER_HAND = 3          # F, CoPx, CoPy (drop the shear moments sxx,syy,sxy)


def _manifest(cfg: Config) -> list[dict]:
    root = cfg.abspath("states_root")
    with open(os.path.join(root, "manifest.jsonl")) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_target(cfg: Config, idx: int) -> np.ndarray:
    """One clip -> (T', 3) raw target at the effective rate."""
    from src.calibration import require_prepared
    require_prepared(cfg)
    root = cfg.abspath("states_root")
    st = np.load(os.path.join(root, f"state_{idx}.npy"))     # (T, 1, 6)
    st = st[:: cfg.downsample]                                 # -> effective rate
    return st[:, 0, :MOMENTS_PER_HAND].astype(np.float64)      # (T', 3)


def load_group(cfg: Config, idxs: list[int]) -> dict[int, np.ndarray]:
    return {i: load_target(cfg, i) for i in idxs}


def category_counts(cfg: Config, field: str) -> dict[str, int]:
    """Corpus-wide count per raw manifest-field value, over EVERY clip regardless of
    which idx subset (train/val/test) is later queried. Computed once over the full
    manifest so the rare-category merge decision (see group_keys) is a stable property
    of the corpus, not of whichever subset happens to be passed to group_keys -- a
    category common in the full corpus must not look "rare" just because a small
    subset was queried.
    """
    return collections.Counter((r.get(field) or "").strip() for r in _manifest(cfg))


def group_keys(cfg: Config, idxs: list[int],
               train_idxs: list[int] | None = None) -> dict[int, str]:
    """Map each clip idx -> its AR fit group.

    'global' scope => one group "ALL". Otherwise cfg.fit_scope names a manifest field
    directly (e.g. "object_category", "action", "scene") -- no string parsing needed,
    unlike ActionSense's parse_label, because the manifest already carries these as
    separate JSON fields (see extract_opentouch.py). Missing/empty values fall into a
    named "unknown" bucket rather than crashing or silently merging into another group.

    RARE-CATEGORY MERGING (OQ-I, SESSION_LOG 2026-08-10): object_category is badly
    long-tailed (110 distinct values measured; 81 have n<30, several n=1). Any value
    with fewer than `baselines.min_group_size` clips is merged into a single "other"
    group instead of being fit alone, because a 1-3 clip group cannot support a stable
    AR fit.

    WHERE THAT COUNT COMES FROM (changed 2026-08-15, user decision). Passing
    `train_idxs` counts within TRAIN; omitting it counts corpus-wide, the original
    behaviour, kept so existing callers and tests are unaffected.

    Counting corpus-wide was meant to deliver OQ-I option (a) -- "a rare category being
    absent from one split's idxs no longer means AR sees a group it never fit" -- and it
    does when the split is fine-grained, because a common category then lands on both
    sides by sheer numbers. It fails exactly when the split is coarse enough to matter:
    objects belong to locations, so holding out a whole location takes its categories
    with it, and a category that is common corpus-wide can still be entirely absent from
    TRAIN. That is what raised KeyError('sports equipment') on 2026-08-13. Counting
    within TRAIN closes it at the definition: a category is its own group only if TRAIN
    can actually fit it, and everything else -- rare, or absent from TRAIN altogether --
    lands in "other", which TRAIN does fit.

    THE CATCH-ALL MUST ITSELF BE FITTABLE. Sending strangers to "other" only helps if
    TRAIN fits "other", and it might not: if every TRAIN category clears the threshold,
    no TRAIN clip is "other" at all, and a held-out category mapped there is as unfitted
    as before -- which is what the splits tests hit on the first run. So under
    TRAIN-relative counting the smallest qualifying categories are pooled into "other",
    smallest first, until it holds `min_group_size` TRAIN clips. It costs one or two
    categories their own AR fit; it buys the guarantee that every group AR is asked to
    score is one it fitted. missing_groups() remains the backstop assertion.
    """
    if cfg.fit_scope == "global":
        return {i: "ALL" for i in idxs}
    field = cfg.fit_scope
    min_size = cfg.raw["baselines"].get("min_group_size", 1)
    rows = {r["idx"]: r for r in _manifest(cfg)}
    demoted: set[str] = set()
    if train_idxs is None:
        counts = category_counts(cfg, field)
    else:
        counts = collections.Counter(
            (rows[i].get(field) or "").strip() for i in train_idxs)
        pooled = sum(n for v, n in counts.items() if n < min_size)
        # smallest first, so the fewest TRAIN clips lose their own fit
        for v, n in sorted(((v, n) for v, n in counts.items() if n >= min_size),
                           key=lambda vn: (vn[1], vn[0])):
            if pooled >= min_size:
                break
            demoted.add(v); pooled += n
    out = {}
    for i in idxs:
        v = (rows[i].get(field) or "").strip()
        if not v:
            out[i] = "unknown" if counts.get("", 0) >= min_size and "" not in demoted \
                else "other"
        elif counts.get(v, 0) < min_size or v in demoted:
            out[i] = "other"
        else:
            out[i] = v
    return out


def missing_groups(train_groups: dict[int, str], other_groups: dict[int, str]) -> set[str]:
    """Groups present in `other_groups` (val or test) that never appear in
    `train_groups`. Empty set = safe: AR.fit(train) will have coefficients for every
    group AR.predict() is later asked to score. Non-empty = AR.predict() will raise
    KeyError deep inside baselines/ar.py rather than failing at split-construction time
    with a clear message -- so splits.py MUST call this after building a split and
    assert the result is empty (OQ-I option (a), SESSION_LOG 2026-08-10). Deliberately
    axis-agnostic (doesn't know or care whether the split is by clip, scene, or
    participant) so it works unmodified once splits.py picks an axis.
    """
    return set(other_groups.values()) - set(train_groups.values())


def eligible_clips(cfg: Config, actions: tuple[str, ...] | None = None) -> list[dict]:
    """Manifest rows that (a) match the configured `actions` filter (empty = no filter)
    and (b) are long enough to yield >= 1 forecast origin at the configured rate/history/
    horizon. Mirrors src.actionsense.eval_harness.splits.eligible_recordings in spirit,
    but reads OpenTouch's manifest schema directly.

    This is a FILTER only -- it does not partition into train/val/test. That partition
    (splits.py) does not exist yet: see the "split" block in configs/opentouch/
    eval_harness.yaml and SESSION_LOG 2026-08-10 for why (p1/p2 semantics unresolved).
    """
    acts = tuple(a.lower() for a in (actions if actions is not None else cfg.raw.get("actions", [])))
    ds = cfg.downsample
    need = cfg.raw["eval"]["min_history"] + cfg.horizon
    out = []
    for r in _manifest(cfg):
        if acts and r.get("action", "").strip().lower() not in acts:
            continue
        if (r["T"] // ds) >= need:
            out.append(r)
    return out
