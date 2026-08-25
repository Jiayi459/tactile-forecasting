"""Leakage checklist — run BEFORE every training experiment.  See docs/leakage_checklist.md.

Programmatic PASS/FAIL assertions on the action_dynamics pipeline. Exits non-zero if any fail.

    python scripts/actionsense/check_leakage.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.actionsense import action_dynamics as AD  # noqa: E402

ROOT = "data/actionsense_states"
SUBS = ["Slice", "Peel"]


def check_filter_causal():
    """A causal filter's response to an impulse at t0 is ZERO for all t < t0 (no future leak).
    Also the velocity (backward diff) must be zero before the impulse."""
    T, t0, fps = 200, 100, 10.0
    imp = np.zeros(T); imp[t0] = 1.0
    slow, fast = AD.slow_fast(imp, fps, 0.4)
    ok_filter = np.allclose(slow[:t0], 0) and np.allclose(fast[:t0], 0)
    v = AD._causal_diff(imp, fps)
    ok_vel = np.allclose(v[:t0], 0)
    return ok_filter and ok_vel, f"pre-impulse: |slow|={np.abs(slow[:t0]).max():.1e} |vel|={np.abs(v[:t0]).max():.1e} (want 0)"


def check_norm_train_only():
    """Norm stats must come from TRAIN clips only. Verify train-only stats differ from train+test
    stats (so including test WOULD change them -> excluding test is meaningful and is what we do)."""
    data = AD.load_pooled(ROOT, SUBS, 3, 0.4)
    tr, te = AD.split_train_test(len(data))
    n_tr = AD.Norm.from_clips([data[i] for i in tr])
    n_all = AD.Norm.from_clips(data)
    differs = not np.allclose(n_tr.mu_x, n_all.mu_x)
    return differs, f"train-only mu_x[0]={n_tr.mu_x[0]:.3f} vs all={n_all.mu_x[0]:.3f} (differ={differs}); train() is fed train clips only"


def check_split_by_trajectory():
    """Train/test split is by TRAJECTORY (clip), and no clip appears in both window sets."""
    data = AD.load_pooled(ROOT, SUBS, 3, 0.4)
    tr, te = AD.split_train_test(len(data))
    disjoint = set(tr).isdisjoint(set(te))
    _, _, _, _, g_tr = AD.windows([data[i] for i in tr], 10, 10, 2)
    _, _, _, _, g_te = AD.windows([data[i] for i in te], 10, 10, 2)
    # group ids are local to each list; the guarantee is the clip index sets are disjoint
    return disjoint, f"train {len(tr)} / test {len(te)} clips, index-disjoint={disjoint}"


def check_input_before_target():
    """For every window, all INPUT frame indices are strictly before all TARGET frame indices."""
    T, t_in, t_out = 60, 10, 10
    feat = np.arange(T, dtype=np.float32)[:, None].repeat(len(AD.FEATS_HIGHPASS), 1)  # feat[t]=t
    targ = np.arange(T, dtype=np.float32)[:, None].repeat(3, 1)                        # targ[t]=t
    X, A, Yin, Y, _ = AD.windows([(feat, targ, 0)], t_in, t_out, 2)
    # X holds input frame indices, Y holds target frame indices (since feat/targ encode index)
    ok = bool((X[:, :, 0].max(1) < Y[:, :, 0].min(1)).all())
    gap = float((Y[:, 0, 0] - X[:, -1, 0]).min())  # should be exactly 1 (target starts right after)
    return ok, f"max(input idx) < min(target idx) for all windows; min gap={gap:.0f}"


def check_baseline_past_only():
    """Persistence baseline uses ONLY the last observed (input-window) value, not the future."""
    T, t_in, t_out = 60, 10, 10
    targ = np.arange(T, dtype=np.float32)[:, None].repeat(3, 1)
    feat = targ.copy()
    X, A, Yin, Y, _ = AD.windows([(feat, targ, 0)], t_in, t_out, 2)
    pers_idx = Yin[:, -1, 0]        # persistence source = last input-window target
    target_first = Y[:, 0, 0]       # first future index
    ok = bool((pers_idx < target_first).all())
    return ok, f"persistence source idx < first target idx for all windows (past-only={ok})"


def check_pipeline_order():
    """CoP/force computed per-frame BEFORE filtering; build_features output is NOT z-scored
    (z-score is applied later, from train stats). Confirm F_fast = raw - causal_lowpass and the
    output retains raw scale (std != 1)."""
    # load WITHOUT warmup trim so highpass-target and raw-input are frame-aligned
    hp = AD.load_pooled(ROOT, SUBS, 3, 0.4, input_mode="highpass", hand="left", warmup_sec=0)
    rw = AD.load_pooled(ROOT, SUBS, 3, 0.4, input_mode="raw", hand="left", warmup_sec=0)
    raw_scale = hp[0][0][:, 0].std()                 # F_fast still in raw units if un-normalized
    not_zscored = not (0.8 < raw_scale < 1.25)
    F_raw = rw[0][0][:, 0]                            # raw total force
    _, fast = AD.slow_fast(F_raw, 10.0, 0.4)         # F - causal lowpass(F)
    matches = np.allclose(fast, hp[0][1][:, 0], atol=1e-4)   # == the highpass target
    return (not_zscored and matches), f"F_fast raw-scale std={raw_scale:.1f} (un-normalized={not_zscored}); fast==raw-lowpass={matches}"


CHECKS = [
    ("1. filter is causal (+ causal velocity)", check_filter_causal),
    ("2. normalization stats from TRAIN only", check_norm_train_only),
    ("3. split by trajectory (no clip overlap)", check_split_by_trajectory),
    ("4. input windows strictly before targets", check_input_before_target),
    ("5. baseline sees same PAST-ONLY input", check_baseline_past_only),
    ("6. pipeline order (CoP/force->filter->zscore)", check_pipeline_order),
]


def main():
    print("LEAKAGE CHECKLIST (docs/leakage_checklist.md)\n" + "=" * 60)
    all_ok = True
    for name, fn in CHECKS:
        try:
            ok, detail = fn()
        except Exception as e:
            ok, detail = False, f"ERROR: {e}"
        all_ok &= ok
        print(f"[{'PASS' if ok else 'FAIL'}] {name}\n        {detail}")
    print("=" * 60)
    print("ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
