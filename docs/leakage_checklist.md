# Leakage checklist

Run **before every training experiment**:
```
python scripts/actionsense/check_leakage.py     # PASS/FAIL for each; non-zero exit if any fail
```
Each check guards against a way that information about the future or the test set could leak into
training/evaluation and inflate the reported skill. What each verifies and why:

### 1. The filter is causal (and so is the velocity)
The slow/fast decomposition must not use future samples. A **causal** filter's output at time *t*
depends only on inputs at times ≤ *t*. `filtfilt` (forward+backward) is **non-causal** — its
backward pass looks into the future, so the "fast" component (both model input *and* target) would
encode information the forecaster is supposed to predict. We use **`sosfilt`** (forward-only) and a
**backward-difference velocity**. *Check:* feed an impulse at `t0`; the filter/velocity response is
0 for all `t < t0` (a non-causal filter rings *before* the impulse).

### 2. Normalization statistics come from TRAIN only
Z-score `mean`/`std` must be computed on the training split, never the test split (otherwise test
statistics leak into the model's inputs). `Norm.from_clips` is fed only train clips; `evaluate`
reuses that norm. *Check:* train-only stats differ from train+test stats — so including test *would*
change them, confirming the exclusion matters and is enforced.

### 3. Train/test split is by trajectory (no clip in both)
If windows from the *same clip* landed in both train and test, the model would be tested on
near-duplicates of what it trained on. We split by **clip index**, then window each split
separately. *Check:* the train and test clip-index sets are disjoint.

### 4. Input windows are strictly before target windows
The model may only see the past. For each window the input frames `[s, s+t_in)` must be strictly
before the target frames `[s+t_in, s+t_in+t_out)`. *Check:* on a clip whose values encode the frame
index, `max(input index) < min(target index)` for every window (gap = 1).

### 5. The baseline sees the same past-only input as the model
For a fair skill score, persistence must use only the last **observed** value, never a future one.
`persistence = Yin[:, -1]` (last input-window target). *Check:* the persistence source index is
strictly before the first target index for every window.

### 6. Pipeline order: CoP/force → causal filter → z-score (train-only)
Total force and center-of-pressure are computed **per frame** from the raw pressure field (no
temporal mixing) *before* filtering; the causal high-pass is applied next; z-scoring happens last,
inside the model, with train-only stats — identically for train and test. *Check:* `build_features`
output is **not** z-scored (retains raw scale) and the fast target equals `raw − causal_lowpass(raw)`.

---
All six currently **PASS** after switching the filter to `sosfilt` and velocities to a causal
backward difference (2026-07-08). See `src/actionsense/action_dynamics.py`.
