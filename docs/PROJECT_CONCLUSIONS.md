# PROJECT CONCLUSIONS — TouchAnything tactile forecasting (Sessions 1 → 2026-07-24)

Consolidated conclusion of everything established so far, written from a full read of
[SESSION_LOG.md](../SESSION_LOG.md) (1,927 lines) and cross-checked against the committed
result tables and source code. Every quantitative claim below was re-verified against the CSV
artifacts in `docs/` at the time of writing (2026-08-05); where a number in the log differs
slightly from the artifact, both are given.

This document **concludes**; it does not replace the running log. Companions:
[STUDY_SUMMARY.md](STUDY_SUMMARY.md) (predictability study),
[RESULTS.md](RESULTS.md) (EgoTouch pixel forecaster),
[ACTION_CATEGORIES.md](ACTION_CATEGORIES.md) (taxonomy),
[REPO_ORGANIZATION.md](REPO_ORGANIZATION.md) (file-by-file map).

---

## 0. What the project became (one paragraph)

The work started as "categorize EgoTouch hand actions and forecast grasp success", and converged
—through four datasets and five modelling phases— onto a single, sharply-defined question:
**how much of the next 1 second of a hand's contact force is predictable from its own tactile
past, which kinds of action are predictable, and what is the honest size of that predictability
once baselines, leakage and overfitting are controlled?** The final experimental object is a
1-second forecast of a 6-dimensional physical state — total force `F` and centre-of-pressure
`(CoP_x, CoP_y)` for each hand — during two kitchen actions (Slice, Peel) recorded with
ActionSense conductive-thread gloves.

---

## 1. Project phases (chronological spine)

| # | Phase | Data | Question | Verdict |
|---|---|---|---|---|
| A | Action categorization | EgoTouch | What actions exist? | 212 tasks / 1,930 traj → 23 verb categories |
| B | Pixel forecaster | EgoTouch | Is a future pressure *map* predictable? | Yes on seen objects; only with pretraining on unseen |
| C | Cross-dataset predictability probe | EgoTouch + OpenTouch + ActionSense (+ Force-Vision from paper) | *Which* actions are predictable and *why*? | A sensor-independent **trait**, not a category |
| D | Physical-state forecaster (v1 → v2) | ActionSense | Predict `F`/`CoP` instead of pixels | v1 failed; v2 (slow/fast + probabilistic) worked, then was corrected downward |
| E | Frozen harness + four-way comparison | ActionSense | What honestly beats what? | **Linear AR wins**; map < aggregate |

---

## 2. Phase A — Action categorization (EgoTouch)

- **Method**: rule-based verb taxonomy — each task is named `verb_object`, so the first known verb
  token assigns the category. Single source of truth is `VERB_CATEGORY` / `categorize()` in
  [src/tactile_pixel/categories.py](../src/tactile_pixel/categories.py); driver is
  [scripts/egotouch/categorize_actions.py](../scripts/egotouch/categorize_actions.py).
- **Result**: **212 real tasks, 1,930 trajectories, 23 categories.** Core grasp-suitable set =
  `Grasp/Hold/Lift` (8 tasks, 82 traj) + `Pick-up` (64 tasks, 635 traj) ≈ 37 % of all trajectories.
- **Critical negative finding**: **the dataset contains no grasp success/failure label.**
  `manual_contact_annotation.json` carries only coarse per-trajectory `left_contact`/`right_contact`
  booleans (True in ~5–6 % of a 120-sample), which is not a usable success signal. The originally
  requested "forecast grasp success" was therefore *not* answerable on this data and was formally
  deferred by the user — this is why the project pivoted to forecasting the tactile signal itself.
- Later extension: `categorize_phrase()` adds gerund stemming (`pulling`→`pull`, `picking up`→`pick`)
  so OpenTouch/ActionSense free-text labels land in the same category space.

## 3. Phase B — Tactile-pixel forecaster (EgoTouch)

Full write-up: [docs/RESULTS.md](RESULTS.md). Code: [src/tactile_pixel/](../src/tactile_pixel/).

- **Setup**: 10 frames (0.33 s) in → 15 frames (0.5 s) out @30 fps, bimanual 2×21×21 grids
  (217 valid taxels/hand after the fixed structural NaN sensor mask), masked loss, headline metric
  **skill = 1 − MSE_model / MSE_persistence**. 82 grasp trajectories, 8 objects.
- **The single most important implementation lesson of this phase**: predicting *absolute* frames
  makes copying-the-last-frame the loss optimum, so the first run scored ≈ 0.004 skill (a tie with
  persistence). Switching to **residual prediction** (`pred = clamp(last + Δ, 0, 1)`) made
  persistence equal to "Δ = 0" and immediately produced +0.174 skill. Residual parameterization
  recurs as the fix in Phase E as well.
- **Cross-validated results** (skill vs persistence, mean ± std):

  | Model | Protocol | Pretrain | Skill |
  |---|---|---|--:|
  | ConvGRU | LTO 5-fold | — | +0.138 ± 0.056 |
  | ConvLSTM | LTO 5-fold | — | +0.152 ± 0.031 |
  | **SimVP** | LTO 5-fold | — | **+0.192 ± 0.044** |
  | SimVP | LOTO 8-fold | — | +0.005 ± 0.111 |
  | **SimVP** | LOTO 8-fold | full, grasp-excluded (~1,851 traj) | **+0.097 ± 0.122** |

- **Three-step conclusion**: (1) future tactile *is* predictable beyond persistence for a seen
  object (+0.192, ~19 % error reduction over 0.5 s); (2) learned from only 8 objects it does **not**
  generalize — leave-one-task-out skill collapses to ≈ 0, i.e. the dynamics are object-specific;
  (3) broad multi-object pretraining (1,851 non-grasp trajectories, held-out object never seen)
  restores generalization to **+0.097**, ~18× the from-scratch LOTO result.
- **Model-family conclusion**: the non-recurrent **SimVP beats both recurrent models at every
  horizon** and trains far faster; this overturned the earlier a-priori preference for ConvGRU.
- **Scope caveat**: TAU was named in the plan but **never implemented** — only SimVP-lite,
  ConvGRU, ConvLSTM exist ([src/tactile_pixel/](../src/tactile_pixel/)). This was flagged, not
  quietly papered over.
- EgoTouch was later **deprecated** by user decision (it is not the target glove hardware); this
  phase stands as a completed, self-contained result.

## 4. Phase C — What makes an action predictable (three sensors)

Method: a **training-free predictability probe** — no GPU, no fitting — computing per clip
`persistence_nMSE@h` (how fast the signal decorrelates), `periodicity` (max total-force
autocorrelation at lag 0.33–1.5 s), `contact_migration` (1 − IoU of the active-taxel mask), and a
z-scored composite `PI = z(−persH15) + z(periodicity) + z(−migration)`.
Shared implementation: [src/tactile_pixel/predictability.py](../src/tactile_pixel/predictability.py);
per-dataset drivers `scripts/probe_{egotouch,opentouch,actionsense}.py`.

- **EgoTouch (1,929 clips, full run)**: easiest **Cut/slice +6.02** ≫ Take +2.90 > Inflate +2.62 >
  Spray +2.46 > Wash/Clean +2.37; hardest **Press/Click −6.67** < Plug/Insert −4.86 < Pinch −4.17
  < Fold −2.34 < Push/Pull −2.28 < Squeeze −2.11 < Grasp/Hold/Lift −2.08. Ranking is robust
  (a 12-per-task subsample, n = 1,493, reproduces it almost exactly).
- **OpenTouch (2,496 usable of 2,958 clips)**: easiest pouring +4.4 / serving +3.6 / eating +3.4 /
  stirring +3.0 / scooping +2.5; hardest cutting(n=4) −3.0 / moving −2.6 / turning −2.2 / pulling −1.8.
  `contact_migration ≈ 0.005` for every category — a single-hand grasp footprint never breaks, so
  the metric is **degenerate on this sensor**.
- **ActionSense (299 clips, S00–S05)**: Pour +2.6 > Cut (slice/peel) +1.8 > Wash/Clean +0.1 >
  Fold/Cloth −0.5 > Organize −1.4 > Open/Close jar −2.6.
- **The durable answer (the project's most portable finding)**: predictability is **not** a property
  of the action *category* — category rankings disagree across datasets — it is a property of the
  **force trace**. Predictable = *smooth, continuous, slowly-varying contact force*
  (pour, slice, wipe, peel, stir, scoop). Unpredictable = *abrupt onset / make-or-break contact*
  (press, click, plug, jar open, stiff turn). **`persistence_nMSE@15` is the sensor-agnostic
  predictor** of this trait.
- **Refutation of a naive prior**: sustained holds are *not* trivially predictable. `Grasp/Hold/Lift`
  sits below the EgoTouch median (worst `persH30` = 1.223) because grips drift and the contact
  footprint is unstable — which retro-explains why the Phase-B grasp-only LOTO skill was ≈ 0.
- **Refutation of the a-priori temporal-pattern axis**: the hand-assigned Axis B (B1 periodic …
  B5 composite) works on ActionSense but **inverts on OpenTouch** (B4 > B2 > B1), because the same
  verb means different mechanics in context ("turning a stiff latch" is a transition, not a rhythm)
  and unmapped verbs pool into "Other". Lesson recorded: **assign the temporal pattern from measured
  periodicity, not from the verb.**
- Force-Vision (ICLR'24, STAG glove, press/hold/squeeze on 89 tools) was **categorized from the
  paper only** — never downloaded or probed. It contributes taxonomy, not numbers.

## 5. Phase D — Physical-state forecasting (ActionSense)

### 5.1 The representation
Instead of pixels, reduce each frame to interpretable **pressure moments** per hand:
`[F, CoP_x, CoP_y, s_xx, s_yy, s_xy]`, plus derived area/orientation/velocity/`dF/dt` and Hilbert
phase — [src/actionsense/physical_state.py](../src/actionsense/physical_state.py). Chosen by the
user over a learned VAE latent, because feedback to a human needs *named* physical variables.

### 5.2 Two bugs that would have invalidated everything
- **P4 — untared glove DC offset.** The conductive-thread glove is not zeroed: every taxel has a
  large resting value (~571/taxel), so total force read ≈ 585,000 ± 0.5 % and CoP was pinned to the
  sensor centre. The first extracted state dataset was **completely degenerate — no motion visible**.
  Fixed by per-taxel 5th-percentile-over-time baseline subtraction
  ([physical_state.py:68](../src/actionsense/physical_state.py#L68)); after the fix, pour force
  genuinely ramps 950 → 9,800 and slice CoP genuinely oscillates. This also retro-explains the
  suspiciously small `persH` values in the ActionSense probe (a constant baseline inflates variance).
- **P5 — resampling artifact.** ActionSense is natively ~6 Hz but was upsampled to 30 Hz to match
  the other datasets, making adjacent frames near-duplicates and persistence artificially unbeatable
  at short horizons. Fixed by forecasting at the native rate (downsample ×3 → 10 Hz).

### 5.3 v1 (raw state → GRU): a productive failure
A GRU seq2seq on the raw physical state scored **≈ persistence (mean skill ≈ −0.1)** with high
seed-to-seed variance ([src/actionsense/state_forecast.py](../src/actionsense/state_forecast.py)).
- **Conclusion, not a bug**: the very smoothness that makes pour/slice "predictable" in absolute
  terms means *persistence already predicts them*, leaving no skill-over-persistence headroom.
  This is the same mechanism as the EgoTouch grasp LOTO ≈ 0 result — the deepest recurring insight
  of the project (logged as problem **P3**).

### 5.4 v2 (slow/fast split + probabilistic): the fix, then the correction
Split each channel into a slow (grip/postural) and fast (action) component; predict the **fast**
component with a probabilistic GRU (mean + log-variance, Gaussian NLL), 5-fold CV by clip —
[src/actionsense/action_dynamics.py](../src/actionsense/action_dynamics.py).
- **First result: +0.725 mean skill, coverage@2σ 0.93.** Pooling four actions gave the same as
  Slice+Peel alone (+0.736), so the *representation*, not extra data, produced the gain.
- **Then it was corrected downward twice, on our own initiative:**
  1. **Causality fix.** `filtfilt` is non-causal — its backward pass lets the "fast" target see the
     future. Replaced with `sosfilt` (forward-only) plus causal backward-difference velocity and a
     5 s warm-up trim ([action_dynamics.py:39-51](../src/actionsense/action_dynamics.py#L39-L51)).
     **Skill fell +0.70 → ≈ +0.40: the leak had inflated the headline by ~0.3.**
  2. **Overfitting fix.** `train()` ran 80 epochs and returned the *final* model with no early
     stopping, while validation loss bottomed at ~epoch 10 (`docs/actionsense/fcop_loss_curve.png`). Adding
     best-validation checkpointing lifted every configuration by **+0.10 … +0.23**.
- **Final early-stopped results** (`docs/actionsense/action_dynamics_results_earlystop.csv`, mean over the 10
  forecast steps, 5-fold CV, calibrated coverage ≈ 0.94–0.95):

  | input / hand | 1 s | 2 s | 3 s | 5 s | 10 s |
  |---|--:|--:|--:|--:|--:|
  | highpass / right | 0.513 | 0.515 | **0.519** | 0.512 | 0.502 |
  | raw / right | 0.513 | 0.515 | 0.508 | 0.508 | 0.495 |
  | highpass / left | 0.467 | 0.468 | 0.466 | 0.470 | 0.458 |
  | raw / left | 0.458 | 0.457 | 0.458 | 0.453 | 0.449 |

- **Ablations settled**:
  - **raw ≈ highpass input** (0.513 vs 0.513 right, 0.458 vs 0.467 left) → the explicit slow/fast
    *input* decomposition buys nothing; only the *target* needs to be the fast component.
  - **right hand > left** by ~0.05 for Slice/Peel (right = dominant/tool hand); right-hand `CoP_x`
    (the knife-stroke direction) is consistently the most predictable channel.
  - **"more history hurts" was an overfitting artifact.** Pre-early-stopping, skill fell monotonically
    1 s → 10 s; after early stopping it is **flat** across history (0.513 → 0.502), and the gain from
    early stopping is *largest at 10 s* — exactly where overfitting was worst. A published-looking
    finding was retracted by fixing the training loop.
- **Calibration**: raw coverage@2σ was overconfident (~0.76–0.86); post-hoc σ-scaling
  (`calibrate_sigma`, fit on a validation slice, target 0.95) lifts it to **0.94–0.95 across the
  whole matrix with skill unchanged** — calibration rescales the band, not the mean.

### 5.5 The metric audit — the most important methodological result
Prompted by the user's question "is F prediction too good?", a full review
([scripts/actionsense/tmp_diag_predictability.py](../scripts/actionsense/tmp_diag_predictability.py)) found **no leakage**,
but found that **the baseline was too weak**:
- The fast target is **anti-correlated with itself at 1 s lag** (ρ ≈ −0.14 … −0.19), so
  `MSE_persistence ≈ 2.4 × var` and *trivial* predictors score high:
  predicting **constant zero** scores +0.56…+0.59 at 1.0 s; damped persistence +0.57…+0.61;
  a linear ridge +0.62 — **all ≥ the GRU**. The GRU genuinely beats both trivial baselines only in
  the **0.2–0.5 s** band.
- Consolidated on one scale (raw/right/3 s, normalized fast target, same split):

  | model | test MSE | skill vs persistence | **variance explained (honest R²)** |
  |---|--:|--:|--:|
  | probGRU early-stopped | 0.737 | +0.55 | **+0.25** |
  | linear AR on fast (p=20) | 0.763 | +0.53 | **+0.22** |
  | probGRU overfit (ep 80) | 0.951 | +0.41 | +0.03 |
  | predict-mean (zero) | 0.978 | +0.40 | 0.00 |
  | persistence-of-fast | 1.624 | 0.00 | −0.66 |

- **Three conclusions**: (a) persistence-of-fast is *worse than predicting the mean*, so
  skill-vs-persistence on this target is structurally inflated — even predict-mean "scores" +0.40;
  (b) the honest signal is **~25 % of variance explained**; (c) the **linear AR essentially ties the
  GRU** (0.763 vs 0.737, ~3 %). Report **R² vs the mean**, not skill vs persistence, when comparing
  across targets.
- Also flagged and recorded (secondary, unfixed by design):
  `physical_state.baseline_correct` uses a **whole-clip** percentile (non-causal, not deployable
  online); `hand="active"` picks the hand from whole-clip mean force; the manifest carries **no
  subject id**, so clip-level splits mix subjects → results are *within-corpus*, **not new-user**,
  generalization.
- **Figure-reading insight** (resolved a persistent confusion): three code paths produce a forecast,
  and only the overlay plot re-anchors. `plot_forecast_overlay.py` tiles a clip into consecutive 1 s
  blocks, each re-seeded from ground truth — visually flattering but metric-neutral.
  `docs/actionsense/horizon_highpass.png` (single anchor) is the honest picture; the reported skill numbers come
  from the honest path.

## 6. Phase E — Frozen evaluation harness and the four-way comparison

### 6.1 The frozen harness
[src/actionsense/eval_harness/](../src/actionsense/eval_harness/), config
[configs/actionsense/eval_harness.yaml](../configs/actionsense/eval_harness.yaml)
(sha256 `8afc249f260894fd`, stamped into every result row).
- **Target redefined for comparability**: raw 6-dim `[F_L, CoPx_L, CoPy_L, F_R, CoPx_R, CoPy_R]`
  at 10 Hz, horizon 1 s = 10 steps, indexed by *target* time `t+h`.
- **Frozen 3-way split** `data/actionsense_states/splits.json`: 60/20/20 **by recording**,
  stratified by (activity, object) → **train 45 / val 15 / test 15** of 75 recordings; both hands
  of a recording always fall in the same partition. Fit on TRAIN → select on VAL → **touch TEST once**.
- **CoP masking**: a CoP target frame is dropped iff that hand's raw force is below the TRAIN 5th
  percentile (force channels never masked) — ~2.5–3 k of ~49.7 k frames removed.
- **Determinism asserted** (two runs must be byte-identical); **7 pytest** unit tests on synthetic
  signals (exact seasonal recovery on a sine, AR(2) coefficient recovery, masking, causality).
- Five spec-vs-repo contradictions (rate, target dimensionality, name collision with the pixel
  `eval.py`/`baselines.py`, missing deps, missing val split) were **surfaced and resolved with the
  user before any code was written**, not silently worked around.

### 6.2 Classical baselines (frozen TEST, `docs/actionsense/harness_baselines.csv`)
| baseline | nRMSE | mean skill vs persistence |
|---|--:|--:|
| persistence | 0.517 | 0 (reference) |
| seasonal-naive | 0.556 | 0 (**falls back to persistence for all 5 groups**) |
| **linear AR** (`AutoReg`, order 20–30 per group) | **0.467** | **+0.18** (artifact recompute +0.189; best channel right-hand `CoP_x` +0.25) |

- **Seasonal-naive is inert, and this is a finding**: raw aggregate force/CoP has **no
  autocorrelation peak in 0.3–3 s** — the slow trend makes the autocorrelation decay monotonically,
  so no fundamental period exists to copy. Closed deliberately (causal detrending would fix period
  *detection* but not *accuracy*, since one period back ≈ now for a drifting signal).

### 6.3 Does the tactile MAP carry extra signal?
[src/actionsense/tactile_map/](../src/actionsense/tactile_map/) — two encoders feeding an
**identical** GRU + one-shot probabilistic head; only the per-frame encoder differs
(flatten `Linear(2048→64)` vs a 3-conv CNN → global-avg-pool → 64).
- **Preprocessing made causal**: per-taxel baseline from the **first N=10 frames** (replacing the
  non-causal whole-clip percentile) + `log1p(α·x)` compression + a single global TRAIN scale.
- **Residual parameterization again decisive**: absolute-level map models **mean-reverted** to the
  training mean and scored −1.9 … −2.2, despite `corr(model, true) = 0.78` (the map *does* carry
  signal — the net was hedging). Predicting the **residual over persistence** made the worst case
  equal persistence and fixed it.
- **5-fold CV, probabilistic, early-stopped, σ-calibrated** (`docs/actionsense/tactile_map_cv_results.csv`):

  | history | CNN(map) | flatten(map) |
  |---|--:|--:|
  | 1 s | +0.052 | −0.040 |
  | 3 s | +0.050 | −0.025 |
  | 10 s | **+0.063** | −0.026 |

  coverage 0.93 raw → 0.95 calibrated.
- **CNN > flatten at every history** → the **spatial structure of the contact patch does contribute**
  to predicting the *change* in F/CoP. Flatten sits at or below persistence.
- **Mechanism established, not assumed** (`docs/actionsense/tactile_map_loss_curve.png`): both encoders overfit
  almost immediately — flatten's validation NLL bottoms at **epoch 1**, the CNN's at **epoch 4**.
  The CNN's spatial inductive bias extracts a little *generalizable* signal before overfitting;
  flatten memorizes instantly. So the modest +0.05–0.06 reflects **data scarcity (45 training
  recordings vs a 2,048-dim input)**, not encoder failure.

### 6.4 The four-way comparison (the project's headline result)
All four forecasters on the **same** raw 6-dim target, same Slice+Peel data, same autoregressive
input convention, and — after a rigor fix — the **same 5-fold folds** (`docs/actionsense/forecaster_comparison.png`):

| history | linear AR | GRU-aggregate | CNN-map | flatten-map | persistence |
|---|--:|--:|--:|--:|--:|
| 1 s | **+0.166** | +0.120 | +0.052 | −0.040 | 0 |
| 3 s | **+0.166** | +0.138 | +0.050 | −0.025 | 0 |
| 10 s | **+0.166** | +0.142 | +0.063 | −0.026 | 0 |

**Ranking: linear AR > GRU-aggregate > CNN-map > flatten-map > persistence.**

- The rigor fix mattered: the AR number originally came from the *frozen split* (+0.180) while the
  GRU came from *5-fold CV*. Re-scoring AR on the identical folds gives **+0.166** (per-fold
  0.15–0.19). The ranking is unchanged but is now apples-to-apples.
- **Conclusion 1 — a linear autoregression is the best 1-second forecaster of raw F/CoP.** Neither
  nonlinearity (GRU) nor a richer input (the map) beats per-channel linear AR. Over 1 s these
  dynamics are essentially linear-autoregressive.
- **Conclusion 2 — the map is an *inferior input* to the aggregate for this target** (+0.05 vs
  +0.14). Routing through the pixel representation loses information: the network must reconstruct
  `F` (a sum) and `CoP` (a centroid) from pixels imperfectly, when those aggregates were available
  directly.
- **Conclusion 3 — within the map, spatial structure still helps** (CNN +0.05 > flatten −0.03),
  robust across history lengths and CV — but not enough to reach the aggregate, let alone AR.
- **Conclusion 4 — all learned models calibrate cleanly** to coverage ≈ 0.94–0.95 after σ-scaling.

---

## 7. Cross-cutting methodological conclusions

These are the transferable lessons; several were corrections against our own earlier headlines.

1. **Residual-over-persistence parameterization is mandatory** when persistence is strong. It made
   the EgoTouch pixel model go 0.004 → 0.174 and rescued the map models from −2.0 → +0.05. At worst
   the model predicts Δ = 0 and *ties* the baseline instead of hedging to the mean.
2. **Never use `filtfilt` in a forecasting pipeline.** Its backward pass leaks the future into the
   target; here it inflated skill by ~0.3 (+0.70 → +0.40). Use `sosfilt` and pay a warm-up trim.
3. **Early stopping is not a detail — it is a result.** Without it the fast-target probGRU had
   *near-zero real skill* (R² +0.03, essentially predict-mean) while displaying a flashy +0.41. It
   also manufactured a false scientific finding ("more history hurts") that vanished on the fix.
4. **Choose the baseline before believing the number.** Skill-vs-persistence is only meaningful when
   persistence is a *strong* baseline. On a zero-mean high-pass target, persistence is *worse than
   predicting the mean*, so trivial shrink-to-zero scores +0.57. **Variance explained (R² vs the
   mean) is the honest, cross-target metric.**
5. **Smoothness is a double-edged property.** The trait that makes an action predictable in absolute
   terms (smooth, slowly-varying force) is exactly the trait that makes it hard to *beat persistence*
   on. This resolved three separate "failures" (grasp LOTO ≈ 0, v1 GRU ≈ persistence, high raw
   accuracy with no skill) as one phenomenon.
6. **Separate what you can measure from what you can compare.** Four gloves with different geometries
   and rates cannot be compared on raw skill; rank *within* a dataset and compare *traits* across.
7. **A-priori semantic taxonomies do not survive contact with data.** The verb→temporal-pattern map
   inverted between datasets; the measured statistic (`persH15`) travelled, the label did not.
8. **Freeze the evaluation before optimizing the model.** The harness (frozen split, config hash,
   determinism assert, causality unit tests, fit/select/score discipline) is what made the four-way
   comparison trustworthy — and it caught the AR frozen-vs-CV protocol mismatch.
9. **Look at the plotting path before trusting the figure.** Re-anchored overlays look far better
   than the honest single-anchor forecast while changing no metric.

---

## 8. Caveats and limits on every conclusion above

- **Data-scarce regime.** 75 recordings (Slice 45 / Peel 30), 45 in training, ~48 min usable after
  warm-up trim. Map models overfit by epoch 1–4. "The map doesn't help" is a statement about *this
  regime*, not about tactile maps in general.
- **1-second horizon only.** Linear AR dominates *at 1 s*. Longer horizons, where linear
  extrapolation must break down, were never tested and could favour the map or nonlinear models.
- **Two targets are in play and their skill numbers are NOT comparable**: raw 6-dim both-hands
  (§6, persistence is strong) vs high-pass fast 3-dim one-hand (§5, persistence is weak). Only
  R²-vs-mean compares across them.
- **Not new-user generalization.** No subject id is stored in the manifest, so splits mix subjects
  and sessions; recovering it requires re-streaming ~88 GB of HDF5 on CRC.
- **One non-causal step remains upstream**: `baseline_correct` uses a whole-clip percentile for the
  *target*. Kept deliberately for comparability with all prior results, and documented; the *map
  input* was made causal (first-N-frames).
- **AR is history-agnostic** in the four-way table — it selects its own order rather than being swept
  over `t_in`, so it appears as a single flat line.
- **Two actions, one sensor, one corpus.** Slice and Peel are both rhythmic stroke actions; nothing
  here has been tested on the abrupt/make-break actions that Phase C identified as hard.
- **Force is uncalibrated** (arbitrary sensor units) and CoP is in normalized grid units [−1, 1],
  not millimetres. "Skill" is dimensionless.
- **Force-Vision was never downloaded** — its contribution is taxonomic only.

---

## 9. Where the project stands, and what is open

**Delivered and reproducible from a plain clone** (states, raw maps, all result CSVs and figures are
committed): the frozen harness, three classical baselines, four forecaster families, the
cross-dataset probe results, and 17 passing unit tests (7 harness + 10 tactile-map).

**Explicitly closed** (recorded 2026-07-21): the seasonal-naive inertness (documented finding, no
code change) and the probGRU one-shot-vs-autoregressive decoder comparison (superseded — the
harness AR replaces the proposed AR(1) baseline, and the probGRU predicts the old target).

**Open / not done** — in rough order of scientific value:
1. **The original goal is still unbuilt: the feedback demo.** Everything needed exists (a calibrated
   probabilistic model of the fast action component, coverage ≈ 0.95, interpretable channels), but
   the normative "expert band → deviation score" application was never implemented.
2. **Test the honest metric everywhere.** Skill-vs-persistence is still the reported number for the
   map/aggregate/AR comparison; those should also be reported as R² vs the mean.
3. **Fair-comparison history sweep** — equalize window counts across history lengths (longer history
   currently trains on up to 22 % fewer windows, confounding the history trend).
4. **Beat AR**, e.g. a hybrid map + aggregate input, more capacity with regularization, or more
   activities/subjects; and test longer horizons where AR should degrade.
5. **Subject-level splits** (needs a CRC re-stream to record subject id) for a new-user claim.
6. **Fully causal `baseline_correct`** for any online/deployable claim.
7. Optional: GPU per-category forecasting on OpenTouch to convert the probe hypothesis into measured
   skill; Force-Vision as a 4th probed dataset.

---

## 10. References

### Primary log
- [SESSION_LOG.md](../SESSION_LOG.md) — source of truth. Key sections: Sessions 1–3 (setup, EgoTouch,
  pixel forecaster); Session 4 (cross-dataset study); *COMPREHENSIVE SUMMARY* (2026-07-06);
  *COLD-START ONBOARDING SNAPSHOT* (2026-07-14); *RIGOROUS CODE + RESULTS REVIEW* (2026-07-10);
  *CONSOLIDATED RESULTS & METHODS* (2026-07-24); *PORTABILITY* (2026-07-24).
- [CLAUDE.md](../CLAUDE.md) — working agreement (plan-before-code, log everything).

### Project documents
- [docs/STUDY_SUMMARY.md](STUDY_SUMMARY.md) — predictability study write-up.
- [docs/ACTION_CATEGORIES.md](ACTION_CATEGORIES.md) — unified cross-dataset taxonomy (Axes A–D).
- [docs/RESULTS.md](RESULTS.md) — EgoTouch pixel forecaster results.
- [docs/REPO_ORGANIZATION.md](REPO_ORGANIZATION.md) — file-by-file dataset grouping.
- [docs/TACTILE_PREDICTION_PLAN.md](TACTILE_PREDICTION_PLAN.md), [docs/TACTILE_FORECAST_PLAN.md](TACTILE_FORECAST_PLAN.md) — design plans (incl. the physics-structured latent world model that was superseded by the explicit-state choice).
- [docs/leakage_checklist.md](leakage_checklist.md) — the 6 leakage checks.

### Code — ActionSense (current thread)
- [src/actionsense/physical_state.py](../src/actionsense/physical_state.py) — pressure moments; baseline correction ([:68](../src/actionsense/physical_state.py#L68)).
- [src/actionsense/action_dynamics.py](../src/actionsense/action_dynamics.py) — the fast-component library: `slow_fast` ([:39](../src/actionsense/action_dynamics.py#L39)), `build_features` ([:54](../src/actionsense/action_dynamics.py#L54)), `windows` ([:100](../src/actionsense/action_dynamics.py#L100)), `ProbGRU` ([:142](../src/actionsense/action_dynamics.py#L142)), `train` w/ early stopping ([:176](../src/actionsense/action_dynamics.py#L176)), `calibrate_sigma` ([:217](../src/actionsense/action_dynamics.py#L217)), `evaluate` ([:226](../src/actionsense/action_dynamics.py#L226)).
- [src/actionsense/state_forecast.py](../src/actionsense/state_forecast.py) — v1 (superseded).
- [src/actionsense/eval_harness/](../src/actionsense/eval_harness/) — `config.py`, `splits.py`, `dataset.py`, `masking.py`, `metrics.py` (`skill` at [:45](../src/actionsense/eval_harness/metrics.py#L45)), `baselines/{persistence,seasonal,ar}.py`, `evaluate.py`, `README.md`.
- [src/actionsense/tactile_map/](../src/actionsense/tactile_map/) — `data.py` (causal baseline, `log1p` at [:51](../src/actionsense/tactile_map/data.py#L51), residual target at [:140](../src/actionsense/tactile_map/data.py#L140)), `models.py`, `train.py`.

### Code — EgoTouch / OpenTouch (earlier thread)
- [src/tactile_pixel/](../src/tactile_pixel/) — `categories.py` (taxonomy), `predictability.py` (probe metrics), `train.py`/`eval.py`/`engine.py`/`baselines.py`, `models/` (SimVP, ConvGRU, ConvLSTM).

### Scripts
- Probes: [probe_egotouch.py](../scripts/egotouch/probe_egotouch.py), [probe_opentouch.py](../scripts/opentouch/probe_opentouch.py), [probe_actionsense.py](../scripts/actionsense/probe_actionsense.py).
- Training CLIs: [train_action_dynamics.py](../scripts/actionsense/train_action_dynamics.py), [train_tactile_map.py](../scripts/actionsense/train_tactile_map.py), [train_state_forecaster.py](../scripts/actionsense/train_state_forecaster.py).
- Integrity: [check_leakage.py](../scripts/actionsense/check_leakage.py), [tmp_diag_predictability.py](../scripts/actionsense/tmp_diag_predictability.py).
- Plots: [plot_forecaster_comparison.py](../scripts/actionsense/plot_forecaster_comparison.py), [plot_tactile_map.py](../scripts/actionsense/plot_tactile_map.py), [plot_tactile_map_loss_curve.py](../scripts/actionsense/plot_tactile_map_loss_curve.py), [plot_fcop_loss_curve.py](../scripts/actionsense/plot_fcop_loss_curve.py), [plot_horizon.py](../scripts/actionsense/plot_horizon.py), [plot_forecast_overlay.py](../scripts/actionsense/plot_forecast_overlay.py), [plot_harness.py](../scripts/actionsense/plot_harness.py).
- CRC: [scripts/crc/](../scripts/crc/) — `README.md`, `stream_actionsense.sh`, `train_state_gpu.job`, `train_tactile_map_gpu.job`.

### Result artifacts (numbers in this document were re-verified against these)
- [docs/actionsense/harness_baselines.csv](harness_baselines.csv) (+ `.parquet`, `_fitparams.csv`) — persistence / seasonal / AR, config hash `8afc249f260894fd`.
- [docs/actionsense/tactile_map_cv_results.csv](tactile_map_cv_results.csv) — CNN vs flatten, 5-fold CV.
- [docs/actionsense/tactile_map_cv_results_aggregate.csv](tactile_map_cv_results_aggregate.csv) — GRU-aggregate.
- [docs/actionsense/action_dynamics_results_earlystop.csv](action_dynamics_results_earlystop.csv) — fast-target probGRU (honest); [docs/actionsense/action_dynamics_results.csv](action_dynamics_results.csv) (overfit, kept for the before/after diff); `_precal.csv` (pre-calibration).
- [docs/actionsense/predictability_by_category_full.csv](predictability_by_category_full.csv) (EgoTouch, n=1,929), [docs/actionsense/predictability_by_category.csv](predictability_by_category.csv) (sampled).
- Figures: `forecaster_comparison.png`, `tactile_map_skill_vs_history.png`, `tactile_map_coverage.png`, `tactile_map_loss_curve.png`, `fcop_earlystop_comparison.png`, `fcop_loss_curve.png`, `horizon_highpass.png`, `harness_skill_{bars,curves}.png`, `results_summary.png`.
- Data: `data/actionsense_states/` — `state_*.npy` (299), `clip_*.npy` (100 raw maps), `manifest.jsonl`, `splits.json` (train 45 / val 15 / test 15, seed 0, stratified by action×object).
- Tests: [tests/test_harness.py](../tests/test_harness.py) (7), [tests/test_tactile_map.py](../tests/test_tactile_map.py) (10).

### External datasets and papers
- **EgoTouch** — HF `zhouzhoujy/EgoTouch`; 21×21 pressure grids, 2 hands, 30 Hz.
- **OpenTouch** — arXiv:2512.16842, opentouch-tactile.github.io; 16×16 FPC, 1 hand, 30 Hz, GRASP-taxonomy grip labels.
- **ActionSense** — NeurIPS 2022 D&B (MIT CSAIL), `delpreto/ActionNet`; 32×32 conductive-thread gloves, ~6 Hz, 20 kitchen activities.
- **Force–Vision** — ICLR 2024, STAG-style glove, press/hold/squeeze on 89 tool instances (taxonomy only, not probed).
- Modelling precedents cited in the plans: SimVP (arXiv:2206.05099), TAU (arXiv:2206.12126, *not implemented here*), ACTP/ACTVP (arXiv:2205.09430), PredFormer (arXiv:2410.04733), tactile-prediction survey (arXiv:2401.14718).
