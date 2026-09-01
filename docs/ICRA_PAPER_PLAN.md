# ICRA paper — research inventory and paper structure

Drafted 2026-09-01 from a full read of [SESSION_LOG.md](../SESSION_LOG.md) (8,628 lines),
[PROJECT_CONCLUSIONS.md](PROJECT_CONCLUSIONS.md), [skill_comparison.md](skill_comparison.md),
[model_comparability.md](model_comparability.md), [d256.md](d256.md) and the committed result
CSVs. **Nothing here is a new measurement.** Every number cited is traceable to an artifact
already in `docs/`; where a claim needs a run that has not happened, it is marked **[MISSING]**.

This is a *plan*, per CLAUDE.md directive 5. The OPEN QUESTIONS in §9 block writing, not
reading — resolve them before any LaTeX is written.

---

## 0. The framing decision that governs everything else

The repo carries **two unrelated papers**:

| | what it is | status |
|---|---|---|
| **EgoTouch / TouchAnything** ([README.md](../README.md)) | multi-view tactile *estimation* from egocentric video; 208 tasks, 1,891 episodes | already on arXiv (2605.13083), authored by the HIT/Meituan team, **not this work** |
| **This work** | 1-second tactile *forecasting*: how much of the near future is predictable from tactile alone, and what current models actually do | 3 sensor corpora, 6 model families, frozen harness, ~10 weeks of results |

The repo was already split off to `Jiayi459/tactile-forecasting` (SESSION_LOG 2026-08-21).
**This plan assumes the ICRA paper is the second one.** If it is a resubmission or extension
of the EgoTouch dataset paper, discard this document — the content map is completely different.

### The honest shape of the result

Across two independent glove sensors, three protocols, six model families and two metrics,
**no model predicts the tactile future.** Every one of them — linear AR, one-shot GRU,
autoregressive probGRU, CNN over the raw map, flatten over the raw map — converges to
predicting the *local mean* of the signal. The positive skill numbers (0.05–0.48) are real
and reproducible, and they measure **how well a model reproduces a local level**, not how well
it anticipates contact dynamics.

That is a defensible ICRA paper only if it is framed as **a measurement paper**: the
contribution is the protocol that makes the limit visible, the quantification of the limit,
and the demonstration that the field's default metric hides it. It is *not* a paper that says
"our model is better." Two framings are viable; §8 recommends one.

---

## 1. Complete inventory of the research content

Ordered by what the paper can use, not chronologically.

### 1.1 Corpora actually processed

| corpus | sensor | rate used | unit | protocol | in paper? |
|---|---|---|---|---|---|
| **ActionSense** (NeurIPS'22) | 32×32 conductive-thread, 2 gloves | 30→10 Hz | 299 recordings (Slice+Peel subset: 75) | stratified 60/20/20 by recording; 5-fold CV | **yes, sensor A** |
| **OpenTouch** (arXiv 2512.16842) | 16×16 FPC, 1 glove | 30 Hz | 2,904 clips / 12 locations | **4-fold group CV, whole locations held out** | **yes, sensor B** |
| **d256** | = ActionSense, repackaged | 6 Hz | 166 reconstructed segments | **LOSO (leave-one-subject-out)** | **as a protocol variant, not a third sensor** |
| **EgoTouch** | 21×21, 2 gloves | 30 Hz | 1,929 trajectories | LTO / LOTO | probably out — different target (pixels) |
| Force–Vision (ICLR'24) | STAG glove | — | — | — | **never obtained** (see §1.7) |

**d256 must not be presented as a fourth dataset.** Frame-level cross-correlation settles it:
a 150-frame probe from one d256 recording matches exactly one of 299 ActionSense recordings at
**0.9949**, runner-up 0.9428, uniquely above 0.95 with the correct label
(`scripts/shared/compare_d256_actionsense.py`, SESSION_LOG 2026-09-01). It is also **not the
dataset of the ICLR 2024 force-vision paper** whose page hosts it — that paper describes
16×16 / single glove / 10 frames / 64×64 video / 89 objects, none of which match. The archive
belongs to **ICCV 2025 MMVidSim** (arXiv 2510.02287). Citing it as "the ICLR force-vision
dataset" would be factually wrong. Correct citation: **ActionSense (DelPreto et al., 2022)**,
optionally noting the MMVidSim repackaging.

### 1.2 The evaluation harness (a contribution in its own right)

`src/actionsense/eval_harness/`, config hash `8afc249f260894fd` stamped into every result row.

- rolling-origin forecasting, target indexed by **target time** `t+h`, horizon = **1 s**
  (10 steps at 10 Hz, 30 at 30 Hz, 6 at 6 Hz)
- frozen split, fit on TRAIN → select on VAL → **TEST touched once**
- CoP masked when that hand's force is below the TRAIN 5th percentile; force never masked
- determinism asserted (two runs byte-identical); 17 unit tests incl. causality and AR(2)
  coefficient recovery on synthetic signals
- five spec-vs-repo contradictions surfaced and resolved *before* code was written

### 1.3 The metrics, and the fact that two of them are named the same thing

- **skill vs persistence** = `1 − MSE_model/MSE_persistence`. **Two non-interchangeable
  estimators**: frame-pooled (driver) and clip-balanced (report). On OpenTouch's F channel they
  give 0.367 and 0.302 for the same arm, and in the backbone comparison **they disagree in
  sign on all three F arms**. Any number in the paper must declare its convention.
- **R² vs the mean** — the honest cross-target metric; skill is only meaningful when
  persistence is strong.
- **Hausdorff shape distance** (`src/shape_metrics.py`, one implementation shared by all
  sensors): forecast and truth as point sets in (scaled time, scaled value); charges a flat
  forecast roughly the amplitude of the oscillation it missed. Reported as a ratio to
  persistence.
- **Predictability floor** `R = E[(y_{t+H}−y_t)²] / (2·Var)`, so `MSE_persist = 2R·Var`.
  `R = 1` means the signal is fully decorrelated at the horizon; `R > 1` means anticorrelated.
  This is the number that makes cross-sensor skill comparison legitimate at all.
- calibration: ±2σ coverage, σ-scaling on a validation slice.

### 1.4 Model families, all under the one harness

| family | decoder | predicts | input |
|---|---|---|---|
| persistence | — | last value | — |
| seasonal-naive | — | one period back | — |
| **linear AR** (`AutoReg`, order 20–30 per group) | — | — | F/CoP history |
| **GRU-aggregate (Seq2Seq)** | one-shot, all H steps | **residual** over last value | F/CoP |
| **probGRU** | autoregressive, own mean fed back | **absolute** | F/CoP (+8-dim action embedding) |
| **map-flatten** | as above | — | 2048-dim flattened map |
| **map-CNN** | as above | — | 3-conv → GAP → 64 |
| *(EgoTouch only)* SimVP / ConvLSTM / ConvGRU | — | residual pixel frames | pixel maps |

### 1.5 Results the paper can stand on

**R1 — the DC pedestal is a first-order confound.**
OpenTouch's array runs at **94.9 % of full scale at rest** (hardware rail 3072, identical
across all 26 shards, longest continuous rail-hit 314 s); **99.78 % of F is DC**. The D1
correction shrinks the target ~350× and **roughly doubles every skill number**
(AR F 0.148 → 0.367; probGRU 0.203 → 0.386). It also collapses the apparent autocorrelation:
r(1) 0.95/0.93/0.88 → **0.318/0.403/0.228**. ActionSense had the same disease in a different
form (untared glove, F ≈ 585,000 ± 0.5 %, CoP pinned to the sensor centre) and the same fix.
**Consequence: a skill number reported without stating the baseline-correction state is
uninterpretable.**

**R2 — persistence is not equally hard across sensors, and the difference is ~2×.**
`docs/predictability_floor.csv`, one definition, one implementation:
ActionSense R = 0.649, d256 R = 0.717, **OpenTouch R = 1.041** (ρ₁ 0.87–0.95, 0.65–0.81,
**0.20–0.36**). OpenTouch's high skill is earned against a *very weak* denominator — the
signal is already decorrelated at 1 s. **Two skill numbers without their R values are two
different questions.**

**R3 — R is governed by segment length, not sampling rate.** Decimating ActionSense to
30/10/6 Hz, with and without anti-aliasing, gives R = 0.605/0.605/0.604/0.602 — the aliasing
hypothesis is refuted. Cutting the same signals into fixed windows:

| window | 10 s | 20 s | 40 s | 80 s | 160 s |
|---|---|---|---|---|---|
| R | 0.807 | 0.692 | 0.616 | 0.560 | 0.569 |

d256's median recording is 12.9 s → R 0.717; ActionSense's is 22.0 s → 0.649. Both land on the
curve. Mechanism is in the definition: a short segment truncates the slow drift, shrinking the
denominator variance. **So the same recordings, cut differently, "become" more or less
predictable.** This alone justifies the protocol paper.

**R4 — every model converges to the local mean.** Three independent lines of evidence:
1. **stitched forecast curves**: after D1 the true F oscillates sharply between 500 and 2700;
   AR, probGRU, seasonal and persistence all draw near-flat lines through the local level.
2. **skill does not decay with horizon** — it *rises* (AR 0.384 → 0.425 from h=1 to h=30). A
   model tracking dynamics must get worse with distance; a mean predictor gains as persistence
   degrades around it.
3. **Hausdorff**: persistence 3.301, best model 2.699. The models' spread from each other
   (0.14) is far smaller than their common distance from actually tracking the oscillation.

**R5 — the predictability ceiling is layered, and short horizons are nearly exhausted.**
Against a conservative noise-derived ceiling on D1-corrected OpenTouch:

| channel | h=1 achieved / ceiling | h=30 achieved / ceiling |
|---|---|---|
| F_R | 0.342 / 0.439 = **78 %** | 0.373 / 0.668 = **56 %** |
| CoPx_R | 0.391 / 0.438 = **89 %** | 0.454 / 0.718 = **63 %** |
| CoPy_R | 0.420 / 0.485 = **87 %** | 0.478 / 0.624 = **77 %** |

Short horizon: 68–77 % of the corrected variance decorrelates within one frame; what is left,
the models already take. **The headroom is entirely at long horizon and is the slow envelope.**

**R6 — the raw tactile map adds nothing over F/CoP aggregates, and this replicates.**
Ordering falls monotonically with how much raw spatial detail an arm is handed:
**AR > GRU-aggregate > CNN-map > flatten-map > persistence.** Holds on ActionSense
(0.181/0.138/−0.042 aggregate/CNN/flatten on F) *and* OpenTouch (0.360/0.333/0.273), on TEST
*and* VAL, and under **both** backbones (`d1_pg` fixes the probGRU backbone and varies only the
input: same ordering). Within the map, spatial structure still helps (CNN > flatten), so the
result is "routing through pixels loses information", not "the map is noise". Mechanism is
measured, not assumed: flatten's val NLL bottoms at **epoch 1**, the CNN's at **epoch 4** —
data scarcity against a 2048-dim input.

**R7 — MSE and shape disagree, systematically and in a way that reverses the ranking.**
Fixing input, data, folds and loss and varying only the decoder: **Δ Hausdorff favours Seq2Seq
in 9 of 9 cells (+0.111 … +0.227)** while frame-pooled skill favours probGRU in 9 of 9. And
`seasonal` — the *only* model that loses to persistence on MSE — is the **best** model on
shape. The backbone effect on shape (0.11–0.23) exceeds the spread between input
representations within a backbone (0.08 / 0.13). **Metric choice changes the conclusion**;
MSE rewards flattening.

**R8 — the a-priori smooth/abrupt trait does not predict skill: a clean null.**
Bootstrap CI on ΔR² (B=2000) **crosses zero for all 4 models × 3 channels**, before *and*
after D1, and survives leave-one-action-out (removing `picking up` or `holding` moves ΔR² by
0.01–0.03). Trait labels were **blind-adjudicated and committed before the join counts were
computed** (commit `7376efd` timestamp is the evidence) — a genuine pre-registration.
Note the tension with R9 and resolve it explicitly in the paper (see §4, "the one real
internal conflict").

**R9 — training-free probe: a trait, not a category (3 sensors).** Predictable = smooth,
continuous, slowly-varying contact force (pour, slice, wipe, peel, stir, scoop); unpredictable
= abrupt make-or-break engagement (press, click, plug, jar open). `persistence_nMSE@15` is the
sensor-agnostic statistic. Category rankings *disagree* across sensors; the a-priori
verb→temporal-pattern axis **inverts** between EgoTouch and OpenTouch. Sustained holds are
*not* trivially predictable (grips drift).

**R10 — calibration.** All learned arms reach coverage 0.94–0.95 after σ-scaling. But the
error is **heavy-tailed, not over-dispersed**: coverage 94.3–94.6 % against nominal 95.4 %
while σ/median|error| = 1.76–1.82 (Gaussian reference 1.48), the same pattern across three
channels spanning ~6 orders of magnitude. The Gaussian likelihood is provably the wrong
family; kept for cross-arm comparability and written into limitations.

**R11 — (EgoTouch, older thread) pretraining unlocks unseen-object pixel forecasting.**
SimVP LTO +0.192; LOTO from scratch +0.005; LOTO with 1,851-trajectory grasp-excluded
pretraining **+0.097**. Different target, different corpus — a separate paper's worth, or one
sentence of related work.

### 1.6 Methodological findings that are transferable (and are the paper's soul)

1. **Residual-over-persistence parameterization is mandatory** when persistence is strong.
   EgoTouch pixels 0.004 → 0.174; map models −2.0 → +0.05.
2. **Never `filtfilt` in a forecasting pipeline** — the backward pass leaked the future and
   inflated skill by ~0.3 (+0.70 → +0.40).
3. **Early stopping is a result, not a detail.** Without it the probGRU had R² +0.03
   (i.e. predict-mean) while displaying +0.41 skill, and it *manufactured a false finding*
   ("more history hurts") that vanished on the fix.
4. **Choose the baseline before believing the number.** On a zero-mean high-pass target,
   persistence is worse than predicting the mean, so trivial shrink-to-zero scores +0.57.
5. **Smoothness is double-edged**: the trait that makes an action predictable in absolute
   terms is exactly the trait that makes persistence hard to beat. Resolves three separate
   "failures" as one phenomenon.
6. **Analytic performance bounds must be run before being called reachable.** Twice an oracle
   quantity was mistaken for an achievable one (AR(1) optimal-predictor "floor" 0.33–0.40 vs
   measured 0.087; whole-segment mean vs causal running mean, which measured −0.300 with the
   sign *reversed* from the derivation).
7. **Look at the plotting path before trusting the figure.** Re-anchored overlays flatter the
   model while changing no metric.
8. **A replicate defines the noise floor.** `d1_map2` → `d1_map3`, same seed and data: CNN
   moved 4.9e−3, flatten 4.2e−3. Any ranking with a smaller margin is not established.

### 1.7 Things that do not exist and must not be implied

- **Force–Vision / the ICLR'24 dataset was never obtained.** Taxonomy only.
- **d256 has no forecaster of its own beyond the ported probGRU/AR arms**, and its absolute
  sampling rate was originally unknown (later fixed at 6 Hz by the ActionSense identification).
- **No subject-level split on ActionSense** — the manifest carries no subject id, so splits
  mix subjects. Claims are *within-corpus*, never *new-user*. (d256's LOSO is the only
  unseen-person evidence in the project.)
- **One non-causal step remains upstream**: `baseline_correct` uses a whole-clip percentile for
  the *target*. Kept for comparability, documented; the map *input* was made causal.
- **TAU was never implemented** despite appearing in an early plan.
- **The feedback/coaching demo — the project's original motivation — was never built.**

---

## 2. Proposed paper structure (6 pages + references, ICRA format)

Working titles, in order of preference:

1. **"Tactile Forecasting Predicts the Mean: Measuring the Limit of One-Second Contact
   Prediction"**
2. "How Hard Is Persistence? A Protocol for Comparable Tactile Forecasting Across Sensors"
3. "What Tactile Forecasters Actually Learn"

### §I Introduction — 0.8 page

- **Hook**: anticipating contact one second ahead is the enabling primitive for reactive
  grasping, slip pre-emption, teleoperation latency hiding and wearable coaching. Recent work
  reports positive "skill over persistence" on tactile forecasting.
- **The gap**: those numbers are not comparable and are rarely interpretable. Skill's
  denominator — how hard persistence is on *that* signal — is never reported, and it varies
  ~2× across sensors and even with how the recordings were segmented. Preprocessing state
  (whether the sensor's DC pedestal was removed) moves skill by a factor of two.
- **What we did**: one frozen protocol, two independent glove sensors, three split protocols,
  six model families, three metrics.
- **What we found**: state R4 up front. Every family converges to predicting the local mean;
  the short-horizon ceiling is 78–89 % exhausted; the raw tactile map contributes nothing
  beyond F/CoP; and the default metric is what hides all of this.
- **Contributions**, four bullets: (C1) the protocol + the predictability floor R;
  (C2) the pedestal/segmentation confounds, quantified; (C3) the local-mean characterization
  with three independent lines of evidence; (C4) the metric-reversal result and the resulting
  reporting guidance.

### §II Related work — 0.5 page

Four short paragraphs: tactile prediction (ACTP/ACTVP, the 2024 survey); spatiotemporal
prediction backbones (SimVP, PredFormer) and why we also carry non-deep baselines; tactile
datasets with real pressure (ActionSense, OpenTouch, EgoPressure, EgoTouch) and their sensor
differences; forecast verification practice (skill scores, persistence/climatology references)
— the meteorology framing is the one that legitimizes R and is worth an explicit citation.

### §III Problem setup and protocol — 1.0 page  ← *the methodological core*

- **III-A Target.** Per hand, per frame, reduce the pressure map to moments
  `[F, CoP_x, CoP_y]` → 6-dim. Motivation: interpretable, sensor-geometry-independent,
  comparable across a 32×32 and a 16×16 glove. Post-hoc justification in R6.
- **III-B Rolling-origin forecasting.** Horizon fixed at **1 s of wall clock**, which is
  10/30/6 steps on the three rates. Note the confound explicitly: the autoregressive decoder
  compounds error three times as far on OpenTouch for the same physical lookahead.
- **III-C Splits.** Location-held-out 4-fold (OpenTouch, the hardest), stratified-by-recording
  (ActionSense), leave-one-subject-out (d256). State that these are three different
  generalization questions and must not be read as one ranking.
- **III-D Metrics.** skill (declare the estimator), R², Hausdorff shape ratio, coverage.
- **III-E The predictability floor R.** Definition, `MSE_persist = 2R·Var`, interpretation of
  R ≷ 1, and Table I (R per sensor per channel).

### §IV Data and the pedestal — 0.5 page

Sensor table (grid, hands, rate, corpus size, split unit) + the D1 correction and its effect
(R1). One figure panel: raw vs corrected F trace, with the r(1) collapse annotated.
**One paragraph, unavoidable, on d256's identity** — it is ActionSense at 6 Hz under LOSO, and
the paper says so plainly rather than presenting three datasets.

### §V Models — 0.5 page

The table from §1.4 plus the training protocol (fit/select/score, early stopping, σ-calibration),
and the one design rule that makes the comparison legal: the backbone is identical across
sensors and only what the data forces may differ (`docs/model_comparability.md`).

### §VI Results — 1.8 pages

- **Table II (main)**: skill per model × channel × sensor, with the **R row at the bottom**.
  Source: `docs/skill_comparison.md`, already generated from CSVs by
  `scripts/shared/build_skill_comparison.py`.
- **Fig. 2**: stitched forecast vs truth, OpenTouch D1, F channel — the single most
  persuasive figure in the project. Existing: `docs/opentouch/d1/opentouch_forecast_d1_F.png`,
  `docs/d256/forecast_*/`.
- **Fig. 3**: skill vs horizon, flat/rising, with the "a real forecaster decays" annotation.
- **Table III**: Hausdorff, with the **persistence row present** (skill divides it out; shape
  does not, so an absolute Hausdorff without its reference means nothing).
- **Table IV**: input-representation ordering on both sensors under both backbones (R6).
- **§VI-E**: the trait null (R8) with bootstrap CIs.
- **§VI-F**: the ceiling table (R5).

### §VII Discussion — 0.6 page

- Why the mean *is* near-optimal here: R, the one-frame noise fraction, and the layered ceiling.
- **R is a property of the corpus, not the sensor**: the segment-length table (R3). Direct
  guidance — report R alongside skill, or the number is not portable.
- MSE rewards flattening; report a shape metric alongside it.
- What would actually move the needle: explicit slow-envelope modelling at long horizon
  (the 0.15–0.30 gap), not larger models. Say plainly that we did not do it.

### §VIII Limitations — 0.3 page

Data-scarce regime (45 training recordings on ActionSense; map arms overfit by epoch 1–4);
1-second horizon only; no new-user claim on ActionSense; one remaining non-causal
preprocessing step; heavy-tailed likelihood mis-specification; force in arbitrary sensor
units; two actions on ActionSense (Slice/Peel), both rhythmic — the abrupt actions R9
identifies as hard are under-tested there.

### §IX Conclusion — 0.15 page

---

## 3. Figure and table budget (ICRA: ~6–8 floats)

| # | content | source | status |
|---|---|---|---|
| Fig. 1 | pipeline: map → moments → rolling-origin harness → 3 metrics | `docs/model_diagram.pdf`, `docs/gru_aggregate_diagram.pdf` | **redraw as one figure** |
| Fig. 2 | stitched forecast vs truth (OpenTouch D1, F) | `opentouch_forecast_d1_F.png` | exists, needs restyling |
| Fig. 3 | skill vs horizon, both sensors | `harness_skill_curves.png` + D1 CSV | **[MISSING] one combined plot** |
| Fig. 4 | pedestal: raw vs D1 trace + r(1) collapse | D1 report artifacts | **[MISSING] figure** |
| Table I | predictability floor R per sensor/channel | `docs/predictability_floor.csv` | exists |
| Table II | main skill table with R row | `docs/skill_comparison.md` | exists |
| Table III | Hausdorff incl. persistence row | `docs/skill_comparison.md` | exists, **one column unusable** (§5) |
| Table IV | input-representation ordering | `skill_comparison.md` | exists |

Fig. 3 and Fig. 4 are plotting work on committed data — no GPU needed.

---

## 4. The one real internal conflict, and how to handle it

**R9 says a trait (smooth vs abrupt force) predicts predictability across three sensors.
R8 says the same trait split gives a clean null on skill.** These are not contradictory, but
a reviewer will read them as such unless the paper says why:

- The **probe** (R9) measures *absolute* predictability — `persistence_nMSE@h`, i.e. how fast
  the signal decorrelates. Smooth actions decorrelate slowly.
- **G2** (R8) measures *skill over persistence* — the headroom **above** a baseline that is
  itself strong precisely because the signal is smooth.

So the trait predicts **R**, and R is exactly what skill divides out. This is the same
mechanism as finding 1.6-5, and stating it converts an apparent inconsistency into one of the
paper's cleaner insights. **It should be a labelled paragraph in §VII, not a footnote.**

A second, smaller one to disclose: `slice`/`clear` are SMOOTH on ActionSense (user
adjudication) while the structurally corresponding `cutting`/`scooping` are ABRUPT on
OpenTouch. Both are in the CONTENTIOUS set and covered by the sensitivity analysis; the
direction does not change. Disclose it in one sentence rather than let a reviewer find it.

---

## 5. What is missing before this can be submitted

Ranked by whether it blocks a claim.

**Blocking**

1. **[MISSING] The ActionSense Hausdorff column is unreadable.** Its CV table carries only the
   `aggregate` encoder and **no persistence row**, so 2.413 has no denominator. Only a
   run-level ratio (0.81× persistence) is comparable. **Fix: re-run that arm with persistence
   scored.** One GPU job. Without it, Table III has a hole in the middle of a headline claim.
2. **[MISSING] The mean-baseline probe on OpenTouch and d256.** It was run on ActionSense
   (hist_mean −0.300, sign opposite to the derivation) but **not** on the two corpora with
   much higher R (0.717, 1.041), where the conclusion may differ. Two zero-GPU commands,
   already written: `scripts/shared/probe_mean_baseline.py --config configs/{d256,opentouch}/...`.
   The claim "even the trivial mean predictor is competitive" cannot be made without them,
   and it is the natural completion of R4.
3. **A decision on whether EgoTouch/SimVP (R11) is in.** It is a different target on a
   different corpus and will cost half a page.

**Strongly desirable**

4. **[MISSING] d256 under a stratified-by-recording split.** Currently d256 differs from
   ActionSense in *two* ways at once (segment length and protocol). R3 isolated segment length;
   the protocol effect is uncontrolled. One run closes it and makes "LOSO costs X" sayable.
5. **[MISSING] The slow-envelope arm (N6).** The only identified path across the long-horizon
   gap. If it works, §VII gains "and here is a way past the limit"; if it fails, the paper is
   unchanged. **High risk before a deadline** — treat as optional.
6. Report R² alongside skill everywhere (currently skill-only for the map/aggregate/AR
   comparison).

**Not needed for this paper**: history-sweep window equalization, subject-level ActionSense
splits (needs an 88 GB re-stream), the feedback demo, Force–Vision.

---

## 6. Risks a reviewer will raise, and the answer

| risk | response |
|---|---|
| "This is a negative result." | It is a *measurement*: R, the ceiling, and the metric reversal are positive, quantitative, reproducible claims. The models are the instrument, not the subject. |
| "Only two sensors." | Two independent glove geometries, three protocols including location-held-out and LOSO, 3,200+ recordings. And we say plainly that d256 is not a third. |
| "Weak models — a transformer would win." | The ceiling analysis answers this: 78–89 % of the achievable short-horizon skill is already taken, and 68–77 % of the corrected variance decorrelates within one frame. Capacity is not the binding constraint; the honest version says so and points to the slow envelope instead. |
| "Why not pixels?" | R6, measured on both sensors under both backbones, with the overfitting mechanism shown. |
| "Is this robotics?" | Needs work — see §7. Currently the framing is closer to a benchmark/analysis paper than to a robot experiment. |

---

## 7. The weakest point: robotics relevance

Every result is about *forecasting a signal*. No robot, no control loop, no downstream task.
ICRA reviewers will ask what changes for a roboticist. Three options, in decreasing honesty
and increasing cost:

- **(a) Reframe the implications, no new experiments.** A 1-second contact forecast is at its
  data ceiling at short horizons — so a controller intending to anticipate slip should not wait
  on a better forecaster; it should either shorten its horizon (where the forecast is nearly
  optimal already) or add exteroception. Plus the reporting guidance (report R; report shape).
  This is defensible and costs nothing.
- **(b) One downstream probe.** E.g. threshold the forecast band for a slip/contact-loss
  pre-alarm and report lead time vs false-alarm rate on held-out recordings. Uses the existing
  calibrated probabilistic arms; a few days of non-GPU work. **This is the highest-value
  addition available before a deadline** and turns "no model predicts the future" into "here
  is what the forecast *is* good enough for."
- **(c) A real robot or teleop experiment.** Out of scope in the time available.

**Recommendation: (a) + (b).**

---

## 8. Two framings — recommendation

**Framing A — measurement/benchmark paper (recommended).** Title #1 above. The contribution is
the protocol, the floor, the ceiling, and the metric result. All evidence exists today; the
blocking gaps in §5 are two commands and one GPU job. Risk: reviewers who want a method.

**Framing B — method paper with the ceiling as motivation.** The slow/fast envelope
decomposition (N6) becomes "our method," motivated by the 0.15–0.30 long-horizon gap. Higher
ceiling if it works, but **it has never been run on the raw target** and could fail; and if it
fails there is no paper, whereas Framing A survives either way.

**Recommendation: Framing A, with N6 attempted in parallel.** If N6 lands before the deadline
it becomes §VII-D, "a path across the gap", strengthening A without A depending on it.

---

## 9. OPEN QUESTIONS — blocking, per CLAUDE.md directive 5

1. **Which paper is this?** The forecasting study (this plan) or something derived from the
   EgoTouch/TouchAnything dataset paper?
2. **Which ICRA, and what is the actual deadline?** ICRA 2027 submission is expected around
   September 2026; the answer decides whether §5 items 1–2 and §7(b) are feasible.
3. **Framing A or B** (§8)?
4. **Is EgoTouch/SimVP (R11) in or out?**
5. **Author list and affiliation.** The repo's README carries the HIT/Meituan EgoTouch author
   block, which is a different work — is any of it carried over?
6. **May the two zero-cost probes (§5.2) and the one GPU job (§5.1) be run?** Both need CRC.
7. **Naming**: keep calling the corpus `d256` (with the identity disclosed), or rename to
   "ActionSense @ 6 Hz"? A 2026-09-01 ruling deferred this; a paper forces it.
8. **Is the downstream probe (§7b) wanted?** It is the single biggest lift in robotics
   relevance per unit of work, but it is new scope.
