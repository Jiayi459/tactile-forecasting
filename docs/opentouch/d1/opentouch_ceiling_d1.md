# OpenTouch predictability ceiling — full results

Produced by `scripts/opentouch_predictability_ceiling.py` on 2026-08-20; raw
output in [opentouch_ceiling_d1.txt](opentouch_ceiling_d1.txt). 600 clips, lags
0-30 frames at 30 fps, median across clips. `D1` is the baseline-corrected cache,
`raw` the uncorrected one.

Skill = 1 - MSE(model)/MSE(persistence), pooled over frames, mean of 4
location-held-out folds, from `docs/opentouch/d1/opentouch_cv4_d1.csv`. Observed skill is shown
only against the D1 ceiling: the models were trained on the corrected target, so
pairing them with the raw ceiling would divide across two different problems. The
`raw` sections are there for the contrast in autocorrelation and ceiling alone.

## 1. Autocorrelation of the target

| cache | channel | r(1) | r(2) | r(3) | r(10) | noise share | first lag r<0.2 |
|---|---|---|---|---|---|---|---|
| D1 | F_R | 0.318 | 0.270 | 0.192 | 0.076 | 68.2% | 3 |
| D1 | CoPx_R | 0.403 | 0.329 | 0.239 | 0.060 | 59.7% | 5 |
| D1 | CoPy_R | 0.228 | 0.146 | 0.082 | 0.013 | 77.2% | 2 |
| raw | F_R | 0.953 | 0.884 | 0.798 | 0.181 | 4.7% | 10 |
| raw | CoPx_R | 0.931 | 0.854 | 0.768 | 0.167 | 6.9% | 10 |
| raw | CoPy_R | 0.880 | 0.776 | 0.673 | 0.107 | 12.0% | 9 |

`noise share` is 1 - r(1), the fraction of variance that decorrelates within one
frame. The raw column's r(1) ~ 0.95 is the DC offset correlating with itself; once
the baseline is removed the true frame-to-frame structure appears, and it is weak.

## 2. Ceiling vs observed skill

skill_max(h) = 1 - v_e / MSE_persistence(h). On pure white noise this is exactly
0.500, so a ceiling near 0.5 means the horizon is noise-dominated. `use` is
observed/ceiling: how much of the achievable skill the model actually took.

### D1 — `/users/jhao3/opentouch/cache_d1`

| channel | h (frames) | MSE persistence | noise var | **ceiling** | ar | use | prob_gru | use |
|---|---|---|---|---|---|---|---|---|
| F_R | 1 | 2.655e+05 | 1.49e+05 | **0.439** | 0.342 | 78% | 0.324 | 74% |
| F_R | 5 | 3.578e+05 | 1.49e+05 | **0.584** | 0.402 | 69% | 0.414 | 71% |
| F_R | 10 | 3.769e+05 | 1.49e+05 | **0.605** | 0.367 | 61% | 0.385 | 64% |
| F_R | 20 | 4.428e+05 | 1.49e+05 | **0.663** | 0.368 | 56% | 0.389 | 59% |
| F_R | 30 | 4.486e+05 | 1.49e+05 | **0.668** | 0.344 | 51% | 0.373 | 56% |
| CoPx_R | 1 | 0.007736 | 0.00435 | **0.438** | 0.391 | 89% | 0.387 | 88% |
| CoPx_R | 5 | 0.01112 | 0.00435 | **0.609** | 0.436 | 72% | 0.434 | 71% |
| CoPx_R | 10 | 0.01294 | 0.00435 | **0.664** | 0.410 | 62% | 0.405 | 61% |
| CoPx_R | 20 | 0.01598 | 0.00435 | **0.728** | 0.440 | 60% | 0.435 | 60% |
| CoPx_R | 30 | 0.01541 | 0.00435 | **0.718** | 0.454 | 63% | 0.450 | 63% |
| CoPy_R | 1 | 0.01693 | 0.008718 | **0.485** | 0.420 | 87% | 0.409 | 84% |
| CoPy_R | 5 | 0.01975 | 0.008718 | **0.559** | 0.484 | 87% | 0.478 | 86% |
| CoPy_R | 10 | 0.02147 | 0.008718 | **0.594** | 0.458 | 77% | 0.456 | 77% |
| CoPy_R | 20 | 0.02378 | 0.008718 | **0.633** | 0.486 | 77% | 0.482 | 76% |
| CoPy_R | 30 | 0.02319 | 0.008718 | **0.624** | 0.478 | 77% | 0.475 | 76% |

### raw — `/users/jhao3/opentouch/cache`

**Ceiling only.** The models were trained on D1, so their skill belongs to the D1
target and dividing it by this ceiling would compare two different problems. What
this section is for is the contrast in the ceiling itself.

| channel | h (frames) | MSE persistence | noise var | **ceiling** |
|---|---|---|---|---|
| F_R | 1 | 7.925e+06 | 4.845e+06 | **0.389** |
| F_R | 5 | 6.539e+07 | 4.845e+06 | **0.926** |
| F_R | 10 | 1.45e+08 | 4.845e+06 | **0.967** |
| F_R | 20 | 2.537e+08 | 4.845e+06 | **0.981** |
| F_R | 30 | 2.406e+08 | 4.845e+06 | **0.980** |
| CoPx_R | 1 | 4.652e-06 | 2.765e-06 | **0.406** |
| CoPx_R | 5 | 2.941e-05 | 2.765e-06 | **0.906** |
| CoPx_R | 10 | 6.139e-05 | 2.765e-06 | **0.955** |
| CoPx_R | 20 | 0.0001023 | 2.765e-06 | **0.973** |
| CoPx_R | 30 | 8.334e-05 | 2.765e-06 | **0.967** |
| CoPy_R | 1 | 3.998e-06 | 2.254e-06 | **0.436** |
| CoPy_R | 5 | 1.72e-05 | 2.254e-06 | **0.869** |
| CoPy_R | 10 | 2.992e-05 | 2.254e-06 | **0.925** |
| CoPy_R | 20 | 4.427e-05 | 2.254e-06 | **0.949** |
| CoPy_R | 30 | 4.017e-05 | 2.254e-06 | **0.944** |

Read against section 2: uncorrected, the ceiling reaches **0.98** by h=30, because
persistence's MSE is inflated ~500x by DC drift while the apparent noise floor is
not. The uncorrected signal therefore looks almost perfectly predictable and the
models look far short of it. Both impressions are the DC offset. After correction
the same horizon's ceiling is 0.67, and the models sit at 51-63% of it -- a real
gap, an order of magnitude smaller than the artefact suggested.

## 3. What it means

**Short horizons are nearly exhausted.** At h=1 the models reach 78-89% of a
ceiling that is itself deliberately conservative. The 0.05-0.10 left is
frame-scale noise, which no forecaster can take.

**Long horizons are not.** The ceiling climbs to 0.62-0.72 by h=30 -- persistence
degrades over a second while the noise floor does not -- but observed skill stays
flat, so the models capture only 56-77% of it. The gap is 0.15-0.30.

**The gap has a named cause.** The 2026-08-20 forecast figures show every model
emitting a nearly flat line: they predict the local mean OF THE PAST. The ceiling
credits predicting the smooth component AT t+h. Nobody is modelling how the slow
envelope evolves, which is where the remaining 0.15-0.30 sits. That calls for a
slow/fast decomposition, not a bigger network and not a regulariser.

**The bound is conservative in one stated direction.** v_e ~ (1-r(1))*var(x)
assumes the smooth part is perfectly correlated frame to frame. If it is not, the
noise is overestimated and the true ceiling is HIGHER -- so the reported headroom
is a floor on how much room remains, never a flattering estimate.

**Caveat on the skill column.** These are the driver's frame-pooled
`SS_vs_persistence`. The report script's per-clip-equal-weight `skill` differs
(ar F: 0.367 vs 0.302) and the two must not be quoted interchangeably; the
frame-pooled one is used here because the ceiling is also computed over frames.
