# One backbone, three datasets — what must match and what may not

The probGRU arm exists so the same forecaster can be run on more than one tactile sensor and
the results compared. That only works if the differences between the three implementations
are the ones the DATA forces, and not ones that crept in. This separates the two.

## Status

| | ActionSense `action_dynamics` | ActionSense harness arm | OpenTouch `prob_gru` | d256 |
|---|---|---|---|---|
| exists | yes (2026-08 earlier) | **to build** | yes | **no model at all** |
| harness-scored | no | yes | yes | — |
| target | FAST (high-passed) | RAW | RAW | — |

d256 is at the inventory stage: `src/d256.py` reads it, `scripts/d256/` probes it, and there
is no forecaster. Its structure already constrains what one can look like (§3).

## 1. The backbone — IDENTICAL, and must stay so

Any change here breaks the comparison and is not a dataset accommodation.

- per-frame input → **encoder GRU**
- **8-dim action embedding**, vocabulary built from TRAIN only, rare actions folded into a
  reserved `other` id
- **autoregressive decoder GRU**, seeded with the last observed target, its own mean fed back
  each step
- **mu and log-variance heads** over `[decoder state ; action embedding]`, logvar clamped to
  `[-6, 4]`
- **Gaussian NLL** as the loss
- prediction is the **ABSOLUTE** target, z-scored by the harness's TRAIN-fitted Norm — not a
  residual over persistence, which would hand one arm a persistence prior the others lack
- windows are the **harness origins**, early ones LEFT zero-padded so a forecast exists at
  every origin the scorer will ask about
- weights selected on **VAL only**; TEST untouched
- reported as **skill vs persistence** and **Hausdorff** (`src/shape_metrics.py`, one
  implementation shared by both sensors)

## 2. What the data forces to differ — legitimate, and each one stated

| | ActionSense | OpenTouch | d256 |
|---|---|---|---|
| rate | 10 Hz (30 ÷ `downsample` 3) | 30 Hz | **unknown** (§3) |
| 1 s horizon | **10 steps** | **30 steps** | undefined without a rate |
| channels `n_out` | 6 (both gloves) | 3 (one glove) | 6 (both gloves) |
| taxel grid | 32×32 × 2 gloves | 16×16 × 1 glove | 32×32 × 2 gloves |
| median recording | long | **2.80 s** | 201 frames reconstructed |
| history sweep | 1/3/10 s | 1/2/3 s (clip length caps it) | ≤48-frame budget keeps 87/94 |
| action vocabulary | 14 verbs, all audited | 66 strings, 30 audited | 20 ActionSense activities |
| split unit | recording, 5-fold | **location**, 4-fold | recording (94), forced |

**The horizon depth is the one to watch.** A one-second forecast is ten autoregressive steps
on ActionSense and thirty on OpenTouch, so the decoder compounds its own error three times as
far on OpenTouch for the same physical lookahead. Any cross-sensor skill difference carries
that confound, and it cannot be removed without either changing the physical horizon or
resampling one sensor to the other's rate. State it wherever the two are compared.

**The split unit is the second.** OpenTouch holds out whole LOCATIONS; ActionSense holds out
recordings, which share environment, subject and habits with the training set. OpenTouch's
numbers are therefore produced under a strictly harder generalisation demand, and are not
flattered by the comparison.

## 3. d256: what its structure will force

- **A clip-level random split leaks catastrophically** — neighbouring clips are 15/16
  identical. The only sound unit is the recording, of which there are 94.
- **`signals`/`signals2` are not extra data**, they are `signals1` decimated by 3 and 2. Use
  `signals1` and reproduce the rest with the harness's existing `downsample` knob.
- **The 16-frame clip length is not the real constraint**: reconstructed recordings have a
  median of 201 frames, and a 48-frame budget (history + horizon) keeps 87/94 recordings and
  25,447 origins.
- **The sampling rate is unknown and not recoverable from the archive** — clips carry no
  timestamps, only a relative 1:2:3 stride between the three groups. Until it is resolved,
  `causal_velocity(sig, fps)` cannot be given an honest fps, "one second" cannot be defined,
  and neither a horizon nor a velocity feature should be trusted. This blocks a d256 arm more
  than any modelling question does.

## 4. Differences that are NOT justified and should be removed

- `action_dynamics.ProbGRU` hardcodes **3** decoder inputs and **3** outputs, so it cannot
  express ActionSense's 6 channels. The OpenTouch fork already parameterises `n_out`; the new
  arm uses the parameterised version rather than a fourth copy.
- `action_dynamics` targets FAST while everything harness-scored targets RAW. The new arm is
  RAW, which is not a preference: **the frozen harness scores RAW and nothing else**, so a
  FAST arm cannot appear in the comparison table at all.
