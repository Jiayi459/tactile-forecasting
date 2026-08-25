# Tactile Action-Predictability Study — Summary of Work

Thorough record of what we investigated, how, and what we found, across the tactile datasets.
Companion to the taxonomy + numbers in [ACTION_CATEGORIES.md](ACTION_CATEGORIES.md). Newest
decisions at the bottom.

## 1. Question
Which *category / kind* of hand action is easiest to predict from its own past tactile signal —
and, more usefully, **what trait makes an action series predictable** — so a predictor can give a
user feedback / adaptive strategies to improve performance. Priors under test: actions with a
**standard procedure** and a **repeatable/periodic pattern** are easier.

## 2. Datasets examined
| dataset | tactile sensor | our use | labels |
|---|---|---|---|
| **EgoTouch** | 21×21 FPC grid, 2 hands, 30 Hz | probed (23 categories) + built GPU forecaster | verb_object task names |
| **OpenTouch** (arXiv 2512.16842) | 16×16 FPC, 1 hand, 30 Hz | probed (2,496 clips) | per-clip action + grip (GRASP taxonomy) |
| **ActionSense** (NeurIPS'22) | 32×32 conductive-thread, 2 hands, ~6 Hz | probed (299 clips, S00-S05) | 20 kitchen activities (Start/Stop intervals) |
| **Force–Vision** (ICLR'24) | STAG glove force map | categorized from paper only (not probed) | press / hold / squeeze on 89 tools |

## 3. Method
Two complementary tools, both sensor-agnostic:
- **Training-free predictability probe** (`src/tactile_pixel/predictability.py`): per clip,
  from the tactile (T,C,H,W) alone — `persistence_nMSE@h` (raw forecastability), `periodicity`
  (max total-force autocorrelation), `contact_migration` (1−IoU of active-taxel mask), and a
  z-scored composite `predictability_index (PI)`. Per-dataset probes segment/label clips and
  group by raw action / mapped category / temporal-pattern axis.
- **Trained forecaster** (`src/tactile_pixel/`): SimVP / ConvLSTM / ConvGRU predicting future
  pressure frames from past, scored as *skill over persistence*; `--category` filter + CRC sweep
  (`run_percategory.sh`) for per-category skill. (Run so far on EgoTouch only.)

Categorization: one shared verb→category taxonomy (`categories.py`) + a category→temporal-pattern
map (B1 periodic … B5 composite). Free-text/gerund labels normalized (`categorize_phrase`).

## 4. Results (per dataset)
- **EgoTouch (probe, 1,929 clips):** easiest Cut(slice)+6.0, Take, Inflate, Spray, Wash/Clean;
  hardest Press/Click −6.7, Plug/Insert, Pinch, Grasp/Hold/Lift. Holds are NOT trivially
  predictable (drift). Trained forecaster (earlier sessions): LTO +0.192, LOTO ~0 → pretrain
  lifts LOTO to +0.097.
- **OpenTouch (probe, 2,496 clips):** raw actions pour/serve/eat/stir/scoop/wipe most predictable;
  turn(latch)/pull/move least. `contact_migration≈0` (single-hand grasp footprint never breaks).
  A-priori temporal-pattern axis *inverts* vs EgoTouch — same verb differs by context.
- **ActionSense (probe, 299 clips):** Pour +2.6 > Cut(slice/peel) +1.8 > Wash/Clean > Fold(spread)
  > Organize(tableware) > Open/Close(jar) −2.6. Cleanest confirmation; a-priori pattern axis works
  (ramp > periodic > composite > transition). Resampled 6→30 Hz, so persH1 & migration degenerate;
  persH15/H30 + periodicity carry the signal.

## 5. Headline finding — a TRAIT, confirmed across 3 sensors
> **Predictable tactile = smooth, continuous, slowly-varying contact force**
> (pour, slice, wipe, peel, stir, scoop). **Unpredictable = abrupt onset / make-or-break
> engagement** (open/close jar, press/click, plug, stiff-latch turn).

`persH15` (decorrelation rate) is the sensor-agnostic predictor. The *category* ranking is
dataset-dependent (verb behaves differently by context); the *trait* is stable. Refinement from
ActionSense: **monotonic ramp (pour) > rhythmic cycle (slice) > sustained hold > transition** — a
cycle has force-reversal turning points a forecaster must anticipate; a ramp does not.

## 6. Artifacts
- Probes: `scripts/egotouch/probe_egotouch.py`, `scripts/opentouch/probe_opentouch.py`, `scripts/actionsense/probe_actionsense.py`
  (+ `scripts/crc/stream_actionsense.sh` streaming driver).
- Forecaster: library `src/actionsense/action_dynamics.py` (model+train+forecast); CLIs
  `scripts/actionsense/train_action_dynamics.py` (train->checkpoint) and `scripts/actionsense/plot_action_forecast.py` (plot).
- Shared metrics: `src/tactile_pixel/predictability.py`; taxonomy `src/tactile_pixel/categories.py`.
- Result CSVs: `docs/predictability_{by_category_full,opentouch,actionsense}.csv`.
- Forecaster + CRC jobs: `src/tactile_pixel/`, `scripts/crc/{percategory_gpu.job,run_percategory.sh}`.

## 7. Decisions & new direction (2026-07-03)
- **Drop EgoTouch going forward.** The usable hardware is the tactile glove behind the 3 linked
  datasets (ActionSense / OpenTouch / Force-Vision). EgoTouch stays only as historical reference.
- **Train a forecaster on the predictable, smooth-force actions** — slice, wipe (clean), pour,
  peel — all present in **ActionSense** (32×32 glove). Goal: a generative latent forecaster whose
  physically-interpretable latent supports user feedback.
- Detailed training design → [TACTILE_FORECAST_PLAN.md](TACTILE_FORECAST_PLAN.md).
