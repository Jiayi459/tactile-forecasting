### EgoTouch test_unseen horizon h0p5 (3 s history, shared 3 s origins)

50 recordings scored. Ranked by **R²**, high → low. Hausdorff is **lower = better**; `HD ratio` < 1 beats persistence.

**model `aggregate_probgru`** — whole dataset (50 recordings): R² **0.6448**, skill **+0.2300**, skill (frame-pooled, OpenTouch's estimator) **+0.2416**, Hausdorff **3.942** (1.223× persistence). This is measured against the whole dataset's mean, so it is **not** the average of the rows below, each of which uses its own action's mean.

per action:

| # | action | n | R² | skill vs pers | Hausdorff | HD ratio |
|---:|---|---:|---:|---:|---:|---:|
| 1 | put | 5 | **0.6615** | -0.0130 | 4.608 | 1.370 |
| 2 | pick | 10 | **0.6566** | 0.2340 | 3.593 | 1.148 |
| 3 | open | 20 | **0.6430** | 0.2165 | 4.494 | 1.360 |
| 4 | fold | 9 | **0.4151** | 0.2700 | 3.345 | 1.026 |
| 5 | rotate | 6 | **0.4126** | 0.1752 | 3.025 | 1.035 |

Lowest Hausdorff: **rotate** (3.025); highest R²: **put** (0.6615).

**model `aggregate_seq2seq`** — whole dataset (50 recordings): R² **0.6281**, skill **+0.1975**, skill (frame-pooled, OpenTouch's estimator) **+0.2054**, Hausdorff **4.105** (1.274× persistence). This is measured against the whole dataset's mean, so it is **not** the average of the rows below, each of which uses its own action's mean.

per action:

| # | action | n | R² | skill vs pers | Hausdorff | HD ratio |
|---:|---|---:|---:|---:|---:|---:|
| 1 | put | 5 | **0.6567** | -0.0219 | 4.353 | 1.294 |
| 2 | open | 20 | **0.6350** | 0.1980 | 5.099 | 1.543 |
| 3 | pick | 10 | **0.6330** | 0.1920 | 3.529 | 1.127 |
| 4 | fold | 9 | **0.3751** | 0.2254 | 3.262 | 1.001 |
| 5 | rotate | 6 | **0.3721** | 0.1784 | 2.808 | 0.961 |

Lowest Hausdorff: **rotate** (2.808); highest R²: **put** (0.6567).


channels: F_L, CoPx_L, CoPy_L, F_R, CoPx_R, CoPy_R · CoP masking: `--thresholds mask_thresholds.json` (TRAIN-fitted)
