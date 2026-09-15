### EgoTouch test_unseen horizon h2p0 (3 s history, shared 3 s origins)

50 recordings scored. Ranked by **R²**, high → low. Hausdorff is **lower = better**; `HD ratio` < 1 beats persistence.

**model `aggregate_probgru`** — whole dataset (50 recordings): R² **0.3925**, skill **+0.2713**, skill (frame-pooled, OpenTouch's estimator) **+0.2900**, Hausdorff **3.492** (1.143× persistence). This is measured against the whole dataset's mean, so it is **not** the average of the rows below, each of which uses its own action's mean.

per action:

| # | action | n | R² | skill vs pers | Hausdorff | HD ratio |
|---:|---|---:|---:|---:|---:|---:|
| 1 | pick | 10 | **0.3705** | 0.2806 | 3.380 | 1.094 |
| 2 | open | 20 | **0.3565** | 0.2557 | 4.268 | 1.392 |
| 3 | rotate | 6 | **0.2633** | 0.1849 | 3.156 | 1.043 |
| 4 | fold | 9 | **0.1407** | 0.3400 | 2.451 | 0.821 |
| 5 | put | 5 | **0.0366** | 0.1407 | 2.890 | 0.929 |

Lowest Hausdorff: **fold** (2.451); highest R²: **pick** (0.3705).

**model `aggregate_seq2seq`** — whole dataset (50 recordings): R² **0.3961**, skill **+0.2759**, skill (frame-pooled, OpenTouch's estimator) **+0.2913**, Hausdorff **3.140** (1.027× persistence). This is measured against the whole dataset's mean, so it is **not** the average of the rows below, each of which uses its own action's mean.

per action:

| # | action | n | R² | skill vs pers | Hausdorff | HD ratio |
|---:|---|---:|---:|---:|---:|---:|
| 1 | open | 20 | **0.3660** | 0.2662 | 3.505 | 1.143 |
| 2 | pick | 10 | **0.3521** | 0.2594 | 3.516 | 1.138 |
| 3 | rotate | 6 | **0.2956** | 0.2296 | 2.665 | 0.880 |
| 4 | fold | 9 | **0.1227** | 0.3262 | 2.403 | 0.804 |
| 5 | put | 5 | **0.0713** | 0.1737 | 2.819 | 0.906 |

Lowest Hausdorff: **fold** (2.403); highest R²: **open** (0.3660).


channels: F_L, CoPx_L, CoPy_L, F_R, CoPx_R, CoPy_R · CoP masking: `--thresholds mask_thresholds.json` (TRAIN-fitted)
