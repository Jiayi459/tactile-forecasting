### EgoTouch test_unseen horizon h1p0 (3 s history, shared 3 s origins)

50 recordings scored. Ranked by **R²**, high → low. Hausdorff is **lower = better**; `HD ratio` < 1 beats persistence.

**model `aggregate_probgru`** — whole dataset (50 recordings): R² **0.5280**, skill **+0.2576**, skill (frame-pooled, OpenTouch's estimator) **+0.2736**, Hausdorff **3.439** (1.142× persistence). This is measured against the whole dataset's mean, so it is **not** the average of the rows below, each of which uses its own action's mean.

per action:

| # | action | n | R² | skill vs pers | Hausdorff | HD ratio |
|---:|---|---:|---:|---:|---:|---:|
| 1 | pick | 10 | **0.5234** | 0.2578 | 3.439 | 1.142 |
| 2 | open | 20 | **0.5114** | 0.2529 | 3.758 | 1.240 |
| 3 | rotate | 6 | **0.3607** | 0.2144 | 3.089 | 1.087 |
| 4 | put | 5 | **0.3576** | 0.0031 | 3.918 | 1.219 |
| 5 | fold | 9 | **0.2650** | 0.3149 | 2.698 | 0.911 |

Lowest Hausdorff: **fold** (2.698); highest R²: **pick** (0.5234).

**model `aggregate_seq2seq`** — whole dataset (50 recordings): R² **0.5109**, skill **+0.2327**, skill (frame-pooled, OpenTouch's estimator) **+0.2450**, Hausdorff **3.385** (1.124× persistence). This is measured against the whole dataset's mean, so it is **not** the average of the rows below, each of which uses its own action's mean.

per action:

| # | action | n | R² | skill vs pers | Hausdorff | HD ratio |
|---:|---|---:|---:|---:|---:|---:|
| 1 | pick | 10 | **0.5035** | 0.2292 | 3.310 | 1.099 |
| 2 | open | 20 | **0.4879** | 0.2116 | 4.011 | 1.323 |
| 3 | put | 5 | **0.3885** | 0.0637 | 3.390 | 1.054 |
| 4 | rotate | 6 | **0.3509** | 0.2150 | 2.517 | 0.886 |
| 5 | fold | 9 | **0.2405** | 0.2938 | 2.653 | 0.896 |

Lowest Hausdorff: **rotate** (2.517); highest R²: **pick** (0.5035).


channels: F_L, CoPx_L, CoPy_L, F_R, CoPx_R, CoPy_R · CoP masking: `--thresholds mask_thresholds.json` (TRAIN-fitted)
