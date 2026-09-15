### EgoTouch test_unseen horizon h3p0 (3 s history, shared 3 s origins)

50 recordings scored. Ranked by **R²**, high → low. Hausdorff is **lower = better**; `HD ratio` < 1 beats persistence.

**model `aggregate_probgru`** — whole dataset (50 recordings): R² **0.3098**, skill **+0.2747**, skill (frame-pooled, OpenTouch's estimator) **+0.2970**, Hausdorff **3.955** (1.266× persistence). This is measured against the whole dataset's mean, so it is **not** the average of the rows below, each of which uses its own action's mean.

per action:

| # | action | n | R² | skill vs pers | Hausdorff | HD ratio |
|---:|---|---:|---:|---:|---:|---:|
| 1 | pick | 10 | **0.2743** | 0.2711 | 3.470 | 1.091 |
| 2 | open | 20 | **0.2425** | 0.2191 | 5.460 | 1.747 |
| 3 | rotate | 6 | **0.2165** | 0.1960 | 3.106 | 0.977 |
| 4 | fold | 9 | **0.0826** | 0.3570 | 2.467 | 0.805 |
| 5 | put | 5 | **-0.0639** | 0.2080 | 2.603 | 0.855 |

Lowest Hausdorff: **fold** (2.467); highest R²: **pick** (0.2743).

**model `aggregate_seq2seq`** — whole dataset (50 recordings): R² **0.3343**, skill **+0.3027**, skill (frame-pooled, OpenTouch's estimator) **+0.3155**, Hausdorff **3.029** (0.970× persistence). This is measured against the whole dataset's mean, so it is **not** the average of the rows below, each of which uses its own action's mean.

per action:

| # | action | n | R² | skill vs pers | Hausdorff | HD ratio |
|---:|---|---:|---:|---:|---:|---:|
| 1 | open | 20 | **0.2858** | 0.2857 | 3.536 | 1.131 |
| 2 | pick | 10 | **0.2811** | 0.2905 | 3.111 | 0.978 |
| 3 | rotate | 6 | **0.2697** | 0.2523 | 2.576 | 0.810 |
| 4 | fold | 9 | **0.0772** | 0.3530 | 2.412 | 0.787 |
| 5 | put | 5 | **-0.0498** | 0.2264 | 2.494 | 0.819 |

Lowest Hausdorff: **fold** (2.412); highest R²: **open** (0.2858).


channels: F_L, CoPx_L, CoPy_L, F_R, CoPx_R, CoPy_R · CoP masking: `--thresholds mask_thresholds.json` (TRAIN-fitted)
