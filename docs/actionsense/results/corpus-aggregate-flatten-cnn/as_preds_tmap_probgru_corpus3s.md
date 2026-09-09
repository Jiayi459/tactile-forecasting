### ActionSense - as_preds_tmap_probgru_corpus3s

290 recordings scored. Ranked by **R²**, high → low. Hausdorff is **lower = better**; `HD ratio` < 1 beats persistence.

**model `probgru_aggregate`** — whole dataset (290 recordings): R² **0.6374**, skill **-0.7225**, skill (frame-pooled, OpenTouch's estimator) **-0.3716**, Hausdorff **2.627** (0.901× persistence). This is measured against the whole dataset's mean, so it is **not** the average of the rows below, each of which uses its own action's mean.

per action:

| # | action | n | R² | skill vs pers | Hausdorff | HD ratio |
|---:|---|---:|---:|---:|---:|---:|
| 1 | slice | 45 | **0.7291** | 0.1153 | 2.552 | 0.882 |
| 2 | peel | 30 | **0.6822** | 0.1916 | 2.464 | 0.847 |
| 3 | spread | 30 | **0.6277** | -0.0162 | 2.494 | 0.851 |
| 4 | clear | 28 | **0.6137** | 0.1257 | 2.582 | 0.882 |
| 5 | set | 6 | **0.5752** | 0.0732 | 2.608 | 0.878 |
| 6 | get/replace | 15 | **0.5473** | 0.1790 | 2.490 | 0.849 |
| 7 | load | 5 | **0.5284** | 0.0775 | 2.699 | 0.906 |
| 8 | get | 30 | **0.5147** | 0.1947 | 2.563 | 0.871 |
| 9 | unload | 5 | **0.4755** | 0.2116 | 2.547 | 0.861 |
| 10 | stack | 5 | **0.4723** | 0.2678 | 2.416 | 0.816 |
| 11 | clean | 55 | **0.4693** | -0.9339 | 2.991 | 1.033 |
| 12 | pour | 21 | **0.4378** | 0.0488 | 2.700 | 0.936 |
| 13 | open/close | 9 | **0.3527** | 0.1835 | 2.429 | 0.847 |
| 14 | open | 6 | **0.2682** | 0.1335 | 2.442 | 0.840 |

Lowest Hausdorff: **stack** (2.416); highest R²: **slice** (0.7291).

**model `probgru_cnn`** — whole dataset (290 recordings): R² **0.6417**, skill **-0.6635**, skill (frame-pooled, OpenTouch's estimator) **-0.3646**, Hausdorff **2.652** (0.910× persistence). This is measured against the whole dataset's mean, so it is **not** the average of the rows below, each of which uses its own action's mean.

per action:

| # | action | n | R² | skill vs pers | Hausdorff | HD ratio |
|---:|---|---:|---:|---:|---:|---:|
| 1 | slice | 45 | **0.7176** | 0.0879 | 2.565 | 0.886 |
| 2 | peel | 30 | **0.6552** | 0.1262 | 2.537 | 0.872 |
| 3 | spread | 30 | **0.6313** | -0.0017 | 2.524 | 0.861 |
| 4 | clear | 28 | **0.6058** | 0.1008 | 2.622 | 0.895 |
| 5 | set | 6 | **0.5556** | 0.0284 | 2.683 | 0.903 |
| 6 | load | 5 | **0.5281** | 0.0815 | 2.684 | 0.901 |
| 7 | get/replace | 15 | **0.5192** | 0.1254 | 2.526 | 0.862 |
| 8 | get | 30 | **0.5003** | 0.1712 | 2.592 | 0.881 |
| 9 | clean | 55 | **0.4810** | -0.8275 | 2.963 | 1.023 |
| 10 | unload | 5 | **0.4523** | 0.1721 | 2.656 | 0.898 |
| 11 | pour | 21 | **0.4479** | 0.0532 | 2.742 | 0.951 |
| 12 | stack | 5 | **0.4245** | 0.2082 | 2.505 | 0.845 |
| 13 | open/close | 9 | **0.3126** | 0.1421 | 2.453 | 0.856 |
| 14 | open | 6 | **0.2519** | 0.1179 | 2.472 | 0.850 |

Lowest Hausdorff: **open/close** (2.453); highest R²: **slice** (0.7176).

**model `probgru_flatten`** — whole dataset (290 recordings): R² **0.6266**, skill **-0.7111**, skill (frame-pooled, OpenTouch's estimator) **-0.4004**, Hausdorff **2.839** (0.974× persistence). This is measured against the whole dataset's mean, so it is **not** the average of the rows below, each of which uses its own action's mean.

per action:

| # | action | n | R² | skill vs pers | Hausdorff | HD ratio |
|---:|---|---:|---:|---:|---:|---:|
| 1 | slice | 45 | **0.7012** | 0.0308 | 2.744 | 0.948 |
| 2 | spread | 30 | **0.6297** | 0.0181 | 2.674 | 0.912 |
| 3 | peel | 30 | **0.6295** | 0.0578 | 2.764 | 0.950 |
| 4 | clear | 28 | **0.5813** | 0.0348 | 2.895 | 0.989 |
| 5 | set | 6 | **0.5415** | 0.0127 | 2.800 | 0.942 |
| 6 | get/replace | 15 | **0.5068** | 0.1092 | 2.702 | 0.922 |
| 7 | load | 5 | **0.4998** | 0.0292 | 2.899 | 0.974 |
| 8 | get | 30 | **0.4821** | 0.1406 | 2.758 | 0.937 |
| 9 | clean | 55 | **0.4578** | -0.9478 | 3.131 | 1.081 |
| 10 | unload | 5 | **0.4475** | 0.1657 | 2.770 | 0.937 |
| 11 | stack | 5 | **0.3891** | 0.1526 | 2.717 | 0.917 |
| 12 | pour | 21 | **0.3441** | -0.0790 | 2.974 | 1.031 |
| 13 | open/close | 9 | **0.2739** | 0.0971 | 2.640 | 0.921 |
| 14 | open | 6 | **0.2530** | 0.1222 | 2.547 | 0.876 |

Lowest Hausdorff: **open** (2.547); highest R²: **slice** (0.7012).


channels: F_L, CoPx_L, CoPy_L, F_R, CoPx_R, CoPy_R · CoP masking: `--mask none`
