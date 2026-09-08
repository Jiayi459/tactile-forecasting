### ActionSense - as_preds_probgru_corpus

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


channels: F_L, CoPx_L, CoPy_L, F_R, CoPx_R, CoPy_R · CoP masking: `--mask none`
