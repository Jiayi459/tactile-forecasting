### ActionSense - as_preds_probgru_corpus_h1

290 recordings scored. Ranked by **R²**, high → low. Hausdorff is **lower = better**; `HD ratio` < 1 beats persistence.

**model `probgru_aggregate`** — whole dataset (290 recordings): R² **0.6428**, skill **-0.7203**, skill (frame-pooled, OpenTouch's estimator) **-0.3800**, Hausdorff **2.629** (0.902× persistence). This is measured against the whole dataset's mean, so it is **not** the average of the rows below, each of which uses its own action's mean.

per action:

| # | action | n | R² | skill vs pers | Hausdorff | HD ratio |
|---:|---|---:|---:|---:|---:|---:|
| 1 | slice | 45 | **0.7292** | 0.1158 | 2.528 | 0.873 |
| 2 | peel | 30 | **0.6711** | 0.1644 | 2.471 | 0.849 |
| 3 | spread | 30 | **0.6216** | -0.0544 | 2.494 | 0.851 |
| 4 | clear | 28 | **0.6085** | 0.1036 | 2.642 | 0.902 |
| 5 | set | 6 | **0.5620** | 0.0323 | 2.670 | 0.899 |
| 6 | load | 5 | **0.5439** | 0.1093 | 2.666 | 0.895 |
| 7 | get/replace | 15 | **0.5331** | 0.1483 | 2.518 | 0.859 |
| 8 | get | 30 | **0.5149** | 0.1952 | 2.552 | 0.867 |
| 9 | unload | 5 | **0.4771** | 0.2123 | 2.580 | 0.873 |
| 10 | clean | 55 | **0.4723** | -0.9902 | 2.973 | 1.027 |
| 11 | stack | 5 | **0.4465** | 0.2361 | 2.476 | 0.836 |
| 12 | pour | 21 | **0.4209** | 0.0238 | 2.713 | 0.941 |
| 13 | open/close | 9 | **0.3467** | 0.1788 | 2.426 | 0.846 |
| 14 | open | 6 | **0.2917** | 0.1613 | 2.382 | 0.819 |

Lowest Hausdorff: **open** (2.382); highest R²: **slice** (0.7292).


channels: F_L, CoPx_L, CoPy_L, F_R, CoPx_R, CoPy_R · CoP masking: `--mask none`
