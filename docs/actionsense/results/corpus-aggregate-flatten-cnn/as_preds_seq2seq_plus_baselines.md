### ActionSense - as_preds_seq2seq_plus_baselines

290 recordings scored. Ranked by **R²**, high → low. Hausdorff is **lower = better**; `HD ratio` < 1 beats persistence.

**model `ar`** — whole dataset (290 recordings): R² **0.7476**, skill **+0.1462**, skill (frame-pooled, OpenTouch's estimator) **+0.1576**, Hausdorff **2.543** (0.873× persistence). This is measured against the whole dataset's mean, so it is **not** the average of the rows below, each of which uses its own action's mean.

per action:

| # | action | n | R² | skill vs pers | Hausdorff | HD ratio |
|---:|---|---:|---:|---:|---:|---:|
| 1 | slice | 45 | **0.7389** | 0.1481 | 2.416 | 0.835 |
| 2 | peel | 30 | **0.6789** | 0.1899 | 2.431 | 0.835 |
| 3 | spread | 30 | **0.6581** | 0.1039 | 2.441 | 0.833 |
| 4 | clean | 55 | **0.6561** | 0.1467 | 2.692 | 0.930 |
| 5 | clear | 28 | **0.6174** | 0.1252 | 2.543 | 0.868 |
| 6 | set | 6 | **0.6069** | 0.1849 | 2.555 | 0.860 |
| 7 | load | 5 | **0.5728** | 0.1680 | 2.559 | 0.859 |
| 8 | get/replace | 15 | **0.5443** | 0.1678 | 2.461 | 0.840 |
| 9 | get | 30 | **0.5048** | 0.1816 | 2.539 | 0.862 |
| 10 | pour | 21 | **0.4920** | 0.1292 | 2.845 | 0.986 |
| 11 | unload | 5 | **0.4569** | 0.1815 | 2.583 | 0.873 |
| 12 | stack | 5 | **0.3866** | 0.1524 | 2.582 | 0.872 |
| 13 | open | 6 | **0.3709** | 0.2566 | 2.253 | 0.775 |
| 14 | open/close | 9 | **0.3230** | 0.1198 | 2.578 | 0.899 |

Lowest Hausdorff: **open** (2.253); highest R²: **slice** (0.7389).

**model `seasonal`** — whole dataset (290 recordings): R² **0.6897**, skill **-0.0084**, skill (frame-pooled, OpenTouch's estimator) **-0.0208**, Hausdorff **2.930** (1.005× persistence). This is measured against the whole dataset's mean, so it is **not** the average of the rows below, each of which uses its own action's mean.

per action:

| # | action | n | R² | skill vs pers | Hausdorff | HD ratio |
|---:|---|---:|---:|---:|---:|---:|
| 1 | slice | 45 | **0.6873** | 0.0000 | 2.895 | 1.000 |
| 2 | peel | 30 | **0.6034** | 0.0000 | 2.910 | 1.000 |
| 3 | spread | 30 | **0.5907** | 0.0000 | 2.932 | 1.000 |
| 4 | clean | 55 | **0.5669** | 0.0000 | 2.896 | 1.000 |
| 5 | clear | 28 | **0.5544** | 0.0000 | 2.928 | 1.000 |
| 6 | load | 5 | **0.4855** | 0.0000 | 2.978 | 1.000 |
| 7 | get/replace | 15 | **0.4377** | 0.0000 | 2.931 | 1.000 |
| 8 | set | 6 | **0.4374** | -0.1113 | 3.242 | 1.091 |
| 9 | pour | 21 | **0.3967** | 0.0000 | 2.884 | 1.000 |
| 10 | get | 30 | **0.3951** | 0.0000 | 2.944 | 1.000 |
| 11 | unload | 5 | **0.3327** | 0.0000 | 2.957 | 1.000 |
| 12 | open/close | 9 | **0.1916** | 0.0000 | 2.866 | 1.000 |
| 13 | open | 6 | **0.1419** | 0.0000 | 2.906 | 1.000 |
| 14 | stack | 5 | **-0.0985** | -0.5479 | 3.531 | 1.192 |

Lowest Hausdorff: **open/close** (2.866); highest R²: **slice** (0.6873).

**model `seq2seq_aggregate`** — whole dataset (290 recordings): R² **0.7414**, skill **+0.1276**, skill (frame-pooled, OpenTouch's estimator) **+0.1453**, Hausdorff **2.419** (0.830× persistence). This is measured against the whole dataset's mean, so it is **not** the average of the rows below, each of which uses its own action's mean.

per action:

| # | action | n | R² | skill vs pers | Hausdorff | HD ratio |
|---:|---|---:|---:|---:|---:|---:|
| 1 | slice | 45 | **0.7361** | 0.1400 | 2.373 | 0.820 |
| 2 | peel | 30 | **0.6808** | 0.1910 | 2.361 | 0.811 |
| 3 | spread | 30 | **0.6501** | 0.1049 | 2.391 | 0.815 |
| 4 | clean | 55 | **0.6339** | 0.1099 | 2.343 | 0.809 |
| 5 | clear | 28 | **0.6132** | 0.1177 | 2.501 | 0.854 |
| 6 | set | 6 | **0.5966** | 0.1611 | 2.496 | 0.840 |
| 7 | load | 5 | **0.5673** | 0.1546 | 2.508 | 0.842 |
| 8 | get/replace | 15 | **0.5376** | 0.1583 | 2.435 | 0.831 |
| 9 | get | 30 | **0.4960** | 0.1651 | 2.442 | 0.830 |
| 10 | stack | 5 | **0.4625** | 0.2537 | 2.325 | 0.785 |
| 11 | unload | 5 | **0.4531** | 0.1821 | 2.440 | 0.825 |
| 12 | pour | 21 | **0.4137** | 0.0121 | 2.747 | 0.952 |
| 13 | open/close | 9 | **0.3475** | 0.1781 | 2.283 | 0.796 |
| 14 | open | 6 | **0.2594** | 0.1237 | 2.347 | 0.808 |

Lowest Hausdorff: **open/close** (2.283); highest R²: **slice** (0.7361).

**model `seq2seq_cnn`** — whole dataset (290 recordings): R² **0.7033**, skill **+0.0346**, skill (frame-pooled, OpenTouch's estimator) **+0.0442**, Hausdorff **2.554** (0.876× persistence). This is measured against the whole dataset's mean, so it is **not** the average of the rows below, each of which uses its own action's mean.

per action:

| # | action | n | R² | skill vs pers | Hausdorff | HD ratio |
|---:|---|---:|---:|---:|---:|---:|
| 1 | slice | 45 | **0.6986** | 0.0385 | 2.509 | 0.867 |
| 2 | peel | 30 | **0.6402** | 0.0866 | 2.504 | 0.860 |
| 3 | spread | 30 | **0.5997** | 0.0146 | 2.543 | 0.867 |
| 4 | clean | 55 | **0.5700** | 0.0010 | 2.518 | 0.869 |
| 5 | clear | 28 | **0.5609** | 0.0067 | 2.675 | 0.914 |
| 6 | set | 6 | **0.5371** | 0.0630 | 2.589 | 0.872 |
| 7 | load | 5 | **0.5126** | 0.0558 | 2.633 | 0.884 |
| 8 | get/replace | 15 | **0.4576** | 0.0364 | 2.598 | 0.886 |
| 9 | pour | 21 | **0.4266** | 0.0315 | 2.550 | 0.884 |
| 10 | get | 30 | **0.4192** | 0.0430 | 2.589 | 0.879 |
| 11 | unload | 5 | **0.3674** | 0.0474 | 2.599 | 0.879 |
| 12 | stack | 5 | **0.3511** | 0.1125 | 2.574 | 0.869 |
| 13 | open/close | 9 | **0.2186** | 0.0227 | 2.556 | 0.892 |
| 14 | open | 6 | **0.1471** | 0.0002 | 2.526 | 0.869 |

Lowest Hausdorff: **peel** (2.504); highest R²: **slice** (0.6986).

**model `seq2seq_flatten`** — whole dataset (290 recordings): R² **0.6946**, skill **+0.0072**, skill (frame-pooled, OpenTouch's estimator) **+0.0084**, Hausdorff **2.554** (0.876× persistence). This is measured against the whole dataset's mean, so it is **not** the average of the rows below, each of which uses its own action's mean.

per action:

| # | action | n | R² | skill vs pers | Hausdorff | HD ratio |
|---:|---|---:|---:|---:|---:|---:|
| 1 | slice | 45 | **0.6818** | -0.0174 | 2.485 | 0.859 |
| 2 | peel | 30 | **0.6208** | 0.0424 | 2.463 | 0.846 |
| 3 | spread | 30 | **0.5881** | -0.0314 | 2.539 | 0.866 |
| 4 | clean | 55 | **0.5629** | -0.0067 | 2.427 | 0.838 |
| 5 | clear | 28 | **0.5451** | -0.0321 | 2.632 | 0.899 |
| 6 | set | 6 | **0.5122** | 0.0151 | 2.725 | 0.917 |
| 7 | load | 5 | **0.4943** | 0.0186 | 2.740 | 0.920 |
| 8 | get/replace | 15 | **0.4101** | -0.0598 | 2.787 | 0.951 |
| 9 | get | 30 | **0.3972** | 0.0047 | 2.650 | 0.900 |
| 10 | pour | 21 | **0.3564** | -0.0646 | 2.684 | 0.930 |
| 11 | unload | 5 | **0.3402** | 0.0107 | 2.648 | 0.895 |
| 12 | stack | 5 | **0.3194** | 0.0670 | 2.593 | 0.875 |
| 13 | open/close | 9 | **0.1704** | -0.0410 | 2.498 | 0.871 |
| 14 | open | 6 | **0.0961** | -0.0665 | 2.525 | 0.869 |

Lowest Hausdorff: **clean** (2.427); highest R²: **slice** (0.6818).


channels: F_L, CoPx_L, CoPy_L, F_R, CoPx_R, CoPy_R · CoP masking: `--mask none`
