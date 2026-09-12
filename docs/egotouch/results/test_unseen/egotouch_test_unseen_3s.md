### EgoTouch test_unseen 3s history

56 recordings scored. Ranked by **R²**, high → low. Hausdorff is **lower = better**; `HD ratio` < 1 beats persistence.

**model `aggregate_probgru`** — whole dataset (56 recordings): R² **0.4908**, skill **+0.2244**, skill (frame-pooled, OpenTouch's estimator) **+0.2595**, Hausdorff **3.935** (1.298× persistence). This is measured against the whole dataset's mean, so it is **not** the average of the rows below, each of which uses its own action's mean.

per action:

| # | action | n | R² | skill vs pers | Hausdorff | HD ratio |
|---:|---|---:|---:|---:|---:|---:|
| 1 | open | 20 | **0.5265** | 0.2319 | 4.332 | 1.421 |
| 2 | pick | 10 | **0.5026** | 0.2450 | 3.600 | 1.194 |
| 3 | put | 5 | **0.3831** | 0.0680 | 4.534 | 1.412 |
| 4 | rotate | 6 | **0.3262** | 0.1757 | 3.028 | 1.066 |
| 5 | fold | 9 | **0.2850** | 0.3015 | 3.136 | 1.051 |
| 6 | wring | 6 | **-1.9512** | -0.3893 | 4.779 | 1.533 |

Lowest Hausdorff: **rotate** (3.028); highest R²: **open** (0.5265).

**model `aggregate_seq2seq`** — whole dataset (56 recordings): R² **0.4807**, skill **+0.2264**, skill (frame-pooled, OpenTouch's estimator) **+0.2473**, Hausdorff **3.607** (1.190× persistence). This is measured against the whole dataset's mean, so it is **not** the average of the rows below, each of which uses its own action's mean.

per action:

| # | action | n | R² | skill vs pers | Hausdorff | HD ratio |
|---:|---|---:|---:|---:|---:|---:|
| 1 | open | 20 | **0.5259** | 0.2322 | 4.178 | 1.371 |
| 2 | pick | 10 | **0.4976** | 0.2412 | 3.225 | 1.070 |
| 3 | put | 5 | **0.3880** | 0.0699 | 4.154 | 1.294 |
| 4 | rotate | 6 | **0.3323** | 0.1900 | 2.588 | 0.911 |
| 5 | fold | 9 | **0.2603** | 0.2791 | 3.010 | 1.009 |
| 6 | wring | 6 | **-1.5242** | -0.0683 | 3.801 | 1.219 |

Lowest Hausdorff: **rotate** (2.588); highest R²: **open** (0.5259).

**model `ar_global`** — whole dataset (56 recordings): R² **0.4607**, skill **+0.1743**, skill (frame-pooled, OpenTouch's estimator) **+0.2303**, Hausdorff **4.407** (1.454× persistence). This is measured against the whole dataset's mean, so it is **not** the average of the rows below, each of which uses its own action's mean.

per action:

| # | action | n | R² | skill vs pers | Hausdorff | HD ratio |
|---:|---|---:|---:|---:|---:|---:|
| 1 | open | 20 | **0.5056** | 0.1978 | 5.251 | 1.723 |
| 2 | pick | 10 | **0.4803** | 0.2068 | 4.124 | 1.368 |
| 3 | rotate | 6 | **0.3788** | 0.2431 | 2.682 | 0.944 |
| 4 | put | 5 | **0.3748** | 0.0466 | 4.784 | 1.490 |
| 5 | fold | 9 | **0.2734** | 0.2907 | 3.089 | 1.035 |
| 6 | wring | 6 | **-2.7749** | -0.5258 | 5.455 | 1.749 |

Lowest Hausdorff: **rotate** (2.682); highest R²: **open** (0.5056).

**model `ar_group`** — whole dataset (56 recordings): R² **0.4533**, skill **+0.1632**, skill (frame-pooled, OpenTouch's estimator) **+0.2023**, Hausdorff **4.723** (1.558× persistence). This is measured against the whole dataset's mean, so it is **not** the average of the rows below, each of which uses its own action's mean.

per action:

| # | action | n | R² | skill vs pers | Hausdorff | HD ratio |
|---:|---|---:|---:|---:|---:|---:|
| 1 | open | 20 | **0.5056** | 0.1978 | 5.251 | 1.723 |
| 2 | pick | 10 | **0.4261** | 0.1150 | 5.895 | 1.955 |
| 3 | rotate | 6 | **0.3788** | 0.2431 | 2.682 | 0.944 |
| 4 | put | 5 | **0.3748** | 0.0466 | 4.784 | 1.490 |
| 5 | fold | 9 | **0.2734** | 0.2907 | 3.089 | 1.035 |
| 6 | wring | 6 | **-2.7749** | -0.5258 | 5.455 | 1.749 |

Lowest Hausdorff: **rotate** (2.682); highest R²: **open** (0.5056).

**model `cnn_probgru`** — whole dataset (56 recordings): R² **0.4628**, skill **+0.1904**, skill (frame-pooled, OpenTouch's estimator) **+0.2400**, Hausdorff **3.925** (1.295× persistence). This is measured against the whole dataset's mean, so it is **not** the average of the rows below, each of which uses its own action's mean.

per action:

| # | action | n | R² | skill vs pers | Hausdorff | HD ratio |
|---:|---|---:|---:|---:|---:|---:|
| 1 | open | 20 | **0.5211** | 0.2142 | 4.301 | 1.411 |
| 2 | pick | 10 | **0.4949** | 0.2395 | 3.339 | 1.108 |
| 3 | put | 5 | **0.4002** | 0.0904 | 4.087 | 1.273 |
| 4 | fold | 9 | **0.2723** | 0.2920 | 3.129 | 1.049 |
| 5 | rotate | 6 | **0.0968** | -0.0062 | 3.711 | 1.306 |
| 6 | wring | 6 | **-1.7208** | -0.3929 | 4.922 | 1.579 |

Lowest Hausdorff: **fold** (3.129); highest R²: **open** (0.5211).

**model `cnn_seq2seq`** — whole dataset (56 recordings): R² **0.4636**, skill **+0.2078**, skill (frame-pooled, OpenTouch's estimator) **+0.2028**, Hausdorff **3.694** (1.219× persistence). This is measured against the whole dataset's mean, so it is **not** the average of the rows below, each of which uses its own action's mean.

per action:

| # | action | n | R² | skill vs pers | Hausdorff | HD ratio |
|---:|---|---:|---:|---:|---:|---:|
| 1 | open | 20 | **0.5012** | 0.2004 | 4.143 | 1.359 |
| 2 | pick | 10 | **0.4514** | 0.1746 | 3.528 | 1.170 |
| 3 | put | 5 | **0.3945** | 0.0788 | 3.961 | 1.234 |
| 4 | rotate | 6 | **0.2410** | 0.0951 | 2.719 | 0.957 |
| 5 | fold | 9 | **0.2099** | 0.2349 | 3.167 | 1.061 |
| 6 | wring | 6 | **-1.2260** | 0.0140 | 4.019 | 1.289 |

Lowest Hausdorff: **rotate** (2.719); highest R²: **open** (0.5012).

**model `flatten_probgru`** — whole dataset (56 recordings): R² **0.4540**, skill **+0.1750**, skill (frame-pooled, OpenTouch's estimator) **+0.2106**, Hausdorff **4.132** (1.363× persistence). This is measured against the whole dataset's mean, so it is **not** the average of the rows below, each of which uses its own action's mean.

per action:

| # | action | n | R² | skill vs pers | Hausdorff | HD ratio |
|---:|---|---:|---:|---:|---:|---:|
| 1 | open | 20 | **0.4850** | 0.1479 | 5.144 | 1.688 |
| 2 | pick | 10 | **0.4666** | 0.1763 | 3.905 | 1.295 |
| 3 | put | 5 | **0.3999** | 0.0795 | 4.469 | 1.392 |
| 4 | rotate | 6 | **0.3070** | 0.1622 | 2.816 | 0.991 |
| 5 | fold | 9 | **0.2536** | 0.2742 | 3.124 | 1.047 |
| 6 | wring | 6 | **-1.3446** | -0.1461 | 3.684 | 1.181 |

Lowest Hausdorff: **rotate** (2.816); highest R²: **open** (0.4850).

**model `flatten_seq2seq`** — whole dataset (56 recordings): R² **0.3968**, skill **+0.1203**, skill (frame-pooled, OpenTouch's estimator) **+0.1312**, Hausdorff **3.609** (1.190× persistence). This is measured against the whole dataset's mean, so it is **not** the average of the rows below, each of which uses its own action's mean.

per action:

| # | action | n | R² | skill vs pers | Hausdorff | HD ratio |
|---:|---|---:|---:|---:|---:|---:|
| 1 | open | 20 | **0.4500** | 0.1125 | 4.451 | 1.460 |
| 2 | pick | 10 | **0.4121** | 0.1172 | 3.339 | 1.108 |
| 3 | put | 5 | **0.3792** | 0.0744 | 3.584 | 1.116 |
| 4 | rotate | 6 | **0.2030** | 0.0809 | 2.758 | 0.971 |
| 5 | fold | 9 | **0.1315** | 0.1601 | 3.010 | 1.009 |
| 6 | wring | 6 | **-1.4803** | 0.0406 | 3.023 | 0.969 |

Lowest Hausdorff: **rotate** (2.758); highest R²: **open** (0.4500).

**model `seasonal_global`** — whole dataset (56 recordings): R² **0.3140**, skill **+0.0000**, skill (frame-pooled, OpenTouch's estimator) **+0.0000**, Hausdorff **3.031** (1.000× persistence). This is measured against the whole dataset's mean, so it is **not** the average of the rows below, each of which uses its own action's mean.

per action:

| # | action | n | R² | skill vs pers | Hausdorff | HD ratio |
|---:|---|---:|---:|---:|---:|---:|
| 1 | open | 20 | **0.3786** | 0.0000 | 3.048 | 1.000 |
| 2 | pick | 10 | **0.3298** | 0.0000 | 3.014 | 1.000 |
| 3 | put | 5 | **0.3258** | 0.0000 | 3.211 | 1.000 |
| 4 | rotate | 6 | **0.1153** | 0.0000 | 2.841 | 1.000 |
| 5 | fold | 9 | **-0.0326** | 0.0000 | 2.984 | 1.000 |
| 6 | wring | 6 | **-1.9896** | 0.0000 | 3.118 | 1.000 |

Lowest Hausdorff: **rotate** (2.841); highest R²: **open** (0.3786).

**model `seasonal_group`** — whole dataset (56 recordings): R² **0.3140**, skill **+0.0000**, skill (frame-pooled, OpenTouch's estimator) **+0.0000**, Hausdorff **3.031** (1.000× persistence). This is measured against the whole dataset's mean, so it is **not** the average of the rows below, each of which uses its own action's mean.

per action:

| # | action | n | R² | skill vs pers | Hausdorff | HD ratio |
|---:|---|---:|---:|---:|---:|---:|
| 1 | open | 20 | **0.3786** | 0.0000 | 3.048 | 1.000 |
| 2 | pick | 10 | **0.3298** | 0.0000 | 3.014 | 1.000 |
| 3 | put | 5 | **0.3258** | 0.0000 | 3.211 | 1.000 |
| 4 | rotate | 6 | **0.1153** | 0.0000 | 2.841 | 1.000 |
| 5 | fold | 9 | **-0.0326** | 0.0000 | 2.984 | 1.000 |
| 6 | wring | 6 | **-1.9896** | 0.0000 | 3.118 | 1.000 |

Lowest Hausdorff: **rotate** (2.841); highest R²: **open** (0.3786).


channels: F_L, CoPx_L, CoPy_L, F_R, CoPx_R, CoPy_R · CoP masking: `--thresholds mask_thresholds.json` (TRAIN-fitted)
