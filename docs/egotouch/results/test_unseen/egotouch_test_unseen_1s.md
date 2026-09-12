### EgoTouch test_unseen 1s history

56 recordings scored. Ranked by **R²**, high → low. Hausdorff is **lower = better**; `HD ratio` < 1 beats persistence.

**model `aggregate_probgru`** — whole dataset (56 recordings): R² **0.4783**, skill **+0.2185**, skill (frame-pooled, OpenTouch's estimator) **+0.2513**, Hausdorff **3.780** (1.247× persistence). This is measured against the whole dataset's mean, so it is **not** the average of the rows below, each of which uses its own action's mean.

per action:

| # | action | n | R² | skill vs pers | Hausdorff | HD ratio |
|---:|---|---:|---:|---:|---:|---:|
| 1 | open | 20 | **0.5223** | 0.2249 | 4.341 | 1.424 |
| 2 | pick | 10 | **0.5007** | 0.2412 | 3.370 | 1.118 |
| 3 | put | 5 | **0.4177** | 0.1162 | 4.087 | 1.273 |
| 4 | rotate | 6 | **0.3319** | 0.1362 | 2.875 | 1.012 |
| 5 | fold | 9 | **0.2838** | 0.3014 | 3.045 | 1.021 |
| 6 | wring | 6 | **-1.4822** | -0.0981 | 4.341 | 1.392 |

Lowest Hausdorff: **rotate** (2.875); highest R²: **open** (0.5223).

**model `aggregate_seq2seq`** — whole dataset (56 recordings): R² **0.4774**, skill **+0.2110**, skill (frame-pooled, OpenTouch's estimator) **+0.2359**, Hausdorff **3.692** (1.218× persistence). This is measured against the whole dataset's mean, so it is **not** the average of the rows below, each of which uses its own action's mean.

per action:

| # | action | n | R² | skill vs pers | Hausdorff | HD ratio |
|---:|---|---:|---:|---:|---:|---:|
| 1 | open | 20 | **0.5075** | 0.1919 | 4.432 | 1.454 |
| 2 | pick | 10 | **0.4922** | 0.2306 | 3.206 | 1.063 |
| 3 | put | 5 | **0.4177** | 0.1196 | 3.870 | 1.205 |
| 4 | rotate | 6 | **0.3273** | 0.1516 | 2.666 | 0.939 |
| 5 | fold | 9 | **0.2711** | 0.2904 | 2.920 | 0.979 |
| 6 | wring | 6 | **-1.4540** | -0.1593 | 4.066 | 1.304 |

Lowest Hausdorff: **rotate** (2.666); highest R²: **open** (0.5075).

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

**model `cnn_probgru`** — whole dataset (56 recordings): R² **0.4413**, skill **+0.1441**, skill (frame-pooled, OpenTouch's estimator) **+0.2017**, Hausdorff **4.710** (1.554× persistence). This is measured against the whole dataset's mean, so it is **not** the average of the rows below, each of which uses its own action's mean.

per action:

| # | action | n | R² | skill vs pers | Hausdorff | HD ratio |
|---:|---|---:|---:|---:|---:|---:|
| 1 | open | 20 | **0.4858** | 0.1523 | 5.684 | 1.865 |
| 2 | pick | 10 | **0.4738** | 0.1848 | 4.393 | 1.457 |
| 3 | put | 5 | **0.3934** | 0.0648 | 4.430 | 1.379 |
| 4 | fold | 9 | **0.2641** | 0.2827 | 2.980 | 0.999 |
| 5 | rotate | 6 | **0.1260** | -0.0939 | 4.157 | 1.463 |
| 6 | wring | 6 | **-2.3060** | -0.7369 | 5.369 | 1.722 |

Lowest Hausdorff: **fold** (2.980); highest R²: **open** (0.4858).

**model `cnn_seq2seq`** — whole dataset (56 recordings): R² **0.4084**, skill **+0.1334**, skill (frame-pooled, OpenTouch's estimator) **+0.1950**, Hausdorff **3.618** (1.193× persistence). This is measured against the whole dataset's mean, so it is **not** the average of the rows below, each of which uses its own action's mean.

per action:

| # | action | n | R² | skill vs pers | Hausdorff | HD ratio |
|---:|---|---:|---:|---:|---:|---:|
| 1 | open | 20 | **0.5064** | 0.2050 | 3.866 | 1.269 |
| 2 | pick | 10 | **0.4514** | 0.1769 | 3.621 | 1.201 |
| 3 | put | 5 | **0.4018** | 0.1103 | 3.613 | 1.125 |
| 4 | rotate | 6 | **0.2277** | 0.0646 | 2.713 | 0.955 |
| 5 | fold | 9 | **0.2042** | 0.2292 | 3.001 | 1.006 |
| 6 | wring | 6 | **-2.3359** | -0.4056 | 4.618 | 1.481 |

Lowest Hausdorff: **rotate** (2.713); highest R²: **open** (0.5064).

**model `flatten_seq2seq`** — whole dataset (56 recordings): R² **0.4079**, skill **+0.1389**, skill (frame-pooled, OpenTouch's estimator) **+0.1487**, Hausdorff **3.638** (1.200× persistence). This is measured against the whole dataset's mean, so it is **not** the average of the rows below, each of which uses its own action's mean.

per action:

| # | action | n | R² | skill vs pers | Hausdorff | HD ratio |
|---:|---|---:|---:|---:|---:|---:|
| 1 | open | 20 | **0.4653** | 0.1342 | 4.167 | 1.367 |
| 2 | pick | 10 | **0.4082** | 0.1150 | 3.919 | 1.300 |
| 3 | put | 5 | **0.3919** | 0.0918 | 3.736 | 1.163 |
| 4 | rotate | 6 | **0.1963** | 0.0455 | 3.052 | 1.074 |
| 5 | fold | 9 | **0.1628** | 0.1901 | 2.924 | 0.980 |
| 6 | wring | 6 | **-1.4445** | 0.0732 | 2.982 | 0.956 |

Lowest Hausdorff: **fold** (2.924); highest R²: **open** (0.4653).

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
