### ActionSense - as_preds_probgru_map

75 recordings scored. Ranked by **R²**, high → low. Hausdorff is **lower = better**; `HD ratio` < 1 beats persistence.

**model `probgru_cnn`**

| # | action | n | R² | skill vs pers | Hausdorff | HD ratio |
|---:|---|---:|---:|---:|---:|---:|
| 1 | slice | 45 | **0.7113** | 0.0665 | 2.589 | 0.895 |
| 2 | peel | 30 | **0.6441** | 0.0969 | 2.543 | 0.874 |

Lowest Hausdorff: **peel** (2.543); highest R²: **slice** (0.7113).


channels: F_L, CoPx_L, CoPy_L, F_R, CoPx_R, CoPy_R · CoP masking: `--mask none`
