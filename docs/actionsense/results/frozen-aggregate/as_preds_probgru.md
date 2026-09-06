### ActionSense - as_preds_probgru

75 recordings scored. Ranked by **R²**, high → low. Hausdorff is **lower = better**; `HD ratio` < 1 beats persistence.

**model `probgru_aggregate`**

| # | action | n | R² | skill vs pers | Hausdorff | HD ratio |
|---:|---|---:|---:|---:|---:|---:|
| 1 | slice | 45 | **0.7328** | 0.1287 | 2.552 | 0.882 |
| 2 | peel | 30 | **0.6502** | 0.1100 | 2.598 | 0.893 |

Lowest Hausdorff: **slice** (2.552); highest R²: **slice** (0.7328).


channels: F_L, CoPx_L, CoPy_L, F_R, CoPx_R, CoPy_R · CoP masking: `--mask none`
