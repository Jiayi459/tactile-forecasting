### ActionSense - as_preds_pg_flat

75 recordings scored. Ranked by **R²**, high → low. Hausdorff is **lower = better**; `HD ratio` < 1 beats persistence.

**model `probgru_flatten`**

| # | action | n | R² | skill vs pers | Hausdorff | HD ratio |
|---:|---|---:|---:|---:|---:|---:|
| 1 | slice | 45 | **0.5763** | -0.4549 | 3.535 | 1.221 |
| 2 | peel | 30 | **0.4698** | -0.3883 | 3.481 | 1.196 |

Lowest Hausdorff: **peel** (3.481); highest R²: **slice** (0.5763).


channels: F_L, CoPx_L, CoPy_L, F_R, CoPx_R, CoPy_R · CoP masking: `--mask none`
