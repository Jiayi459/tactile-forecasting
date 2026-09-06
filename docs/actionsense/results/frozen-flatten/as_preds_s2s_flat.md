### ActionSense - as_preds_s2s_flat

75 recordings scored. Ranked by **R²**, high → low. Hausdorff is **lower = better**; `HD ratio` < 1 beats persistence.

**model `seq2seq_flatten`**

| # | action | n | R² | skill vs pers | Hausdorff | HD ratio |
|---:|---|---:|---:|---:|---:|---:|
| 1 | slice | 45 | **0.6691** | -0.0660 | 2.588 | 0.894 |
| 2 | peel | 30 | **0.6087** | 0.0118 | 2.560 | 0.880 |

Lowest Hausdorff: **peel** (2.560); highest R²: **slice** (0.6691).


channels: F_L, CoPx_L, CoPy_L, F_R, CoPx_R, CoPy_R · CoP masking: `--mask none`
