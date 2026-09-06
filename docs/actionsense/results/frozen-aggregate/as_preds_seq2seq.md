### ActionSense - as_preds_seq2seq

75 recordings scored. Ranked by **R²**, high → low. Hausdorff is **lower = better**; `HD ratio` < 1 beats persistence.

**model `seq2seq_aggregate`**

| # | action | n | R² | skill vs pers | Hausdorff | HD ratio |
|---:|---|---:|---:|---:|---:|---:|
| 1 | slice | 45 | **0.7337** | 0.1306 | 2.331 | 0.805 |
| 2 | peel | 30 | **0.6788** | 0.1829 | 2.327 | 0.800 |

Lowest Hausdorff: **peel** (2.327); highest R²: **slice** (0.7337).


channels: F_L, CoPx_L, CoPy_L, F_R, CoPx_R, CoPy_R · CoP masking: `--mask none`
