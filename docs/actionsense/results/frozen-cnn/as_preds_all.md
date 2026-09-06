### ActionSense frozen — all arms + baselines (TEST split only)

15 recordings scored. Ranked by **R²**, high → low. Hausdorff is **lower = better**; `HD ratio` < 1 beats persistence.

**model `ar`**

| # | action | n | R² | skill vs pers | Hausdorff | HD ratio |
|---:|---|---:|---:|---:|---:|---:|
| 1 | slice | 9 | **0.7348** | 0.1125 | 2.386 | 0.828 |
| 2 | peel | 6 | **0.6654** | 0.2087 | 2.391 | 0.828 |

Lowest Hausdorff: **slice** (2.386); highest R²: **slice** (0.7348).

**model `pg_probgru_cnn`**

| # | action | n | R² | skill vs pers | Hausdorff | HD ratio |
|---:|---|---:|---:|---:|---:|---:|
| 1 | slice | 9 | **0.7306** | 0.0955 | 2.532 | 0.879 |
| 2 | peel | 6 | **0.6324** | 0.1109 | 2.500 | 0.865 |

Lowest Hausdorff: **peel** (2.500); highest R²: **slice** (0.7306).

**model `s2s_seq2seq_cnn`**

| # | action | n | R² | skill vs pers | Hausdorff | HD ratio |
|---:|---|---:|---:|---:|---:|---:|
| 1 | slice | 9 | **0.7058** | 0.0308 | 2.481 | 0.862 |
| 2 | peel | 6 | **0.6139** | 0.0802 | 2.503 | 0.867 |

Lowest Hausdorff: **slice** (2.481); highest R²: **slice** (0.7058).

**model `seasonal`**

| # | action | n | R² | skill vs pers | Hausdorff | HD ratio |
|---:|---|---:|---:|---:|---:|---:|
| 1 | slice | 9 | **0.6993** | 0.0000 | 2.880 | 1.000 |
| 2 | peel | 6 | **0.5739** | 0.0000 | 2.889 | 1.000 |

Lowest Hausdorff: **slice** (2.880); highest R²: **slice** (0.6993).


channels: F_L, CoPx_L, CoPy_L, F_R, CoPx_R, CoPy_R · CoP masking: `--mask none`
