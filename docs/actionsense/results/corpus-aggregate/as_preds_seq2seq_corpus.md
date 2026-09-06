### ActionSense - as_preds_seq2seq_corpus

290 recordings scored. Ranked by **R²**, high → low. Hausdorff is **lower = better**; `HD ratio` < 1 beats persistence.

**model `seq2seq_aggregate`**

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


channels: F_L, CoPx_L, CoPy_L, F_R, CoPx_R, CoPy_R · CoP masking: `--mask none`
