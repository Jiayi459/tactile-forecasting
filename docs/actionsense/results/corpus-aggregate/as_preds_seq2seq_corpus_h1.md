### ActionSense - as_preds_seq2seq_corpus_h1

290 recordings scored. Ranked by **R²**, high → low. Hausdorff is **lower = better**; `HD ratio` < 1 beats persistence.

**model `seq2seq_aggregate`** — whole dataset (290 recordings): R² **0.7410**, skill **+0.1243**, Hausdorff **2.408** (0.826× persistence). This is measured against the whole dataset's mean, so it is **not** the average of the rows below, each of which uses its own action's mean.

per action:

| # | action | n | R² | skill vs pers | Hausdorff | HD ratio |
|---:|---|---:|---:|---:|---:|---:|
| 1 | slice | 45 | **0.7327** | 0.1322 | 2.368 | 0.818 |
| 2 | peel | 30 | **0.6753** | 0.1758 | 2.377 | 0.817 |
| 3 | spread | 30 | **0.6515** | 0.1004 | 2.368 | 0.808 |
| 4 | clean | 55 | **0.6391** | 0.1259 | 2.329 | 0.804 |
| 5 | clear | 28 | **0.6125** | 0.1168 | 2.510 | 0.857 |
| 6 | set | 6 | **0.5826** | 0.1391 | 2.503 | 0.842 |
| 7 | load | 5 | **0.5617** | 0.1437 | 2.495 | 0.838 |
| 8 | get/replace | 15 | **0.5312** | 0.1510 | 2.414 | 0.824 |
| 9 | get | 30 | **0.4915** | 0.1578 | 2.448 | 0.831 |
| 10 | unload | 5 | **0.4417** | 0.1638 | 2.464 | 0.833 |
| 11 | stack | 5 | **0.4358** | 0.2205 | 2.344 | 0.791 |
| 12 | pour | 21 | **0.4199** | 0.0208 | 2.649 | 0.918 |
| 13 | open/close | 9 | **0.3258** | 0.1546 | 2.268 | 0.791 |
| 14 | open | 6 | **0.2692** | 0.1315 | 2.309 | 0.795 |

Lowest Hausdorff: **open/close** (2.268); highest R²: **slice** (0.7327).


channels: F_L, CoPx_L, CoPy_L, F_R, CoPx_R, CoPy_R · CoP masking: `--mask none`
