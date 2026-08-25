# Results — Tactile→Tactile Forecasting (Grasp/Hold/Lift)

**Question.** Can future tactile pressure maps be predicted from past tactile alone (no vision,
no actions), for grasp/hold/lift interactions — and does it generalize to unseen objects?

**Setup.** Input 10 frames (0.33 s) → predict 15 frames (0.5 s) @30 fps. Bimanual 2×21×21 grids
(217 valid taxels/hand), residual prediction (Δ from last frame), masked loss over valid taxels.
Headline metric: **skill = 1 − MSE_model / MSE_persistence** (>0 means it beats copying the last
frame), averaged over horizons and CV folds. Data: 82 grasp trajectories, 8 objects/tasks.

## Cross-validation results (skill vs persistence, mean ± std)

| Model | Protocol | Pretrain | Skill | Notes |
|---|---|---|--:|---|
| ConvGRU | LTO (5-fold) | — | +0.138 ± 0.056 | h1 negative |
| ConvLSTM | LTO (5-fold) | — | +0.152 ± 0.031 | |
| **SimVP** | **LTO (5-fold)** | — | **+0.192 ± 0.044** | best; positive at every horizon |
| SimVP | LOTO (8-fold) | — | +0.005 ± 0.111 | unseen object: ≈ persistence (fails) |
| **SimVP** | **LOTO (8-fold)** | **full (≈1,851 traj, grasp excluded)** | **+0.097 ± 0.122** | **pretraining unlocks generalization** |

- **LTO** = Leave-Trajectory-Out (test = new recordings, objects overlap train).
- **LOTO** = Leave-One-Task-Out (test = a whole held-out object, never seen in train; pretraining
  excludes all 8 grasp tasks so the held-out object is truly unseen → no leakage).

### Per-horizon skill (SimVP)
| | h1 | h5 | h10 | h15 |
|---|--:|--:|--:|--:|
| LTO | +0.065 | +0.187 | +0.217 | +0.235 |
| LOTO scratch | +0.010 | +0.010 | −0.012 | +0.027 |
| LOTO pretrained | +0.057 | +0.109 | +0.092 | +0.109 |

LTO skill *rises* with horizon (persistence decays faster than learned dynamics). LOTO-pretrained
is positive and roughly flat (~0.10); LOTO-scratch hovers at zero / dips negative mid-horizon.

## Conclusions
1. **Future tactile is predictable from past** beyond persistence (SimVP LTO +0.192 / 0.5 s).
2. **From few objects it does not generalize** to an unseen object (LOTO scratch ≈ 0).
3. **Broad multi-object pretraining unlocks unseen-object prediction** (LOTO +0.005 → +0.097,
   ~18×, positive at all horizons; helped 6/8 held-out objects). Demonstrates transferable
   tactile dynamics learned from object diversity.

## Caveats
- Small data: 82 trajectories / **8 objects**; LOTO per-fold variance is large (±0.12) and one
  object (fold 5) stays negative.
- SimVP is over-parameterized (30.5 M vs ~5 k training windows) — a smaller model ablation is
  open. Recurrent baselines (ConvGRU 0.34 M, ConvLSTM 0.45 M) trail SimVP on LTO.
- Deterministic prediction only; ≥1 s horizon not targeted (tactile-only autocorrelation → 0 by ~1 s).

## Reproduce
```bash
# CV from scratch (LTO/LOTO):   python -m src.tactile_pixel.train --config configs/tactile_pixel/simvp.yaml --protocol {lto,loto} --fold N --scope grasp
# Pretrain (GPU batch):         qsub scripts/crc/pretrain_gpu.job        # -> runs/simvp_pretrain/best.pt
# Fine-tune from pretrain:      python -m src.tactile_pixel.train ... --protocol loto --fold N --pretrained runs/simvp_pretrain/best.pt --out runs/simvp_ft_grasp_loto_fN
# Aggregate:                    python scripts/shared/aggregate_results.py
```
