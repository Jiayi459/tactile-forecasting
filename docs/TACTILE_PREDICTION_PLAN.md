# Implementation Plan — Tactile→Tactile Forecasting on the Grasp/Hold/Lift Subset

**Scientific question.** Can *future* tactile pressure maps be predicted from *past* tactile
pressure maps alone (no vision, no action conditioning), for grasp/hold/lift interactions?
This is a **spatiotemporal sequence forecasting** problem on a sparse, masked 2×21×21 signal.

**Status:** plan for review (no model trained yet). Data subset prepared; baseline/feasibility
probe done. Decisions needing sign-off are in §10.

---

## 1. Data (prepared) and what the signal actually looks like

Subset: `datasets/grasp_hold_lift_tactile/` (+ `manifest.csv`), built by
`scripts/egotouch/prepare_grasp_tactile.py` from the 8 core-grasp tasks:
`grasp_body_lotion, grasp_cola, grasp_floral_water, grasp_power_adapter, grasp_sunscreen,
grip_hand_dynamometer, hold_teapot, lift_towel`.

Measured facts (from EDA, `scripts/egotouch/prepare_grasp_tactile.py`):
- **82 trajectories, 31,577 frames @30 fps.** Lengths very skewed: min 71, **median 125
  (~4.2 s), mean 385**, max 2206. → windowing is mandatory; can't treat as fixed-length.
- Per hand: 21×21 grid, but **~50.8% of cells are structurally NaN** — a *fixed sensor-layout
  mask* (constant across frames/trajectories), i.e. "no taxel here", **not** missing data.
  ~220 valid taxels/hand.
- Values normalized to **[0,1]** (attr `tactile_max`). Spatially **sparse**: ~33% of valid
  taxels active per frame.
- Both hands always present in the array; grasp tasks are often one-hand-dominant (manifest
  has per-hand `active_frac` to pick the working hand if needed).

**Predictability probe** (`scripts/egotouch/tactile_predictability_probe.py`) — persistence baseline
(ŷ[t+h]=y[t]), normalized MSE, and total-force autocorrelation vs horizon:

| Horizon | nMSE (persistence) | force autocorr r |
|---|--:|--:|
| 1 fr (33 ms)  | 0.040 | 0.987 |
| 3 (100 ms)    | 0.127 | 0.935 |
| 5 (167 ms)    | 0.226 | 0.861 |
| 10 (333 ms)   | 0.473 | 0.651 |
| 15 (500 ms)   | 0.724 | 0.410 |
| 30 (1000 ms)  | 1.343 | −0.167 |

**Implications that drive every later choice:**
1. Tactile is *highly* autocorrelated short-term → the **honest target horizon is ~0.1–0.5 s**
   (3–15 frames). Beyond ~1 s, persistence is worse than the mean and the signal is near
   unpredictable from tactile-only history — report it but don't optimize for it.
2. **Persistence is a strong baseline.** Success must be defined as *beating persistence*
   (skill score), not raw MSE — a model with low MSE that merely copies the last frame is
   worthless.
3. Dataset is **small** (82 sequences). This rules out large data-hungry models as the primary
   choice and forces strong regularization, augmentation, and leave-out cross-validation.

---

## 2. Literature review (focused, decision-oriented)

**Tactile prediction (the closest prior art).**
- **ACTP / ACTVP** — Nazari et al., *Action-Conditioned Tactile Prediction: a case study on
  slip prediction* ([arXiv:2205.09430](https://arxiv.org/pdf/2205.09430),
  [code](https://github.com/imanlab/action_conditioned_tactile_prediction)). Predicts future
  tactile images from past tactile + (future) actions; encodes to a latent space then a
  **Conv-LSTM** chain for spatiotemporal rollout; evaluates on downstream **slip** detection.
  Most directly comparable; our task is the *unconditioned* variant (no action input).
- *Deep Functional Predictive Control for Strawberry Cluster Manipulation using Tactile
  Prediction* ([arXiv:2303.05393](https://arxiv.org/pdf/2303.05393)) — uses tactile prediction
  inside an MPC loop; confirms short-horizon tactile rollout is the useful regime.
- *Dream-Tac: A Unified Tactile World Action Model*
  ([arXiv:2606.08737](https://arxiv.org/html/2606.08737v1)) and *Tactile-Conditioned Diffusion
  Policy* ([arXiv:2510.13324](https://arxiv.org/html/2510.13324v1)) — recent generative/world-model
  framings; relevant only as a later, longer-horizon/probabilistic extension.

**Spatiotemporal / video-prediction backbones (a 2×21×21 tactile clip *is* a tiny video).**
- **OpenSTL** benchmark ([code](https://github.com/chengtan9907/OpenSTL)) — unifies ConvLSTM,
  PredRNN/++/v2, E3D-LSTM, SimVP/v2, TAU, SwinLSTM, etc. Our reference implementation source.
- **ConvLSTM** (Shi et al., NeurIPS'15) — canonical recurrent spatiotemporal model; strong,
  compact, the ACTP backbone.
- **SimVP** (Gao et al., CVPR'22, [arXiv:2206.05099](https://ar5iv.labs.arxiv.org/html/2206.05099))
  and **SimVPv2 / TAU** (CVPR'23) — pure-CNN encoder→translator→decoder; **state-of-the-art-ish
  with far fewer params and no recurrence**, trains fast on CPU. Best fit for small data + modest
  compute.
- **PredFormer** (*Video Prediction Transformers without Recurrence or Convolution*,
  [arXiv:2410.04733](https://arxiv.org/html/2410.04733v3)) and the
  *Survey on Video Prediction* ([arXiv:2401.14718](https://arxiv.org/html/2401.14718v3)) —
  transformers are strong but data-hungry; deprioritized for 82 sequences.

**Takeaway for method choice.** The field converges on (a) **ConvLSTM** as the compact recurrent
baseline (and the tactile-specific precedent), and (b) **SimVP/TAU** as the efficient CNN SOTA.
Both are small enough for CPU and appropriate for limited data. Transformers/diffusion are
extensions, not the starting point.

---

## 3. Preprocessing (this is where most of the result is won)

1. **Sensor mask.** Build the fixed validity mask `M∈{0,1}^{2×21×21}` (NaN→0 in mask). Store once.
   Zero-fill NaN in the data. **Compute the loss only over valid taxels** (masked loss); never
   let the model spend capacity on structurally-dead cells.
2. **Channels.** Stack hands as 2 channels → tensor `(T, 2, 21, 21)`. Keep both hands (bimanual
   coupling may help); also support a single-dominant-hand mode (1 channel) as an ablation.
3. **Amplitude transform.** Data is sparse with heavy tails. Compare three on val: raw [0,1],
   `sqrt`, `log1p(α·x)`. Sparse heavy-tailed pressure usually benefits from a concave transform;
   pick by val skill. Keep normalization **global** (consistent with dataset-level `tactile_max`),
   not per-clip, to preserve magnitude semantics.
4. **Windowing.** Sliding windows: **input `Tin` frames → predict `Tout` frames**. Start
   `Tin=10` (0.33 s), `Tout=5` (0.17 s); also evaluate `Tout=10` (0.33 s). Stride 2–5 to limit
   redundancy. This converts 82 variable-length clips into thousands of training windows
   (mitigates small-N).
5. **Frame rate.** Optionally downsample 30→15 fps so a fixed `Tout` covers a longer horizon at
   equal compute; decide from the horizon goal (§10).
6. **Augmentation (critical at N=82).** Left↔right **hand mirror** (doubles data; physically
   valid if mask mirrored), small temporal crops/jitter, mild multiplicative magnitude scaling,
   low additive noise on active taxels. Avoid spatial rotations (break sensor geometry).
7. **Splits / leakage control.** Split **by trajectory** (never window-level) so no window from a
   test trajectory leaks into train. Provide two protocols (§5).

---

## 4. Models (staged; each must beat the previous and beat persistence)

- **B0 Persistence** — ŷ[t+h]=y[t]. The bar (numbers in §1).
- **B1 Last-velocity / linear extrapolation** — ŷ[t+h]=y[t]+h·(y[t]−y[t−1]); and a per-taxel
  VAR(p). Cheap, often beats persistence at h≥5.
- **M1a ConvGRU** (**primary recurrent — recommended**) — encoder + ConvGRU + decoder, masked
  loss. *Chosen over ConvLSTM for this data:* 3 gates vs 4 ⇒ ~25% fewer params, no separate cell
  state ⇒ less overfitting and faster convergence at N=82, while preserving spatial structure
  (conv state) — a plain (non-conv) GRU is rejected because flattening 21×21 discards contact
  geometry. ConvGRU+CNN is an established tactile-force backbone.
- **M1b ConvLSTM** (precedent baseline) — same harness; included because it is the published
  tactile-prediction architecture (ACTP), for apples-to-apples comparison.
- **M2 SimVP / TAU** (primary CNN, **recommended headline model**) — encoder–translator–decoder,
  fewer params, no recurrence, strong on small data. Reuse OpenSTL implementation.
- **M3 (optional extension)** transformer (PredFormer) or generative (VAE/diffusion) **only** if
  M1/M2 plateau. (Deferred: scope is **deterministic** prediction per decision §10.6.)

Rollout: train one-step + scheduled-sampling for multi-step, **or** direct multi-horizon heads;
compare. Report autoregressive error growth vs horizon.

---

## 5. Validation protocol

- **Protocol A — Leave-Trajectory-Out (k-fold, k=5 by trajectory):** can the model interpolate
  within seen object/grasp types? Primary metric.
- **Protocol B — Leave-Task-Out (8-fold, one task held out each time):** does it generalize to an
  **unseen object/grasp**? The honest generalization test; expect a drop — report it.
- **Metrics (all masked to valid taxels, per horizon h):**
  - MSE / MAE on the grid; **Skill = 1 − MSE_model/MSE_persistence** (the headline — must be >0).
  - **Total-force error** (sum over taxels) — captures grip-force trend.
  - **Contact-region IoU / F1** on binarized maps (threshold τ) — spatial contact correctness.
  - **SSIM** on the grid for structural similarity.
  - Per-horizon curves for h∈{1,3,5,10,15} frames; report autoregressive degradation.
- **Statistics:** mean ± std across folds; paired test (model vs persistence) across test
  trajectories. With N=82, report effect sizes, don't over-claim.

**Success criterion (feasibility verdict):** statistically significant positive skill over
persistence at h≥5 (≥167 ms) under Protocol A, and a quantified (even if smaller) skill under
Protocol B.

---

## 6. Training / compute

**Two-tier workflow (local dev + cluster training).** The school **CRC GPU cluster** is the
training target; the Windows box is for data prep, baselines, and dataloader/metric debugging.
- **Local (Windows, `touchanything` conda env):** plain **PyTorch CPU wheel** is enough to build
  and unit-test the dataloader, masking, B0/B1 baselines, and a 1-epoch smoke test of M1/M2
  (model+data are tiny: 2×21×21, ~31k frames).
- **Cluster (ND CRC, Linux + CUDA):** real training, k-fold CV, ablations.
  - **Scheduler = Univa Grid Engine (UGE), `qsub`** (NOT SLURM). GPU jobs: `#$ -q gpu` +
    `#$ -l gpu_card=1`, 4-day wallclock limit; GPU nodes ~32 cores / 4 GPUs.
  - **Env:** one-time `module load conda; conda init; source ~/.bashrc; module unload conda`,
    then a dedicated **CUDA env** (lean: PyTorch+CUDA, no DINOv2/xformers needed for this task).
    In job scripts use `conda activate <env>` directly (do **not** `module load conda` in jobs).
    Setup automated in `scripts/crc/` (see below).
  - One `qsub` per CV fold → folds run in parallel.
- **Data transfer:** subset is small (only `pressure_grids.npz`, no video) → a few hundred MB;
  `rsync`/`scp` `datasets/grasp_hold_lift_tactile/` (and, for pretraining, the npz of all 1,930
  trajectories) to CRC scratch. No MP4s needed.
- **GPU enables two extensions worth taking:**
  1. **Pretrain → fine-tune** to beat small-N: pretrain the forecaster on **all 1,930
     trajectories / 23 action categories** (self-supervised next-frame tactile prediction), then
     fine-tune on the 82 grasp/hold/lift clips. Strong candidate to lift skill.
  2. Feasible to run the heavier M3 models (transformer/generative) and broad ablations quickly.

- **Loss:** masked MSE, optionally + active-taxel weighting (counter sparsity) + small
  gradient/temporal-consistency term. Tune on val.
- **Regularization:** weight decay, dropout, early stopping on val skill, small model widths
  (data-limited — still the binding constraint even with GPU).
- **Logging:** TensorBoard (already in env); save per-fold metrics to CSV + qualitative GIFs of
  predicted vs GT grids. On SLURM, log to a run dir under cluster scratch + sync back.

---

## 7. Risks and mitigations

| Risk | Evidence | Mitigation |
|---|---|---|
| Model just copies last frame | persistence nMSE 0.04@33ms | Report **skill vs persistence**, not MSE |
| Overfitting (N=82) | small data | SimVP (few params), augmentation, LTO/LOTO CV, early stop |
| NaN mask mishandled | 50.8% NaN cells | Fixed mask + masked loss; unit-test mask |
| Horizon too ambitious | autocorr<0 @1s | Target 0.1–0.5 s; treat ≥1 s as stretch/generative |
| Long-clip imbalance | max 2206 vs median 125 | Window + stride; cap windows/clip |
| Per-hand emptiness | one-hand grasps | Dominant-hand mode + per-hand metrics |

---

## 8. Milestones (efficient path)

1. **M0 (½ day):** finalize mask + dataloader (windowing, masked loss, augmentation) + B0/B1
   baselines + metric harness. *Deliver baseline skill table.*
2. **M1 (1 day):** ConvLSTM, Protocol A. Beat persistence?
3. **M2 (1 day):** SimVP/TAU, Protocols A & B. Headline numbers + qualitative GIFs.
4. **M3 (optional):** transform/loss ablations, frame-rate study, transformer/generative if
   warranted.
5. **Write-up:** feasibility verdict + skill curves.

---

## 9. Deliverables — BUILT (2026-06-18, "go")

Implemented `src/tactile_pixel/`:
- `tactile_utils.py` — torch-free core (sensor mask, amplitude transforms, windowing, LTO/LOTO
  splits, numpy per-horizon metrics). **Verified locally** vs real data (mask 217/hand, 5,955
  windows, splits, metrics).
- `data.py` — `TactileWindows` dataset (trajectory-level splits, mask-safe augmentation).
- `models/` — `ConvGRU` + `ConvLSTM` (`conv_rnn.py`, seq2seq + scheduled sampling), `SimVP`
  (`simvp.py`); `build_model` factory.
- `engine.py` — masked MSE (active-taxel weighting), train/eval loops, SS schedule.
- `baselines.py` — persistence + last-velocity.
- `train.py` / `eval.py` — CLI: protocol (lto/loto), fold, scope (grasp/full), pretrain &
  `--pretrained` fine-tune; writes `best.pt`, `train_log.csv`, `test_metrics.csv`, `summary.json`.
- `configs/tactile_pixel/{convgru,convlstm,simvp}.yaml`; `scripts/crc/smoke_test.py` (synthetic
  end-to-end check). All 11 modules byte-compile; torch path to be smoke-tested on CRC.

Run recipe in `scripts/crc/README.md` (smoke → pretrain on full → fine-tune LTO/LOTO CV → eval).
Headline metric = **mean skill vs persistence**.

---

## 10. DECISIONS — RESOLVED (2026-06-18)

1. **Target horizon → 0.5 s (15 frames @30 fps).** Train/eval up to 15-frame rollout; report
   intermediate horizons (1/3/5/10/15). No ≥1 s goal ⇒ stay deterministic.
2. **Hands → BOTH.** Bimanual 2-channel model is primary; dominant-hand single-channel as ablation.
3. **Method → ConvGRU primary recurrent + ConvLSTM precedent baseline + SimVP/TAU headline CNN.**
   (GRU question answered: ConvGRU preferred for N=82 — fewer params, less overfit; plain GRU
   rejected as it discards spatial structure. See §4.)
4. **Compute → ND CRC (UGE/`qsub`, `-q gpu -l gpu_card=1`).** Add env-setup + GPU job scripts
   (`scripts/crc/`). Local Windows env = dev only.
5. **Pretrain→fine-tune → YES.** Pretrain on all 1,930 trajectories (self-supervised next-frame),
   fine-tune on the 82 grasp/hold/lift clips.
6. **Scope → deterministic** next-frame map prediction. Generative/uncertainty deferred.
7. **Install → CUDA.** Set up a CUDA PyTorch env on CRC (no local CPU torch needed).

### Next phase after env setup (awaiting "go"): build `src/tactile_pixel/` (dataset/masking,
ConvGRU/ConvLSTM/SimVP, train/eval, CV harness) per §8 milestones.
