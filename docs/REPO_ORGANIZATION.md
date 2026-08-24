# Repository organization — file-by-file dataset categorization

Rigorous classification of every tracked source file by (what it does, which dataset), built by
inspecting code headers + imports + SESSION_LOG (§ "Datasets and what we did with each", ~line 695).
**Nothing has been moved yet** — this is the proposal/map for the reorg. See "Open questions" at the end.

## Key finding: it is NOT a clean 2-way split
There are **three** dataset bodies of work plus shared infra:
- **A. TouchAnything (upstream)** — multi-view video → hand-pose + tactile-pixel prediction (DINOv2,
  MANO, WiLoR/HaMeR; trains on EgoDex video + glove; EgoPressure). The original fork.
- **B. ActionSense (ours)** — aggregate physical-state (F/CoP) forecasting: v1 state, v2 probGRU,
  frozen eval harness. Data in `data/actionsense_states/`.
- **C. EgoTouch/OpenTouch (ours)** — the cross-dataset predictability study + the tactile-PIXEL
  forecaster (SimVP/ConvLSTM/ConvGRU) on the grasp subset. EgoTouch later deprecated.
- **D. Shared / infra / root.**

**Coupling caveat:** B and C both live inside the single package `src/tactile_forecast/` and share
its `__init__.py` (and `categories.py`/`predictability.py` are used across the study). Splitting B
from C means untangling one Python package — the main source of import risk in any physical move.

---

## A. TouchAnything (upstream) — video + pose + tactile-pixel pipeline
Datasets it uses: **EgoDex** (video), **EgoPressure**, TouchAnything glove tactile.

### src/ (model + data + losses + utils for the pose/tactile net)
- `src/data/touchanything_dataset.py` — main TouchAnything dataset (multi-view frames + pose + tactile).
- `src/data/egodex_dataset_v2.py` — EgoDex video dataset loader.
- `src/data/hdf5_dataset.py` — generic HDF5 clip dataset.
- `src/data/transforms.py` — image/tensor transforms.
- `src/data/glove_augmentation.py` — tactile-glove augmentation (data-side).
- `src/datasets/egopressure_dataset.py` — EgoPressure dataset loader.
- `src/models/touch_anything.py` — top-level TouchAnything model.
- `src/models/vision_encoder.py` / `multi_view_encoder.py` — DINOv2 image + multi-view encoders.
- `src/models/temporal_transformer.py` — temporal transformer backbone.
- `src/models/pose_encoder.py` / `pose_decoder.py` — 42-joint hand-pose head.
- `src/models/fusion.py` — vision/pose/tactile fusion.
- `src/losses/pose_loss.py` / `tactile_loss.py` — training losses.
- `src/utils/{config,logger,metrics,pressure_map,vis_pressure,visualization}.py` — pipeline utils.
- `src/resources/mano_right_neutral_subdiv.obj`, `src/resources/pressure_mappings/*.json` — MANO mesh + taxel↔position maps.

### scripts/ (data prep, training, inference, visualization)
- `scripts/core/train.py` — TouchAnything trainer (DDP).
- `scripts/core/convert_to_hdf5.py`, `run_convert_to_hdf5.sh` — raw → HDF5.
- `scripts/core/create_dataset_split.py`, `run_create_split.sh` — train/val/test split of TA data.
- `scripts/core/load_data.py` — data loading helpers.
- `scripts/core/inference_from_video.py`, `inference_tactile_parallel.py`,
  `inference_tactile_parallel_mano_style.py`, `run_inference.sh` — inference from video.
- `scripts/core/batch_process_hamer.py`, `scripts/batch_process_wilor_simple.py` — HaMeR/WiLoR hand-pose extraction.
- `scripts/core/visualize_hdf5.py`, `scripts/visualize_cleaned_data.py`, `run_visualize_cleaned_data.sh` — data viz.
- `scripts/data_processing/glove_augmentation.py`, `glove_augmentation_realistic.py` — glove aug (script-side).
- `scripts/tools/mano_visualization/*` — MANO renderer + mapping JSONs + mesh.
- `scripts/utils/sample_lite_trajectories.py`, `visualize_wilor_from_json.py` — misc TA utils.
- `scripts/run_train_ddp.sh` — DDP training launcher.

### configs/ + assets/
- `configs/base.yaml`, `configs/touchanything_with_glove_aug_wilor.yaml` — TA model/training configs.
- `configs/hand_joint_positions.json`, `configs/pressure_position_mapping_{left,right}.json` — geometry.
- `assets/*.gif` (6) — tactile demo GIFs for the README.

---

## B. ActionSense (ours) — physical-state (F/CoP) forecasting + eval harness
Data: `data/actionsense_states/` (state_N.npy + manifest.jsonl + splits.json).

### src/tactile_forecast/ (interleaved with C — see coupling caveat)
- `physical_state.py` — analytic moment extraction (F, CoP, shear) from a pressure clip; builds the
  ActionSense states (called by `probe_actionsense.py`). NUMPY-only.
- `state_forecast.py` — **v1** forecaster over the raw physical state (GRU vs structured baselines).
- `action_dynamics.py` — **v2** slow/fast + probabilistic GRU (probGRU); the library behind the CLIs.
- `eval_harness/` — the FROZEN evaluation harness (whole package): `config.py`, `splits.py`,
  `dataset.py`, `masking.py`, `metrics.py`, `evaluate.py`, `baselines/{persistence,seasonal,ar,base}.py`,
  `README.md`. Scores the raw 6-dim both-hands target; persistence/seasonal/AR baselines.

### scripts/actionsense/
- `probe_actionsense.py` — stream ActionSense HDF5, segment by Start/Stop, extract states + manifest.
- `train_state_forecaster.py` — v1 CLI (pour/slice).
- `train_action_dynamics.py` — v2 CLI (sweep input×hand×history, CV, calibration).
- `check_leakage.py` — 6 leakage assertions on the action_dynamics pipeline.
- `plot_action_forecast.py`, `plot_forecast_overlay.py`, `plot_horizon.py`, `plot_results_summary.py`,
  `plot_signal_decomposition.py`, `plot_test_results.py`, `plot_harness.py` — figures.
- `crc/download_actionsense.sh`, `crc/stream_actionsense.sh`, `crc/train_state_gpu.job` — CRC.

### configs/ + docs/ + tests/
- `configs/eval_harness.yaml` — frozen-harness config.
- `docs/TACTILE_FORECAST_PLAN.md` — v1/v2 plan.
- `docs/action_dynamics_results{,_precal}.csv`, `docs/harness_baselines{,_fitparams}.csv/.parquet` — results.
- `docs/leakage_checklist.md` — the 6 checks.
- `docs/*.png` (harness_*, forecast_*, horizon_*, results_summary) — figures.
- `tests/test_harness.py` — pytest for the harness.

---

## C. EgoTouch / OpenTouch (ours) — predictability study + tactile-PIXEL forecaster
Data: `datasets/EgoTouch/`, `datasets/grasp_hold_lift_tactile/` (both gitignored blobs).

### src/tactile_forecast/ (interleaved with B)
- `data.py` — torch dataset for windowed tactile→tactile PIXEL forecasting (grasp).
- `engine.py` — masked train/eval loops in [0,1] pressure space.
- `train.py` — trains one CV fold of a pixel forecaster (ConvGRU/ConvLSTM/SimVP), LTO/LOTO, grasp.
- `eval.py` — evaluate a pixel checkpoint vs pixel baselines.
- `baselines.py` — pixel baselines (persistence, last-velocity) on (B,t,C,H,W) tensors.
- `tactile_utils.py` — numpy tactile clip/mask/split/metric utils (21×21 EgoTouch).
- `models/conv_rnn.py`, `models/simvp.py` — the pixel forecaster architectures.
- `categories.py` — action-category taxonomy (verb→category, category→temporal pattern). Cross-dataset.
- `predictability.py` — training-free predictability metrics (persistence nMSE, periodicity, ...). Cross-dataset.

### scripts/ (top level = not one of the three arms, or shared)
- `probe_egotouch.py` — EgoTouch predictability probe (`probe_opentouch.py` now lives in
  `scripts/opentouch/`).
- `categorize_actions.py` — EgoTouch task→category classification.
- `prepare_grasp_tactile.py` — build the Grasp/Hold/Lift subset (EgoTouch).
- `tactile_predictability_probe.py` — feasibility probe on grasp subset.
- `aggregate_results.py` — aggregate CV results across runs/.
- `download_egotouch.py`, `crc/download_opentouch.sh` — dataset download.
- `crc/percategory_gpu.job`, `run_percategory.sh`, `pretrain_gpu.job`, `train_gpu.job` — CRC pixel training.

### configs/ + docs/
- `configs/tactile/{convgru,convlstm,simvp}.yaml` — pixel forecaster configs.
- `docs/ACTION_CATEGORIES.md` — taxonomy doc.
- `docs/predictability_by_category{,_full}.csv` — probe results.
- `docs/TACTILE_PREDICTION_PLAN.md`, `docs/RESULTS.md`, `docs/STUDY_SUMMARY.md` — study write-ups.

---

## D. Shared / infra / root (dataset-agnostic)
- `CLAUDE.md`, `AGENTS.md` (untracked), `SESSION_LOG.md`, `README.md`, `LICENSE`, `.gitignore`,
  `.gitmodules`, `environment.yaml`.
- `src/__init__.py`, `src/tactile_forecast/__init__.py` — package roots (the latter spans B+C).
- `scripts/crc/README.md`, `environment_tactile_cuda.yaml`, `setup_crc_env.sh`, `smoke_test.py` — CRC env.

---

## APPLIED structure (2026-07-20)
Decision: 3 groups + shared; move (staged + tested); keep single `src/` import root.

**DONE + verified — `src/` packages:**
```
src/actionsense/    # B: physical_state, state_forecast, action_dynamics (probGRU), eval_harness/
src/tactile_pixel/  # C: data, engine, train, eval, baselines, tactile_utils, categories, predictability, models/
src/touchanything/  # A: data/, models/, losses/, utils/, datasets/, resources/
```
- Stage 1 (`3c13dfe`): B → `src/actionsense/`. Verified: pytest 7/7, `python -m src.actionsense.eval_harness.evaluate` identical hash + determinism.
- Stage 2: C → `src/tactile_pixel/`. Verified: imports + `--help` entry points + probe cross-imports.
- Stage 3: A → `src/touchanything/`. Verified: 25 files py_compile (full runtime needs DINOv2/MANO, CRC-only).

**DONE — `configs/` grouped:** `configs/{actionsense,tactile_pixel,touchanything}/`. Harness re-verified.

**DONE — `scripts/` grouped by dataset (2026-08-24):** `scripts/{actionsense,opentouch,d256}/`.
Both hazards flagged below were real and were handled: 23 scripts had their
`sys.path.insert(...)` depth corrected for the extra level, and 101 invocation references were
rewritten across 50 files (docs, configs, CRC `.job`s, `src/`, `tests/`). Two cross-script
imports (`scripts.train_action_dynamics`, `scripts.plot_opentouch_fcop`) also needed
repointing. Verified by running `--help` on all 35 moved scripts: 35/35 import cleanly.
`SESSION_LOG.md` was deliberately NOT rewritten — it is a dated record of what was run at the
time, so its paths stay as they were; see the mapping table in `scripts/README.md`.

**NOT moved (intentional):**
- `docs/`, `data/` — referenced by path from the FROZEN harness config (`out_csv: docs/...`,
  `states_root: data/actionsense_states`) and many doc links; moving them breaks outputs for no benefit.
- `scripts/crc/`, root files — shared/infra, stay at their locations.

## Risks of a physical move (why staging + approval first)
1. **Import rewrites everywhere** — `from src.tactile_forecast import ...`, `from src.models ...`,
   `python -m src.tactile_forecast.evaluate`, `sys.path.insert(repo_root)`.
2. **eval_harness path logic** — `config.py::REPO_ROOT` counts `__file__` parents; moving changes depth.
3. **Config/data hardcoded paths** — `data/actionsense_states`, `configs/...`, split files.
4. **B/C entanglement** — one package `src/tactile_forecast/` holds both; `categories.py`/
   `predictability.py`/`__init__.py` are shared.
5. **CRC jobs** — `.job` files `cd` to repo paths and `python -m` modules.
Mitigation: move in verified stages (one bucket at a time), fix imports, run `pytest` + a smoke
`python -m ...evaluate` after each stage.

## Resolved (all confirmed by user 2026-07-20)
- Q1 → **3 groups + shared** (A/B/C). Q2 → **move, staged + tested**. Q3 → **single `src/` root**.

## Remaining (optional, awaiting go-ahead)
- **docs/ + data/ grouping** — deferred: they are referenced by path from the frozen harness config
  and would break output/data paths. Would require editing the (frozen) config + hash.
