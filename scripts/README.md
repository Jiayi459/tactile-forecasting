# scripts/ layout

Grouped by dataset arm on 2026-08-24. Three arms have their own directory; everything still at
top level either belongs to a *different* dataset or is shared, and is pending a decision.

| directory | arm | n |
|---|---|--:|
| `actionsense/` | ActionSense (MIT CSAIL) — the original probGRU / physical-state work | 18 |
| `opentouch/` | OpenTouch (single right-hand FPC glove, 16×16) | 15 |
| `d256/` | d256 / `Dataset256` (ICLR force-vision release; ActionSense-derived) | 2 |
| `crc/` | ND CRC cluster infrastructure — grouped by *cluster*, not dataset, so left alone | — |
| `core/`, `data_processing/`, `tools/`, `utils/` | upstream TouchAnything repo, untouched | — |

## If a documented command stopped working

Paths moved. `SESSION_LOG.md` was **not** rewritten — it records what was actually run on a
given date, and editing it would falsify that record. Translate with this rule:

    scripts/<name>.py   ->   scripts/<arm>/<name>.py

`probe_*`, `plot_*`, `train_*`, `extract_*` follow their dataset. Everything below stayed put.

## Still at top level (awaiting a decision)

**EgoTouch — a fourth dataset, not one of the three arms**
`download_egotouch.py`, `probe_egotouch.py`, `categorize_actions.py`,
`prepare_grasp_tactile.py`, `tactile_predictability_probe.py`

**Genuinely cross-dataset**
- `build_skill_comparison.py` — reads both `docs/actionsense/*` and `docs/opentouch/*` to
  rebuild `docs/skill_comparison.md`. Belongs to no single arm by construction.
- `aggregate_results.py` — aggregates `runs/*/summary.json` by (model, scope, protocol);
  dataset-agnostic.
- `check_leakage.py` — a general leakage checklist, but its assertions currently run against
  `src.actionsense` and `data/actionsense_states`, so it is ActionSense-specific *in
  implementation* while general *in intent*.

**Upstream TouchAnything repo (inherited from the fork parent)**
`batch_process_wilor_simple.py`, `visualize_cleaned_data.py`, `run_convert_to_hdf5.sh`,
`run_create_split.sh`, `run_inference.sh`, `run_train_ddp.sh`, `run_visualize_cleaned_data.sh`

## Pre-existing dangling references (not caused by the move)

`scripts/download_data.sh`, `scripts/predictability_by_category.py`,
`scripts/run_inference_mano.sh`, and the `scripts/X.py` placeholder in
`docs/REPO_ORGANIZATION.md` are referenced but have never existed in this repo.
