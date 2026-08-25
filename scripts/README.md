# scripts/ layout

Grouped by dataset arm on 2026-08-24. Three arms have their own directory; everything still at
top level either belongs to a *different* dataset or is shared, and is pending a decision.

| directory | arm | n |
|---|---|--:|
| `actionsense/` | ActionSense (MIT CSAIL) — the original probGRU / physical-state work | 20 |
| `opentouch/` | OpenTouch (single right-hand FPC glove, 16×16) | 15 |
| `d256/` | d256 / `Dataset256` (ICLR force-vision release; ActionSense-derived) | 2 |
| `egotouch/` | EgoTouch (HF `zhouzhoujy/EgoTouch`) | 5 |
| `shared/` | genuinely cross-dataset | 2 |
| `crc/` | ND CRC cluster infrastructure — grouped by *cluster*, not dataset, so left alone | — |
| `core/`, `data_processing/`, `tools/`, `utils/` | upstream TouchAnything repo, untouched | — |

## If a documented command stopped working

Paths moved. `SESSION_LOG.md` was **not** rewritten — it records what was actually run on a
given date, and editing it would falsify that record. Translate with this rule:

    scripts/<name>.py   ->   scripts/<arm>/<name>.py

`probe_*`, `plot_*`, `train_*`, `extract_*` follow their dataset. Everything below stayed put.

## `shared/` — and why `check_leakage.py` is not in it

- `build_skill_comparison.py` — reads both `docs/actionsense/*` and `docs/opentouch/*` to
  rebuild `docs/skill_comparison.md`. Belongs to no single arm by construction.
- `aggregate_results.py` — aggregates `runs/*/summary.json` by (model, scope, protocol);
  dataset-agnostic.

`check_leakage.py` is a general leakage checklist *in intent*, but its assertions run against
`src.actionsense` and `data/actionsense_states`, so it can only be run on one arm. It is filed
under `actionsense/` by implementation (user decision 2026-08-25): filing it under `shared/`
would advertise a generality it does not have. Move it if it ever becomes dataset-agnostic.

## Still at top level: upstream TouchAnything repo (inherited from the fork parent)
`batch_process_wilor_simple.py`, `visualize_cleaned_data.py`, `run_convert_to_hdf5.sh`,
`run_create_split.sh`, `run_inference.sh`, `run_train_ddp.sh`, `run_visualize_cleaned_data.sh`

## Pre-existing dangling references (not caused by the move)

`scripts/download_data.sh`, `scripts/predictability_by_category.py`,
`scripts/run_inference_mano.sh`, and the `scripts/X.py` placeholder in
`docs/REPO_ORGANIZATION.md` are referenced but have never existed in this repo.

## One stale path that must STAY stale

`configs/opentouch/eval_harness_d1.yaml` line 1 credits `scripts/opentouch_apply_baseline.py`,
which now lives at `scripts/opentouch/opentouch_apply_baseline.py`. **Do not fix it.** That
file's `config_hash` is `sha256` over its exact bytes and is stamped into the results table for
traceability, so editing even a comment would orphan every existing result row from the config
that produced it. The same applies to `configs/opentouch/eval_harness.yaml` and
`configs/actionsense/eval_harness.yaml`. Current hashes, re-verified after the move:

    001dcee8e81efda3  configs/opentouch/eval_harness_d1.yaml
    916820c096c7666a  configs/opentouch/eval_harness.yaml
    947e650076742574  configs/actionsense/eval_harness.yaml
