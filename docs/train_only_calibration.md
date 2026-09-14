# TRAIN-only calibration, version 1

The approved protocol provides **no extra calibration segment at a new location**.
For each fold, split recording IDs first, fit one sensor template using TRAIN raw
pressure, freeze it, and apply it unchanged to TRAIN, VAL and TEST. Model fitting may
use all of TRAIN; prediction of a held-out recording must not estimate statistics from
that recording's future. VAL remains available for model/history selection and sigma
calibration, but not for fitting the pressure template.

## What changed

- ActionSense: a per-hand, per-taxel TRAIN p5 template replaces the whole-clip p5.
  Raw extraction uses previous-sample hold on `start + n / fps`; it never interpolates
  against a later sensor sample. Query/source timestamps are saved in `time_N.npz`.
- OpenTouch: one shared TRAIN template replaces per-shard calibration. It uses the
  existing median, one-sided MAD and quantization floor, with `k=1` by default.
  Sampling balances TRAIN shards and then clips; an unseen shard needs no adaptation.
- Both: corrected pressure maps and framewise physical states come from the same
  correction. Aggregate, flatten, CNN, ProbGRU and classical baseline arms consume
  those states/maps. Strict mode skips legacy first-N/shard baseline estimation and
  rejects whole-recording active-hand selection in action-dynamics.
- Existing checkpoint exporters include actual TRAIN/VAL/TEST IDs and calibration
  provenance. `plot_action_forecast.py --ckpt` uses the saved TEST IDs and verifies
  the cache; it cannot invent a new test split for a model trained on those recordings.

The shared template can leave a location-dependent offset or remove weak pressure.
Those are generalization errors to measure on TEST; they are not grounds to refit the
template using TEST pressure. This protocol does not establish participant independence
across locations, or rule out unknown duplication in upstream datasets.

## Prepare real data

ActionSense must be re-extracted from timestamped HDF5. Old linearly interpolated maps
cannot be repaired by recalibrating their output, and old state-only caches are
insufficient. Use a **new** directory; all extracted clips now retain float32 pressure:

```sh
python scripts/actionsense/probe_actionsense.py \
  --data-dir /path/to/actionsense_hdf5 \
  --extract-states data/actionsense_raw_causal_v1 \
  --out docs/actionsense/train_only_v1/raw_probe.csv
```

OpenTouch needs uncorrected `clip_N.npy` maps plus `manifest.jsonl`. The original raw
cache can be used if complete; the D1 corrected cache cannot. Fresh extraction retains
float32 pressure:

```sh
python scripts/opentouch/extract_opentouch.py \
  --shard /path/to/shard.hdf5 --labels /path/to/final_annotations \
  --out data/opentouch_raw_v1
```

Extract each source once. Verify source identities before reusing old split IDs:
re-extraction can change which recordings are eligible or how their IDs are assigned.
Create a new split file for the new manifest, or explicitly map the old split through
the source file and activity interval. Never copy numeric IDs across manifests without
checking that they still identify the same recordings.

Copy the appropriate `configs/{actionsense,opentouch}/eval_harness.yaml` to a run config.
Set `paths.states_root` to the new raw cache and `paths.split_file` to a new split JSON
for that cache. Keep `calibration.mode: train_only`. Optional
`paths.calibration_cache` selects where derived caches live; the default is
`runs/calibration`. Relative paths are resolved from the repository root.

## Run and replay

Updated training/evaluation entrypoints prepare each fold automatically. For explicit
TRAIN/VAL/TEST JSON, the cache can also be prepared independently:

```sh
python -m src.calibration --config /path/to/run.yaml --splits /path/to/splits.json
```

The command prints the resolved configuration path. Each derived cache contains
`calibration.json`, `manifest.jsonl`, `splits.json`, corrected maps/states, and
`config_<hash>.yaml`. It records the fitted template, TRAIN IDs, raw map hashes,
sampling rule and version. Every evaluation config has a separate immutable filename,
so two horizon settings can share pressure data without overwriting each other's config.
An already prepared config is bound to its split; using it with another fold raises.

For ActionSense corpus CV, use the same config, recording population, seed and fold
count for neural models and baseline exports:

```sh
python scripts/actionsense/train_tactile_map.py --config /path/to/actionsense_run.yaml \
  --scope corpus --folds 5 --backbone probgru --encoders aggregate,flatten,cnn \
  --save-preds runs/train_only_v1/actionsense/neural
python scripts/actionsense/export_baseline_forecasts.py --config /path/to/actionsense_run.yaml \
  --scope corpus --folds 5 --seed 0 --out runs/train_only_v1/actionsense/baselines
```

The ActionSense `frozen` baseline export evaluates one frozen split; neural CV over the
frozen population is a different collection of folds. Do not merge their predictions
as if they shared a calibration. The new calibration-ID check rejects this mismatch.

For OpenTouch, retain the location split and use the TRAIN scope:

```sh
python scripts/opentouch/run_opentouch_exploratory.py --config /path/to/opentouch_run.yaml \
  --split-mode location --folds 5 --baseline-scope train --model pg_all \
  --save-preds runs/train_only_v1/opentouch/preds \
  --save-model runs/train_only_v1/opentouch/models
```

Use fresh output directories. Neural weights, normalization, classical baselines and
VAL sigma calibration must all be refitted before scoring the new targets. Default
metric paths now use `docs/<corpus>/train_only_v1/`. Historical D1 results/configs remain
legacy; their reported scores are not certified by the new tests. Do not overwrite old
results or relabel old checkpoints as TRAIN-only.

The ActionSense scorer's `--model-preds` archive contains numeric recording-ID keys
plus scalar `calibration_id` and JSON `split_ids`. Strict scoring requires matching
provenance and exactly the configured TEST IDs. Per-clip prediction exports also carry
calibration/split provenance; prediction merging rejects different calibration IDs.

## Validation and remaining real-data work

```sh
python -m pytest -q tests/test_train_only_calibration.py
python scripts/actionsense/check_leakage.py
```

Tests start from raw synthetic pressure: only TRAIN signal files may be opened by the
fitter; changing held-out future pressure leaves earlier maps/states/predictions
unchanged; VAL pressure cannot change the template; a new shard uses the fixed template;
training IDs must match calibration IDs; overlapping or invented TEST IDs are rejected.
Training/export integration checks exercise ActionSense's six encoder/backbone
combinations and OpenTouch's model families.

These are implementation regressions, not certification of an external corpus. At
implementation time this workstation had only incomplete legacy ActionSense maps and
no OpenTouch raw cache. Full re-extraction, retraining, evaluation and old/new score
comparison still require the actual raw-data host and paths. No real-corpus rerun has
been performed as part of the synthetic checks.

EgoTouch is outside this fix. A subsequent [EgoTouch audit](egotouch/calibration_audit.md)
found strong sample-based evidence that its released grids already use whole-episode
normalization upstream. Disabling our downstream baseline subtraction does not certify
those grids as causal; their raw-to-grid preprocessing needs separate verification.
