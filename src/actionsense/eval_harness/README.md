# Frozen tactile-forecast evaluation harness

The single, frozen way every tactile-forecasting model is scored. Metrics, masking, splits, and
the classical baselines are defined **once** here and imported everywhere — do not re-implement
them per model.

## What it does

- **Target**: the RAW 6-dim both-hands vector `[F_L, CoPx_L, CoPy_L, F_R, CoPx_R, CoPy_R]`
  (physical moments straight from `state_N.npy`), at **10 Hz**. Horizon = **1 s = 10 steps**.
- **Split**: frozen 3-way `data/actionsense_states/splits.json` (60/20/20 by recording,
  stratified by activity × object). Fit on TRAIN, select hyperparameters on VAL, TEST touched once.
- **Baselines**: persistence, seasonal-naive (period per activity×object from TRAIN autocorrelation,
  fallback→persistence), linear AR (statsmodels AutoReg, order selected on VAL).
- **Masking**: CoP metrics exclude target frames where that hand's raw force is below the TRAIN
  5th-percentile; force channels are never masked. One function, `masking.valid_mask`.
- **Metrics**: MSE, MAE, and skill `1 - MSE/MSE_baseline` vs persistence, seasonal, and AR — all on
  identical masked frame sets. Output is a tidy long table (CSV **+** parquet):
  `[model, channel, hand, horizon_step, metric, value, n_frames, config_hash]`.

All operations are **causal** (no `filtfilt`; a forecast issued at origin *t* reads only data ≤ *t*)
and **deterministic** (the run asserts two passes are identical). Everything is controlled by
`configs/actionsense/eval_harness.yaml`; its sha256 is stamped into every result row.

## Run the baselines

```bash
python -m src.actionsense.eval_harness.evaluate      # -> docs/actionsense/harness_baselines.csv (+ .parquet)
python scripts/actionsense/plot_harness.py                            # -> docs/harness_skill_{bars,curves}.png
pytest tests/test_harness.py                              # synthetic-signal unit tests
```

## Evaluating a NEW model against this harness

The harness scores **predictions**, so it never needs your model code. Your model must produce, for
each TEST recording, a forecast for every rolling origin in the standard **target-time-indexed**
format:

- For recording `idx`, the valid origins are `baselines.origins(len(Y), cfg)` (in order).
- Emit `yhat[idx]` of shape `(n_origins, H, 6)`: for the j-th origin `t`, `yhat[idx][j]` is the
  forecast for target times `t+1 … t+H` (h = 1..H), in raw target units, channel order
  `[F_L, CoPx_L, CoPy_L, F_R, CoPx_R, CoPy_R]`. Predictions must use only data ≤ t (causal).

Save them as an `.npz` keyed by recording index and score:

```python
import numpy as np
np.savez("preds.npz", **{str(idx): yhat_idx for idx, yhat_idx in preds.items()})
```
```bash
python -m src.actionsense.eval_harness.evaluate --model-preds preds.npz --model-name gru
```

This scores your model with the **same** mask, splits, and metrics, and reports its skill vs all
three baselines in the same table. (The existing probGRU predicts a different target — the 3-dim
high-pass component of one hand — so it must first be re-scoped to the raw 6-dim both-hands target
before it can be scored here.)
