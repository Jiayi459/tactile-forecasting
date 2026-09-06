# ActionSense run artifacts — how these folders are named

The 94 files here vary along **two independent axes**. The folder names are
`<scope>-<input>`, and it is worth being explicit that `map`/`flat` and `corpus` are not the
same kind of word: the first two say *what the encoder reads*, the third says *which
recordings were scored*.

## Axis 1 — input representation (what the per-frame encoder reads)

| name | input per frame | encoder | what it tests |
|---|---|---|---|
| `aggregate` | the 6-dim F/CoP signal itself, `[F, CoPx, CoPy] × {L,R}` | `Linear(6→64)+ReLU` | the neural AR: can the target's own history predict it? |
| `flatten` | the raw pressure map, `2×32×32 = 2048` values | `Linear(2048→64)+ReLU` | does the map help **without** exploiting spatial structure? |
| `cnn` (a.k.a. `map`) | the same map | small conv stack, stride-2 twice | does spatial structure help on top of that? |

All three feed an **identical** GRU and head, so the encoder is the only variable
([`models.py`](../../../src/actionsense/tactile_map/models.py)). `aggregate` never reads
`clip_*.npy` at all; the other two do, which is why they are ~340× more expensive per sample
and why only they need the raw maps to exist on disk.

## Axis 2 — population scope (which recordings were scored)

| name | recordings | actions | protocol |
|---|---|---|---|
| `frozen` | **75** | **2** (slice, peel) | the frozen harness: `actions: [slice, peel]` in [`eval_harness.yaml`](../../../configs/actionsense/eval_harness.yaml). Every previously published ActionSense number lives here. |
| `corpus` | **290** | **14** | every manifest recording with a state file, 5-fold CV by recording. **Exploratory.** |

`corpus` is **not** a superset result. Widening the population changes the `Norm`, the
class-mean denominator and the CV folds, so a corpus number and a frozen number must never
share a table. Nothing about the frozen protocol was altered to produce it:
`cross_validate` takes its recording list directly and never reads `cfg.raw["actions"]`
(only `splits.py` does).

Only `aggregate` has a corpus run. The map arms cannot: just **100** of the 299 recordings
have `clip_*.npy`, so `--scope corpus` with a map encoder would silently score a different,
smaller population than the aggregate arm and the two would not be comparable.

## What is in each folder

| folder | runs |
|---|---|
| `corpus-aggregate/` | the four new runs — `{seq2seq, probgru} × {1 s, 3 s}` — plus their `cv_*.csv` (run-level skill / Hausdorff / coverage per forecast step) and the `compare_backbones_*` overlays |
| `frozen-aggregate/` | `as_preds_seq2seq`, `as_preds_probgru` |
| `frozen-flatten/` | `as_preds_s2s_flat`, `as_preds_pg_flat` |
| `frozen-cnn/` | `as_preds_probgru_map`, and `as_preds_all` (both CNN arms merged, with the baselines on the 15-recording TEST split only) |
| `loss-curves/` | per-epoch train/val/test NLL and MSE, one figure per corpus run |

Per run: `<run>.csv` + `<run>.md` are the per-action table
([`score_preds_per_action.py`](../../../scripts/shared/score_preds_per_action.py)) and
`<run>_<channel>.png` are the forecast overlays, one figure per channel — six here, because
ActionSense instruments **two** gloves.

## Two traps recorded in the file names

- **`as_preds_all` carries its baselines on 15 of 75 recordings**, not 75: `ar`, `persistence`
  and `seasonal` are exported for the frozen TEST split alone, while a cross-validated arm
  covers every recording. Scoring them together therefore drops to 15 recordings, and the
  scorer says so rather than silently mixing populations.
- **`_h1` means 1 s of history, not "run 1."** History is `t_in` frames at 10 Hz: 1 s = 10,
  3 s = 30. Both are below the harness's `min_history = 40`, so **neither ever zero-pads a
  window** — which is why 10 s was deliberately not run here (it pads 96 % of `pour`).
