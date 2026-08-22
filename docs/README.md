# docs/ layout

Reorganised 2026-08-22. Results are filed by sensor, and OpenTouch results by the run that
produced them, because "which run is this number from" had become the question every table
raised and none answered.

```
docs/
  *.md                      project-level: conclusions, plans, the cross-sensor comparison
  actionsense/              every ActionSense result (harness, tactile_map, action_dynamics)
  opentouch/
    raw/                    4-fold, uncorrected target                    (2026-08-17)
    df/                     4-fold, raw+df feature ablation               (2026-08-18)
    d1/                     4-fold, D1 baseline-corrected, NLL-selected   (2026-08-20)
    d1_mse/                 as d1 but weights selected on VAL MSE         (2026-08-21)
    d1_map/                 map arms: aggregate / flatten / cnn, on D1    (2026-08-22)
    exploratory/            geometry and signal probes, not tied to a fold run
```

## Reading numbers across runs

**Absolute errors are not comparable between `raw`/`df` and any `d1*` run.** The D1
correction removes a per-taxel baseline that was 99.78% of F, so the target itself shrank
about 350-fold and MSE with it. Skill against persistence *is* comparable, because numerator
and denominator come from the same data.

**Two different "skill" numbers exist and are not interchangeable.** `opentouch_cv4*.csv`
carries the driver's frame-pooled `SS_vs_persistence`; `opentouch_report*.csv` carries the
report's per-clip-equal-weight `skill`. For the same arm and channel they differ (ar on F:
0.367 against 0.302). The report convention is the one that matches the R2 and dR2 tables.

Run names match the `--save-preds` / `--out` tags used on CRC, so a figure can be traced to
`runs/preds_<name>` and to the SESSION_LOG entry of the same date.
