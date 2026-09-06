#!/bin/bash
# Score and plot every finished ActionSense run. NO GPU, NO retraining -- reads only what
# --save-preds wrote, so it belongs on a login node, not in a queue.
#
#   bash scripts/crc/score_and_plot_runs.sh
#   bash scripts/crc/score_and_plot_runs.sh runs/as_preds_probgru_corpus     # one run
#
# Discovers runs by glob, so it does not need to know how each was named. For every
# runs/as_preds_*/ it writes:
#   docs/actionsense/per_action/<run>.csv|.md    per-action R2 / skill / Hausdorff
#   docs/actionsense/forecast/<run>_<chan>.png   real signal vs the 1 s forecast, per channel
#
# WHY THE FORECAST PLOTTER IS THE OPENTOUCH ONE. src/actionsense/tactile_map/train.py::
# save_predictions writes "one clip_<idx>.npz per recording, in the OpenTouch overlay format"
# precisely so one plotter serves both sensors. scripts/actionsense/plot_forecast_overlay.py is
# NOT the tool here: it drives the older action_dynamics pipeline (fast-component R^3 target)
# and retrains, so it would draw a different model than the one these runs produced.
set -euo pipefail

PY="${PY:-python}"
# Destination is DERIVED from the predictions, not hardcoded: docs/actionsense/results is
# laid out as <scope>-<input>, and a run knows both. Hardcoding one output directory is what
# made this script write into the pre-reorganisation paths after the files had been moved.
OUT_ROOT="${OUT_ROOT:-docs/actionsense/results}"

TOUCHED=""
if [ "$#" -gt 0 ]; then DIRS=("$@"); else DIRS=(runs/as_preds_*/); fi
[ -e "${DIRS[0]}" ] || { echo "FATAL: no runs/as_preds_*/ found. Did --save-preds write here?"; exit 1; }

# INVENTORY FIRST. /runs/ is gitignored and survives between jobs, so a stale directory from
# an earlier, differently-scoped run sits here looking exactly like a fresh one. Printing the
# recording count, the action count and the model list up front makes a wrong scope obvious
# before an hour of scoring: the corpus scope is ~290 recordings over 14 actions, the frozen
# slice+peel split is 75 over 2.
echo "=================================================================="
echo "INVENTORY -- check the counts before trusting anything below"
echo "=================================================================="
for d in "${DIRS[@]}"; do
    d="${d%/}"
    $PY - "$d" <<'PYINV'
import glob, os, sys, collections, numpy as np
d = sys.argv[1]
fs = sorted(glob.glob(os.path.join(d, "clip_*.npz")))
if not fs:
    print(f"  {os.path.basename(d):34s} EMPTY"); raise SystemExit
acts, models = collections.Counter(), collections.Counter()
for f in fs:
    z = np.load(f, allow_pickle=True)
    acts[str(z["action"]) if "action" in z.files else ""] += 1
    for k in z.files:
        if k.startswith("mu_"):
            models[k[3:]] += 1
z0 = np.load(fs[0], allow_pickle=True)
tag = str(z0["tag"]) if "tag" in z0.files else "?"
print(f"  {os.path.basename(d):34s} {len(fs):4d} recordings  {len(acts):2d} actions  "
      f"tag={tag}  C={len(z0['channels'])}")
print(f"      models: " + ", ".join(f"{m}({n}/{len(fs)})" for m, n in sorted(models.items())))
print(f"      actions: " + ", ".join(f"{a or '?'}({n})" for a, n in acts.most_common()))
PYINV
done
echo ""

for d in "${DIRS[@]}"; do
    d="${d%/}"
    run="$(basename "$d")"
    n=$(ls "$d"/clip_*.npz 2>/dev/null | wc -l)
    if [ "$n" -eq 0 ]; then echo "SKIP $run: no clip_*.npz"; continue; fi
    sub=$($PY - "$d" <<'PYDEST'
import glob, os, sys, collections, numpy as np
d = sys.argv[1]
fs = sorted(glob.glob(os.path.join(d, "clip_*.npz")))
acts, models = set(), set()
for f in fs:
    z = np.load(f, allow_pickle=True)
    acts.add(str(z["action"]) if "action" in z.files else "")
    models |= {k[3:] for k in z.files if k.startswith("mu_")}
# scope: the frozen harness is slice+peel over 75 recordings; anything wider is the corpus run
scope = "corpus" if len(acts) > 2 else "frozen"
joined = " ".join(sorted(models))
inp = ("aggregate" if "aggregate" in joined else
       "flatten" if "flatten" in joined else
       "cnn" if "cnn" in joined else "other")
print(f"{scope}-{inp}")
PYDEST
)
    OUT="$OUT_ROOT/$sub"
    mkdir -p "$OUT"
    TOUCHED="$TOUCHED $OUT/$run"
    echo ""
    echo "=================================================================="
    echo "$run  ($n recordings of forecasts)  ->  $OUT"
    echo "=================================================================="

    # Per-action R2 / skill / Hausdorff. --mask none is the default and is stated in the
    # output: the harness's per-fold TRAIN force threshold cannot be rebuilt from a preds
    # directory, which records no fold membership.
    $PY scripts/shared/score_preds_per_action.py \
        --preds "$d" --label "ActionSense - $run" --out "$OUT"

    # One figure per channel. ActionSense has six (two hands); the plotter falls back to the
    # channel name for the ones its LABELS table does not know, so all six come out.
    # --band matters here: these are probabilistic arms, and mu alone understates them.
    $PY scripts/opentouch/plot_opentouch_forecast_overlay.py \
        --preds "$d" --n-clips 3 --band --out-prefix "$OUT/$run"
done

echo ""
echo "=================================================================="
echo "wrote (under $OUT_ROOT):"
for stem in $TOUCHED; do
    ls -1 "$stem".csv "$stem".md "$stem"_*.png 2>/dev/null | sed 's/^/  /'
done
echo ""
echo "Commit the SMALL artifacts only. /runs/ is gitignored on purpose"
echo "('Training outputs / job logs (rsynced back from CRC)'), so the .npz forecasts"
echo "should travel by rsync, not by git -- roughly 30 MB per run, and git history is forever."
echo ""
echo "  git add docs/actionsense/results"
echo "  git commit -m 'ActionSense corpus runs: per-action tables, forecast overlays, loss curves'"
echo "  git push origin main"
echo ""
echo "Then, from the LOCAL machine, for the forecasts themselves:"
echo "  rsync -avz -e 'ssh -J jhao3@bastion.crc.nd.edu' \\"
echo "    jhao3@crcfe01.crc.nd.edu:'~/TouchAnything/runs/as_preds_*' runs/"
