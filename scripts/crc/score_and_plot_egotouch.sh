#!/bin/bash
# Score and plot the finished EgoTouch runs. NO GPU, NO retraining -- reads only what the two
# drivers wrote, so this belongs on a login node (or locally after rsync), not in a queue.
#
#   bash scripts/crc/score_and_plot_egotouch.sh
#   bash scripts/crc/score_and_plot_egotouch.sh 1s          # one history only
#
# For each (split, history) it MERGES the reference ladder into the neural arms and scores the
# merged directory ONCE. That merge is the point: W6 requires baselines and neural models to be
# judged by one scorer, one mask, over one set of recordings and origins. Scoring them from
# separate directories would reintroduce the two-path problem the merge exists to close --
# merge_preds.py asserts y, origins and channel order are equal across sources, so a mismatch
# is a hard error rather than a table quietly comparing unlike things.
#
# TWO DELIBERATE DIFFERENCES FROM scripts/crc/score_and_plot_runs.sh (ActionSense):
#   1. --thresholds, not --mask none. ActionSense cannot rebuild its per-fold TRAIN force
#      threshold from a preds directory (no fold membership is recorded), so it scores unmasked.
#      EgoTouch has ONE split and a frozen mask_thresholds.json, so CoP is masked by the
#      harness's own TRAIN-fitted rule -- the protocol the config declares.
#   2. test_seen and test_unseen are scored SEPARATELY and never merged. They are different
#      populations answering different questions, and test_unseen covers only 8 (action,object)
#      groups at 34.7% of the official unseen split -- see the caveats printed at the end.
set -euo pipefail

PY="${PY:-python}"
NEURAL_ROOT="${NEURAL_ROOT:-runs/egotouch_tactile_map}"
BASE_ROOT="${BASE_ROOT:-runs/egotouch_baselines}"
OUT_ROOT="${OUT_ROOT:-docs/egotouch/results}"
MERGE_ROOT="${MERGE_ROOT:-runs/egotouch_merged}"
THR="${THR:-data/egotouch_states/mask_thresholds.json}"

[ -f "$THR" ] || { echo "FATAL: $THR missing -- run scripts/shared/export_mask_thresholds.py"; exit 1; }
HISTS=("$@"); [ "${#HISTS[@]}" -gt 0 ] || HISTS=(1s 3s)

echo "=================================================================="
echo "INVENTORY -- check the counts before trusting anything below"
echo "=================================================================="
echo "mask thresholds: $($PY -c "import json;d=json.load(open('$THR'));print(d['force_thresholds'],'pct',d['percentile'],'fit_on',d['fit_on'],'n_train',d['n_train_recordings'])")"
for d in "$BASE_ROOT"/test_seen "$BASE_ROOT"/test_unseen "$NEURAL_ROOT"/*_*s; do
    [ -d "$d" ] || continue
    $PY - "$d" <<'PYINV'
import glob, os, sys, collections, numpy as np
d = sys.argv[1]
fs = sorted(glob.glob(os.path.join(d, "clip_*.npz")))
if not fs:
    print(f"  {d:44s} EMPTY"); raise SystemExit
acts, models, nor = collections.Counter(), collections.Counter(), 0
for f in fs:
    z = np.load(f, allow_pickle=True)
    acts[str(z["action"]) if "action" in z.files else ""] += 1
    nor += len(z["origins"])
    for k in z.files:
        if k.startswith("mu_"):
            models[k[3:]] += 1
z0 = np.load(fs[0], allow_pickle=True)
print(f"  {d:44s} {len(fs):4d} recordings  {nor:7d} origins  {len(acts):3d} verbs  "
      f"C={len(z0['channels'])}")
print(f"      arms: " + ", ".join(f"{m}({n}/{len(fs)})" for m, n in sorted(models.items())))
PYINV
done

for split in test_seen test_unseen; do
    for h in "${HISTS[@]}"; do
        NEU="$NEURAL_ROOT/${split}_${h}"
        BAS="$BASE_ROOT/$split"
        [ -d "$NEU" ] || { echo "SKIP $split $h: $NEU missing"; continue; }
        [ -d "$BAS" ] || { echo "SKIP $split $h: $BAS missing"; continue; }
        run="egotouch_${split}_${h}"
        MERGED="$MERGE_ROOT/$run"
        OUT="$OUT_ROOT/$split"
        mkdir -p "$OUT"

        echo ""
        echo "=================================================================="
        echo "$run  ->  $OUT"
        echo "=================================================================="
        rm -rf "$MERGED"
        $PY scripts/shared/merge_preds.py --out "$MERGED" "$BAS" "$NEU"

        # Per-action R2 / skill / Hausdorff, clip-balanced, TRAIN-fitted CoP mask.
        # Persistence is synthesized by the scorer from y and origins, so every arm above --
        # baselines and neural alike -- is measured against the identical reference.
        $PY scripts/shared/score_preds_per_action.py \
            --preds "$MERGED" --label "EgoTouch $split ${h} history" \
            --thresholds "$THR" --out "$OUT"

        # One overlay figure per channel; --band because every neural arm is probabilistic and
        # mu alone understates them. Same plotter as OpenTouch and ActionSense by construction.
        $PY scripts/opentouch/plot_opentouch_forecast_overlay.py \
            --preds "$MERGED" --n-clips 3 --band --diverse --out-prefix "$OUT/$run"
    done
done

echo ""
echo "=================================================================="
echo "CAVEATS THAT MUST TRAVEL WITH THESE NUMBERS"
echo "=================================================================="
echo "  * F_L/F_R are AGGREGATE NORMALIZED PRESSURE (P_Sigma), not newtons: EgoTouch grids ship"
echo "    normalized by tactile_max and may mix tactile and bending channels. Skill and R2 are"
echo "    scale-free so they compare across sensors; raw F magnitudes do not."
echo "  * test_unseen covers 8 (action,object) groups and 56 of the official 147 recordings"
echo "    (38.1% at min_history=30). Report it as ONE aggregate number, never a per-action table,"
echo "    and split it"
echo "    by verb-known vs verb-OTHER: 35 of its 85 recordings ride the OTHER action embedding,"
echo "    which only 7 of 1458 TRAIN recordings ever trained. Only the probgru arm reads it."
echo ""
echo "  git add docs/egotouch/results && git commit && git push origin main"
echo "  # /runs/ is gitignored: rsync the .npz forecasts, do not commit them."
