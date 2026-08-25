#!/bin/bash
# Submit the per-category tactile-forecasting sweep on ND CRC (UGE).
# Run from the repo ROOT on a CRC front-end, AFTER: env built, full EgoTouch npz staged,
# `mkdir -p logs`. One qsub per (category, fold).
#
#   bash scripts/crc/run_percategory.sh
#   CONFIG=configs/tactile_pixel/convgru.yaml bash scripts/crc/run_percategory.sh   # different model
#   FOLDS="0 1 2" CATS="Cut Spray" bash scripts/crc/run_percategory.sh        # subset
#
# Categories ranked most/least predictable by the training-free probe (docs/ACTION_CATEGORIES.md).
# This sweep MEASURES the real skill-over-persistence to confirm or break that ranking.
set -euo pipefail

CONFIG="${CONFIG:-configs/tactile_pixel/simvp.yaml}"
PROTOCOL="${PROTOCOL:-lto}"
FOLDS="${FOLDS:-0 1 2 3 4}"
# Informative spread: top-PI (periodic) ... bottom-PI (holds / make-break contact).
CATS="${CATS:-Cut Spray Wash/Clean Take/Retrieve Grasp/Hold/Lift Squeeze Pick-up Press/Click Plug/Unplug/Insert}"

echo "config=$CONFIG protocol=$PROTOCOL folds=[$FOLDS]"
n=0
for c in $CATS; do
  for f in $FOLDS; do
    qsub -v CATEGORY="$c",FOLD="$f",CONFIG="$CONFIG",PROTOCOL="$PROTOCOL" \
         scripts/crc/percategory_gpu.job
    n=$((n + 1))
  done
done
echo "submitted $n jobs. monitor: qstat -u \$USER   |   collect: python scripts/shared/aggregate_results.py"
