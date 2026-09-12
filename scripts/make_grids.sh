#!/usr/bin/env bash
# Build one 3x3 forecast grid per sensor/split, picking each clip by the same criteria.
#
# Run it from the repo root with the environment already active (`conda activate tactile`).
# With no arguments it builds every figure; name one or more to rebuild only those, e.g.
#   bash scripts/make_grids.sh ego_seen ego_unseen
# Pure numpy + matplotlib, seconds per figure, so the login node is fine -- no qsub.
#
# A source directory that is absent or empty is REPORTED AND SKIPPED, not fatal: the three
# ActionSense encoder directories have been empty on the cluster since Sep 8, and one missing
# sensor should not cost you the other three figures.
#
# Deliberately NOT listed below: runs/preds_d1_map. Its flatten/cnn numbers were withdrawn
# after an encoder bug (SESSION_LOG 2026-08-23) and d1_map2 is the rerun, but the withdrawn
# npz still sit on the cluster under a near-identical name.
set -uo pipefail

OUT=${OUT:-docs/grids}
CHANNEL=${CHANNEL:-F_R}
mkdir -p "$OUT"

# Which figures to build: all of them, or only the names given on the command line.
WANT=("$@")
wanted () {
  [ ${#WANT[@]} -eq 0 ] && return 0
  local w
  for w in "${WANT[@]}"; do [ "$w" = "$1" ] && return 0; done
  return 1
}

grid () {
  local ds=$1 name=$2; shift 2
  wanted "$name" || return 0
  local preds=() d
  for d in "$@"; do
    if [ -n "$(ls "$d"/clip_*.npz 2>/dev/null | head -1)" ]; then
      preds+=(--preds "$d")
    else
      echo "  [skip source] $d is missing or holds no clip_*.npz"
    fi
  done
  echo "=== $name ($ds) ==="
  if [ ${#preds[@]} -eq 0 ]; then
    echo "  [SKIP] no usable source directory; no figure for $name"
    return 0
  fi
  python scripts/pick_grid_clip.py --dataset "$ds" "${preds[@]}" \
      --top 10 --stage "runs/grid_stage_$name" || { echo "  [FAIL] picker"; return 0; }
  local clip
  clip=$(ls "runs/grid_stage_$name"/*/clip_*.npz 2>/dev/null | head -1 \
         | sed 's/.*clip_//; s/\.npz//')
  if [ -z "$clip" ]; then echo "  [FAIL] nothing staged"; return 0; fi
  local out="$OUT/${name}_clip${clip}_${CHANNEL}.png"
  python scripts/actionsense/plot_clip_model_grid.py --dataset "$ds" "${preds[@]}" \
      --clip "$clip" --channel "$CHANNEL" --out "$out" || { echo "  [FAIL] plot"; return 0; }
  # The filename carries the clip number, so a rebuild that picks a DIFFERENT clip would
  # otherwise leave the old figure sitting beside the new one, both looking current. Drop the
  # stale ones only after the new figure exists.
  local old
  for old in "$OUT/${name}_clip"*"_${CHANNEL}.png"; do
    [ -e "$old" ] || continue
    [ "$old" = "$out" ] && continue
    rm -f "$old" && echo "  [replaced] $(basename "$old")"
  done
}

grid opentouch   opentouch   runs/preds_d1_map2 runs/preds_d1_pg
grid egotouch    ego_seen    runs/egotouch_merged/egotouch_test_seen_3s
grid egotouch    ego_unseen  runs/egotouch_merged/egotouch_test_unseen_3s
grid actionsense actionsense runs/as_preds_seq2seq_corpus runs/as_preds_probgru_corpus \
                             runs/as_preds_baselines_corpus

echo
if [ ${#WANT[@]} -gt 0 ]; then echo "rebuilt only: ${WANT[*]}"; fi
echo "figures now in $OUT:"
ls -1 "$OUT" 2>/dev/null | sed 's/^/  /'
