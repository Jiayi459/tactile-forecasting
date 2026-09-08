#!/bin/bash
# STREAMING ActionSense probe: download one wearables HDF5 (~2-4 GB each) -> probe it ->
# DELETE it -> next. Bounds disk to a single file, so it fits a small home quota.
# Per-clip metrics accumulate in acc.jsonl; a final --report-only pass prints the ranking.
# Subjects S00-S05 (tactile). Usage (from anywhere):
#   bash scripts/crc/stream_actionsense.sh [DEST_DIR]        # default ~/actionsense
#   CLIPS=all  bash scripts/crc/stream_actionsense.sh          # save EVERY activity's raw map
#   CLIPS=Pour,Slice,Peel bash scripts/crc/stream_actionsense.sh   # the old, narrow behaviour
#
# CLIPS defaults to `all`. It used to be hardcoded to "Pour,Slice,Peel", which is why the corpus
# carries F/CoP for all 299 recordings but raw maps for only 100 of them -- F/CoP is computed
# from the map in memory, and the map was then thrown away for every other activity, so the
# flatten/cnn arms could never run at corpus scope. The maps cost ~4 KB per frame (2 hands x
# 32 x 32 x float16) = ~1.3 GB for the whole corpus, which is nothing beside the ~47 GB this
# script downloads to produce them; discarding them was never the saving it looked like.
set -uo pipefail

REPO="$HOME/TouchAnything"
PROBE="$REPO/scripts/actionsense/probe_actionsense.py"
DEST="${1:-$HOME/actionsense}"
ACC="$DEST/acc.jsonl"

# numpy/h5py/scipy live in the `tactile` conda env, NOT in (base). Activate it, then FAIL FAST if
# the deps are missing -- otherwise every ~24 GB download is wasted on a probe that can't import numpy.
source "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate tactile 2>/dev/null || echo "WARN: 'conda activate tactile' failed; activate it manually"
python -c "import numpy, h5py, scipy" 2>/dev/null || {
  echo "ERROR: the tactile env is not active (import numpy/h5py/scipy failed)."
  echo "       Run 'conda activate tactile' first, then re-run this script."; exit 1; }
mkdir -p "$DEST"

CLIPS="${CLIPS:-all}"
if [ "$CLIPS" = "all" ]; then CLIP_ARG=(--save-all-clips); else CLIP_ARG=(--save-clips-for "$CLIPS"); fi
echo "clip policy: ${CLIP_ARG[*]}"

# One HDF5 is ~4.3 GB and is deleted before the next is fetched, so peak disk is a single file
# plus the outputs (~1.3 GB of maps + ~15 MB of states). Fail here rather than 40 GB into a
# download that cannot finish.
AVAIL_KB=$(df -Pk "$DEST" | awk 'NR==2{print $4}')
if [ "${AVAIL_KB:-0}" -lt 12000000 ]; then
  echo "ERROR: only $((AVAIL_KB / 1024)) MB free at $DEST; need ~12 GB (one 4.3 GB HDF5 at a"
  echo "       time + ~1.3 GB of maps + headroom). Point DEST_DIR at a larger allocation."
  exit 1
fi

# The recording index is a counter that increments across files, and splits.json plus every
# result under docs/ is keyed by it. Re-streaming rebuilds the manifest from scratch, so the
# numbering only survives if the URL list and the accept/reject logic are unchanged. Keep the
# old manifest and diff (idx, label) against the new one at the end: a renumbering must fail
# loudly here, not show up later as results silently attached to the wrong recordings.
OLD_MANIFEST=""
if [ -f "$DEST/states/manifest.jsonl" ]; then
  OLD_MANIFEST="$DEST/manifest.jsonl.before-restream"
  cp "$DEST/states/manifest.jsonl" "$OLD_MANIFEST"
  echo "kept previous manifest ($(wc -l < "$OLD_MANIFEST") rows) at $OLD_MANIFEST for comparison"
fi

rm -f "$DEST"/*.hdf5 "$ACC"          # clear any partial files + old accumulator
rm -rf "$DEST/states"                # clear old state extraction (avoid duplicate append)

URLS=(
  "https://data.csail.mit.edu/ActionNet/wearable_data/2022-06-07_experiment_S00/2022-06-07_18-10-55_actionNet-wearables_S00/2022-06-07_18-11-37_streamLog_actionNet-wearables_S00.hdf5"
  "https://data.csail.mit.edu/ActionNet/wearable_data/2022-06-13_experiment_S01_recordingStopped/2022-06-13_18-13-12_actionNet-wearables_S01/2022-06-13_18-14-59_streamLog_actionNet-wearables_S01.hdf5"
  "https://data.csail.mit.edu/ActionNet/wearable_data/2022-06-13_experiment_S02/2022-06-13_21-39-50_actionNet-wearables_S02/2022-06-13_21-40-16_streamLog_actionNet-wearables_S02.hdf5"
  "https://data.csail.mit.edu/ActionNet/wearable_data/2022-06-13_experiment_S02/2022-06-13_21-47-57_actionNet-wearables_S02/2022-06-13_21-48-24_streamLog_actionNet-wearables_S02.hdf5"
  "https://data.csail.mit.edu/ActionNet/wearable_data/2022-06-13_experiment_S02/2022-06-13_22-34-45_actionNet-wearables_S02/2022-06-13_22-35-11_streamLog_actionNet-wearables_S02.hdf5"
  "https://data.csail.mit.edu/ActionNet/wearable_data/2022-06-13_experiment_S02/2022-06-13_23-22-21_actionNet-wearables_S02/2022-06-13_23-22-44_streamLog_actionNet-wearables_S02.hdf5"
  "https://data.csail.mit.edu/ActionNet/wearable_data/2022-06-14_experiment_S03/2022-06-14_13-11-44_actionNet-wearables_S03/2022-06-14_13-12-07_streamLog_actionNet-wearables_S03.hdf5"
  "https://data.csail.mit.edu/ActionNet/wearable_data/2022-06-14_experiment_S03/2022-06-14_13-52-21_actionNet-wearables_S03/2022-06-14_13-52-57_streamLog_actionNet-wearables_S03.hdf5"
  "https://data.csail.mit.edu/ActionNet/wearable_data/2022-06-14_experiment_S04/2022-06-14_16-38-18_actionNet-wearables_S04/2022-06-14_16-38-43_streamLog_actionNet-wearables_S04.hdf5"
  "https://data.csail.mit.edu/ActionNet/wearable_data/2022-06-14_experiment_S05/2022-06-14_20-36-27_actionNet-wearables_S05/2022-06-14_20-36-54_streamLog_actionNet-wearables_S05.hdf5"
  "https://data.csail.mit.edu/ActionNet/wearable_data/2022-06-14_experiment_S05/2022-06-14_20-45-43_actionNet-wearables_S05/2022-06-14_20-46-12_streamLog_actionNet-wearables_S05.hdf5"
)
# (S00 first session is calibration-only / no activities; omitted.)

n=0
for URL in "${URLS[@]}"; do
  n=$((n + 1))
  f="$DEST/$(basename "$URL")"
  echo "=== [$n/${#URLS[@]}] downloading $(basename "$f") ==="
  if ! curl -fL --retry 3 -o "$f" "$URL"; then
    echo "  WARN: download failed (disk? net?), skipping"; rm -f "$f"; continue
  fi
  python "$PROBE" --data-dir "$DEST" --jsonl "$ACC" --extract-states "$DEST/states" \
      "${CLIP_ARG[@]}" || echo "  WARN: probe error on this file"
  # KEEP=1 retains the HDF5 (download once to a large /temp180 or /bluefs allocation and
  # re-process for free); default deletes each file to bound disk on the 100 GB home quota.
  [ "${KEEP:-0}" = "1" ] || rm -f "$f"
done

echo ""
echo "=== inventory ==="
echo "  recordings (manifest rows): $(wc -l < "$DEST/states/manifest.jsonl")"
echo "  state_*.npy: $(ls "$DEST"/states/state_*.npy 2>/dev/null | wc -l)"
echo "  clip_*.npy (raw maps): $(ls "$DEST"/states/clip_*.npy 2>/dev/null | wc -l)"

if [ -n "$OLD_MANIFEST" ]; then
  echo ""
  echo "=== recording-index check against the previous manifest ==="
  python "$REPO/scripts/actionsense/check_manifest_indices.py" \
      "$OLD_MANIFEST" "$DEST/states/manifest.jsonl" || {
    echo "FATAL: recording indices changed; see above."; exit 1; }
fi

echo ""
echo "=== aggregating all streamed clips ==="
python "$PROBE" --report-only --jsonl "$ACC" --out "$REPO/docs/predictability_actionsense.csv"
