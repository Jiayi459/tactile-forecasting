#!/bin/bash
# Stream the OpenTouch corpus: download one shard -> extract the tiny cache -> DELETE it.
#
# Peak disk stays ~1 GB instead of 14.6 GB. This is the pattern that rescued the
# ActionSense run after repeated ENOSPC/truncation failures (SESSION_LOG 2026-07-03) --
# and note that the previous full-download OpenTouch copy is exactly the one that got
# deleted during that saga. Do not "simplify" this back to download-all-then-extract.
#
# Google Drive rate-limits bulk pulls, so a failed shard is LOGGED and skipped, never
# fatal; rerunning picks up only what is missing (extraction is per-shard and append-only).

# Usage:
#   pip install gdown h5py numpy
#   bash scripts/crc/stream_opentouch.sh [WORKDIR]     # default ~/opentouch
# Rerun the same command to retry whatever failed.
#
# Output: $WORKDIR/cache/{state_N.npy, clip_N.npy, pose_N.npy, manifest.jsonl}  (~100-150 MB)
# That cache is the deliverable -- rsync it to data/opentouch_states/ and the shards are
# never needed again.
set -uo pipefail

WORK="${1:-$HOME/opentouch}"
CACHE="$WORK/cache"
LABELS="$WORK/final_annotations"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PYTHON:-python}"

# --- single-instance lock -----------------------------------------------------
# 2026-08-10: two concurrent runs corrupted the first cache. Both saw an empty
# done_ids.txt and re-extracted the same shards; both computed the same next index
# from the same manifest and OVERWROTE each other's state_N.npy; and each wiped the
# shared shard dir at the top of its loop, deleting the other's download (which is
# what the 19 spurious "failures" were). Result: 4,511 manifest rows for a 2,958-clip
# corpus, with a broken row->file mapping. Never again: mkdir is atomic on NFS.
LOCK="$WORK/.stream.lock"
mkdir -p "$WORK"
if ! mkdir "$LOCK" 2>/dev/null; then
    echo "FATAL: another run holds $LOCK"
    echo "  If no stream_opentouch.sh is running (check: pgrep -fa stream_opentouch.sh),"
    echo "  remove it with:  rmdir $LOCK"
    exit 1
fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT INT TERM

# Per-instance shard dir, so even a forced second run cannot delete our download.
SHARDS="$WORK/shards.$$"
mkdir -p "$SHARDS" "$CACHE"
trap 'rm -rf "$SHARDS"; rmdir "$LOCK" 2>/dev/null' EXIT INT TERM

IDS=(
    "1EjMOzs45devBo0TqhuhZTT_Ll7HZ1lrW" "1fAmmieSr0yFm7ldhW7Smld7jUxBCw8fu"
    "1cUhgYbredkIRswanUiM5uDixiFLq4WCC" "1PCzWMJxtbD2HJLCl2WFzOIB-5RN3X81G"
    "1jFlYmCFb6GldbPJ-zLzJSCY-BKipdjPE" "1reSqa8v8RaY2kZXLw0_g7Amvq7lJl6Cu"
    "1atXpcctoHs4dbXhyAAO9EY88D2f1JYfT" "1Z3b-I6BMPgNlpiKw8gISkUi3VULUtLFN"
    "1u-6WGn3eMQJe3eh6lCFahlIcEVmkULna" "17wF0aBIH6RRtRGRaXeiI-Y4Lh5bnDFBL"
    "1KICpqtfmbnKhgHi-CIR9XAp24TE1945M" "1vkl6wat_dgF5NQs9QVDfCyJGyjEjd2FW"
    "1BbKU5vSH-wOrCnOjRWNe3H7niJP_uJrb" "1GCX4mAgCvOvmIQ0uXotqpoNdXgYzp4ki"
    "1rxsLWGw_diPvRnALxOYakCIweG90O28I" "1zAYfcMt2hqcG1bPtCOAkWu0zsd6lfvrX"
    "1tQh21z8KRxYHsh69dW6VcSw5Wux67R6_" "1jeA1bEit-tDQpfwt3NmTeC8iwM6I1qiE"
    "1UT5htydKCfBCO57On-mRJRz7mSi57K4u" "1h9Bl8CTGJWvU2XPr93fptBTpB2cwYgwq"
    "1SAbxWQZDEyTZ-ESVi9G5bxEc7ov-EO28" "1jKyVNsi7fsofSho_xoRi0Kgqem4zrk5F"
    "11LQ28c6jPhNfiu9fPDu5diruNUCa0bGM" "1dwlVYtBfyNUHg7Qxnxa_iYBYPCcn9VeX"
    "1X4-MS7Qodhtmn6zcY9a5cMq02eDLvOJq" "1VAKXJPO4j_40hpqslNJ4_WbgWfKaGLQC"
)
LABELS_ID="1cM-816vcCnkgWVIGXZrR1o8TPsDvRVCZ"
FAILED="$WORK/failed_ids.txt"
DONE="$WORK/done_ids.txt"
touch "$DONE"; : > "$FAILED"
sort -u -o "$DONE" "$DONE"          # collapse any duplicates left by earlier runs

# ---- labels first (459 KB) -------------------------------------------------
if [ ! -d "$LABELS" ]; then
    echo "== labels =="
    ( cd "$WORK" && gdown "$LABELS_ID" && unzip -o -q final_annotation.zip )
    [ -d "$WORK/final_annotations" ] || { echo "FATAL: annotations missing"; exit 1; }
fi

# ---- shards, one at a time -------------------------------------------------
n=0
for ID in "${IDS[@]}"; do
    n=$((n + 1))
    if grep -qx "$ID" "$DONE"; then echo "[$n/${#IDS[@]}] $ID already done"; continue; fi
    echo "== [$n/${#IDS[@]}] $ID =="
    rm -f "$SHARDS"/*.hdf5 "$SHARDS"/*.h5 2>/dev/null

    if ! ( cd "$SHARDS" && gdown "$ID" ); then
        echo "  WARN download failed (Drive quota?) -> logged, continuing"
        echo "$ID" >> "$FAILED"; continue
    fi
    SHARD="$(ls -1 "$SHARDS"/*.hdf5 "$SHARDS"/*.h5 2>/dev/null | head -1)"
    if [ -z "$SHARD" ]; then
        echo "  WARN no hdf5 produced -> logged, continuing"
        echo "$ID" >> "$FAILED"; continue
    fi
    echo "  extracting $(basename "$SHARD") ($(du -h "$SHARD" | cut -f1))"

    # --taxel-stats only on the first shard: it answers the dead-taxel and DC-offset
    # questions once, and is pure overhead thereafter.
    EXTRA=""; [ "$n" -eq 1 ] && EXTRA="--taxel-stats"
    if $PY "$REPO/scripts/opentouch/extract_opentouch.py" --shard "$SHARD" \
            --labels "$LABELS" --out "$CACHE" $EXTRA; then
        echo "$ID" >> "$DONE"
    else
        echo "  WARN extraction failed -> logged, continuing"
        echo "$ID" >> "$FAILED"
    fi
    rm -f "$SHARD"                       # <-- the whole point
done

echo
echo "== summary =="
echo "  cache:  $CACHE  ($(du -sh "$CACHE" 2>/dev/null | cut -f1), \
$(ls "$CACHE"/state_*.npy 2>/dev/null | wc -l) clips)"
echo "  done:   $(wc -l < "$DONE") / ${#IDS[@]} shards"
if [ -s "$FAILED" ]; then
    echo "  FAILED: $(wc -l < "$FAILED") -> rerun this same script to retry:"
    sed 's/^/    /' "$FAILED"
else
    echo "  FAILED: none"
fi
echo
echo "Next: rsync -avz <netid>@crcfe01.crc.nd.edu:$CACHE/ data/opentouch_states/"
