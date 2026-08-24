#!/bin/bash
# Stream-download shards from an EXTERNAL id list (one Drive file ID or share-link per line),
# instead of the hardcoded 26-ID array in stream_opentouch.sh.
#
# Use this after "Make a copy" of the shared OpenTouch shards into your OWN Google Drive: the
# copies are brand-new files under your own quota, independent of the original shared file's
# "too many users have downloaded this recently" throttle (SESSION_LOG 2026-08-12). Order in
# the id file does not matter -- extraction keys off the HDF5's own internal scene name, not
# the Drive file ID or filename.
#
# Same safety design as stream_opentouch.sh: single-instance lock, per-instance shard dir,
# one shard on disk at a time (download -> extract -> delete -> next), resumable via
# done_ids.txt/failed_ids.txt. Labels are NOT re-fetched here -- the small labels zip never
# hit the quota problem; run stream_opentouch.sh once (or by hand) first if
# $WORK/final_annotations doesn't exist yet.
#
# Usage:
#   bash scripts/crc/download_own_copies.sh IDS_FILE [WORKDIR]
#   IDS_FILE: one line per shard, "<Drive file ID or share-link>  [original filename]".
#             Blank lines and '#' comments (whole-line or trailing) are ignored.
#
# Give the filename column whenever the Drive copies were renamed. gdown otherwise names
# the download after the Drive title, and extract_opentouch.py takes the shard's identity
# -- which is also its clip dedup key (`<stem>::<group>`) -- from that local filename's
# stem. Copies titled e.g. "opentouch_shard_07" would therefore both break the *.hdf5
# glob below and, if forced to a single generic name, collide every shard onto one stem
# and silently drop clips. data/own_copy_ids_full.txt carries the real names.
set -uo pipefail

IDS_FILE="${1:?usage: download_own_copies.sh IDS_FILE [WORKDIR]}"
WORK="${2:-$HOME/opentouch}"
CACHE="$WORK/cache"
LABELS="$WORK/final_annotations"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PYTHON:-python}"

[ -f "$IDS_FILE" ] || { echo "FATAL: id file not found: $IDS_FILE"; exit 1; }
[ -d "$LABELS" ] || {
    echo "FATAL: $LABELS missing. Run stream_opentouch.sh once first (labels are small and"
    echo "never hit the quota problem), or: gdown 1cM-816vcCnkgWVIGXZrR1o8TPsDvRVCZ && unzip -o final_annotation.zip -d $WORK"
    exit 1
}

# --- single-instance lock (same reasoning as stream_opentouch.sh: 2026-08-10 corruption) ---
LOCK="$WORK/.stream.lock"
mkdir -p "$WORK"
if ! mkdir "$LOCK" 2>/dev/null; then
    echo "FATAL: another run holds $LOCK (check: pgrep -fa download_own_copies.sh)"
    echo "  If nothing is actually running: rmdir $LOCK"
    exit 1
fi
SHARDS="$WORK/shards.$$"
mkdir -p "$SHARDS" "$CACHE"
trap 'rm -rf "$SHARDS"; rmdir "$LOCK" 2>/dev/null' EXIT INT TERM

FAILED="$WORK/failed_own_ids.txt"
DONE="$WORK/done_own_ids.txt"
touch "$DONE"; : > "$FAILED"
sort -u -o "$DONE" "$DONE"

# Extract a bare Drive file ID out of either a raw ID or a full share-link.
extract_id() {
    local line="$1" trimmed
    if [[ "$line" =~ /d/([A-Za-z0-9_-]+) ]]; then echo "${BASH_REMATCH[1]}"
    elif [[ "$line" =~ id=([A-Za-z0-9_-]+) ]]; then echo "${BASH_REMATCH[1]}"
    else
        read -r trimmed <<< "$line"    # trim leading/trailing whitespace without eating the newline
        echo "$trimmed"
    fi
}

RAW_IDS=()
while IFS= read -r LINE; do RAW_IDS+=("$LINE"); done \
    < <(sed 's/#.*//' "$IDS_FILE" | grep -v '^[[:space:]]*$')
# ^ portable array-fill (not `mapfile`/`readarray`): macOS ships bash 3.2 (mapfile needs 4.0+);
#   CRC's bash is unverified from here (no SSH access), so this avoids betting on its version.
n=0; ok=0
for RAW in "${RAW_IDS[@]}"; do
    read -r IDFIELD NAME <<< "$RAW"      # NAME is "" when the line is an id/link only
    ID="$(extract_id "$IDFIELD")"
    n=$((n + 1))
    if grep -qx "$ID" "$DONE"; then
        ok=$((ok + 1)); echo "[$n/${#RAW_IDS[@]}] $ID already done"; continue
    fi
    echo "== [$n/${#RAW_IDS[@]}] $ID ${NAME:+-> $NAME} =="
    rm -f "$SHARDS"/* 2>/dev/null       # per-instance dir: only ever holds one download

    if ! ( cd "$SHARDS" && gdown "$ID" ${NAME:+-O "$NAME"} ); then
        echo "  WARN download failed -> logged, continuing"
        echo "$ID" >> "$FAILED"; continue
    fi
    if [ -n "$NAME" ]; then
        SHARD="$SHARDS/$NAME"
        [ -s "$SHARD" ] || SHARD=""
    else
        SHARD="$(ls -1 "$SHARDS"/*.hdf5 "$SHARDS"/*.h5 2>/dev/null | head -1)"
    fi
    if [ -z "$SHARD" ]; then
        echo "  WARN no hdf5 produced -> logged, continuing"
        echo "$ID" >> "$FAILED"; continue
    fi
    echo "  extracting $(basename "$SHARD") ($(du -h "$SHARD" | cut -f1))"

    if $PY "$REPO/scripts/opentouch/extract_opentouch.py" --shard "$SHARD" --labels "$LABELS" --out "$CACHE"; then
        echo "$ID" >> "$DONE"; ok=$((ok + 1))
    else
        echo "  WARN extraction failed -> logged, continuing"
        echo "$ID" >> "$FAILED"
    fi
    rm -f "$SHARD"
done

echo
echo "== summary =="
echo "  cache:  $CACHE  ($(du -sh "$CACHE" 2>/dev/null | cut -f1), \
$(ls "$CACHE"/state_*.npy 2>/dev/null | wc -l) clips)"
echo "  done:   $ok / ${#RAW_IDS[@]} ids in $IDS_FILE"
if [ -s "$FAILED" ]; then
    echo "  FAILED: $(wc -l < "$FAILED") -> $FAILED"
else
    echo "  FAILED: none"
fi
