"""Fail loudly if re-streaming renumbered the recordings.

The recording index is a counter that increments across the streamed HDF5 files, and
`splits.json` plus every result under `docs/` is keyed by it. Re-streaming rebuilds the
manifest from scratch, so the numbering survives only while the URL list and the accept/reject
rules (`--min-frames`, the activity-interval parsing, `seq_metrics` returning empty) are
unchanged. If it does not survive, existing splits and results silently refer to different
recordings than the ones they were computed on -- the failure is invisible at the point it
happens and only shows up as results that no longer mean what they say.

    python scripts/actionsense/check_manifest_indices.py OLD.jsonl NEW.jsonl

Exit 0 when every idx maps to the same label as before, 1 otherwise.
"""
from __future__ import annotations

import json
import sys


def rows(path: str) -> list[tuple[int, str]]:
    out = []
    with open(path) as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                out.append((int(r["idx"]), r["label"]))
    return out


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    old, new = rows(sys.argv[1]), rows(sys.argv[2])
    if old == new:
        print(f"  OK: {len(new)} recordings, every idx maps to the same label as before.")
        return 0

    print(f"  MISMATCH: {len(old)} recordings before, {len(new)} now.")
    for (i_o, l_o), (i_n, l_n) in zip(old, new):
        if (i_o, l_o) != (i_n, l_n):
            print(f"  first divergence: idx {i_o} was {l_o!r}, is now idx {i_n} {l_n!r}")
            break
    else:
        longer, which = (old, "before") if len(old) > len(new) else (new, "now")
        print(f"  the shorter run is a prefix of the longer; {which} has "
              f"{len(longer) - min(len(old), len(new))} extra recording(s), "
              f"first at idx {longer[min(len(old), len(new))][0]}")
    print("  splits.json and every result under docs/ is keyed by this index, so they no longer")
    print("  refer to the recordings they were computed on. Do NOT train on this until the")
    print("  cause is understood (a changed URL list or accept/reject rule would do it).")
    return 1


if __name__ == "__main__":
    sys.exit(main())
