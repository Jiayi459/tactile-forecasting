"""List ActionSense's action vocabulary, so the smooth/abrupt audit is over the real labels.

src/opentouch/trait.py is a PRE-REGISTRATION artifact: its rubric is stated as sensor-
agnostic, but its TRAIT_CLASS table covers OpenTouch's vocabulary alone. Extending it to
ActionSense means auditing ActionSense's verbs against that same rubric, and trait.py's own
discipline says the verdict is committed BEFORE anything is scored with it. Auditing from
memory of a documentation table is how a vocabulary item gets missed or invented, hence this.

    python scripts/actionsense_action_inventory.py --config configs/actionsense/eval_harness.yaml
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.actionsense.eval_harness.config import load_config           # noqa: E402
from src.actionsense.eval_harness.splits import parse_label           # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/actionsense/eval_harness.yaml")
    ap.add_argument("--out", default="docs/actionsense/action_inventory.md")
    a = ap.parse_args()

    cfg = load_config(a.config)
    root = cfg.abspath("states_root")
    rows = [json.loads(l) for l in open(os.path.join(root, "manifest.jsonl")) if l.strip()]

    by_verb = collections.defaultdict(list)
    frames = collections.Counter()
    for r in rows:
        v, o = parse_label(r["label"])
        by_verb[v].append((r["label"], o))
        frames[v] += int(r.get("T", 0))

    print(f"{len(rows)} recordings, {len(by_verb)} distinct verbs\n")
    L = ["# ActionSense action inventory", "",
         f"{len(rows)} recordings, {len(by_verb)} distinct verbs, from `{root}`.",
         "One activity per recording, so the verb is available per window -- which is what",
         "lets a probGRU arm carry the same action embedding here as on OpenTouch.", "",
         "| verb | recordings | frames | objects | example label |",
         "|---|---|---|---|---|"]
    for v in sorted(by_verb, key=lambda k: -len(by_verb[k])):
        items = by_verb[v]
        objs = sorted({o for _, o in items})
        print(f"  {v:12s} {len(items):4d} recordings  {frames[v]:8d} frames  "
              f"objects={objs}")
        L.append(f"| `{v}` | {len(items)} | {frames[v]} | {', '.join(objs)} "
                 f"| {items[0][0]} |")
    L += ["", "## Next step", "",
          "Audit each verb against the rubric in `src/opentouch/trait.py` (Layer 1) and",
          "commit the verdict BEFORE scoring anything by it. Note that a training-free",
          "predictability probe over these actions already exists",
          "(`docs/actionsense/predictability_by_category*.csv`), so the audit has to stand on",
          "the physical rubric alone and be readable as such by someone who has seen it."]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out, "w").write("\n".join(L) + "\n")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    raise SystemExit(main())
