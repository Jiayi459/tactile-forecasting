"""TRAIT CLASS (smooth vs abrupt) for ActionSense — the pre-registered table.

THE RUBRIC IS NOT RESTATED HERE. It lives in `src/opentouch/trait.py` (Layer 1, with its
refinements R1 and R2) and is written there as a priori, physical and SENSOR-AGNOSTIC. What
is sensor-specific is the vocabulary, so this file carries ActionSense's table and nothing
else. Read that rubric before reading this table; a second paraphrase of it would be a second
definition, and two definitions is exactly what the cross-sensor comparison cannot survive.

WHY A SEPARATE FILE. `src/opentouch/trait.py` is an OpenTouch pre-registration artifact whose
git timestamp is evidence about that dataset. Appending another dataset's vocabulary to it
would muddy what that timestamp attests to.

PRE-REGISTRATION STATUS. Committed 2026-08-24, BEFORE any ActionSense number was computed by
trait class. One honest qualification: a training-free predictability probe over these same
actions already exists (docs/actionsense/predictability_by_category*.csv), so this audit had
to stand on the physical rubric alone. It is written to be checkable as such.

AMENDMENT 2026-09-02 -- ALIGNED TO OpenTouch's VERDICTS [U2]
===========================================================
THE ORIGINAL DIVERGENCE, RECORDED HERE VERBATIM SO THE CHANGE IS AUDITABLE. As committed on
2026-08-24 this table had `slice` and `clear` as SMOOTH, while OpenTouch's table has the
structurally corresponding `cutting` and `scooping` as ABRUPT. That was the user's explicit
ruling of 2026-08-24, and it diverged from the rubric's own R1, whose worked example is a
knife: "a knife striking the cutting board ... is what forces `cutting` -> abrupt".

THE USER'S NEW RULING (2026-09-02): "保持和 opentouch 一样的分类 abrupt 和 smooth 方法" --
keep ActionSense's classification identical to OpenTouch's. Applying it moves `slice` ->
ABRUPT (= `cutting`) and `clear` -> ABRUPT (= `scooping`). `peel` STAYS SMOOTH: its
OpenTouch correspondent is `scraping`, which that table rules SMOOTH [U], so alignment does
not touch it. Those are the only two entries the amendment changes; the other twelve already
agreed with their correspondents.

WHY THIS IS NOT A POST-HOC RELABELLING. The HARD DISCIPLINE clause of the rubric forbids
editing a class definition in response to a measured outcome. No ActionSense number has ever
been computed by trait class: as of this amendment `src/actionsense/trait.py` is imported by
nothing but its own test (checked by grep across the repo, 2026-09-02), and no artifact in
docs/actionsense/ is broken down by smooth/abrupt. The amendment is therefore still made
BEFORE any result it could be tuned to, which is the property the pre-registration exists to
establish. The training-free probe qualification below is unchanged and still applies.

WHAT THE AMENDMENT COSTS, STATED PLAINLY. The frozen harness scores two verbs
(`actions: [slice, peel]` in configs/actionsense/eval_harness.yaml). Under the 2026-08-24
table both were SMOOTH, so a trait contrast on the scored corpus was not merely unmeasured
but ARITHMETICALLY IMPOSSIBLE -- one class was empty. Under this amendment the same corpus
splits slice (ABRUPT) vs peel (SMOOTH). That makes the contrast computable, and it also
means the amendment is what creates the measurement: a reader is entitled to know that, and
to know that the direction of the eventual result was not consulted, only its existence.

A LIMITATION THE AMENDMENT CANNOT REMOVE. On the scored corpus each class then holds exactly
ONE verb, so trait class and verb identity are perfectly confounded: any smooth-vs-abrupt
difference measured on ActionSense is equally a slice-vs-peel difference. OpenTouch's G2 had
many verbs per class and does not have this problem. Any ActionSense trait number must be
reported as "slice vs peel", with the trait reading offered as interpretation rather than as
what was measured.

CONTENTIOUS IS UNCHANGED BY THE AMENDMENT. Membership there is about variability across
typical instances of an action (Layer 3), not about cross-sensor agreement, and the
instance-level ambiguity of `slice`, `clear`, `peel`, `open` and `open/close` is the same
after alignment as before. Note that on the scored corpus BOTH verbs are contentious, so the
Layer-3 sensitivity analysis is empty there and cannot be run; that is a limitation to report
beside the primary number, not a reason to shrink the contentious set.

PROVENANCE
==========
  [U] the user's explicit ruling, 2026-08-24 (recorded verbatim, not re-derived)
  [U2] the user's ruling of 2026-09-02: match OpenTouch's verdict for the corresponding
       action (see the AMENDMENT section); supersedes the [U] verdict it replaces
  [R] derived by applying the rubric in src/opentouch/trait.py
  +   also in CONTENTIOUS (Layer 3)

Every entry names the OpenTouch action it corresponds to. OT_CORRESPONDENT below makes that
mapping machine-checkable, so "classified the same way as OpenTouch" is enforced by a test
rather than asserted in a docstring.

VOCABULARY COMPLETENESS
=======================
ActionSense has 14 distinct verbs over 299 recordings (measured 2026-08-24 by
scripts/actionsense/actionsense_action_inventory.py). ALL 14 are audited below -- there is no
long tail here, unlike OpenTouch's 66 -- so `trait_class` should never raise on this corpus.
It still raises rather than defaulting, because a silent default would quietly define a class
if the corpus ever grows.
"""
from __future__ import annotations

from ..opentouch.trait import ABRUPT, SMOOTH, UnauditedAction, normalize_action

# The verb is the first token of the manifest label (eval_harness.splits.parse_label), so
# these keys are verbs, not whole labels: "Slice a cucumber" -> "slice".
TRAIT_CLASS: dict[str, str] = {
    # -- SMOOTH ---------------------------------------------------------------------------
    "clean":       SMOOTH,   # [R] sponge/towel on a surface: sustained contact, continuous
                             #     force modulation, no collision. Same structure as the
                             #     rubric's `wiping`/`cleaning`.
    "spread":      SMOOTH,   # [R] knife face held against bread, force modulated along the
                             #     stroke; nothing is struck.
    "pour":        SMOOTH,   # [R] the sustained tilt IS the action; the grasp of the jug is
                             #     preparatory and excluded by R2. Matches the rubric's own
                             #     ruling on `pouring`.
    "peel":        SMOOTH,   # [U]+ = OpenTouch `scraping` (smooth). The blade leaves the
                             #     surface into air rather than striking anything, so R1
                             #     finds no impact. CONTENTIOUS because a stroke's
                             #     engage/disengage is a contact-surface change and some
                             #     participants peel in discrete strokes rather than
                             #     continuous rotation. Unchanged by the 2026-09-02
                             #     amendment: OpenTouch already rules this class smooth.

    # -- ABRUPT ---------------------------------------------------------------------------
    "slice":       ABRUPT,   # [U2]+ = OpenTouch `cutting` (abrupt): "刀每次下压到砧板是一次
                             #      impact,锯切动作里刀刃反复脱离/重新咬合材料". This is R1's
                             #      own worked example. Was SMOOTH until 2026-09-02.
    "clear":       ABRUPT,   # [U2]+ = OpenTouch `scooping` (abrupt): utensil-container
                             #      collisions under R1, plus the grasp/release of what is
                             #      cleared. Was SMOOTH until 2026-09-02.
    "get":         ABRUPT,   # [R] the grasp is constitutive (R2): remove it and nothing was
                             #     retrieved. Opening a drawer or door adds a second one.
    "get/replace": ABRUPT,   # [R] as `get`, with a release as well.
    "open":        ABRUPT,   # [R]+ the lid parting from the tub is a discrete transition.
                             #     CONTENTIOUS: a twist-off can instead be sustained rotation,
                             #     which is why the rubric lists `twisting`/`unscrewing` as
                             #     contentious.
    "open/close":  ABRUPT,   # [R]+ as `open`, twice.
    "set":         ABRUPT,   # [R] placing: "cannot `place` without release" (rubric, R2).
    "stack":       ABRUPT,   # [R] grasp, place, release, plus bowl meeting bowl.
    "load":        ABRUPT,   # [R] repeated grasp/place/release into the rack.
    "unload":      ABRUPT,   # [R] repeated grasp/lift/release out of it.
}

CONTENTIOUS: frozenset[str] = frozenset({
    # verdict varies across typical instances of the action (Layer 3). Membership is about
    # that variability, not about cross-sensor agreement, so the 2026-09-02 alignment did not
    # change it: a sawing stroke that never lifts, a peel done in discrete strokes and a cap
    # unscrewed in one sustained rotation are all still real instances.
    "slice", "clear", "peel", "open", "open/close",
})
assert CONTENTIOUS <= set(TRAIT_CLASS), "CONTENTIOUS must only name audited actions"

# The OpenTouch action each ActionSense verb corresponds to. This is the whole content of the
# user's 2026-09-02 ruling: the two tables must agree wherever they describe the same physical
# action, so the correspondence is stated once, here, and asserted in tests rather than being
# left as prose that can drift away from the table beneath it.
OT_CORRESPONDENT: dict[str, str] = {
    "clean":       "cleaning",
    "spread":      "spreading",
    "pour":        "pouring",
    "peel":        "scraping",
    "slice":       "cutting",
    "clear":       "scooping",
    "get":         "picking up",
    "get/replace": "picking up",   # + a `placing` on the return leg; both are ABRUPT
    "open":        "turning",      # twist-off cap
    "open/close":  "turning",
    "set":         "placing",
    "stack":       "placing",
    "load":        "placing",
    "unload":      "picking up",
}
assert set(OT_CORRESPONDENT) == set(TRAIT_CLASS), \
    "every audited verb needs the OpenTouch action it is aligned to"


def trait_class(action: str | None) -> str:
    """-> "smooth" | "abrupt". Raises UnauditedAction for unknown/missing verbs."""
    key = normalize_action(action)
    if not key:
        raise UnauditedAction("empty action label (missing annotation) has no trait class")
    if key not in TRAIT_CLASS:
        raise UnauditedAction(
            f"verb {key!r} has not been audited against the trait rubric; apply the rubric "
            f"in src/opentouch/trait.py and commit the verdict BEFORE scoring it")
    return TRAIT_CLASS[key]


def is_contentious(action: str | None) -> bool:
    """True iff in the pre-registered contentious subset (Layer 3 of the rubric)."""
    return normalize_action(action) in CONTENTIOUS


def unaudited(actions) -> set[str]:
    """Verbs present in `actions` that the table does not cover (empty labels excluded)."""
    seen = {normalize_action(a) for a in actions}
    return {a for a in seen if a and a not in TRAIT_CLASS}


def partition(actions: dict[int, str]) -> dict[str, list[int]]:
    """recording idx -> verb  ==>  {smooth, abrupt, unlabeled, unaudited} -> [idx].

    Every input lands in exactly one bucket, so the drops are COUNTABLE and must be reported
    beside any G2 number rather than vanishing.
    """
    out: dict[str, list[int]] = {SMOOTH: [], ABRUPT: [], "unlabeled": [], "unaudited": []}
    for i, a in sorted(actions.items()):
        key = normalize_action(a)
        if not key:
            out["unlabeled"].append(i)
        elif key not in TRAIT_CLASS:
            out["unaudited"].append(i)
        else:
            out[TRAIT_CLASS[key]].append(i)
    return out


def contentious_ids(actions: dict[int, str]) -> list[int]:
    """Recording idxs the Layer-3 sensitivity analysis DROPS before recomputing."""
    return sorted(i for i, a in actions.items() if is_contentious(a))
