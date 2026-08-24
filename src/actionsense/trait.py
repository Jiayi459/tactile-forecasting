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

A KNOWN CROSS-SENSOR DIVERGENCE, RECORDED IN ADVANCE
====================================================
`slice` and `clear` are SMOOTH here, while OpenTouch's table has the structurally
corresponding `cutting` and `scooping` as ABRUPT. The rubric's own R1 uses cutting as its
worked example -- "a knife striking the cutting board ... is what forces `cutting` ->
abrupt" -- so the divergence is real and is NOT an application of R1 as written. It stands
because it is the user's explicit ruling of 2026-08-24, marked [U] below, and this file
records the user's rulings verbatim rather than re-deriving them.

Both are therefore placed in CONTENTIOUS, which is what that subset is for: every G2 result
is reported alongside a recomputation with the contentious actions dropped. If the direction
survives, no conclusion depends on how these two were assigned; if it does not, the
disagreement is exactly what needs reporting. Any cross-sensor comparison of a G2 result must
state this divergence, because a smooth-vs-abrupt contrast means something different on the
two sensors while it stands.

PROVENANCE
==========
  [U] the user's explicit ruling, 2026-08-24 (recorded verbatim, not re-derived)
  [R] derived by applying the rubric in src/opentouch/trait.py
  +   also in CONTENTIOUS (Layer 3)

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
    "peel":        SMOOTH,   # [U]+ the blade leaves the surface into air rather than striking
                             #     anything, so R1 finds no impact; structurally this is the
                             #     rubric's `scraping`, which it classes smooth. CONTENTIOUS
                             #     because a stroke's engage/disengage is a contact-surface
                             #     change and some participants peel in discrete strokes
                             #     rather than continuous rotation.
    "slice":       SMOOTH,   # [U]+ DIVERGES FROM R1 AND FROM OpenTouch's `cutting` (abrupt).
                             #     See the header. Contentious by construction.
    "clear":       SMOOTH,   # [U]+ DIVERGES FROM OpenTouch's `scooping` (abrupt), which the
                             #     rubric names as forced by R1 when the tool strikes the
                             #     bowl. See the header. Contentious by construction.

    # -- ABRUPT ---------------------------------------------------------------------------
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
    # verdict varies across typical instances of the action, or (slice/clear) diverges from
    # the same rubric's verdict on the corresponding OpenTouch action
    "slice", "clear", "peel", "open", "open/close",
})
assert CONTENTIOUS <= set(TRAIT_CLASS), "CONTENTIOUS must only name audited actions"


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
