"""The ActionSense trait table: coverage, the contentious subset, and countable drops."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.actionsense import trait as T                                # noqa: E402
from src.opentouch import trait as OT                                 # noqa: E402

# the 14 verbs and their recording counts, measured 2026-08-24
CORPUS = {"clean": 60, "slice": 45, "get": 30, "peel": 30, "spread": 30, "clear": 28,
          "pour": 25, "get/replace": 15, "open/close": 9, "open": 6, "set": 6,
          "stack": 5, "load": 5, "unload": 5}


def test_every_verb_in_the_corpus_is_audited():
    """No long tail here, unlike OpenTouch's 66 strings, so nothing should be unaudited."""
    assert T.unaudited(CORPUS) == set()
    assert sum(CORPUS.values()) == 299


def test_an_unknown_verb_raises_rather_than_defaulting():
    """Defaulting the unknown into abrupt would silently define the majority class."""
    with pytest.raises(OT.UnauditedAction, match="audited"):
        T.trait_class("julienne")
    with pytest.raises(OT.UnauditedAction):
        T.trait_class("")


def test_the_classes_share_the_rubric_constants():
    """One rubric, one pair of class names: a second SMOOTH string would silently split the
    G2 tables between the two sensors."""
    assert {T.trait_class(v) for v in CORPUS} == {OT.SMOOTH, OT.ABRUPT}


def test_every_verb_agrees_with_its_opentouch_correspondent():
    """The user's ruling of 2026-09-02: ActionSense is classified the same way OpenTouch is.

    Stated as a test rather than as prose because the failure mode is silent -- the two
    tables live in different files, and a verdict edited on one side would otherwise leave
    the docstring claiming an agreement that no longer holds. This is the assertion that
    `slice` -> abrupt (= `cutting`) and `clear` -> abrupt (= `scooping`) exist to satisfy.
    """
    for verb, ot_action in T.OT_CORRESPONDENT.items():
        assert T.trait_class(verb) == OT.trait_class(ot_action), (
            f"{verb!r} is {T.trait_class(verb)} here but its OpenTouch correspondent "
            f"{ot_action!r} is {OT.trait_class(ot_action)}")


def test_the_two_verbs_the_harness_scores_land_in_opposite_classes():
    """configs/actionsense/eval_harness.yaml scores `slice` and `peel` and nothing else.

    Under the superseded 2026-08-24 table both were SMOOTH, so the abrupt class on the scored
    corpus was EMPTY and no trait contrast could be computed at all -- which is why none ever
    was. This test pins the property the amendment bought; if a future edit collapses the two
    back into one class, every ActionSense trait result silently becomes uncomputable again.
    """
    assert T.trait_class("slice") == OT.ABRUPT
    assert T.trait_class("peel") == OT.SMOOTH


def test_contentious_still_covers_the_boundary_actions():
    """Layer 3 is about within-action variability, so the alignment did not shrink it. Both
    scored verbs are contentious, so the sensitivity analysis is empty on the scored corpus
    -- asserted here so the limitation is visible in the tests, not only in the docstring."""
    assert T.is_contentious("slice") and T.is_contentious("clear") and T.is_contentious("peel")
    assert T.CONTENTIOUS <= set(T.TRAIT_CLASS)
    assert not (({"slice", "peel"}) - T.CONTENTIOUS)


def test_partition_accounts_for_every_recording():
    """Drops have to be countable, not silent."""
    acts = {i: v for i, v in enumerate(CORPUS)}
    acts[99], acts[98] = "", "julienne"
    p = T.partition(acts)
    assert sum(len(v) for v in p.values()) == len(acts)
    assert p["unlabeled"] == [99] and p["unaudited"] == [98]
    assert set(p[OT.SMOOTH]) | set(p[OT.ABRUPT]) == set(range(len(CORPUS)))


def test_dropping_contentious_leaves_both_classes_populated():
    """The sensitivity analysis is only informative if it can still be computed."""
    acts = {i: v for i, v in enumerate(CORPUS)}
    keep = {i: v for i, v in acts.items() if i not in set(T.contentious_ids(acts))}
    p = T.partition(keep)
    assert p[OT.SMOOTH] and p[OT.ABRUPT]
