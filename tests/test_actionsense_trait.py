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


def test_contentious_covers_the_cross_sensor_divergence():
    """slice and clear are SMOOTH here and ABRUPT on OpenTouch under the same rubric (the
    user's ruling, 2026-08-24). They must be contentious, so every result is reported with a
    recomputation that drops them."""
    assert T.trait_class("slice") == OT.SMOOTH and OT.trait_class("cutting") == OT.ABRUPT
    assert T.is_contentious("slice") and T.is_contentious("clear")
    assert T.CONTENTIOUS <= set(T.TRAIT_CLASS)


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
