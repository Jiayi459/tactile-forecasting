"""Tests for the G2 wiring: fit once on TRAIN, score smooth/abrupt separately on TEST."""
import json, os, sys
import numpy as np
import pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.actionsense.eval_harness.config import load_config
from src.opentouch import aggregate, evaluate as EV, trait


def make_cfg(tmp_path, actions):
    recs = []
    for i, act in enumerate(actions):
        T = 90
        t = np.arange(T) / 30.0
        st = np.zeros((T, 1, 6), np.float32)
        st[:, 0, 0] = 50 + 10 * np.sin(2 * np.pi * 0.7 * t + i)
        st[:, 0, 1] = 0.2 * np.sin(2 * np.pi * 0.5 * t + i)
        st[:, 0, 2] = 0.2 * np.cos(2 * np.pi * 0.3 * t + i)
        np.save(tmp_path / f"state_{i}.npy", st)
        recs.append({"idx": i, "shard": f"loc{i % 6}", "clip_id": f"c{i}",
                     "scene": f"loc{i % 6}", "action": act, "object_category": "cup",
                     "environment": "e", "T": T, "fps_est": 30.0,
                     "has_clip": False, "has_pose": False})
    (tmp_path / "manifest.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    text = (open("configs/opentouch/eval_harness.yaml").read()
            .replace("states_root: data/opentouch_states", f"states_root: {tmp_path}")
            .replace("min_group_size: 30", "min_group_size: 1")
            .replace("ar_orders: [6, 15, 30, 45, 60, 90]", "ar_orders: [4]"))
    p = tmp_path / "h.yaml"; p.write_text(text)
    # These fixtures exercise the historical model/state contract; new RAW-to-model
    # TRAIN-only coverage lives in test_train_only_calibration.py.
    p.write_text(p.read_text().replace("mode: train_only", "mode: legacy"))
    return load_config(str(p))


def _splits(n):
    ids = list(range(n))
    return {"train": ids[: n // 2], "val": ids[n // 2: n // 2 + n // 4],
            "test": ids[n // 2 + n // 4:]}


def test_models_are_fit_once_and_scored_per_class(tmp_path):
    """holding is SMOOTH, picking up is ABRUPT in the frozen table."""
    acts = ["holding", "picking up"] * 12
    cfg = make_cfg(tmp_path, acts)
    sp = _splits(len(acts))
    st, _ = EV.collect_clip_stats(cfg, sp)
    rows = EV.trait_rows(cfg, st)
    assert len(rows[trait.SMOOTH]) and len(rows[trait.ABRUPT])
    # every TEST clip lands in exactly one bucket, and with an audited vocabulary the two
    # trait classes account for all of them
    assert not rows.get("unaudited", []).size if hasattr(rows.get("unaudited", []), "size") else True
    assert len(rows[trait.SMOOTH]) + len(rows[trait.ABRUPT]) == len(st.clip_ids)
    assert not set(rows[trait.SMOOTH]) & set(rows[trait.ABRUPT])
    # and the per-class R2 uses the SAME fitted model, just different rows
    for cls in (trait.SMOOTH, trait.ABRUPT):
        r = aggregate.r2(st, "persistence", rows=rows[cls])
        assert np.isfinite(r.per_channel).all()


def test_unaudited_action_refuses_to_score(tmp_path):
    """The 36 unadjudicated actions must block scoring, not default silently."""
    # a string that is not in the table and never will be -- "lifting" served here until
    # the 2026-08-15 blind pass audited it, which is exactly the completeness this guards
    cfg = make_cfg(tmp_path, ["holding", "levitating"] * 12)
    st, _ = EV.collect_clip_stats(cfg, _splits(24))
    with pytest.raises(trait.UnauditedAction):
        EV.trait_rows(cfg, st)


def test_external_predictions_join_the_same_stats(tmp_path):
    cfg = make_cfg(tmp_path, ["holding", "picking up"] * 12)
    sp = _splits(24)
    st0, _ = EV.collect_clip_stats(cfg, sp)
    from src.opentouch.baselines import origins
    from src.opentouch.dataset import load_group
    test = load_group(cfg, sp["test"])
    fake = {i: np.zeros((len(origins(len(Y), cfg)), cfg.horizon, len(cfg.channels)))
            for i, Y in test.items()}
    st1, _ = EV.collect_clip_stats(cfg, sp, external={"zero": fake})
    assert "zero" in st1.sse and "persistence" in st1.sse
    assert np.allclose(st0.sse["persistence"], st1.sse["persistence"])
