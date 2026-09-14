"""Regressions at the RAW-pressure boundary, with fitted artifacts held fixed."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from src import calibration as C
from src.actionsense.eval_harness.config import Config, load_config
from scripts.actionsense.probe_actionsense import resample, query_times


def make_raw(tmp_path, corpus="opentouch", n=8):
    root = tmp_path / corpus
    root.mkdir()
    raw = copy.deepcopy(load_config(f"configs/{corpus}/eval_harness.yaml").raw)
    raw["paths"] = {"states_root": str(root), "split_file": str(root / "splits.json"),
                    "calibration_cache": str(tmp_path / "calibrated")}
    raw["rate"] = {"fps_raw": 10., "downsample": 1, "horizon_s": .2}
    raw["eval"] = {"min_history": 3, "stride": 2, "seed": 0}
    raw["calibration"]["max_frames"] = 64
    raw["baselines"]["min_group_size"] = 1
    raw["baselines"]["ar_orders"] = [1, 2]
    raw["actions"] = [] if corpus == "opentouch" else ["slice"]
    shape = (1, 16, 16) if corpus == "opentouch" else (2, 32, 32)
    rows = []
    for i in range(n):
        a = np.random.default_rng(i).uniform(5, 20, (20,) + shape).astype(np.float32)
        np.save(root / f"clip_{i}.npy", a)
        # Deliberately wrong preexisting states: preparation MUST recompute them.
        np.save(root / f"state_{i}.npy", np.full((20, shape[0], 6), 999, np.float32))
        rows.append({"idx": i, "T": 20, "fps": 10., "label": "Slice a potato", "action": "slicing",
                     "object_category": "food", "shard": f"place{i // 2}_p1", "has_clip": True,
                     "pressure_calibration": "none",
                     "resampling": "native" if corpus == "opentouch" else "previous_sample_hold_v1"})
    (root / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    split = {"train": list(range(n - 4)), "val": [n - 4, n - 3], "test": [n - 2, n - 1]}
    return Config(raw, "synthetic", "test"), root, split


def test_previous_sample_hold_never_reads_future():
    t = np.array([0., .4, 1.])
    a = np.array([2., 4., 8.])[:, None, None]
    b = a.copy(); b[2] = 999
    q = np.array([0., .4, .9])
    np.testing.assert_array_equal(resample(a, t, q), resample(b, t, q))
    np.testing.assert_array_equal(resample(a, t, q).ravel(), [2, 4, 4])
    with pytest.raises(ValueError, match="precedes"):
        resample(a, t, [-.01])
    np.testing.assert_array_equal(query_times(0, 1, 10), query_times(0, 2, 10)[:10])


@pytest.mark.parametrize("corpus", ["actionsense", "opentouch"])
def test_only_training_signals_are_read_by_fit(tmp_path, monkeypatch, corpus):
    cfg, root, split = make_raw(tmp_path, corpus)
    original = np.load
    reads = []
    def guarded(path, *a, **kw):
        i = int(Path(path).stem.split("_")[-1])
        reads.append(i)
        assert i in split["train"], "fitter read a held-out signal"
        return original(path, *a, **kw)
    monkeypatch.setattr(np, "load", guarded)
    cal = C.fit_calibration(root, split["train"], corpus, max_frames=64)
    assert sorted(reads) == split["train"]
    assert cal.metadata["train_ids"] == split["train"]
    assert C.Calibration.from_payload(cal.payload()).id == cal.id


@pytest.mark.parametrize("corpus", ["actionsense", "opentouch"])
def test_full_pipeline_future_invariance_and_fold_isolation(tmp_path, corpus):
    cfg, root, split = make_raw(tmp_path, corpus)
    f1 = C.prepare_fold(cfg, split)
    r1 = Path(f1.abspath("states_root"))
    i = split["test"][0]
    original = np.load(root / f"clip_{i}.npy")
    modified = original.copy(); modified[8:] += 500
    np.save(root / f"clip_{i}.npy", modified)
    f2 = C.prepare_fold(cfg, split)
    r2 = Path(f2.abspath("states_root"))
    assert r1 != r2                     # output-cache identity follows changed raw source
    assert f1.raw["calibration"]["id"] == f2.raw["calibration"]["id"]
    assert f1.raw["calibration"]["artifact"] == f2.raw["calibration"]["artifact"]
    for kind in ("clip", "state"):
        np.testing.assert_array_equal(np.load(r1 / f"{kind}_{i}.npy")[:8],
                                      np.load(r2 / f"{kind}_{i}.npy")[:8])
        for j in split["train"]:
            np.testing.assert_array_equal(np.load(r1 / f"{kind}_{j}.npy"),
                                          np.load(r2 / f"{kind}_{j}.npy"))
    assert C.prepare_fold(f1, split) is f1
    assert C.prepare_fold(cfg, split).abspath("states_root") == str(r2)
    wrong = {**split, "train": split["val"], "val": split["train"]}
    with pytest.raises(ValueError, match="another split"):
        C.prepare_fold(f1, wrong)


def test_validation_values_do_not_fit_calibration(tmp_path):
    cfg, root, sp = make_raw(tmp_path)
    a = C.prepare_fold(cfg, sp)
    for i in sp["val"]:
        np.save(root / f"clip_{i}.npy", np.full((20, 1, 16, 16), 1000, np.float32))
    b = C.prepare_fold(cfg, sp)
    assert a.raw["calibration"]["id"] == b.raw["calibration"]["id"]


def test_rejects_legacy_interpolation_and_double_correction(tmp_path):
    cfg, root, sp = make_raw(tmp_path, "actionsense")
    rows = list(C.read_manifest(root).values())
    rows[0].pop("resampling")
    (root / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(ValueError, match="Re-extract"):
        C.prepare_fold(cfg, sp)
    rows[0]["resampling"] = "previous_sample_hold_v1"
    rows[0]["d1_k"] = 1
    (root / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(ValueError, match="already calibrated"):
        C.prepare_fold(cfg, sp)


def test_shared_template_serves_unseen_shard_and_matches_targets(tmp_path):
    from src.opentouch import tactile_map as TM
    from src.opentouch.dataset import load_group, Norm
    cfg, root, sp = make_raw(tmp_path)
    f = C.prepare_fold(cfg, sp)
    norm = Norm.from_train(load_group(f, sp["train"]))
    _, mn = TM.build_inputs(f, "flatten", sp["train"], sp["train"], norm, 10)
    maps, _ = TM.build_inputs(f, "flatten", sp["test"], sp["train"], norm, 10, mn)
    assert set(maps) == set(sp["test"])
    for i in sp["test"]:
        p = np.load(Path(f.abspath("states_root")) / f"clip_{i}.npy")
        st = np.load(Path(f.abspath("states_root")) / f"state_{i}.npy")
        np.testing.assert_allclose(st, C.moments(p))
        np.testing.assert_allclose(maps[i], mn.apply(p))
        assert not np.all(st == 999)


def test_action_map_uses_exact_same_correction_as_state(tmp_path):
    from src.actionsense.tactile_map.data import load_map
    cfg, _, sp = make_raw(tmp_path, "actionsense")
    f = C.prepare_fold(cfg, sp)
    i = sp["test"][0]
    p = np.load(Path(f.abspath("states_root")) / f"clip_{i}.npy")
    np.testing.assert_array_equal(load_map(f, i, baseline_frames=10), p)


def test_moments_match_existing_framewise_definition():
    from src.actionsense.physical_state import clip_states
    a = np.random.default_rng(0).uniform(0, 20, (5, 2, 8, 8)).astype(np.float32)
    a[0] = 0
    np.testing.assert_allclose(C.moments(a), clip_states(a, baseline_pct=None), rtol=1e-6, atol=1e-6)


def test_checkpoint_cannot_relabel_training_as_test():
    good = {"split_ids": {"train": [0, 1], "val": [2], "test": [3]}}
    assert C.held_out_ids(good) == [3]
    with pytest.raises(ValueError, match="provenance"):
        C.held_out_ids({})
    with pytest.raises(ValueError, match="held-out"):
        C.held_out_ids(good, [0])
    with pytest.raises(ValueError, match="overlap"):
        C.held_out_ids({"split_ids": {"train": [0, 1], "val": [2], "test": [1]}})


@pytest.mark.parametrize("arm", ["raw", "flatten", "cnn"])
def test_probgru_predictions_ignore_test_future_after_preprocessing(tmp_path, arm):
    from src.opentouch import prob_gru as P
    cfg, root, sp = make_raw(tmp_path)
    f1 = C.prepare_fold(cfg, sp)
    hp = {"input": arm, "epochs": 1, "hidden": 4, "d": 4, "batch": 64, "seed": 1}
    m, norm, fn, vocab, by_idx, _ = P.train(f1, sp["train"], sp["val"], 4, hp,
                                         device="cpu", verbose=False, base_scope="train")
    pred1 = P.predict(m, f1, norm, fn, vocab, by_idx, sp["test"], 4, hp=hp)
    for i in sp["test"]:
        a = np.load(root / f"clip_{i}.npy"); a[8:] += 100
        np.save(root / f"clip_{i}.npy", a)
    f2 = C.prepare_fold(cfg, sp)
    pred2 = P.predict(m, f2, norm, fn, vocab, by_idx, sp["test"], 4, hp=hp)
    # Origins 3,5,7: all are before the changed future, including their anchor.
    for i in sp["test"]:
        np.testing.assert_array_equal(pred1[i][:3], pred2[i][:3])


def test_unprepared_config_cannot_read_old_targets(tmp_path):
    from src.opentouch.dataset import load_target
    cfg, _, _ = make_raw(tmp_path)
    with pytest.raises(ValueError, match="prepare_fold"):
        load_target(cfg, 0)


@pytest.mark.parametrize("encoder", ["aggregate", "flatten", "cnn"])
@pytest.mark.parametrize("backbone", ["seq2seq", "probgru"])
def test_action_cv_trains_each_arm_with_fold_calibration(tmp_path, encoder, backbone):
    from src.actionsense.tactile_map import train as T
    from scripts.actionsense.export_baseline_forecasts import corpus_folds
    cfg, _, _ = make_raw(tmp_path, "actionsense", n=12)
    hp = {"baseline_frames": 10, "alpha": 10, "d": 4, "hidden": 4, "epochs": 1,
          "lr": .003, "batch": 64, "backbone": backbone}
    cv = T.cross_validate(cfg, hp, encoder, 4, list(range(12)), folds=2, save_preds="yes")
    assert len(cv["preds"]) == 12
    for tr, va, te in corpus_folds(list(range(12)), 2, 0):
        f = C.prepare_fold(cfg, {"train": tr, "val": va, "test": te})
        assert load_config(f.path).config_hash == f.config_hash
        for i in te:
            meta = cv["provenance"][i]
            assert i in C.held_out_ids(meta)
            assert meta["calibration"]["id"] == f.raw["calibration"]["id"]
            C.verify_checkpoint_cache(meta)
    T.save_predictions({encoder: cv["preds"]}, cfg, str(tmp_path / "preds"), {},
                       provenance=cv["provenance"])
    z = np.load(tmp_path / "preds" / "clip_0.npz")
    assert str(z["calibration_id"]) == cv["provenance"][0]["calibration"]["id"]


def test_training_split_must_match_fitted_calibrator(tmp_path):
    cfg, _, sp = make_raw(tmp_path)
    f = C.prepare_fold(cfg, sp)
    with pytest.raises(ValueError, match="training IDs"):
        C.validate_training(f, sp["train"] + sp["val"], [])


def test_shared_pressure_cache_preserves_each_resolved_protocol(tmp_path):
    cfg, _, sp = make_raw(tmp_path)
    first = C.prepare_fold(cfg, sp)
    contents = Path(first.path).read_bytes()
    other = copy.deepcopy(cfg.raw)
    other["rate"]["horizon_s"] = .3
    second = C.prepare_fold(Config(other, cfg.path, "other"), sp)
    assert first.abspath("states_root") == second.abspath("states_root")
    assert first.path != second.path
    assert Path(first.path).read_bytes() == contents
    assert load_config(first.path).config_hash == first.config_hash


def test_external_scorer_rejects_predictions_from_another_protocol(tmp_path):
    cfg, _, sp = make_raw(tmp_path)
    f = C.prepare_fold(cfg, sp)
    path = tmp_path / "external.npz"
    preds = {str(i): np.zeros((8, 2, 3)) for i in sp["test"]}
    np.savez(path, **preds)
    with pytest.raises(ValueError, match="lack"):
        C.load_external_predictions(path, f, sp)
    np.savez(path, **preds, calibration_id="old", split_ids=json.dumps(sp))
    with pytest.raises(ValueError, match="different"):
        C.load_external_predictions(path, f, sp)
    np.savez(path, **preds, calibration_id=f.raw["calibration"]["id"], split_ids=json.dumps(sp))
    assert set(C.load_external_predictions(path, f, sp)) == set(sp["test"])


def test_action_dynamics_fold_training_and_future_invariance(tmp_path, monkeypatch):
    from src.actionsense import action_dynamics as AD
    from src.actionsense.eval_harness import config as config_module
    cfg, root, sp = make_raw(tmp_path, "actionsense")
    monkeypatch.setattr(config_module, "load_config", lambda: cfg)
    parts, f = AD.fold_data(root, ["Slice"], sp, ds=1, warmup_sec=0)
    assert [len(parts[p]) for p in C.PARTS] == [4, 2, 2]
    model, norm = AD.train(parts["train"], 1, 4, 2, hidden=4, epochs=1,
                           val_clips=parts["val"])
    before = AD._predict(model, norm, [parts["test"][0]], 4, 2)[0]
    i = sp["test"][0]
    a = np.load(root / f"clip_{i}.npy"); a[8:] += 100
    np.save(root / f"clip_{i}.npy", a)
    changed, f2 = AD.fold_data(root, ["Slice"], sp, ds=1, warmup_sec=0)
    assert f.raw["calibration"]["id"] == f2.raw["calibration"]["id"]
    after = AD._predict(model, norm, [changed["test"][0]], 4, 2)[0]
    np.testing.assert_array_equal(before[:3], after[:3])
    with pytest.raises(ValueError, match="active-hand"):
        AD.load_pooled(f.abspath("states_root"), ["Slice"], 1, .4, hand="active")


@pytest.mark.parametrize("family", ["pg_all", "map_all", "gru_aggregate"])
def test_opentouch_driver_trains_scores_and_exports_one_protocol(tmp_path, family):
    from types import SimpleNamespace
    from scripts.opentouch.run_opentouch_exploratory import run_split
    cfg, _, sp = make_raw(tmp_path)
    args = SimpleNamespace(baseline_scope="train", skip_gru=False, model=family,
                           gru_config="configs/opentouch/gru_aggregate.yaml", histories="0.4",
                           epochs=1, device="cpu", features="raw", weight_decay=0., dropout=0.,
                           select_on="nll", save_preds=str(tmp_path / "preds"),
                           save_model=str(tmp_path / "models"))
    rows, results = run_split(cfg, sp, args, "synthetic")
    assert rows and "ar" in results and len(results) > 3
    for i in sp["test"]:
        z = np.load(tmp_path / "preds" / f"clip_{i}.npz")
        assert str(z["calibration_id"]) != "legacy"
        assert json.loads(str(z["split_ids"])) == sp
    import torch
    for path in (tmp_path / "models").glob("*.pt"):
        meta = torch.load(path, weights_only=False)
        assert C.held_out_ids(meta) == sp["test"]
        C.verify_checkpoint_cache(meta)
