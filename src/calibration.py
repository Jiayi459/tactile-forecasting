"""Frozen TRAIN-only sensor calibration and fold-specific, reproducible caches.

Calibration is fitted before forecasting, like a training normalizer. Transforming a
clip never estimates anything from that clip. Old, non-causal ActionSense resampling
cannot be repaired from its output: strict runs require a fresh raw extraction.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

VERSION = "train_only_v1"
PARTS = ("train", "val", "test")


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


@lru_cache(maxsize=8192)
def _file_digest(path: str, size: int, mtime_ns: int) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_digest(path) -> str:
    p = Path(path).resolve()
    st = p.stat()
    return _file_digest(str(p), st.st_size, st.st_mtime_ns)


def read_manifest(root) -> dict[int, dict]:
    rows = [json.loads(l) for l in (Path(root) / "manifest.jsonl").read_text().splitlines() if l.strip()]
    out = {int(r["idx"]): r for r in rows}
    if len(out) != len(rows):
        raise ValueError("duplicate recording IDs in manifest")
    return out


def validate_splits(splits: dict) -> dict[str, list[int]]:
    out = {p: sorted(int(i) for i in splits.get(p, [])) for p in PARTS}
    for p, ids in out.items():
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate IDs in {p}")
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if set(out[a]) & set(out[b]):
            raise ValueError(f"{a}/{b} overlap")
    if not out["train"]:
        raise ValueError("calibration requires nonempty TRAIN")
    return out


def validate_raw(row: dict, corpus: str):
    if row.get("d1") or row.get("d1_k") is not None or row.get("calibration_id"):
        raise ValueError("already calibrated pressure: use the original raw maps")
    if row.get("pressure_calibration", "none") != "none":
        raise ValueError("expected uncalibrated raw pressure")
    if corpus == "actionsense" and row.get("resampling") != "previous_sample_hold_v1":
        raise ValueError("ActionSense requires previous_sample_hold_v1 raw extraction; "
                         "old linearly interpolated maps cannot be made causal. "
                         "Re-extract the timestamped HDF5 with probe_actionsense.py.")
    if corpus not in ("actionsense", "opentouch"):
        raise ValueError(f"unsupported calibration corpus: {corpus}")


def moments(clip: np.ndarray) -> np.ndarray:
    """Framewise physical moments of (T,C,H,W); no temporal statistic."""
    T, C, H, W = clip.shape
    yy, xx = np.meshgrid(np.linspace(-1, 1, H), np.linspace(-1, 1, W), indexing="ij")
    out = np.zeros((T, C, 6), np.float32)
    for s in range(0, T, 128):
        p = np.maximum(np.asarray(clip[s:s + 128], np.float64), 0)
        f = p.sum((-2, -1))
        safe = np.where(f > 1e-9, f, 1)
        x = (p * xx).sum((-2, -1)) / safe
        y = (p * yy).sum((-2, -1)) / safe
        dx, dy = xx - x[..., None, None], yy - y[..., None, None]
        a = np.stack([f, x, y, (p * dx * dx).sum((-2, -1)) / safe,
                      (p * dy * dy).sum((-2, -1)) / safe,
                      (p * dx * dy).sum((-2, -1)) / safe], -1)
        a[..., 1:] = np.where((f > 1e-9)[..., None], a[..., 1:], 0)
        out[s:s + len(p)] = a
    return out


@dataclass(frozen=True)
class Calibration:
    base: np.ndarray
    sigma: np.ndarray
    metadata: dict

    def apply(self, raw: np.ndarray) -> np.ndarray:
        x = np.asarray(raw, dtype=np.float32)
        if x.ndim != 4 or x.shape[1:] != self.base.shape:
            raise ValueError(f"pressure shape {x.shape} does not match {self.base.shape}")
        if not np.isfinite(x).all():
            raise ValueError("nonfinite raw pressure")
        return np.maximum(x - (self.base + self.metadata["k"] * self.sigma), 0)

    def payload(self) -> dict:
        return {"base": self.base.tolist(), "sigma": self.sigma.tolist(), "metadata": self.metadata}

    @property
    def id(self) -> str:
        return digest(self.payload())

    @classmethod
    def from_payload(cls, data):
        return cls(np.asarray(data["base"], np.float32), np.asarray(data["sigma"], np.float32),
                   data["metadata"])


def fit_calibration(root, train_ids, corpus: str, *, max_frames=20000, k=1.0) -> Calibration:
    """Only TRAIN signal files are read. Equal shard/recording sampling, fixed in advance."""
    ids = sorted(int(i) for i in train_ids)
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("TRAIN IDs must be nonempty and unique")
    if max_frames < len(ids) or not np.isfinite(k) or k < 0:
        raise ValueError("max_frames must cover TRAIN IDs; k must be finite and nonnegative")
    rows = read_manifest(root)
    groups = {}
    for i in ids:
        validate_raw(rows[i], corpus)
        group = rows[i].get("shard", "") if corpus == "opentouch" else "all"
        groups.setdefault(group, []).append(i)
    samples, sources, shape = [], {}, None
    for group in sorted(groups):
        members = groups[group]
        n = max_frames // len(groups) // len(members)
        if n < 1:
            raise ValueError("max_frames is too small for balanced sampling of every TRAIN clip")
        for i in members:
            path = Path(root) / f"clip_{i}.npy"
            a = np.load(path, mmap_mode="r", allow_pickle=False)
            if a.ndim != 4 or not len(a):
                raise ValueError(f"invalid raw map: {path}")
            shape = shape or a.shape[1:]
            if a.shape[1:] != shape:
                raise ValueError("sensor grid/hand count changed within TRAIN")
            # Repeated indices for short clips give every member the same weight.
            fr = np.asarray(a[np.linspace(0, len(a) - 1, n, dtype=int)], np.float32)
            if not np.isfinite(fr).all():
                raise ValueError(f"nonfinite TRAIN pressure: {path}")
            samples.append(fr.reshape(n, -1))
            sources[str(i)] = file_digest(path)
    frames = np.concatenate(samples)
    if corpus == "actionsense":
        base = np.percentile(frames, 5, axis=0)
        sigma = np.zeros_like(base)
        k = 0.0
        estimator = "p5"
    else:
        from src.opentouch.baseline import estimate
        base, sigma = estimate(frames)
        estimator = "median_one_sided_mad_quantum_floor"
    metadata = {"version": VERSION, "corpus": corpus, "train_ids": ids, "sources": sources,
                "estimator": estimator, "k": float(k), "max_frames": int(max_frames),
                "sampling": "equal_shard_then_clip_linspace_with_repetition",
                "train_rows": {str(i): rows[i] for i in ids}}
    return Calibration(base.reshape(shape).astype(np.float32),
                       sigma.reshape(shape).astype(np.float32), metadata)


def enabled(cfg) -> bool:
    return cfg.raw.get("calibration", {}).get("mode") == "train_only"


def require_prepared(cfg):
    if enabled(cfg) and not cfg.raw["calibration"].get("artifact"):
        raise ValueError("TRAIN-only data must be prepared for its split with prepare_fold first")


def prepared(cfg) -> bool:
    require_prepared(cfg)
    return enabled(cfg)


def validate_training(cfg, train_ids, val_ids):
    if prepared(cfg):
        sp = cfg.raw["calibration"]["split"]
        if sorted(train_ids) != sp["train"] or sorted(val_ids) != sp["val"]:
            raise ValueError("model training IDs do not match calibration TRAIN/VAL")


def prepare_fold(cfg, splits: dict):
    """Bind all arms to the same calibrated cache, or validate an already bound fold.

    Original configs/caches remain intact. The resulting config hashes both the original
    protocol and this fold's split, calibration and raw-data provenance.
    """
    if not enabled(cfg):
        return cfg
    sp = validate_splits(splits)
    options = cfg.raw["calibration"]
    if options.get("artifact"):
        if options["split"] != sp:
            raise ValueError("calibrated cache belongs to another split")
        return cfg
    corpus = options["corpus"]
    root = cfg.abspath("raw_root") if "raw_root" in cfg.raw["paths"] else cfg.abspath("states_root")
    rows = read_manifest(root)
    ids = sorted(set().union(*(set(sp[p]) for p in PARTS)))
    for i in ids:
        validate_raw(rows[i], corpus)
    sources = {str(i): file_digest(Path(root) / f"clip_{i}.npy") for i in ids}
    signature = {"version": VERSION, "split": sp, "sources": sources,
                 "rows": {str(i): rows[i] for i in ids}, "options": options}
    key = digest(signature)
    cache_root = Path(cfg.abspath("calibration_cache")) if "calibration_cache" in cfg.raw["paths"] \
        else Path(__file__).resolve().parents[1] / "runs" / "calibration"
    dest = cache_root / corpus / key
    ready = dest / "calibration.json"
    if ready.exists():
        record = json.loads(ready.read_text())
        cal = Calibration.from_payload(record["artifact"])
        if record["signature"] != key or cal.id != record["calibration_id"]:
            raise ValueError(f"invalid calibration cache: {dest}")
        if any(not (dest / f"{kind}_{i}.npy").exists() for i in ids for kind in ("clip", "state")):
            raise ValueError(f"incomplete calibration cache: {dest}")
    else:
        cal = fit_calibration(root, sp["train"], corpus,
                              max_frames=int(options.get("max_frames", 20000)),
                              k=float(options.get("k", 1.0)))
        dest.mkdir(parents=True, exist_ok=True)
        new_rows = []
        for i in ids:
            raw = np.load(Path(root) / f"clip_{i}.npy", allow_pickle=False)
            corr = cal.apply(raw)
            np.save(dest / f"clip_{i}.npy", corr)
            np.save(dest / f"state_{i}.npy", moments(corr))
            new_rows.append({**rows[i], "T": len(raw), "has_clip": True,
                             "pressure_calibration": VERSION, "calibration_id": cal.id,
                             "raw_sha256": sources[str(i)]})
        (dest / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in new_rows))
        (dest / "splits.json").write_text(json.dumps(sp, indent=2))
        record = {"signature": key, "artifact": cal.payload(), "calibration_id": cal.id,
                  "split": sp, "raw_root": str(root), "sources": sources}
        # Write the completion marker last; interrupted construction is never reused.
        temp = dest / f"calibration.{os.getpid()}.tmp"
        temp.write_text(json.dumps(record, indent=2))
        temp.replace(ready)
    raw_cfg = copy.deepcopy(cfg.raw)
    raw_cfg["paths"]["states_root"] = str(dest)
    raw_cfg["paths"]["split_file"] = str(dest / "splits.json")
    raw_cfg["calibration"] = {**options, "artifact": cal.payload(), "id": cal.id,
                               "split": sp, "raw_root": str(root)}
    from src.actionsense.eval_harness.config import Config
    import yaml
    # This self-contained config also lets diagnostic/scoring tools consume the fold.
    # Multiple horizons/input protocols can share the same calibrated pressure.
    # Keep each resolved config immutable instead of overwriting another run's config.
    config_path = dest / f"config_{digest(raw_cfg)[:16]}.yaml"
    config_path.write_text(yaml.safe_dump(raw_cfg, sort_keys=False))
    return Config(raw_cfg, str(config_path), file_digest(config_path)[:16])


def checkpoint_provenance(cfg) -> dict:
    require_prepared(cfg)
    if not enabled(cfg):
        return {"calibration_protocol": "legacy"}
    return {"calibration_protocol": VERSION, "calibration": cfg.raw["calibration"],
            "split_ids": cfg.raw["calibration"]["split"], "config_hash": cfg.config_hash,
            "states_root": cfg.abspath("states_root")}


def load_external_predictions(path, cfg, splits):
    """Load an external scorer archive without silently accepting another protocol."""
    require_prepared(cfg)
    with np.load(path, allow_pickle=False) as archive:
        if enabled(cfg):
            if "calibration_id" not in archive or "split_ids" not in archive:
                raise ValueError("external predictions lack calibration/split provenance")
            if str(archive["calibration_id"]) != cfg.raw["calibration"]["id"] \
                    or validate_splits(json.loads(str(archive["split_ids"]))) != validate_splits(splits):
                raise ValueError("external predictions have different calibration/split provenance")
        preds = {int(k): archive[k] for k in archive.files if k.isdecimal()}
    if enabled(cfg) and set(preds) != set(splits["test"]):
        raise ValueError("external predictions must cover exactly the held-out TEST IDs")
    return preds


def held_out_ids(meta: dict, requested=None) -> list[int]:
    """A checkpoint consumer cannot invent a test split after training."""
    if "split_ids" not in meta:
        raise ValueError("checkpoint lacks split provenance; cannot claim held-out evaluation")
    sp = validate_splits(meta["split_ids"])
    ids = sp["test"] if requested is None else [int(i) for i in requested]
    if not ids or not set(ids) <= set(sp["test"]):
        raise ValueError("requested recordings are not checkpoint-held-out TEST IDs")
    return ids


def verify_checkpoint_cache(meta: dict):
    held_out_ids(meta)
    record = json.loads((Path(meta["states_root"]) / "calibration.json").read_text())
    if record["calibration_id"] != meta.get("calibration", {}).get("id") \
            or record["split"] != validate_splits(meta["split_ids"]):
        raise ValueError("checkpoint and cache calibration/split provenance differ")


def main():
    import argparse
    from src.actionsense.eval_harness.config import load_config
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--splits", required=True, help="explicit train/val/test JSON")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if not enabled(cfg):
        raise ValueError("config must enable calibration.mode=train_only")
    bound = prepare_fold(cfg, json.loads(Path(args.splits).read_text()))
    print(f"protocol={VERSION} calibration={bound.raw['calibration']['id']} "
          f"split sizes={[len(bound.raw['calibration']['split'][p]) for p in PARTS]}")
    print(f"resolved config: {bound.path}")


if __name__ == "__main__":
    main()
