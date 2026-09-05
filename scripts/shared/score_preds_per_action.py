"""Per-action R2 / skill / Hausdorff from saved forecasts. Works on BOTH sensors.

WHY ONE SCRIPT SERVES BOTH. src/actionsense/tactile_map/train.py::save_predictions writes
"one clip_<idx>.npz per recording, in the OpenTouch overlay format" -- same keys, same
layout as scripts/opentouch/run_opentouch_exploratory.py --save-preds. So the same reader
scores an ActionSense run and an OpenTouch run, and the OpenTouch per-action Hausdorff that
does not exist today needs no new code, only its preds directory.

WHAT IT COMPUTES, AND WITH WHOSE DEFINITIONS
  * R2 and skill come from src/opentouch/aggregate.py -- `r2` (clip-balanced, against the
    CLASS MEAN of the scored subset) and `skill` (clip-balanced, against persistence). Those
    are the canonical definitions already used by every OpenTouch number; they are imported,
    not reimplemented, so this table cannot drift from them.
  * Hausdorff comes from src/shape_metrics.py::hausdorff_scaled, averaged clip-equal to match
    the R2 weighting. Absolute and residual curves give the SAME value here: the metric scales
    each pair by the truth's own per-horizon sd, and subtracting the shared anchor y[t] changes
    neither that sd nor the distance -- so there is no absolute-vs-residual choice to get wrong.
  * PERSISTENCE IS SYNTHESIZED, not read: the npz stores only learned arms, so persistence is
    rebuilt as y[origin] repeated across the horizon -- the same definition
    src/actionsense/tactile_map/train.py::_predict uses for its absolute-space reference.

MASKING -- READ THIS BEFORE QUOTING CoP NUMBERS. The frozen harnesses drop a CoP target frame
whose hand force is below the TRAIN 5th percentile, and that threshold is fitted PER FOLD.
A preds directory does not record fold membership, so this script cannot reproduce it. The
options are both honest and neither is the harness's:
  --mask none    (default) no masking at all; CoP is scored at every origin, including
                 near-zero-force frames where CoP is close to undefined.
  --mask corpus  one threshold from the 5th percentile of force over ALL clips present.
                 Transductive -- it sees every clip -- so it is a descriptive convenience,
                 not a protocol. Flagged in the output either way.
Force channels are never masked under either option.

    python scripts/shared/score_preds_per_action.py --preds runs/as_preds_seq2seq_corpus \
        --label "ActionSense seq2seq/aggregate (corpus scope)" --out docs/actionsense/per_action
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.opentouch import aggregate                       # noqa: E402
from src.shape_metrics import hausdorff_scaled            # noqa: E402

PERS = "persistence"


def load_clip(path):
    """-> (idx, y (T,C), origins (n,), {model: mu (n,H,C)}, action, channels)."""
    z = np.load(path, allow_pickle=True)
    idx = int(os.path.basename(path).split("_")[1].split(".")[0])
    y = np.asarray(z["y"], dtype=np.float64)
    ors = np.asarray(z["origins"], dtype=np.int64)
    mus = {k[3:]: np.asarray(z[k], dtype=np.float64) for k in z.files if k.startswith("mu_")}
    act = str(z["action"]) if "action" in z.files else ""
    chans = tuple(str(c) for c in z["channels"]) if "channels" in z.files else ()
    return idx, y, ors, mus, act, chans


def actions_from_manifest(path):
    """{idx: action} from a jsonl manifest -- for preds that do not carry the label."""
    out = {}
    with open(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            a = r.get("action") or r.get("label", "")
            out[int(r["idx"])] = a.split()[0].lower() if a else ""
    return out


def scan(preds_dir):
    """-> (paths, {path: set(model names)}). One cheap pass, so coverage is known BEFORE any
    stacking.

    Model coverage is not uniform in practice, and assuming it was is what broke this script
    on 2026-09-05: baselines are exported for one frozen test split while a cross-validated
    arm covers every recording, so a clip can carry the neural arm and none of the baselines
    (the same asymmetry plot_opentouch_forecast_overlay.py already had to learn). Stacking
    blindly produced preds['seasonal'] with 4,967 rows against a ytrue of 28,694 and a bare
    ValueError from clip_stats.
    """
    paths = sorted(glob.glob(os.path.join(preds_dir, "clip_*.npz")),
                   key=lambda q: int(os.path.basename(q)[5:-4]))
    if not paths:
        raise SystemExit(f"no clip_*.npz under {preds_dir}")
    have = {}
    for q in paths:
        z = np.load(q, allow_pickle=True)
        have[q] = {k[3:] for k in z.files if k.startswith("mu_")}
    return paths, have


def choose(have, want=None):
    """-> (models, kept_paths, every, common).

    Every scored model must exist on every scored clip. Scoring a model on the clips it
    happens to cover, next to another model on a different set, would put two numbers with
    different populations in the same column -- which is exactly the comparison this table
    exists to make. So either the models shrink (default: those present everywhere) or the
    clips do (--models pins the set and drops clips that lack any of them).
    """
    every = sorted(set().union(*have.values())) if have else []
    common = sorted(set.intersection(*have.values())) if have else []
    models = [m.strip() for m in want.split(",")] if want else common
    if not models:
        raise SystemExit(
            f"no model is present on every clip, so nothing can be scored on one population.\n"
            f"  models seen anywhere: {every}\n"
            f"  pick a subset with --models, e.g. --models {every[0] if every else 'NAME'}")
    unknown = [m for m in models if m not in every]
    if unknown:
        raise SystemExit(f"models {unknown} appear in no clip; available: {every}")
    kept = sorted((q for q, s in have.items() if set(models) <= s),
                  key=lambda q: int(os.path.basename(q)[5:-4]))
    return models, kept, every, common


def gather(paths, models, manifest=None):
    """Read the chosen clips into the (N,H,C) stacks aggregate.clip_stats wants, plus per-clip
    Hausdorff. Only `models` are read, so every stack has identical length by construction."""
    fallback = actions_from_manifest(manifest) if manifest else {}
    yts, ids, acts, chans = [], [], {}, ()
    per_model, hd_rows = {}, {}
    for p in paths:
        idx, y, ors, mus, act, ch = load_clip(p)
        if not len(ors) or not mus:
            continue
        chans = chans or ch
        H = next(iter(mus.values())).shape[1]
        keep = ors[(ors + H) < len(y)]
        if not len(keep):
            continue
        sel = np.isin(ors, keep)
        true = np.stack([y[t + 1:t + 1 + H] for t in keep])          # (n,H,C)
        pers = np.repeat(y[keep][:, None, :], H, axis=1)             # (n,H,C)
        yts.append(true)
        ids += [idx] * len(keep)
        acts[idx] = (act or fallback.get(idx, "") or "").strip().lower() or "unlabelled"
        # OpenTouch runs already store mu_persistence (EV.MODELS includes it); ActionSense runs
        # store learned arms only. Synthesizing a second copy alongside the stored one appended
        # each clip twice and desynchronized the stacks from ytrue -- so take the stored one
        # when it is there, and only rebuild it when it is not.
        arms = {m: mus[m][sel] for m in models}
        arms.setdefault(PERS, pers)
        for m, arr in arms.items():
            per_model.setdefault(m, []).append(arr)
        # per-clip, per-channel Hausdorff (clip-equal, matching the R2 weighting)
        C = true.shape[-1]
        hd_rows[idx] = {m: np.array([np.nanmean(hausdorff_scaled(arr[:, :, c], true[:, :, c]))
                                     for c in range(C)])
                        for m, arr in arms.items()}

    if not yts:
        raise SystemExit("every chosen clip was too short for its horizon")
    ytrue = np.concatenate(yts, 0)
    preds = {m: np.concatenate(v, 0) for m, v in per_model.items()}
    chans = chans or tuple(f"ch{i}" for i in range(ytrue.shape[-1]))
    return ytrue, np.array(ids), preds, acts, hd_rows, chans


def force_channels(chans):
    """Indices of the force channels, read from the channel NAMES rather than hardcoded.

    ActionSense is [F_L, CoPx_L, CoPy_L, F_R, CoPx_R, CoPy_R] -> (0, 3); OpenTouch is
    [F_R, CoPx_R, CoPy_R] -> (0,). These match `force_idx` in the two harness configs, so a
    CoP channel is masked against ITS OWN hand's force instead of whichever force came first.
    """
    idx = tuple(i for i, c in enumerate(chans) if str(c).upper().startswith("F"))
    return idx or (0,)


def build_mask(ytrue, mode, chans, pct=5):
    """(N,H,C) boolean. Force channels always valid; CoP masked only under --mask corpus."""
    mask = np.ones_like(ytrue, dtype=bool)
    if mode != "corpus":
        return mask
    fidx = force_channels(chans)
    thr = {c: np.percentile(ytrue[:, :, c], pct) for c in fidx}
    for c in range(ytrue.shape[-1]):
        if c in fidx:
            continue
        anchor = max(f for f in fidx if f <= c)      # that CoP channel's OWN hand
        mask[:, :, c] = ytrue[:, :, anchor] >= thr[anchor]
    return mask


def score(preds_dir, label, mask_mode, manifest, min_clips, want_models=None):
    paths, have = scan(preds_dir)
    models, kept, every, common = choose(have, want_models)
    print(f"  {len(paths)} clips; models seen: {every}")
    if len(kept) < len(paths):
        print(f"  scoring {len(kept)} clips that carry all of {models}; "
              f"{len(paths) - len(kept)} dropped for missing one")
    dropped = sorted(set(every) - set(models))
    if dropped:
        print(f"  NOT scored (absent from some clip): {dropped} — "
              f"pass --models to pin them instead and drop the clips they miss")
    ytrue, ids, preds, acts, hd_rows, chans = gather(kept, models, manifest)
    st = aggregate.clip_stats(ytrue, build_mask(ytrue, mask_mode, chans), ids, preds, chans)
    models = [m for m in sorted(preds) if m != PERS]

    by_action = {}
    for idx, a in acts.items():
        by_action.setdefault(a, []).append(idx)

    rows = []
    for a, clips in sorted(by_action.items()):
        if len(clips) < min_clips:
            continue
        sel = st.rows_of(clips)
        for m in models + [PERS]:
            r2 = float(np.nanmean(aggregate.r2(st, m, rows=sel).per_channel))
            sk = float(np.nanmean(aggregate.skill(st, m, PERS, rows=sel)))
            hd = float(np.nanmean([np.nanmean(hd_rows[i][m]) for i in clips]))
            hp = float(np.nanmean([np.nanmean(hd_rows[i][PERS]) for i in clips]))
            rows.append(dict(label=label, action=a, n_clips=len(clips), model=m,
                             r2=r2, skill=sk, hausdorff=hd,
                             hausdorff_ratio=hd / hp if hp else float("nan")))
    return rows, models, chans, len(acts)


def to_markdown(rows, models, label, n_clips_total):
    """One table per model, ranked by R2 desc; Hausdorff shown alongside (lower = better)."""
    out = [f"### {label}", "",
           f"{n_clips_total} recordings scored. Ranked by **R²**, high → low. "
           f"Hausdorff is **lower = better**; `HD ratio` < 1 beats persistence.", ""]
    for m in models:
        sub = sorted([r for r in rows if r["model"] == m], key=lambda r: -r["r2"])
        if not sub:
            continue
        out += [f"**model `{m}`**", "",
                "| # | action | n | R² | skill vs pers | Hausdorff | HD ratio |",
                "|---:|---|---:|---:|---:|---:|---:|"]
        for i, r in enumerate(sub, 1):
            out.append(f"| {i} | {r['action']} | {r['n_clips']} | **{r['r2']:.4f}** | "
                       f"{r['skill']:.4f} | {r['hausdorff']:.3f} | {r['hausdorff_ratio']:.3f} |")
        out.append("")
        best_hd = min(sub, key=lambda r: r["hausdorff"])
        out.append(f"Lowest Hausdorff: **{best_hd['action']}** ({best_hd['hausdorff']:.3f}); "
                   f"highest R²: **{sub[0]['action']}** ({sub[0]['r2']:.4f}).")
        out.append("")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preds", required=True, help="directory of clip_*.npz (--save-preds)")
    ap.add_argument("--label", default=None, help="section title; defaults to the dir name")
    ap.add_argument("--out", default="docs/per_action", help="output directory")
    ap.add_argument("--mask", default="none", choices=["none", "corpus"],
                    help="CoP masking; see the module docstring before quoting CoP numbers")
    ap.add_argument("--manifest", default=None,
                    help="jsonl to read actions from, when the npz does not carry them")
    ap.add_argument("--models", default=None,
                    help="comma list to score. Default: the models present on EVERY clip. "
                         "Pinning a model that only some clips carry drops the others, so "
                         "every number in the table still comes from one population.")
    ap.add_argument("--min-clips", type=int, default=1,
                    help="drop actions with fewer recordings than this (OpenTouch uses 30)")
    a = ap.parse_args()

    label = a.label or os.path.basename(a.preds.rstrip("/"))
    rows, models, chans, n_acts = score(a.preds, label, a.mask, a.manifest,
                                       a.min_clips, a.models)
    os.makedirs(a.out, exist_ok=True)
    stem = os.path.join(a.out, os.path.basename(a.preds.rstrip("/")))

    import csv
    with open(stem + ".csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    with open(stem + ".md", "w") as fh:
        fh.write(to_markdown(rows, models, label, n_acts))
        fh.write(f"\n\nchannels: {', '.join(chans)} · CoP masking: `--mask {a.mask}`"
                 f"{' (transductive, not the harness protocol)' if a.mask == 'corpus' else ''}\n")
    print(f"wrote {stem}.csv and {stem}.md  ({len(rows)} rows, models={models})")


if __name__ == "__main__":
    raise SystemExit(main())
