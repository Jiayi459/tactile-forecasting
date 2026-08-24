"""Reader for the d256 / `Dataset256` release (ICLR force-vision submission).

Layout on disk, after `scripts/crc/fetch_d256.py`:

    <root>/Dataset256/
        signals{,1,2}/<split>/<subject>/<session>/<clip>.p     pickle, one 16-frame clip
        signals/ego_4d_verb.npy                                148 Ego4D verbs
        signals/ego_4d_noun.npy                                112 Ego4D nouns
        videos{,1,2}/<split>/<subject>/<session>/video_<k>_{256,32}.npz   (not fetched by default)

Each `.p` is `{'signal': {...}, 'label_text': str, 'label_idx': int}` carrying the ActionSense
wearable suite for 16 frames -- see `SIGNAL_SHAPES`. Values arrive **pre-scaled to ~[0,1]**;
this reader does not rescale them.

Two properties of the release that callers keep needing, both measured rather than assumed
(`scripts/d256/probe_d256.py` re-checks them against the data):

* **The session directory name equals `label_idx`.** `<...>/S05/3/0.p` is class 3. So the class
  of a clip is readable from its path, and a mismatch means the tree was reorganised.
* **`val` is three held-out S05 sessions, and which three differs per group.** It is *not* a
  subject-level split -- S05 dominates `train` too -- and it covers 3 of the 20 classes, so
  `val` accuracy is not a whole-task number. See docs/d256.md.

No location/scene dimension exists: ActionSense was recorded in one instrumented kitchen.
The axes are group / split / subject / session(class) / clip.
"""
from __future__ import annotations

import os
import pickle
from collections import Counter
from typing import Iterator, NamedTuple

import numpy as np

ROOT_DIRNAME = "Dataset256"
SIGNAL_GROUPS = ("signals", "signals1", "signals2")
VIDEO_GROUPS = ("videos", "videos1", "videos2")
SPLITS = ("train", "val")
SUBJECTS = ("S01", "S02", "S03", "S04", "S05")
N_CLASSES = 20
CLIP_FRAMES = 16

# Per-clip array shapes, T=16 leading. Verified identical across a 275-session sample.
SIGNAL_SHAPES = {
    "tactile-glove-left": (16, 32, 32),
    "tactile-glove-right": (16, 32, 32),
    "myo-emg-left": (16, 8),
    "myo-emg-right": (16, 8),
    "myo-acc-left": (16, 3),
    "myo-acc-right": (16, 3),
    "joint-position": (16, 28, 3),
    "right-hand-pose": (16, 24, 3),
    "left-hand-pose": (16, 24, 3),
}
TACTILE_KEYS = ("tactile-glove-left", "tactile-glove-right")

# The clips carry no timestamps, so the capture rate cannot be recovered from the data. Anything
# that converts to a physical rate (velocity, acceleration) must be told the fps explicitly
# rather than inheriting a default from another dataset -- see docs/d256.md, open question 1.
FPS_UNKNOWN = None


class Clip(NamedTuple):
    path: str
    group: str
    split: str
    subject: str
    session: int
    clip_id: int
    label_idx: int
    label_text: str
    signal: dict


def root_of(path: str) -> str:
    """Accept either the fetch --dest or the Dataset256 dir itself."""
    inner = os.path.join(path, ROOT_DIRNAME)
    return inner if os.path.isdir(inner) else path


def iter_paths(root: str, groups=SIGNAL_GROUPS, splits=SPLITS, subjects=None,
               sessions=None) -> Iterator[str]:
    """Yield clip paths without unpickling -- path alone gives group/split/subject/class."""
    root = root_of(root)
    for g in groups:
        for sp in splits:
            base = os.path.join(root, g, sp)
            if not os.path.isdir(base):
                continue
            for subj in sorted(os.listdir(base)):
                if subjects and subj not in subjects:
                    continue
                sdir = os.path.join(base, subj)
                if not os.path.isdir(sdir):
                    continue
                for sess in sorted(os.listdir(sdir), key=_int_or_inf):
                    if sessions is not None and int(sess) not in sessions:
                        continue
                    cdir = os.path.join(sdir, sess)
                    if not os.path.isdir(cdir):
                        continue
                    for fn in sorted(os.listdir(cdir), key=_int_or_inf):
                        if fn.endswith(".p"):
                            yield os.path.join(cdir, fn)


def _int_or_inf(name: str):
    stem = name[:-2] if name.endswith(".p") else name
    return int(stem) if stem.isdigit() else float("inf")


def parse_path(path: str) -> dict:
    """group/split/subject/session/clip_id from the path, no I/O."""
    parts = os.path.normpath(path).split(os.sep)
    return {"group": parts[-5], "split": parts[-4], "subject": parts[-3],
            "session": int(parts[-2]), "clip_id": int(parts[-1][:-2])}


def load_clip(path: str, check: bool = True) -> Clip:
    with open(path, "rb") as fh:
        obj = pickle.load(fh)
    meta = parse_path(path)
    if check and meta["session"] != obj["label_idx"]:
        raise ValueError(
            f"{path}: session dir {meta['session']} != label_idx {obj['label_idx']}; "
            "the path-encodes-class invariant no longer holds, so any code deriving labels "
            "from paths is now wrong. Re-run scripts/d256/probe_d256.py.")
    return Clip(path=path, label_idx=obj["label_idx"], label_text=obj["label_text"],
                signal=obj["signal"], **meta)


def iter_clips(root: str, **kw) -> Iterator[Clip]:
    for p in iter_paths(root, **kw):
        yield load_clip(p)


def tactile(clip: Clip) -> np.ndarray:
    """Both gloves stacked -> (16, 2, 32, 32), matching the (T,2,H,W) layout the EgoTouch and
    OpenTouch probes already consume."""
    return np.stack([clip.signal[k] for k in TACTILE_KEYS], axis=1)


def label_map(root: str, groups=SIGNAL_GROUPS) -> dict:
    """{label_idx: label_text}, read from one clip per session directory."""
    out, seen = {}, set()
    for p in iter_paths(root, groups=groups):
        meta = parse_path(p)
        key = (meta["group"], meta["split"], meta["subject"], meta["session"])
        if key in seen:
            continue
        seen.add(key)
        c = load_clip(p)
        out.setdefault(c.label_idx, c.label_text)
    return dict(sorted(out.items()))


def ego4d_vocab(root: str):
    """(verbs, nouns) as string arrays, or (None, None) if the .npy files weren't fetched."""
    base = os.path.join(root_of(root), "signals")
    v = os.path.join(base, "ego_4d_verb.npy")
    n = os.path.join(base, "ego_4d_noun.npy")
    if not (os.path.exists(v) and os.path.exists(n)):
        return None, None
    return np.load(v, allow_pickle=True), np.load(n, allow_pickle=True)


def counts(root: str, groups=SIGNAL_GROUPS) -> Counter:
    """Clip counts keyed (group, split, subject, session) -- path walk only, no unpickling."""
    c = Counter()
    for p in iter_paths(root, groups=groups):
        m = parse_path(p)
        c[(m["group"], m["split"], m["subject"], m["session"])] += 1
    return c
