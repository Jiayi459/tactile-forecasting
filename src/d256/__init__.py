"""The d256 arm.

`raw.py` reads the shipped 16-frame clips. Everything downstream works on the *recordings*
rebuilt from them by `scripts/d256/extract_d256_states.py` -- see `dataset.py`.

Re-exported at package level so `from src import d256; d256.load_clip(...)` keeps working
from before this was split into a package.
"""
from .raw import (  # noqa: F401
    CLIP_FRAMES, FPS_UNKNOWN, N_CLASSES, ROOT_DIRNAME, SIGNAL_GROUPS, SIGNAL_SHAPES,
    SPLITS, SUBJECTS, TACTILE_KEYS, VIDEO_GROUPS,
    Clip, counts, ego4d_vocab, iter_clips, iter_paths, label_map, load_clip, parse_path,
    root_of, tactile,
)
