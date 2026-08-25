"""Download the EgoTouch dataset from Hugging Face into datasets/EgoTouch/.

By default this skips the large *.mp4 video files (~72.5 GB), downloading only
the JSON annotations, .npz arrays, and split.json (~16 GB). Pass --videos to
include the mp4 files for a full pull.
"""
import argparse
import os

from huggingface_hub import snapshot_download

REPO_ID = "zhouzhoujy/EgoTouch"
DEST = os.path.join("datasets", "EgoTouch")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--videos", action="store_true",
                   help="also download the *.mp4 video files (full ~88.5 GB pull)")
    p.add_argument("--pressure-only", action="store_true",
                   help="download ONLY pressure_grids.npz (+ split.json), ~1.7 GB; "
                        "enough for tactile->tactile pretraining")
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    allow = ignore = None
    if args.pressure_only:
        allow = ["*/*/*/pressure_grids.npz", "split.json"]
    elif not args.videos:
        ignore = ["*.mp4"]
    os.makedirs(DEST, exist_ok=True)
    path = snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=DEST,
        allow_patterns=allow,
        ignore_patterns=ignore,
        max_workers=args.workers,
    )
    print("Downloaded to:", path)


if __name__ == "__main__":
    main()
