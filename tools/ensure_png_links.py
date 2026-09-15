"""
-------------------------------------------------------------------------------
Author: dan64
Date: 2026-09-10
-------------------------------------------------------------------------------
Description:
-------------------------------------------------------------------------------
DAVISVidevoDataset (colormnet/dataset/vos_dataset.py) opens, for each frame
'NNNNN.jpg', also a 'NNNNN.png' file in the same folder (gt_root == im_root
in our case): 'png_name = jpg_name.replace(".jpg", ".png")'. In the original
dataset (upstream README) the frames were already .png, so that substitution
was a no-op (the same file was reopened). Our folders instead use the .jpg
convention, so without this script DAVISVidevoDataset would raise
FileNotFoundError at the first __getitem__.

The content of the 'gt' file is never used: it only needs Image.open() to
open a valid file successfully at that path. To avoid duplicating 43k+
files on disk (jpg already compressed, ~2GB), an NTFS HARD LINK (os.link)
is used - same content, zero extra space, no special privilege required
(unlike symlinks on Windows).

Idempotent: skips the .png that already exist. Does NOT touch/rename any
existing .jpg.

Usage:
    python tools/ensure_png_links.py [root_dir ...]
    (default: datasets/train_root relative to the working directory)
"""
import os
import sys
from pathlib import Path


def ensure_png_links(root_dir: str) -> tuple[int, int]:
    root = Path(root_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"root not found: {root}")

    created = 0
    skipped = 0
    for video_dir in sorted(root.iterdir()):
        if not video_dir.is_dir():
            continue
        for jpg_path in video_dir.glob("*.jpg"):
            png_path = jpg_path.with_suffix(".png")
            if png_path.exists():
                skipped += 1
                continue
            os.link(jpg_path, png_path)
            created += 1
    return created, skipped


if __name__ == "__main__":
    roots = sys.argv[1:] or ["datasets/train_root"]
    for root_dir in roots:
        created, skipped = ensure_png_links(root_dir)
        print(f"{root_dir}: {created} .png hardlinks created, {skipped} already present")
