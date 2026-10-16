"""Build the canonical MOT view of GrapeMOTS from its per-frame instance masks.

Each sequence unzips to `raw/<seq>/default/{images,instances}/frame_%06d.*`: frames
are 0-based and every second raw frame, and each mask is an int32 PNG with
background 0 and ids `class * 1000 + n`. Writes `<data_root>/grapemots/mot/<split>/<seq>/`:

    img1           symlink to raw/<seq>/default/images, so frame k is the k-th sorted file
    gt/gt.txt      frame, track_id, x, y, w, h, conf, class, visibility
    seqinfo.ini    MOTChallenge sequence metadata

Usage:
    python scripts/data/grapemots_to_mot.py [--seqs PathPlanning_7 ...] [--split test]
"""

from __future__ import annotations

import argparse
import configparser
import io
import os
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import find_objects

REPO_ROOT = Path(__file__).resolve().parents[2]
FRAME_RATE = 15  # every second frame of a 30 fps capture


def mask_boxes(mask: np.ndarray) -> list[tuple[int, int, int, int, int, int]]:
    """`(track_id, class_id, x, y, w, h)` per instance, as tight mask bounds."""
    rows = []
    for inst_id, sl in enumerate(find_objects(mask), start=1):
        if sl is None:
            continue
        ys, xs = sl
        # n is 0-based upstream; MOT track ids start at 1.
        rows.append((inst_id % 1000 + 1, inst_id // 1000, xs.start, ys.start,
                     xs.stop - xs.start, ys.stop - ys.start))
    return rows


def link_images(raw_images: Path, dest: Path) -> None:
    if dest.is_symlink() or dest.exists():
        dest.unlink()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.symlink_to(os.path.relpath(raw_images, dest.parent))


def write_seqinfo(dest: Path, name: str, length: int, width: int, height: int, ext: str) -> None:
    cfg = configparser.ConfigParser()
    cfg.optionxform = str
    cfg["Sequence"] = {
        "name": name,
        "imDir": "img1",
        "frameRate": str(FRAME_RATE),
        "seqLength": str(length),
        "imWidth": str(width),
        "imHeight": str(height),
        "imExt": ext,
    }
    buf = io.StringIO()
    cfg.write(buf, space_around_delimiters=False)
    dest.write_text(buf.getvalue().rstrip("\n") + "\n")


def convert(src: Path, seq_dir: Path) -> tuple[int, int, int, int]:
    images = sorted((src / "images").iterdir())
    lines, width, height = [], 0, 0
    for frame, image in enumerate(images, start=1):
        mask = np.array(Image.open(src / "instances" / f"{image.stem}.png"))
        height, width = mask.shape
        for track_id, class_id, x, y, w, h in mask_boxes(mask):
            lines.append(f"{frame},{track_id},{x},{y},{w},{h},1,{class_id},1.0")

    gt = seq_dir / "gt" / "gt.txt"
    gt.parent.mkdir(parents=True, exist_ok=True)
    gt.write_text("\n".join(lines) + "\n")
    link_images(src / "images", seq_dir / "img1")
    write_seqinfo(seq_dir / "seqinfo.ini", seq_dir.name, len(images), width, height, images[0].suffix)
    return len(images), width, height, len(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, default=REPO_ROOT / "data")
    ap.add_argument("--split", default="test")
    ap.add_argument("--seqs", nargs="*", help="default: every unzipped sequence in raw/")
    args = ap.parse_args()

    raw = args.data_root / "grapemots" / "raw"
    if not raw.exists():
        raise SystemExit(f"raw symlink does not resolve: {raw} (see data/README.md)")

    seqs = args.seqs or sorted(p.name for p in raw.iterdir() if (p / "default" / "images").is_dir())
    for seq in seqs:
        seq_dir = args.data_root / "grapemots" / "mot" / args.split / seq
        frames, width, height, rows = convert(raw / seq / "default", seq_dir)
        print(f"{seq}  -> mot/{args.split}  frames={frames}  {width}x{height}  gt_rows={rows}")


if __name__ == "__main__":
    main()
