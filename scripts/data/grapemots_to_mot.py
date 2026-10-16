"""Build the canonical MOT view of GrapeMOTS from its per-frame instance masks.

Each sequence unzips to `raw/<seq>/default/{images,instances}/frame_%06d.*`: frames
are 0-based raw frame numbers of a 30 fps capture (stride 2 in PP2/PP7, 1 in PP1), and
each mask is an int32 PNG with background 0 and ids `class * 1000 + n`. Only frames with
both image and mask are kept (PP1 has no masks for 331-337).
Writes `<data_root>/grapemots/mot/<split>/<seq>/`:

    img1           per-frame symlinks into raw/<seq>/default/images; frame k is the k-th sorted file
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
CAPTURE_FPS = 30


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


def link_images(images: list[Path], dest: Path) -> None:
    if dest.is_symlink():
        dest.unlink()
    dest.mkdir(parents=True, exist_ok=True)
    for old in dest.iterdir():
        old.unlink()
    for image in images:
        (dest / image.name).symlink_to(os.path.relpath(image, dest))


def write_seqinfo(dest: Path, name: str, length: int, width: int, height: int, ext: str,
                  frame_rate: float) -> None:
    cfg = configparser.ConfigParser()
    cfg.optionxform = str
    cfg["Sequence"] = {
        "name": name,
        "imDir": "img1",
        "frameRate": f"{frame_rate:g}",
        "seqLength": str(length),
        "imWidth": str(width),
        "imHeight": str(height),
        "imExt": ext,
    }
    buf = io.StringIO()
    cfg.write(buf, space_around_delimiters=False)
    dest.write_text(buf.getvalue().rstrip("\n") + "\n")


def convert(images_dir: Path, instances_dir: Path, seq_dir: Path,
            frame_rate: float) -> tuple[int, int, int, int, int]:
    """Frames are the images that have a mask; returns the number dropped last."""
    all_images = sorted(images_dir.iterdir())
    images = [i for i in all_images if (instances_dir / f"{i.stem}.png").exists()]
    lines, width, height = [], 0, 0
    for frame, image in enumerate(images, start=1):
        mask = np.array(Image.open(instances_dir / f"{image.stem}.png"))
        height, width = mask.shape
        for track_id, class_id, x, y, w, h in mask_boxes(mask):
            lines.append(f"{frame},{track_id},{x},{y},{w},{h},1,{class_id},1.0")

    gt = seq_dir / "gt" / "gt.txt"
    gt.parent.mkdir(parents=True, exist_ok=True)
    gt.write_text("\n".join(lines) + "\n")
    link_images(images, seq_dir / "img1")
    write_seqinfo(seq_dir / "seqinfo.ini", seq_dir.name, len(images), width, height,
                  images[0].suffix, frame_rate)
    return len(images), width, height, len(lines), len(all_images) - len(images)


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
        src = raw / seq / "default"
        f0, f1 = (int(p.stem.split("_")[1]) for p in sorted((src / "images").iterdir())[:2])
        frames, width, height, rows, dropped = convert(
            src / "images", src / "instances", seq_dir, CAPTURE_FPS / (f1 - f0))
        print(f"{seq}  -> mot/{args.split}  frames={frames}  {width}x{height}  gt_rows={rows}  dropped={dropped}")


if __name__ == "__main__":
    main()
