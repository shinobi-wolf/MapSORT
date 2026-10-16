"""Build the canonical MOT view of AppleMOTS from its per-frame instance masks.

Raw layout is `raw/APPLE_MOTS/{train,testing}/{images,instances}/<seq>/%06d.png`, masks
uint16 with ids `class * 1000 + n`, as in GrapeMOTS. `testing` maps to split `test`.
Writes `<data_root>/applemots/mot/<split>/<seq>/{img1,gt/gt.txt,seqinfo.ini}`.
Testing 0010 has a mask but no image for frame 96, so that frame is dropped.

Usage:
    python scripts/data/applemots_to_mot.py [--seqs 0006 ...]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from grapemots_to_mot import convert

REPO_ROOT = Path(__file__).resolve().parents[2]
SPLITS = {"train": "train", "testing": "test"}
FRAME_RATE = 30  # not published; only used to time rendered videos


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, default=REPO_ROOT / "data")
    ap.add_argument("--seqs", nargs="*", help="default: every sequence in raw/")
    args = ap.parse_args()

    raw = args.data_root / "applemots" / "raw" / "APPLE_MOTS"
    if not raw.exists():
        raise SystemExit(f"raw symlink does not resolve: {raw} (see data/README.md)")

    for src_split, split in SPLITS.items():
        for images_dir in sorted((raw / src_split / "images").iterdir()):
            seq = images_dir.name
            if args.seqs and seq not in args.seqs:
                continue
            seq_dir = args.data_root / "applemots" / "mot" / split / seq
            frames, width, height, rows, dropped = convert(
                images_dir, raw / src_split / "instances" / seq, seq_dir, FRAME_RATE)
            print(f"{seq}  -> mot/{split}  frames={frames}  {width}x{height}  gt_rows={rows}  dropped={dropped}")


if __name__ == "__main__":
    main()
