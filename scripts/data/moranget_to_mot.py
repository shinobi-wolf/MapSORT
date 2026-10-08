"""Build the canonical MOT view of MOrangeT.

The download is already MOTChallenge-shaped, so this only normalises it: `img1`
symlinks to the untouched frames, `gt.txt` loses the spaces after its commas,
and `seqinfo.ini` is rewritten with the frames' real dimensions (upstream's says
1080x1940; the JPEGs and the GT boxes are 1083x1944).

`gt.txt` is the visible-only ground truth used for evaluation; `gt.all.txt` is
copied alongside it and also holds boxes >50% occluded.

Usage:
    python scripts/data/moranget_to_mot.py [--seqs V01 ...] [--split test]
"""

from __future__ import annotations

import argparse
import configparser
import io
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def normalise_gt(src: Path, dest: Path) -> int:
    """Rewrite a MOT16 gt file as bare comma-separated fields."""
    lines = [
        ",".join(f.strip() for f in line.split(","))
        for line in src.read_text().splitlines()
        if line.strip()
    ]
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(lines) + "\n")
    return len(lines)


def link_images(raw_images: Path, dest: Path) -> None:
    """Point img1 at the download; frames are already 1-based `frame-%04d.jpg`."""
    if dest.is_symlink() or dest.exists():
        dest.unlink()
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Relative, so the tree survives being moved as long as raw/ resolves.
    dest.symlink_to(os.path.relpath(raw_images, dest.parent))


def write_seqinfo(src: Path, dest: Path, width: int, height: int, length: int) -> None:
    cfg = configparser.ConfigParser()
    cfg.optionxform = str
    cfg.read(src)
    cfg["Sequence"].update(
        imDir="img1", imWidth=str(width), imHeight=str(height), seqLength=str(length)
    )
    buf = io.StringIO()
    cfg.write(buf, space_around_delimiters=False)
    dest.write_text(buf.getvalue().rstrip("\n") + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, default=REPO_ROOT / "data")
    ap.add_argument("--split", default="test", help="upstream ships every sequence as test")
    ap.add_argument("--seqs", nargs="*", help="default: every sequence present in raw/")
    args = ap.parse_args()

    import cv2

    raw = args.data_root / "moranget" / "raw"
    if not raw.exists():
        raise SystemExit(f"raw symlink does not resolve: {raw} (see data/README.md)")

    seqs = args.seqs or sorted(p.name for p in raw.glob("V*") if (p / "seqinfo.ini").exists())
    for seq in seqs:
        src, seq_dir = raw / seq, args.data_root / "moranget" / "mot" / args.split / seq
        frames = sorted((src / "images").glob("*.jpg"))
        height, width = cv2.imread(str(frames[0])).shape[:2]

        rows = normalise_gt(src / "gt" / "gt.txt", seq_dir / "gt" / "gt.txt")
        all_rows = normalise_gt(src / "gt" / "gt.all.txt", seq_dir / "gt" / "gt.all.txt")
        link_images(src / "images", seq_dir / "img1")
        write_seqinfo(src / "seqinfo.ini", seq_dir / "seqinfo.ini", width, height, len(frames))

        print(
            f"{seq}  -> mot/{args.split:5s}  frames={len(frames):4d}  "
            f"{width}x{height}  gt_rows={rows:5d}  (all={all_rows})"
        )


if __name__ == "__main__":
    main()
