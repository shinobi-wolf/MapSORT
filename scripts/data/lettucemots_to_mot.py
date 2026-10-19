"""Build the canonical MOT view of LettuceMOTS from its MOTS annotations.

LettuceMOTS ships ground truth as MOTS instance files, `instances/<split>/<seq>.txt`,
one row per object instance:

    frame id class height width RLE

with 0-based frames and ids encoded as `class * 1000 + n`. Only three sequences
ever got converted to MOT format upstream, so this script does the conversion
for all of them, writing into the project's own data root and never touching
the download.

Sequences land under the split their annotations come from, so `mot/test/` holds
the upstream held-out sequences (0003, 0007, 0011) and `mot/train/` the other
nine. For each it writes `<data_root>/lettucemots/mot/<split>/<seq>/`:

    img1/%06d.png    symlinks into raw/<split>/images/<seq>, renamed to 1-based
    gt/gt.txt        frame, track_id, x, y, w, h, conf, class, visibility
    seqinfo.ini      MOTChallenge sequence metadata

Boxes are the tight axis-aligned bounds of each instance mask. The output is
byte-identical to the three sequences converted upstream; `--verify` checks that.

Usage:
    python scripts/data/lettucemots_to_mot.py            # all sequences
    python scripts/data/lettucemots_to_mot.py --verify   # also diff vs upstream
    python scripts/data/lettucemots_to_mot.py --split test   # force one split
"""

from __future__ import annotations

import argparse
import configparser
import io
from pathlib import Path

from pycocotools import mask as mask_utils

REPO_ROOT = Path(__file__).resolve().parents[2]
FRAME_RATE = 30  # matches the seqinfo.ini of the upstream-converted sequences


def parse_instances(path: Path) -> list[tuple[int, int, int, float, float, float, float]]:
    """Decode a MOTS instance file into `(frame, track_id, class_id, x, y, w, h)`
    rows, with frames and boxes still in the source's own conventions."""
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        frame, obj_id, class_id, height, width, rle = line.split(" ", 5)
        # Ignore-regions are id 10000 in the MOTS spec; LettuceMOTS has none,
        # but skip them rather than emitting them as tracks.
        if int(obj_id) == 10000:
            continue
        box = mask_utils.toBbox(
            {"size": [int(height), int(width)], "counts": rle.strip().encode()}
        )
        x, y, w, h = (float(v) for v in box)
        rows.append((int(frame), int(obj_id) % 1000, int(class_id), x, y, w, h))
    return rows


def write_gt(rows, dest: Path) -> None:
    """Write MOT `gt.txt`: 1-based frames, top-left xywh, all boxes fully visible."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"{frame + 1},{track_id},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1,{class_id},1.0"
        for frame, track_id, class_id, x, y, w, h in sorted(rows)
    ]
    dest.write_text("\n".join(lines) + "\n")


def link_images(raw_images: Path, dest: Path, num_frames: int) -> None:
    """Symlink the sequence's frames into `img1/`, renumbered from 1.

    The source is 0-based, MOTChallenge expects 1-based, so frame `t` on disk
    becomes `%06d.png % (t + 1)`.
    """
    dest.mkdir(parents=True, exist_ok=True)
    for t in range(num_frames):
        link = dest / f"{t + 1:06d}.png"
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(raw_images / f"{t:06d}.png")


def write_seqinfo(dest: Path, name: str, length: int, width: int, height: int) -> None:
    cfg = configparser.ConfigParser()
    cfg.optionxform = str  # MOTChallenge keys are camelCase; don't lowercase them
    cfg["Sequence"] = {
        "name": name,
        "imDir": "img1",
        "frameRate": str(FRAME_RATE),
        "seqLength": str(length),
        "imWidth": str(width),
        "imHeight": str(height),
        "imExt": ".png",
    }
    buf = io.StringIO()
    cfg.write(buf, space_around_delimiters=False)
    # configparser ends a section with a blank line; the upstream files don't.
    dest.write_text(buf.getvalue().rstrip("\n") + "\n")


def convert(seq: str, src_split: str, raw: Path, out_split_dir: Path) -> Path:
    instances = raw / "instances" / src_split / f"{seq}.txt"
    raw_images = raw / src_split / "images" / seq

    rows = parse_instances(instances)
    # Mask dimensions are per-row but constant within a sequence. The frame count
    # comes from the images rather than the annotations, since a trailing frame
    # with no visible instance would otherwise be dropped.
    height, width = [
        int(v) for v in instances.read_text().splitlines()[0].split(" ")[3:5]
    ]
    num_frames = len(list(raw_images.glob("*.png")))

    seq_dir = out_split_dir / seq
    write_gt(rows, seq_dir / "gt" / "gt.txt")
    link_images(raw_images, seq_dir / "img1", num_frames)
    write_seqinfo(seq_dir / "seqinfo.ini", seq, num_frames, width, height)
    return seq_dir


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--data-root",
        type=Path,
        default=REPO_ROOT / "data",
        help="project data root (default: ./data)",
    )
    ap.add_argument(
        "--split",
        help="write every sequence under this one split, instead of mirroring "
        "the upstream train/test division",
    )
    ap.add_argument("--seqs", nargs="*", help="sequences to convert (default: all)")
    ap.add_argument(
        "--verify",
        action="store_true",
        help="diff generated gt.txt against the upstream-converted copy where one exists",
    )
    args = ap.parse_args()

    raw = args.data_root / "lettucemots" / "raw"
    if not raw.exists():
        raise SystemExit(f"raw symlink does not resolve: {raw} (see data/README.md)")

    # Which upstream split each sequence's annotations live in.
    src_split = {
        seq: split
        for split in ("train", "test")
        for seq in sorted(p.stem for p in (raw / "instances" / split).glob("*.txt"))
    }
    seqs = args.seqs or sorted(src_split)

    mot = args.data_root / "lettucemots" / "mot"
    for seq in seqs:
        if seq not in src_split:
            raise SystemExit(f"no instance file for sequence {seq}")

        out_split_dir = mot / (args.split or src_split[seq])

        # A sequence left over from an earlier layout may be a symlink into raw/,
        # which has to go before we can write a real directory in its place.
        seq_dir = out_split_dir / seq
        if seq_dir.is_symlink():
            seq_dir.unlink()

        seq_dir = convert(seq, src_split[seq], raw, out_split_dir)
        rows = len((seq_dir / "gt" / "gt.txt").read_text().splitlines())
        frames = len(list((seq_dir / "img1").glob("*.png")))
        split = seq_dir.parent.name
        print(f"{seq}  -> mot/{split:5s}  frames={frames:4d}  gt_rows={rows:5d}")

        if args.verify:
            upstream = raw / "mot" / src_split[seq] / seq / "gt" / "gt.txt"
            if not upstream.exists():
                print(f"       no upstream copy to verify against")
                continue
            ours = (seq_dir / "gt" / "gt.txt").read_text()
            match = ours == upstream.read_text()
            print(f"       verify vs upstream: {'identical' if match else 'DIFFERS'}")


if __name__ == "__main__":
    main()
