"""Write GT-derived detections: `mot/<split>/<seq>/gt/gt.txt` -> `dets/<seq>.txt`.

Rows are `frame,-1,x,y,w,h,conf,class` (1-based frames, top-left xywh), with
conf fixed at 1.0. A real detector's output replaces these later.

`--masks` also writes `masks/<seq>/%06d.png` (1-based frame) for MOTS datasets: a
uint16 label image where pixel value k is the k-th detection of that frame in
`dets/<seq>.txt` and 0 is background, the format a segmenting detector writes too.

Usage:
    python scripts/data/gt_to_dets.py [--data-root DIR] [--dataset lettucemots] [--seqs ...] [--masks]
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]


def instance_mask(image: Path) -> Path:
    """The raw instance mask behind an img1 symlink: the `images` path component becomes `instances`."""
    parts = list(image.resolve().parts)
    parts[len(parts) - 1 - parts[::-1].index("images")] = "instances"
    return Path(*parts).with_suffix(".png")


def write_masks(seq_dir: Path, out: Path, per_frame: Counter) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for frame, image in enumerate(sorted((seq_dir / "img1").iterdir()), start=1):
        raw = np.array(Image.open(instance_mask(image)))
        # gt.txt lists a frame's instances in ascending raw label order (find_objects).
        labels = np.unique(raw[raw > 0])
        if len(labels) != per_frame.get(frame, 0):
            raise SystemExit(f"{seq_dir.name} frame {frame}: {len(labels)} masks vs {per_frame.get(frame, 0)} dets")
        relabel = np.zeros(int(raw.max()) + 1, dtype=np.uint16)
        relabel[labels] = np.arange(1, len(labels) + 1)
        Image.fromarray(relabel[raw]).save(out / f"{frame:06d}.png")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, default=REPO_ROOT / "data")
    ap.add_argument("--dataset", default="lettucemots")
    ap.add_argument("--seqs", nargs="*", help="default: every sequence under mot/")
    ap.add_argument("--masks", action="store_true", help="also write masks/<seq>/ from the raw instance masks")
    args = ap.parse_args()

    root = args.data_root / args.dataset
    dets = root / "dets"
    dets.mkdir(exist_ok=True)
    for gt in sorted(root.glob("mot/*/*/gt/gt.txt")):
        seq = gt.parent.parent.name
        if args.seqs and seq not in args.seqs:
            continue
        lines = []
        per_frame: Counter = Counter()
        for line in gt.read_text().splitlines():
            frame, _id, x, y, w, h, _conf, class_id, _vis = line.split(",")
            lines.append(f"{frame},-1,{x},{y},{w},{h},1.0,{class_id}")
            per_frame[int(frame)] += 1
        (dets / f"{seq}.txt").write_text("\n".join(lines) + "\n")
        if args.masks:
            write_masks(gt.parent.parent, root / "masks" / seq, per_frame)
        print(f"{seq}  dets={len(lines)}{'  masks' if args.masks else ''}")


if __name__ == "__main__":
    main()
