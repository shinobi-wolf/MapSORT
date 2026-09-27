"""Write GT-derived detections: `mot/<split>/<seq>/gt/gt.txt` -> `dets/<seq>.txt`.

Rows are `frame,-1,x,y,w,h,conf,class` (1-based frames, top-left xywh), with
conf fixed at 1.0. A real detector's output replaces these later.

Usage:
    python scripts/data/gt_to_dets.py [--data-root DIR] [--dataset lettucemots]
"""

from __future__ import annotations

import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, default=REPO_ROOT / "data")
    ap.add_argument("--dataset", default="lettucemots")
    args = ap.parse_args()

    root = args.data_root / args.dataset
    dets = root / "dets"
    dets.mkdir(exist_ok=True)
    for gt in sorted(root.glob("mot/*/*/gt/gt.txt")):
        seq = gt.parent.parent.name
        lines = []
        for line in gt.read_text().splitlines():
            frame, _id, x, y, w, h, _conf, class_id, _vis = line.split(",")
            lines.append(f"{frame},-1,{x},{y},{w},{h},1.0,{class_id}")
        (dets / f"{seq}.txt").write_text("\n".join(lines) + "\n")
        print(f"{seq}  dets={len(lines)}")


if __name__ == "__main__":
    main()
