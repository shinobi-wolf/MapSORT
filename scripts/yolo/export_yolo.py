"""Export a YOLO train/val/test view of each dataset to data/<dataset>/yolo/. Splits and sizes: config/yolo.yaml.

Frames are scaled to `long_side` and saved as JPEG; labels are normalised, so they hold at any scale.
`segment` datasets take one polygon per instance from the raw instance masks, a fragmented instance's
pieces merged as Ultralytics' COCO converter does. `detect` datasets take gt.txt boxes clipped to the frame.
The output directory is rebuilt from scratch.

    data/<dataset>/yolo/
      images/<split>/<seq>_<frame:06d>.jpg   frame is the 1-based MOT frame
      labels/<split>/<seq>_<frame:06d>.txt
      data.yaml                              no `path`: pass it by absolute path so splits resolve beside it
      manifest.json

Usage:
    python scripts/yolo/export_yolo.py --datasets applemots grapemots moranget [--workers 6]
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from collections import defaultdict
from datetime import datetime, timezone
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image
from scipy.ndimage import find_objects
from ultralytics.data.converter import merge_multi_segment

from mapsort.data import REPO_ROOT, data_root, sequence, splits
from mapsort.runs import git_state

CONFIG = REPO_ROOT / "config" / "yolo.yaml"
SPLITS = ("train", "val", "test")


def instance_mask(image: Path) -> Path:
    """The raw instance mask behind an img1 symlink, as in scripts/data/gt_to_dets.py."""
    parts = list(image.resolve().parts)
    parts[len(parts) - 1 - parts[::-1].index("images")] = "instances"
    return Path(*parts).with_suffix(".png")


def polygon_lines(mask: np.ndarray, eps: float) -> list[str]:
    height, width = mask.shape
    lines = []
    for inst_id, sl in enumerate(find_objects(mask), start=1):
        if sl is None:
            continue
        ys, xs = sl
        contours, _ = cv2.findContours((mask[sl] == inst_id).astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        segs = [cv2.approxPolyDP(c, eps, True).reshape(-1, 2) for c in contours]
        segs = [s for s in segs if len(s) >= 3]
        if not segs:  # sliver too thin to survive simplification
            segs = [np.array([[0, 0], [xs.stop - xs.start - 1, 0], [xs.stop - xs.start - 1, ys.stop - ys.start - 1],
                              [0, ys.stop - ys.start - 1]])]
        poly = np.concatenate(merge_multi_segment([s.reshape(-1).tolist() for s in segs])) if len(segs) > 1 else segs[0]
        # +0.5: contour vertices are pixel indices, polygon vertices are pixel centres.
        poly = (poly + (xs.start + 0.5, ys.start + 0.5)) / (width, height)
        lines.append("0 " + " ".join(f"{v:.6f}" for v in poly.clip(0, 1).reshape(-1)))
    return lines


def box_lines(rows: np.ndarray, width: int, height: int) -> list[str]:
    lines = []
    for x, y, w, h in rows:
        x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, width), min(y + h, height)
        if x1 - x0 < 1 or y1 - y0 < 1:
            continue
        lines.append(f"0 {(x0 + x1) / 2 / width:.6f} {(y0 + y1) / 2 / height:.6f} "
                     f"{(x1 - x0) / width:.6f} {(y1 - y0) / height:.6f}")
    return lines


def export_frame(job: dict) -> int:
    """Write one image and its label file; returns the number of labels."""
    bgr = cv2.imread(job["image"])
    if bgr is None:
        raise RuntimeError(f"could not read {job['image']}")
    height, width = bgr.shape[:2]
    scale = min(1.0, job["long_side"] / max(height, width))

    if job["task"] == "segment":
        mask = np.array(Image.open(job["mask"]))
        if mask.shape != (height, width):
            raise RuntimeError(f"{job['mask']}: mask {mask.shape} vs image {(height, width)}")
        lines = polygon_lines(mask, eps=1.0 / scale)
        if len(lines) != job["expected"]:
            raise RuntimeError(f"{job['image']}: {len(lines)} masks vs {job['expected']} gt rows")
    else:
        lines = box_lines(job["boxes"], width, height)

    if scale < 1.0:
        bgr = cv2.resize(bgr, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(job["out_image"], bgr, [cv2.IMWRITE_JPEG_QUALITY, job["quality"]])
    Path(job["out_label"]).write_text("".join(f"{line}\n" for line in lines))
    return len(lines)


def jobs_for(dataset: str, cfg: dict, quality: int, out: Path) -> tuple[list[dict], dict]:
    seqs = {"train": cfg["train"], "val": cfg["val"], "test": splits(dataset)["test"]}
    jobs, summary = [], {}
    for split in SPLITS:
        (out / "images" / split).mkdir(parents=True)
        (out / "labels" / split).mkdir(parents=True)
        stride = cfg["stride"][split]
        summary[split] = {"stride": stride, "sequences": {}}
        for name in seqs[split]:
            seq = sequence(dataset, "train" if split == "val" else split, name)
            images = seq.image_paths()
            if len(images) != seq.num_frames:
                raise RuntimeError(f"{dataset}/{name}: {len(images)} frames vs seqLength {seq.num_frames}")
            rows = defaultdict(list)
            for frame, _id, x, y, w, h, *_ in seq.gt_rows():
                rows[int(frame)].append((x, y, w, h))
            frames = list(range(1, len(images) + 1, stride))
            summary[split]["sequences"][name] = {"frames": len(frames), "of": len(images),
                                                 "size": [int(seq.info["imWidth"]), int(seq.info["imHeight"])]}
            for frame in frames:
                stem = f"{name}_{frame:06d}"
                image = images[frame - 1]
                jobs.append({
                    "split": split, "task": cfg["task"], "long_side": cfg["long_side"], "quality": quality,
                    "image": str(image), "expected": len(rows[frame]),
                    "mask": str(instance_mask(image)) if cfg["task"] == "segment" else None,
                    "boxes": np.array(rows[frame], dtype=float).reshape(-1, 4),
                    "out_image": str(out / "images" / split / f"{stem}.jpg"),
                    "out_label": str(out / "labels" / split / f"{stem}.txt"),
                })
    return jobs, summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    config = yaml.safe_load(CONFIG.read_text())
    for dataset in args.datasets:
        cfg = config["datasets"][dataset]
        out = data_root() / dataset / "yolo"
        if out.exists():
            shutil.rmtree(out)
        start = time.time()
        jobs, summary = jobs_for(dataset, cfg, config["quality"], out)
        with Pool(args.workers) as pool:
            counts = pool.map(export_frame, jobs, chunksize=4)

        for split in SPLITS:
            n = [c for job, c in zip(jobs, counts) if job["split"] == split]
            mb = sum(p.stat().st_size for p in (out / "images" / split).iterdir()) / 2**20
            summary[split].update(images=len(n), labels=sum(n), empty=n.count(0), megabytes=round(mb, 1))
        (out / "data.yaml").write_text(yaml.safe_dump(
            {**{s: f"images/{s}" for s in SPLITS}, "names": {0: cfg["name"]}}, sort_keys=False))
        (out / "manifest.json").write_text(json.dumps({
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git": git_state(), "config": {**cfg, "quality": config["quality"]},
            "splits": summary, "seconds": round(time.time() - start, 1),
        }, indent=2) + "\n")
        print(dataset, {s: {k: summary[s][k] for k in ("images", "labels", "megabytes")} for s in SPLITS})


if __name__ == "__main__":
    main()
