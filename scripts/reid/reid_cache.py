"""Cache per-detection ReID embeddings, computed exactly as the live tracker does.

Each frame's stored detections go through boxmot's own embedding path
(`_resolve_input_embeddings`, one batch per frame, on the BGR frame), so a tracker
fed these rows sees the same features it would have computed itself. Valid only
while `dets/<seq>.txt` is unchanged: the file's sha256 is stored and must match.

Writes `reid/<seq>/<model>.npz`, one row per line of `dets/<seq>.txt`, in that order.

Usage:
    python scripts/reid/reid_cache.py --dataset grapemots --seqs PathPlanning_1 PathPlanning_7
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time

import cv2
import numpy as np
import yaml
from boxmot.reid.specs import ReIDConfig
from boxmot.trackers.strongsort.config import StrongSortConfig
from boxmot.trackers.strongsort.tracker import StrongSort

from mapsort.data import DEFAULT_CONFIG, data_root, sequence


def build(seq, reid: dict, force: bool) -> None:
    out = data_root() / seq.dataset / "reid" / seq.name / f"{reid['model']}.npz"
    digest = hashlib.sha256(seq.dets_path.read_bytes()).hexdigest()
    if out.exists() and not force and str(np.load(out)["dets_sha256"]) == digest:
        print(f"{seq.name}  up to date -> {out}")
        return

    rows = np.atleast_2d(np.loadtxt(seq.dets_path, delimiter=","))
    frames = rows[:, 0].astype(np.int32)
    det_idx = np.empty(len(frames), dtype=np.int32)
    seen: dict[int, int] = {}
    for i, frame in enumerate(frames):
        det_idx[i] = seen.get(frame, 0)
        seen[frame] = det_idx[i] + 1

    # Same arithmetic and dtypes as data.stored_detections -> Tracker.__call__.
    x, y, w, h = rows[:, 2], rows[:, 3], rows[:, 4], rows[:, 5]
    geometry = np.stack([x, y, x + w, y + h], axis=1).astype(np.float32)
    scores = rows[:, 6].astype(np.float32)
    class_ids = rows[:, 7].astype(np.float32)

    tracker = StrongSort(config=StrongSortConfig(), reid=ReIDConfig(**reid))
    images = seq.image_paths()
    embedding = None
    start = time.time()
    for n, frame in enumerate(sorted(seen)):
        sel = np.flatnonzero(frames == frame)
        bgr = cv2.imread(str(images[frame - 1]))
        feats = tracker._resolve_input_embeddings(
            geometry=geometry[sel], embeddings=None, frame=bgr,
            scores=scores[sel], class_ids=class_ids[sel],
        )
        if embedding is None:
            embedding = np.full((len(frames), feats.shape[1]), np.nan, dtype=np.float32)
        embedding[sel] = feats
        if (n + 1) % 50 == 0:
            print(f"{seq.name}  {n + 1}/{len(seen)} frames  {time.time() - start:.0f}s", flush=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        embedding=embedding,
        frame=frames,
        det_idx=det_idx,
        dets_sha256=np.str_(digest),
        reid=np.str_(json.dumps(reid, sort_keys=True)),
    )
    print(f"{seq.name}  {len(frames)} dets  {embedding.shape[1]}-d  {time.time() - start:.0f}s -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--seqs", nargs="+", required=True)
    ap.add_argument("--force", action="store_true", help="rebuild even if the dets hash matches")
    args = ap.parse_args()

    reid = yaml.safe_load(DEFAULT_CONFIG.read_text())["pipeline"]["tracker"]["reid"]
    for name in args.seqs:
        build(sequence(args.dataset, args.split, name), reid, args.force)


if __name__ == "__main__":
    main()
