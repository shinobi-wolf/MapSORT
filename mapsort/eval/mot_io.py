"""Reading and writing MOTChallenge tracker files.

Evaluation never imports the pipeline: it reads what a run wrote to disk. This
module is the boundary. A tracker file has one row per tracked detection,

    frame, track_id, x, y, w, h, conf, class, visibility

with 1-based frames and top-left `xywh` -- the same format as `gt.txt`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..schemas import Track


def write_tracks(tracks: list[Track], path: Path) -> Path:
    """Write a sequence's `Track` list (one per frame) as a MOT tracker file.

    `Track.frame_id` is 0-based, as produced by the pipeline loop, and boxes are
    `xyxy`; both are converted to MOT's conventions here.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for track in tracks:
        for det, track_id in zip(track.detections, track.track_ids):
            x1, y1, x2, y2 = det.bbox
            lines.append(
                f"{track.frame_id + 1},{track_id},{x1:.2f},{y1:.2f},"
                f"{x2 - x1:.2f},{y2 - y1:.2f},{det.confidence:.4f},"
                f"{det.class_id},-1"
            )
    path.write_text("\n".join(lines) + "\n" if lines else "")
    return path


def read_rows(path: Path) -> np.ndarray:
    """A MOT file as an (N, >=6) float array, empty-file safe."""
    if path.stat().st_size == 0:
        return np.empty((0, 9))
    return np.atleast_2d(np.loadtxt(path, delimiter=","))


def filter_frames(src: Path, dest: Path, keep: set[int]) -> Path:
    """Copy a MOT file keeping only rows whose (1-based) frame is in `keep`.

    Used to score a metric over a restricted set of frames without
    reimplementing it: the subset is written out and handed to the same engine.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        line
        for line in src.read_text().splitlines()
        if line.strip() and int(float(line.split(",")[0])) in keep
    ]
    dest.write_text("\n".join(lines) + "\n" if lines else "")
    return dest
