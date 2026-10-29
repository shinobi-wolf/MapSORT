"""The multiple orange tracker of Santos et al. (arXiv:2312.16724v1, Sec. 3.3): IoU tracking plus 3-D relocalization."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .cameras import Camera
from .geometry import assign, estimate_orange, reprojected_box


@dataclass
class Params:
    # Algorithm 1; the paper gives no values for these three.
    max_geom_error: float = 10.0  # px
    inliers_ratio: float = 0.5
    max_iters: int = 1000
    order: str = "random"  # visiting order of Combinations(T, 3): random | lexicographic
    # Sec. 3.3.4 says "more than 5" boxes, its summary "at least 5"; only 5 reproduces Table 3 at 60% and 40%.
    min_boxes: int = 5
    c_ray: float = 0.9  # Algorithm 2, line 13
    relocalization: bool = True  # False is the paper's Table 4 ablation
    keep_failed: bool = False  # True keeps the last sphere when a re-estimate fails
    count_unmodelled: bool = False  # True also reports tracks that never got a sphere
    seed: int = 0


@dataclass
class OrangeTrack:
    frames: list[int] = field(default_factory=list)
    indices: list[int] = field(default_factory=list)  # row of the box in its frame's input
    boxes: list[np.ndarray] = field(default_factory=list)
    X: np.ndarray | None = None
    r3d: float | None = None

    def add(self, frame: int, index: int, box: np.ndarray) -> None:
        self.frames.append(frame)
        self.indices.append(index)
        self.boxes.append(box)


class Reloc3D:
    """Runs a whole sequence: `run(boxes)` with xyxy boxes [n_i, 4] per frame, one Camera (or None) per frame."""

    def __init__(self, cameras: list[Camera | None], params: Params | None = None) -> None:
        self.cameras, self.p = cameras, params or Params()
        self.rng = np.random.default_rng(self.p.seed)

    def _estimate(self, track: OrangeTrack) -> None:
        seen = [(cam, box) for f, box in zip(track.frames, track.boxes) if (cam := self.cameras[f]) is not None]
        if len(seen) < 3:
            X, r3d = None, None
        else:
            X, r3d = estimate_orange(
                np.array([b for _, b in seen]),
                np.array([c.P for c, _ in seen]),
                np.array([c.C for c, _ in seen]),
                np.array([c.focal for c, _ in seen]),
                c_ray=self.p.c_ray,
                max_geom_error=self.p.max_geom_error,
                inliers_ratio=self.p.inliers_ratio,
                max_iters=self.p.max_iters,
                order=self.p.order,
                rng=self.rng,
            )
        if X is not None or not self.p.keep_failed:
            track.X, track.r3d = X, r3d

    def _relocalize(self, i: int, boxes: np.ndarray, owner: dict[int, OrangeTrack], tracks: list[OrangeTrack]) -> None:
        cam = self.cameras[i]
        lost = [t for t in tracks if t.frames[-1] != i and t.X is not None]
        free = [k for k in range(len(boxes)) if k not in owner]
        if cam is None or not lost or not free:
            return
        projected = np.array([reprojected_box(t.X, t.r3d, cam.P, cam.C, cam.focal) for t in lost])
        for a, b in assign(projected, boxes[free]):
            lost[a].add(i, free[b], boxes[free[b]])
            owner[free[b]] = lost[a]

    def run(self, boxes: list[np.ndarray]) -> list[OrangeTrack]:
        """Tracks the counting rule keeps: those with a sphere, unless `count_unmodelled`."""
        tracks: list[OrangeTrack] = []
        owner: dict[int, OrangeTrack] = {}  # box index in the current frame -> its track
        for i in range(len(boxes) - 1):
            if self.p.relocalization:
                self._relocalize(i, boxes[i], owner, tracks)
            following: dict[int, OrangeTrack] = {}
            for p, q in assign(boxes[i], boxes[i + 1]):
                track = owner.get(p)
                if track is None:
                    track = OrangeTrack()
                    track.add(i, p, boxes[i][p])
                    tracks.append(track)
                track.add(i + 1, q, boxes[i + 1][q])
                following[q] = track
            for track in following.values():
                if len(track.boxes) >= self.p.min_boxes:
                    self._estimate(track)
            owner = following
        return [t for t in tracks if self.p.count_unmodelled or t.X is not None]
