"""Tracker. See docs/modules.yaml#Tracker."""

from __future__ import annotations

import numpy as np
from boxmot.reid.specs import ReIDConfig
from boxmot.trackers.strongsort.config import StrongSortConfig
from boxmot.trackers.strongsort.tracker import StrongSort

from .schemas import Detection, ObjectPose, Track


class Tracker:
    """Associates per-frame detections (via appearance and estimated
    map-frame pose) against the existing track bank, updates matched tracks'
    state estimates, and emits the running set of identity-persistent tracks
    T.

    T is the cumulative trajectory view of the track bank as it evolves
    across frames.
    """

    def __init__(
        self,
        strongsort: dict | None = None,
        reid: dict | None = None,
    ) -> None:
        self.tracker = StrongSort(
            config=StrongSortConfig(**(strongsort or {})),
            reid=ReIDConfig(**(reid or {})),
        )
        self.tracks: list[Track] = []

    def __call__(
        self,
        frame_idx: int,
        frame: np.ndarray,
        detections: list[Detection],
        poses: list[ObjectPose],
    ) -> Track:
        # poses are unused: StrongSORT associates on appearance and image-space
        # motion only. They are carried through for the pose-aware tracker.
        dets = np.array(
            [[*d.bbox, d.confidence, d.class_id] for d in detections],
            dtype=np.float32,
        ).reshape(-1, 6)

        # StrongSORT's ReID encoder and ECC expect BGR; pipeline frames are RGB.
        bgr = np.ascontiguousarray(frame[..., ::-1])
        outputs = np.asarray(self.tracker.update(dets, bgr))

        # Nx8 rows: x1, y1, x2, y2, track_id, conf, cls, det_ind
        det_inds = outputs[:, 7].astype(int)
        track = Track(
            frame_id=frame_idx,
            detections=[detections[i] for i in det_inds],
            track_ids=outputs[:, 4].astype(int).tolist(),
        )
        self.tracks.append(track)
        return track
