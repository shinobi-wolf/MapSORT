"""MapSORT top-level pipeline. See docs/modules.yaml#MapSORT.

Wiring (matches docs/LS_v2_diagram):

    I_t -> Detector -> D_t -+-> Tracker -> T
                             |              ^
                             +-> Mapping -> P_t -+
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from .data import DEFAULT_CONFIG
from .detector import Detector
from .mapping import Mapping
from .schemas import Detection, LandmarkObservation, Track
from .tracker import Tracker


class MapSORT:
    """Top-level online tracking pipeline.

    Consumes a monocular RGB frame I_t and produces the running set of
    tracks T, by detecting objects, estimating each detection's landmark
    position in the map frame, and associating detections into identity-persistent tracks.
    """

    def __init__(self, config_path: str | Path = DEFAULT_CONFIG, frame_rate: int | None = None) -> None:
        self.config = yaml.safe_load(Path(config_path).read_text())["pipeline"]

        self.detector = Detector(**self.config["detector"])
        self.mapping = Mapping(**self.config["mapping"])
        # BoT-SORT scales its lost-track buffer by the sequence's frame rate.
        self.tracker = Tracker(**self.config["tracker"], frame_rate=frame_rate)

    def step(
        self,
        frame_idx: int,
        frame: np.ndarray,
        detections: list[Detection] | None = None,
        landmarks: list[LandmarkObservation | None] | None = None,
    ) -> Track:
        """Advance the pipeline by one frame.

        Passing `detections` bypasses the Detector and passing `landmarks`
        bypasses Mapping, so stored or externally-produced values can be fed
        straight to the Tracker.
        """
        if detections is None:
            detections = self.detector(frame)
        if landmarks is None:
            landmarks = self.mapping(frame_idx, frame, detections)

        return self.tracker(frame_idx, frame, detections, landmarks)

    def run(
        self,
        frames: list[np.ndarray],
        detections: list[list[Detection]] | None = None,
        landmarks: list[list[LandmarkObservation | None]] | None = None,
    ) -> list[Track]:
        for frame_idx, frame in enumerate(frames):
            self.step(
                frame_idx,
                frame,
                None if detections is None else detections[frame_idx],
                None if landmarks is None else landmarks[frame_idx],
            )

        return self.tracker.tracks