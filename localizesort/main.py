"""LocalizeSORTv2 top-level pipeline. See docs/modules.yaml#LocalizeSORTv2.

Wiring (matches docs/LS_v2_diagram):

    I_t -> Detector -> D_t -+-> Tracker -> T
                             |              ^
                             +-> Localization -> P_t -+
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from .detector import Detector
from .localization import Localization
from .schemas import Detection, Track
from .tracker import Tracker

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config" / "localizesort.yaml"


class LocalizeSORT:
    """Top-level online tracking pipeline.

    Consumes a monocular RGB frame I_t and produces the running set of
    tracks T, by detecting objects, estimating each detection's pose in the
    map frame, and associating detections into identity-persistent tracks.
    """

    def __init__(self, config_path: str | Path = DEFAULT_CONFIG) -> None:
        self.config = yaml.safe_load(Path(config_path).read_text())

        self.detector = Detector(**self.config["detector"])
        self.localization = Localization(**self.config["localization"])
        self.tracker = Tracker(**self.config["tracker"])

    def step(
        self,
        frame_idx: int,
        frame: np.ndarray,
        detections: list[Detection] | None = None,
    ) -> Track:
        """Advance the pipeline by one frame.

        Passing `detections` bypasses the Detector, so ground-truth or
        externally-produced boxes can be fed straight to Localization and the
        Tracker.
        """
        if detections is None:
            detections = self.detector(frame)

        object_poses = self.localization(frame_idx, frame, detections)

        return self.tracker(frame_idx, frame, detections, object_poses)

    def run(
        self,
        frames: list[np.ndarray],
        detections: list[list[Detection]] | None = None,
    ) -> list[Track]:
        for frame_idx, frame in enumerate(frames):
            frame_detections = None if detections is None else detections[frame_idx]
            self.step(frame_idx, frame, frame_detections)

        return self.tracker.tracks