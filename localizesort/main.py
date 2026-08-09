"""LocalizeSORTv2 top-level pipeline. See docs/modules.yaml#LocalizeSORTv2.

Wiring (matches docs/LS_v2_diagram):

    I_t -> Detector -> D_t -+-> Tracker -> T
                             |              ^
                             +-> Localization -> P_t -+
"""

from __future__ import annotations

from .detector import Detector
from .localization import Localization
from .schemas import LocalizationInput, RGBFrame, Tracks, TrackerInput
from .tracker import Tracker


class LocalizeSORT:
    """Top-level online tracking pipeline.

    Consumes a monocular RGB frame I_t and produces the running set of
    tracks T, by detecting objects, estimating each detection's pose in the
    map frame, and associating detections into identity-persistent tracks.
    """

    def __init__(self, *args, **kwargs) -> None:
        self.detector = Detector()
        self.localization = Localization()
        self.tracker = Tracker()

    def __call__(self, frame: RGBFrame) -> Tracks:
        detections = self.detector(frame)

        poses = self.localization(
            LocalizationInput(frame=frame, detections=detections)
        )

        return self.tracker(TrackerInput(detections=detections, poses=poses))
