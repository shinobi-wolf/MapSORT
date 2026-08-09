"""Detector submodule. See docs/modules.yaml#Detector."""

from __future__ import annotations

from .schemas import Detections, RGBFrame


class Detector:
    """YOLO-based object detector.

    Produces per-frame bounding-box detections D_t from the raw RGB frame I_t.
    """

    def __init__(self, *args, **kwargs) -> None:
        pass

    def __call__(self, frame: RGBFrame) -> Detections:
        raise NotImplementedError
