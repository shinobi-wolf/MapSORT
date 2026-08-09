"""AppearanceEncoder submodule. See docs/modules.yaml#Tracker.AppearanceEncoder."""

from __future__ import annotations

from ..schemas import AppearanceFeatures, Detections


class AppearanceEncoder:
    """Encodes each detection's cropped image bbox D_t^i into an appearance
    embedding A_t^i."""

    def __init__(self, *args, **kwargs) -> None:
        pass

    def __call__(self, detections: Detections) -> AppearanceFeatures:
        raise NotImplementedError
