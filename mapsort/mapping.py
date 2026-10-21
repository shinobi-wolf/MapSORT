"""Mapping submodule. See docs/modules.yaml#Mapping and docs/problem.md
("Sub-task: Online Mapping")."""

from __future__ import annotations

import numpy as np

from .schemas import Detection, LandmarkObservation

class Mapping:
    """Online, feed-forward mapping module.

    Estimates each detection's 3D landmark position in the map frame, P_t,
    directly from the current RGB frame and its detections - causally, using
    frames 1..t only, with no offline per-scene optimization.
    """

    def __init__(self, *args, **kwargs) -> None:
        pass

    def __call__(self, frame_idx: int, frame: np.ndarray, detections: list[Detection]) -> list[LandmarkObservation]:
        # Placeholder until the online backend lands; stored landmarks are passed to the Tracker directly.
        return []
