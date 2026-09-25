"""Localization submodule. See docs/modules.yaml#Localization and docs/problem.md
("Sub-task: Online Localization")."""

from __future__ import annotations

import numpy as np

from .schemas import Detection, ObjectPose

class Localization:
    """Online, feed-forward localization module.

    Estimates each detection's 3D pose in the map frame, P_t, directly from
    the current RGB frame and its detections - causally, using frames 1..t
    only, with no offline per-scene optimization.
    """

    def __init__(self, *args, **kwargs) -> None:
        pass

    def __call__(self, frame_idx: int, frame: np.ndarray, detections: list[Detection]) -> list[ObjectPose]:
        # Placeholder: StrongSORT doesn't consume poses, so emit none for now.
        return []
