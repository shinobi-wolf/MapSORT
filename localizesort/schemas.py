

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

# ─────────────────────────────────────────────
# DETECTION SCHEMAS
# ─────────────────────────────────────────────

@dataclass
class Detection:
    """Single detection d_t^i produced by the Detector for frame t."""

    bbox: np.ndarray  # float[4], xyxy image coordinates
    confidence: float
    class_id: int

# ─────────────────────────────────────────────
# POSE SCHEMAS
# ─────────────────────────────────────────────
@dataclass
class ObjectPose:
    """Estimated map-frame pose P_t^i of a single detection d_t^i."""

    position: np.ndarray  # float[3], p_t^i in R^3
    covariance: np.ndarray  # float[3, 3]

# ─────────────────────────────────────────────
# TRACKER SCHEMAS
# ─────────────────────────────────────────────

@dataclass
class Track:
    """A single track in the tracker."""

    frame_id: int
    detections: list[Detection]
    track_ids: list[int]
