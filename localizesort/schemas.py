"""Data contracts that flow between LocalizeSORTv2 modules.

Mirrors docs/schema.yaml field-for-field. Referenced by docs/modules.yaml.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

# ─────────────────────────────────────────────
# INPUT SCHEMAS  (fed into the system)
# ─────────────────────────────────────────────


@dataclass
class RGBFrame:
    """Single monocular RGB video frame I_t, the sole raw input at time t."""

    image: np.ndarray  # uint8[H, W, 3]
    frame_index: int


# ─────────────────────────────────────────────
# DETECTION SCHEMAS
# ─────────────────────────────────────────────


@dataclass
class Detection:
    """Single detection d_t^i produced by the Detector for frame t."""

    detection_id: int
    bbox: np.ndarray  # float[4], xyxy image coordinates
    frame: RGBFrame
    confidence: float
    class_id: int


@dataclass
class Detections:
    """Full per-frame detection set D_t = [D_t^1, ..., D_t^{n_t}]."""

    frame_index: int
    detections: list[Detection]


# ─────────────────────────────────────────────
# LOCALIZATION SCHEMAS
# ─────────────────────────────────────────────


@dataclass
class ObjectPose:
    """Estimated map-frame pose P_t^i of a single detection d_t^i."""

    detection_id: int
    position: np.ndarray  # float[3], p_t^i in R^3
    covariance: np.ndarray  # float[3, 3]


@dataclass
class ObjectPoses:
    """Full per-frame map-frame pose set P_t = [P_t^1, ..., P_t^{n_t}]."""

    frame_index: int
    poses: list[ObjectPose]


# ─────────────────────────────────────────────
# COMPOSITE INPUT SCHEMAS  (module inputs with more than one source)
# ─────────────────────────────────────────────


@dataclass
class LocalizationInput:
    """Joint input to the Localization module."""

    frame: RGBFrame
    detections: Detections


@dataclass
class TrackerInput:
    """Joint input to the Tracker."""

    detections: Detections
    poses: ObjectPoses


@dataclass
class TrackAssociationInput:
    """Joint input to the TrackAssociation submodule."""

    appearance_features: AppearanceFeatures
    poses: ObjectPoses
    track_bank: TrackBank


# ─────────────────────────────────────────────
# TRACKER SCHEMAS
# ─────────────────────────────────────────────


@dataclass
class AppearanceFeature:
    """Appearance embedding A_t^i for a single detection."""

    detection_id: int
    embedding: np.ndarray  # float[d]


@dataclass
class AppearanceFeatures:
    """Full per-frame appearance embedding set A_t = [A_t^1, ..., A_t^{n_t}]."""

    frame_index: int
    features: list[AppearanceFeature]


@dataclass
class Associations:
    """Result of TrackAssociation for frame t."""

    frame_index: int
    matched: list[tuple[int, int]]  # (detection_id, track_id)
    unmatched_detections: list[int]
    unmatched_tracks: list[int]


# ─────────────────────────────────────────────
# STATE SCHEMAS  (persistent across frames)
# ─────────────────────────────────────────────


class TrackStatus(Enum):
    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    DELETED = "deleted"


@dataclass
class KalmanState:
    mean: np.ndarray  # float[3]
    covariance: np.ndarray  # float[3, 3]


@dataclass
class Track:
    """Single persistent track (identity) k maintained in the track bank."""

    track_id: int
    kalman_state: KalmanState
    appearance_state: np.ndarray  # float[d]
    hits: int
    time_since_update: int
    status: TrackStatus


@dataclass
class TrackBank:
    """Full track bank state gamma_t."""

    frame_index: int
    tracks: list[Track]


# ─────────────────────────────────────────────
# OUTPUT SCHEMAS  (produced by the system)
# ─────────────────────────────────────────────


@dataclass
class Trajectory:
    track_id: int
    assignments: list[tuple[int, int]]  # (frame_index, detection_id)


@dataclass
class Tracks:
    """Final pipeline output T = {tau_1, ..., tau_K}."""

    trajectories: list[Trajectory]
