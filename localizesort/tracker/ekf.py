"""EKF submodule. See docs/modules.yaml#Tracker.EKF."""

from __future__ import annotations

from ..schemas import Associations, TrackBank


class EKF:
    """Extended Kalman Filter.

    Updates each matched track's state estimate from its associated
    detection's map-frame pose, initializes new tracks from unmatched
    detections, and ages/prunes unmatched tracks. Writes the updated track
    bank state gamma_{t+1}, which is read back by TrackAssociation at frame
    t+1.
    """

    def __init__(self, *args, **kwargs) -> None:
        pass

    def __call__(self, associations: Associations) -> TrackBank:
        raise NotImplementedError
