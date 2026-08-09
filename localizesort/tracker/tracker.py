"""Tracker. See docs/modules.yaml#Tracker."""

from __future__ import annotations

from ..schemas import TrackAssociationInput, TrackBank, Tracks, TrackerInput
from .appearance_encoder import AppearanceEncoder
from .ekf import EKF
from .track_association import TrackAssociation


class Tracker:
    """Associates per-frame detections (via appearance and estimated
    map-frame pose) against the existing track bank, updates matched tracks'
    state estimates, and emits the running set of identity-persistent tracks
    T.

    T is the cumulative trajectory view of the track bank as it evolves
    across frames.
    """

    def __init__(self, *args, **kwargs) -> None:
        self.appearance_encoder = AppearanceEncoder()
        self.track_association = TrackAssociation()
        self.ekf = EKF()
        self.track_bank = TrackBank(frame_index=0, tracks=[])

    def __call__(self, tracker_input: TrackerInput) -> Tracks:
        appearance_features = self.appearance_encoder(tracker_input.detections)

        association_input = TrackAssociationInput(
            appearance_features=appearance_features,
            poses=tracker_input.poses,
            track_bank=self.track_bank,
        )
        associations = self.track_association(association_input)

        self.track_bank = self.ekf(associations)

        raise NotImplementedError
