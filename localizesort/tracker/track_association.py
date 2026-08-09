"""TrackAssociation submodule. See docs/modules.yaml#Tracker.TrackAssociation."""

from __future__ import annotations

from ..schemas import Associations, TrackAssociationInput


class TrackAssociation:
    """Matches current-frame detections against the existing track bank
    (state gamma_t), using a joint cost over appearance features A_t and
    estimated map-frame poses P_t.

    Produces frame-t assignments (matched detection-track pairs, unmatched
    detections, unmatched tracks) for the state update step.
    """

    def __init__(self, *args, **kwargs) -> None:
        pass

    def __call__(self, association_input: TrackAssociationInput) -> Associations:
        raise NotImplementedError
