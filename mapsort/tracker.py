"""Tracker. See docs/modules.yaml#Tracker."""

from __future__ import annotations

import numpy as np
from boxmot.reid.specs import ReIDConfig
from boxmot.trackers.common.appearance import normalize_embedding
from boxmot.trackers.common.association.matching import chi2inv95
from boxmot.trackers.common.association.strongsort import (
    INFTY_COST,
    gate_cost_matrix,
    matching_cascade,
    min_cost_matching,
)
from boxmot.trackers.strongsort.config import StrongSortConfig
from boxmot.trackers.strongsort.track import Track as StrongSortTrack
from boxmot.trackers.strongsort.track import TrackState
from boxmot.trackers.strongsort.tracker import StrongSort

from . import association
from .association import LandmarkState
from .schemas import Detection, LandmarkObservation, Track


class MapTrack(StrongSortTrack):
    """A StrongSORT track that also carries a fused landmark, and may sleep instead of die."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.landmark = LandmarkState()
        self.is_dormant = False

    def mark_missed(self) -> None:
        # A stationary object's landmark outlives its image-space prediction.
        if (
            self.state == TrackState.Confirmed
            and self.time_since_update > self._max_age
            and self.landmark.observed
        ):
            self.is_dormant = True
            return
        super().mark_missed()

    def revive(self, detection) -> None:
        """Wake a dormant track: identity and landmark persist, the Kalman state restarts."""
        self.is_dormant = False
        self.bbox = detection.to_measurement()
        self.mean, self.covariance = self.kf.initiate(self.bbox)
        self.conf, self.cls, self.det_ind = detection.conf, detection.cls, detection.det_ind
        self.age += 1
        self.hits += 1
        self.time_since_update = 0
        if detection.feat is not None:
            self.features = [normalize_embedding(detection.feat)]
        self._append_current_history()


class MapSort(StrongSort):
    """StrongSORT with a landmark association stage and dormant-track revival.

    `observations` holds the current frame's landmark observations, parallel to the
    frame's detections, and is set by the caller before each update.
    """

    tracks: list[MapTrack]

    def __init__(
        self,
        *,
        gate: float = chi2inv95[3],
        use_appearance: bool = True,
        dormancy: dict | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.gate = float(gate)
        self.use_appearance = bool(use_appearance)
        dormancy = dormancy or {}
        self.dormancy = bool(dormancy.get("enabled", True))
        self.max_dormant = dormancy.get("max_dormant")
        self.dormant: list[MapTrack] = []
        self.observations: list[LandmarkObservation | None] = []

    def _observation(self, detection) -> LandmarkObservation | None:
        index = int(detection.det_ind)
        return self.observations[index] if index < len(self.observations) else None

    def _fuse(self, track: MapTrack, detection) -> None:
        observation = self._observation(detection)
        if observation is not None:
            track.landmark.fuse(observation)

    def _landmark_cost(self, tracks, detections, track_indices, detection_indices) -> np.ndarray:
        return association.cost_matrix(
            [tracks[int(index)].landmark for index in track_indices],
            [self._observation(detections[int(index)]) for index in detection_indices],
            INFTY_COST,
        )

    def _appearance_cost(self, tracks, detections, track_indices, detection_indices) -> np.ndarray:
        """StrongSORT's gated appearance metric, lifted out of its `_match` closure."""
        features = np.asarray([detections[index].feat for index in detection_indices])
        targets = np.asarray([tracks[index].id for index in track_indices])
        cost_matrix = self.metric.distance(features, targets)
        gated_mask = np.empty(cost_matrix.shape, dtype=bool) if self._mask_guidance is not None else None
        cost_matrix = gate_cost_matrix(
            cost_matrix,
            tracks,
            detections,
            track_indices,
            detection_indices,
            self.mc_lambda,
            gated_mask=gated_mask,
        )
        return self._condition_gated_association(
            cost_matrix,
            [tracks[index] for index in track_indices],
            [detections[index] for index in detection_indices],
            threshold=self.metric.matching_threshold,
            gated_mask=gated_mask,
        )

    def _match(self, detections) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        """Associate on appearance, then landmark, then IoU."""
        confirmed = [index for index, track in enumerate(self.tracks) if track.is_confirmed()]
        unconfirmed = [index for index, track in enumerate(self.tracks) if not track.is_confirmed()]

        if self.use_appearance:
            matches_a, unmatched_tracks_a, unmatched_detections = matching_cascade(
                self._appearance_cost,
                self.metric.matching_threshold,
                self.max_age,
                self.tracks,
                detections,
                confirmed,
            )
        else:
            matches_a = []
            unmatched_tracks_a = confirmed
            unmatched_detections = list(range(len(detections)))

        # Stale tracks stay candidates here: recovering them is the point.
        matches_l, unmatched_tracks_a, unmatched_detections = min_cost_matching(
            self._landmark_cost,
            self.gate,
            self.tracks,
            detections,
            unmatched_tracks_a,
            unmatched_detections,
        )

        iou_track_candidates = unconfirmed + [
            index for index in unmatched_tracks_a if self.tracks[index].time_since_update == 1
        ]
        unmatched_tracks_a = [
            index for index in unmatched_tracks_a if self.tracks[index].time_since_update != 1
        ]
        matches_b, unmatched_tracks_b, unmatched_detections = min_cost_matching(
            self._association_cost,
            self.max_iou_dist,
            self.tracks,
            detections,
            iou_track_candidates,
            unmatched_detections,
        )

        matches = matches_a + matches_l + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections

    def _revive_dormant(self, detections, detection_indices: list[int]) -> list[int]:
        """Match leftover detections against sleeping tracks before any new id is minted."""
        if not (self.dormancy and self.dormant and len(detection_indices)):
            return detection_indices

        matches, _, unmatched_detections = min_cost_matching(
            self._landmark_cost,
            self.gate,
            self.dormant,
            detections,
            list(range(len(self.dormant))),
            detection_indices,
        )
        for track_idx, detection_idx in sorted(matches, reverse=True):
            track = self.dormant.pop(track_idx)
            track.revive(detections[detection_idx])
            self._fuse(track, detections[detection_idx])
            self.tracks.append(track)
        return unmatched_detections

    def _sleep_expired(self) -> None:
        asleep = [track for track in self.tracks if track.is_dormant]
        if not asleep:
            return
        self.tracks = [track for track in self.tracks if not track.is_dormant]
        self.dormant.extend(asleep)
        if self.max_dormant is not None:
            del self.dormant[: max(0, len(self.dormant) - int(self.max_dormant))]

    def _update_tracks(self, detections) -> None:
        matches, unmatched_tracks, unmatched_detections = self._match(detections)

        StrongSortTrack.multi_update(
            (self.tracks[track_idx], detections[detection_idx])
            for track_idx, detection_idx in matches
        )
        for track_idx, detection_idx in matches:
            self._fuse(self.tracks[track_idx], detections[detection_idx])

        unmatched_detections = self._revive_dormant(detections, unmatched_detections)

        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx])
            self._fuse(self.tracks[-1], detections[detection_idx])

        self._sleep_expired()
        self.tracks = [track for track in self.tracks if not track.is_deleted()]
        self.active_tracks = self.tracks

        active_targets = [track.id for track in self.tracks if track.is_confirmed()]
        features: list[np.ndarray] = []
        targets: list[int] = []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features.extend(track.features)
            targets.extend([track.id] * len(track.features))
        self.metric.partial_fit(np.asarray(features), np.asarray(targets), active_targets)

    def _initiate_track(self, detection) -> None:
        self.tracks.append(
            MapTrack(
                detection,
                self.id_allocator.alloc(),
                self.n_init,
                self.max_age,
                self.max_obs,
                self.ema_alpha,
                is_obb=detection.is_obb,
                noise_config=self.kalman_noise_config,
            )
        )

    def reset(self) -> None:
        super().reset()
        self.dormant = []


class Tracker:
    """Associates per-frame detections (via appearance and estimated
    map-frame landmark) against the existing track bank, updates matched tracks'
    state estimates, and emits the running set of identity-persistent tracks
    T.

    T is the cumulative trajectory view of the track bank as it evolves
    across frames.
    """

    def __init__(
        self,
        strongsort: dict | None = None,
        reid: dict | None = None,
        landmarks: dict | None = None,
    ) -> None:
        landmarks = dict(landmarks or {})
        enabled = bool(landmarks.pop("enabled", False))
        kwargs = {
            "config": StrongSortConfig(**(strongsort or {})),
            "reid": ReIDConfig(**(reid or {})),
        }
        self.tracker = MapSort(**kwargs, **landmarks) if enabled else StrongSort(**kwargs)
        self.tracks: list[Track] = []

    def __call__(
        self,
        frame_idx: int,
        frame: np.ndarray,
        detections: list[Detection],
        landmarks: list[LandmarkObservation | None],
    ) -> Track:
        dets = np.array(
            [[*d.bbox, d.confidence, d.class_id] for d in detections],
            dtype=np.float32,
        ).reshape(-1, 6)

        if isinstance(self.tracker, MapSort):
            self.tracker.observations = landmarks or []

        # StrongSORT's ReID encoder and ECC expect BGR; pipeline frames are RGB.
        bgr = np.ascontiguousarray(frame[..., ::-1])
        outputs = np.asarray(self.tracker.update(dets, bgr))

        # Nx8 rows: x1, y1, x2, y2, track_id, conf, cls, det_ind
        det_inds = outputs[:, 7].astype(int)
        track = Track(
            frame_id=frame_idx,
            detections=[detections[i] for i in det_inds],
            track_ids=outputs[:, 4].astype(int).tolist(),
        )
        self.tracks.append(track)
        return track
