"""Tracker. See docs/modules.yaml#Tracker."""

from __future__ import annotations

from collections import deque

import numpy as np
from boxmot.reid.protocols import AppearanceEncoder
from boxmot.reid.specs import ReIDConfig
from boxmot.trackers.botsort.config import BotSortConfig
from boxmot.trackers.botsort.track import STrack, TrackState
from boxmot.trackers.botsort.tracker import BotSort
from boxmot.trackers.common.appearance import resolve_batch_embeddings
from boxmot.trackers.common.association import AssociationStage, run_association_stage
from boxmot.trackers.common.association.matching import linear_assignment
from boxmot.trackers.common.tracking.lifecycle import joint_stracks

from . import association
from .association import LandmarkState
from .camera import project, resect
from .schemas import Detection, LandmarkObservation, Track

_NO = 1e6  # assignment cost of a forbidden pair


class MapTrack(STrack):
    """A BoT-SORT track that also carries a fused landmark and where that landmark sits in its box."""

    def __init__(self, *args, observation: LandmarkObservation | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.observation = observation  # the landmark of the detection this track was built from
        self.landmark = LandmarkState()
        self.last = LandmarkState()  # the latest observation alone
        self.offset = np.zeros(2)  # box centre minus projected landmark, in box sizes (EMA)
        self.offsets = 0
        self.residual = 1.0  # EMA of the offset-corrected reprojection error, in box sizes
        self.wh = np.asarray(self.xywh[2:4], dtype=float).copy()  # last width, height not cut by the frame edge

    def fuse(self, observation: LandmarkObservation) -> None:
        self.landmark.fuse(observation)
        self.last = LandmarkState()
        self.last.fuse(observation)


def _cut(xywh: np.ndarray, w: float, h: float) -> np.ndarray:
    """Whether each xywh box [N, 4] touches the frame edge."""
    lo, hi = xywh[:, :2] - xywh[:, 2:] / 2, xywh[:, :2] + xywh[:, 2:] / 2
    return (lo <= 1).any(1) | (hi[:, 0] >= w - 1) | (hi[:, 1] >= h - 1)


class MapSort(BotSort):
    """BoT-SORT whose stationary tracks are located through the frame's camera.

    Each frame the camera is resected from the tracks BoT-SORT matched (plus 3D nearest-neighbour
    landmark pairs). Reliable tracks the first stage left unmatched are tried again where it projects
    their landmarks, and lost and dormant tracks are revived where theirs project. Without a camera,
    lost and dormant tracks fall back to the Mahalanobis gate. `observations` holds the frame's landmark
    observations, parallel to its detections, and is set by the caller before each update.
    """

    def __init__(
        self,
        *,
        gate: float = 200.0,
        short_gate: float = 10.0,
        distance: str = "mahalanobis",
        reprojection: float | None = 1.0,
        camera: dict | None = None,
        map_motion: dict | None = None,
        revival: dict | None = None,
        dormancy: dict | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if distance not in association.DISTANCES:
            raise ValueError(f"distance must be one of {association.DISTANCES}, got {distance!r}")
        self.gate = float(gate)  # squared Mahalanobis (or map units): dormant fallback and camera pairs
        self.short_gate = float(short_gate)  # the same, against a lost track's latest observation
        self.distance = distance
        self.reprojection = reprojection  # revival gate on reprojection error, in box sizes; None: 3D only
        camera = camera or {}
        self.camera_iters = int(camera.get("iters", 300))
        self.camera_inlier = float(camera.get("inlier", 0.5))
        self.camera_min = int(camera.get("min_inliers", 8))
        self.planar_margin = float(camera.get("planar_margin", 0.1))
        self.rng = np.random.default_rng(camera.get("seed", 0))
        map_motion = map_motion or {}
        self.map_motion = bool(map_motion.get("enabled", True))
        self.min_offsets = int(map_motion.get("min_offsets", 2))
        self.max_residual = float(map_motion.get("max_residual", 0.3))
        self.offset_alpha = float(map_motion.get("alpha", 0.3))
        revival = revival or {}
        edge = revival.get("edge") or {}
        self.edge_iou = float(edge.get("iou", 0.3))
        self.edge_visible = float(edge.get("visible", 0.5))
        self.edge_wait = int(edge.get("wait", 8))  # frames a new track at the edge may wait for a revival
        dormancy = dormancy or {}
        self.dormancy = bool(dormancy.get("enabled", True))
        self.max_dormant = dormancy.get("max_dormant")
        self.dormant: list[MapTrack] = []
        self._revived: set[int] = set()
        self._waiting: set[int] = set()
        self._camera_prev: np.ndarray | None = None
        self._camera: np.ndarray | None = None
        self.observations: list[LandmarkObservation | None] = []

    # ── landmarks and camera ──────────────────────────────────────────────────

    def _observation(self, det_ind) -> LandmarkObservation | None:
        index = int(det_ind)
        return self.observations[index] if 0 <= index < len(self.observations) else None

    def _landmark_cost(self, tracks, detections, which: str = "landmark") -> np.ndarray:
        return association.cost_matrix(
            [getattr(t, which) for t in tracks], [d.observation for d in detections], np.inf, self.distance
        )

    def _landmark_pairs(self, tracks, detections) -> list:
        """Mutually nearest track landmark and detection observation within the gate."""
        tracks = [t for t in tracks if t.landmark.observed]
        detections = [d for d in detections if d.observation is not None]
        if not tracks or not detections:
            return []
        d2 = self._landmark_cost(tracks, detections)
        rows, cols = d2.argmin(1), d2.argmin(0)
        return [(tracks[i], detections[j]) for i, j in enumerate(rows) if cols[j] == i and d2[i, j] <= self.gate]

    def _resect(self, pairs) -> np.ndarray | None:
        """The frame's camera from (track, detection) pairs: fused landmark <-> box centre."""
        pairs = [(t, d) for t, d in pairs if t.landmark.observed]
        if len(pairs) < max(6, self.camera_min):
            return None
        X = np.array([t.landmark.gaussian[0] for t, _ in pairs])
        x = np.array([d.xywh[:2] for _, d in pairs])
        size = np.array([np.sqrt(d.xywh[2] * d.xywh[3]) for _, d in pairs])
        P = resect(X, x, size, self.rng, iters=self.camera_iters, inlier=self.camera_inlier,
                   min_inliers=self.camera_min, prior=self._camera_prev, planar_margin=self.planar_margin)
        if P is not None:
            self._camera_prev = P
        return P

    def _matched_pairs(self, detections) -> list:
        by_index = {int(d.det_ind): d for d in detections}
        return [
            (t, by_index[int(t.det_ind)])
            for t in self.active_tracks + self.lost_stracks
            if t.frame_id == self.frame_count and t.state == TrackState.Tracked and int(t.det_ind) in by_index
        ]

    def _reliable(self, track: MapTrack) -> bool:
        return track.offsets >= self.min_offsets and track.residual <= self.max_residual

    def _learn_offsets(self, P: np.ndarray, detections) -> None:
        """Where each matched track's landmark projects in its box, and how steadily."""
        for t, d in self._matched_pairs(detections):
            if not t.landmark.observed:
                continue
            r = (d.xywh[:2] - project(P, t.landmark.gaussian[0][None])[0]) / np.sqrt(d.xywh[2] * d.xywh[3])
            if not np.isfinite(r).all():
                continue
            if t.offsets:
                t.residual += self.offset_alpha * (float(np.linalg.norm(r - t.offset)) - t.residual)
                t.offset = t.offset + self.offset_alpha * (r - t.offset)
            else:
                t.offset = r
            t.offsets += 1

    # ── association ──────────────────────────────────────────────────────────

    def _create_detections(self, dets_first, features_high):
        if len(dets_first) == 0:
            return []
        features = features_high if self.use_embeddings else [None] * len(dets_first)
        return [
            MapTrack(det, feat, id_allocator=self.id_allocator, max_obs=self.max_obs, is_obb=self.is_obb,
                     observation=self._observation(det[6]))
            for det, feat in zip(dets_first, features)
        ]

    def _first_association(self, dets, dets_first, active_tracks, unconfirmed, img, detections, activated, refind, strack_pool):
        """BoT-SORT's first stage, then its leftovers re-solved with reliable tracks where the camera places them."""
        STrack.multi_predict(strack_pool, dt=self._prediction_dt)
        self._apply_camera_motion_compensation(dets, img, strack_pool, unconfirmed)
        stage = AssociationStage(name="botsort_high", cost=self._first_association_cost, threshold=self.match_thresh)
        matches = np.asarray(run_association_stage(stage, strack_pool, detections).matches, dtype=int).reshape(-1, 2)
        pairs = [(strack_pool[i], detections[j]) for i, j in matches] + self._landmark_pairs(strack_pool, detections)
        self._camera = self._resect(pairs) if (self.map_motion or self.reprojection is not None) else None

        if self.map_motion and self._camera is not None:
            rows, cols = set(matches[:, 0].tolist()), set(matches[:, 1].tolist())
            left = [j for j in range(len(detections)) if j not in cols]
            placed, kalman = [], []
            for i, t in enumerate(strack_pool):
                if i in rows or not self._reliable(t):
                    continue
                size = np.sqrt(max(t.mean[2] * t.mean[3], 1e-6))
                centre = project(self._camera, t.landmark.gaussian[0][None])[0] + t.offset * size
                if np.isfinite(centre).all():
                    kalman.append(t.mean.copy())
                    t.mean[:2] = centre
                    placed.append(i)
            extra = []
            if placed and left:
                cost = self._first_association_cost([strack_pool[i] for i in placed], [detections[j] for j in left])
                found, _, _ = linear_assignment(cost, self.match_thresh)
                extra = [(placed[a], left[b]) for a, b in np.asarray(found, dtype=int).reshape(-1, 2)]
            # A track still unmatched keeps BoT-SORT's own prediction.
            moved = {i for i, _ in extra}
            for i, mean in zip(placed, kalman):
                if i not in moved:
                    strack_pool[i].mean = mean
            matches = np.array(matches.tolist() + extra, dtype=int).reshape(-1, 2)

        tracked, lost = [], []
        for i, j in matches:
            track = strack_pool[i]
            if track.state == TrackState.Tracked:
                tracked.append((track, detections[j]))
                activated.append(track)
            else:
                lost.append((track, detections[j]))
                refind.append(track)
        STrack.multi_update(tracked, self.frame_count)
        STrack.multi_update(lost, self.frame_count, reactivate=True)
        rows, cols = set(matches[:, 0].tolist()), set(matches[:, 1].tolist())
        return (
            matches,
            np.array([i for i in range(len(strack_pool)) if i not in rows], dtype=int),
            np.array([j for j in range(len(detections)) if j not in cols], dtype=int),
        )

    def _reacquire(self, track: MapTrack, detection: MapTrack) -> None:
        """Identity, landmark and offset persist; the Kalman state restarts at the detection."""
        track.kalman_filter = self.kalman_filter
        track.mean, track.covariance = self.kalman_filter.initiate(detection.xywh)
        track._finish_reactivate(detection, self.frame_count, False)

    def _reproject(self, P, shape, lost_tracks, detections, pool):
        """Revive lost and dormant tracks where their landmarks project.

        Each candidate's box, at its last full size and clipped to the frame as the detector sees it, is
        compared with the unmatched detections, so an object re-entering at the edge can be revived.
        """
        sleeping = [t for t in self.dormant if t.state == TrackState.Removed] if self.dormancy else []
        if self.map_motion:
            # A track whose landmark has rarely been seen in its box is not trusted to revive.
            lost_tracks = [t for t in lost_tracks if t.offsets >= self.min_offsets]
            sleeping = [t for t in sleeping if t.offsets >= self.min_offsets]
        candidates = lost_tracks + sleeping
        if not candidates or not pool:
            return [], pool
        h, w = shape
        wh = np.array([t.wh for t in candidates], dtype=float)
        size = np.sqrt(wh.prod(1))
        offset = np.array([t.offset if (self.map_motion and t.offsets) else np.zeros(2) for t in candidates])
        centre = project(P, np.array([t.landmark.gaussian[0] for t in candidates])) + offset * size[:, None]
        lo, hi = np.clip(centre - wh / 2, 0, [w, h]), np.clip(centre + wh / 2, 0, [w, h])
        with np.errstate(invalid="ignore"):
            visible = np.isfinite(centre).all(1) & ((hi - lo) > 0).all(1)
        dets = [detections[j] for j in pool]
        box = np.array([d.xywh for d in dets], dtype=float)
        err = np.linalg.norm((lo + hi)[:, None] / 2 - box[None, :, :2], axis=2) / size[:, None]
        err = np.where(visible[:, None], np.nan_to_num(err, nan=_NO), _NO)
        dlo, dhi = box[:, :2] - box[:, 2:] / 2, box[:, :2] + box[:, 2:] / 2
        inter = np.clip(np.minimum(hi[:, None], dhi[None]) - np.maximum(lo[:, None], dlo[None]), 0, None).prod(2)
        # A detection cut by the frame edge only shows a sliver: it must overlap the clipped prediction, which in
        # turn must be mostly inside the frame.
        cut = _cut(box, w, h)
        if cut.any():
            iou = inter / np.maximum((hi - lo).prod(1)[:, None] + box[:, 2:].prod(1)[None] - inter, 1e-9)
            inside = (hi - lo).prod(1) / np.maximum(wh.prod(1), 1e-9)
            err = np.where(cut[None] & ((iou < self.edge_iou) | (inside[:, None] < self.edge_visible)), _NO, err)
        # A new track on a detection that one of these tracks could be waits for its revival.
        cover = inter / np.maximum(box[:, 2:].prod(1)[None], 1e-9)
        self._waiting = {int(d.det_ind) for d, ok in zip(dets, ((cover >= 0.5) & visible[:, None]).any(0)) if ok}
        matches, _, left = linear_assignment(err, self.reprojection)
        revived = [(candidates[i], dets[j], i >= len(lost_tracks)) for i, j in np.asarray(matches, dtype=int).reshape(-1, 2)]
        return revived, [pool[j] for j in left]

    def _landmark_revive(self, lost_tracks, detections, pool):
        """Without a camera: lost tracks on their latest observation, dormant ones on the fused landmark."""
        revived = []
        for tracks, which, gate, dormant in (
            (lost_tracks, "last", self.short_gate, False),
            ([t for t in self.dormant if t.state == TrackState.Removed] if self.dormancy else [], "landmark", self.gate, True),
        ):
            if not tracks or not pool:
                continue
            dets = [detections[j] for j in pool]
            cost = self._landmark_cost(tracks, dets, which)
            cost = np.where(np.isfinite(cost) & (cost <= gate), cost, _NO)
            matches, _, left = linear_assignment(cost, gate)
            for i, j in np.asarray(matches, dtype=int).reshape(-1, 2):
                revived.append((tracks[i], dets[j], dormant))
            pool = [pool[j] for j in left]
        return revived, pool

    def _handle_unconfirmed_tracks(self, u_detection, detections, activated, removed, unconfirmed):
        out = super()._handle_unconfirmed_tracks(u_detection, detections, activated, removed, unconfirmed)
        if self.edge_wait and self._waiting:
            # A track born at the frame edge where a revival is expected stays unconfirmed, and unreported,
            # until the object shows enough of itself to be revived or the wait runs out.
            h, w = self._shape
            by_index = {int(d.det_ind): d for d in detections}
            for t in unconfirmed:
                d = by_index.get(int(t.det_ind))
                if (t.is_activated and t.frame_id == self.frame_count and d is not None
                        and self.frame_count - t.start_frame < self.edge_wait and int(d.det_ind) in self._waiting
                        and _cut(d.xywh[None], w, h)[0]):
                    t.is_activated = False
        return out

    def _track_detections(self, dets, img, embs=None, masks=None):
        self.frame_count += 1
        self._shape, self._waiting = img.shape[:2], set()
        activated, refind, lost, removed = [], [], [], []

        dets, dets_first, first_batch, dets_second = self._split_detections(dets, embs)
        features = resolve_batch_embeddings(first_batch, enabled=self.use_embeddings, placeholder_value=1.0)
        detections = self._create_detections(dets_first, features)
        unconfirmed, active_tracks = self._separate_tracks()
        strack_pool = joint_stracks(active_tracks, self.lost_stracks)

        _, unmatched_tracks, pool = self._first_association(
            dets, dets_first, active_tracks, unconfirmed, img, detections, activated, refind, strack_pool
        )
        pool = list(pool)
        self._second_association(dets_second, activated, lost, refind, unmatched_tracks, strack_pool)

        lost_tracks = [
            strack_pool[i]
            for i in unmatched_tracks
            if strack_pool[i].is_activated and strack_pool[i].landmark.observed
            and strack_pool[i].state == TrackState.Lost and strack_pool[i].frame_id != self.frame_count
        ]
        # Stationary objects revive before any unconfirmed track is confirmed or a new id is minted.
        P = self._resect(self._matched_pairs(detections)) if self.reprojection is not None else None
        if P is None and self.reprojection is not None:
            P = self._camera
        if P is not None:
            revived, pool = self._reproject(P, img.shape[:2], lost_tracks, detections, pool)
        else:
            revived, pool = self._landmark_revive(lost_tracks, detections, pool)
        for track, detection, dormant in revived:
            self._reacquire(track, detection)
            if track in lost:
                lost.remove(track)
            refind.append(track)
            if dormant:
                self._revived.add(track.id)

        _, _, new = self._handle_unconfirmed_tracks(pool, detections, activated, removed, unconfirmed)
        self._initialize_new_tracks(new, activated, [detections[j] for j in pool])
        self._update_track_states(removed)
        out = self._prepare_output(activated, refind, lost, removed)

        P = P if P is not None else self._camera
        if self.map_motion and P is not None:
            self._learn_offsets(P, detections)
        h, w = img.shape[:2]
        for t, d in self._matched_pairs(detections):
            # A box cut by the frame edge understates the object, so it only ever grows the remembered size.
            wh = np.asarray(d.xywh[2:4], dtype=float)
            t.wh = np.maximum(t.wh, wh) if _cut(d.xywh[None], w, h)[0] else wh.copy()
        for track in self.active_tracks:
            if track.frame_id == self.frame_count and track.state == TrackState.Tracked:
                observation = self._observation(track.det_ind)
                if observation is not None:
                    track.fuse(observation)
        return out

    # ── lifecycle ────────────────────────────────────────────────────────────

    def _update_track_states(self, removed_stracks):
        for track in self.lost_stracks:
            if self.frame_count - track.end_frame > self.max_time_lost:
                # BotSort keeps a removed track in lost_stracks one more frame; put it to sleep once.
                if self.dormancy and track.state != TrackState.Removed and track.is_activated and track.landmark.observed:
                    self.dormant.append(track)
                    if self.max_dormant is not None:
                        del self.dormant[: max(0, len(self.dormant) - int(self.max_dormant))]
                track.mark_removed()
                removed_stracks.append(track)

    def _prepare_output(self, activated, refind, lost, removed):
        out = super()._prepare_output(activated, refind, lost, removed)
        # A revived id must leave removed_stracks, or losing it again would drop it at once.
        if self._revived:
            self.removed_stracks = deque(
                (t for t in self.removed_stracks if t.id not in self._revived), maxlen=self.removed_stracks.maxlen
            )
            self._revived = set()
        self.dormant = [t for t in self.dormant if t.state == TrackState.Removed]
        return out

    def reset(self) -> None:
        super().reset()
        self.dormant = []
        self._camera_prev = None


class Tracker:
    """Associates per-frame detections (via appearance, image motion and the map-frame landmark)
    against the existing track bank and emits the running set of identity-persistent tracks T.

    T is the cumulative trajectory view of the track bank as it evolves across frames.
    """

    def __init__(
        self,
        botsort: dict | None = None,
        reid: dict | AppearanceEncoder | None = None,
        landmarks: dict | None = None,
        frame_rate: int | None = None,
    ) -> None:
        landmarks = dict(landmarks or {})
        enabled = bool(landmarks.pop("enabled", False))
        config = dict(botsort or {})
        if frame_rate is not None:
            config["frame_rate"] = int(frame_rate)
        kwargs = {
            "config": BotSortConfig(**config),
            # An encoder object, e.g. one serving cached embeddings, is used as is.
            "reid": ReIDConfig(**(reid or {})) if reid is None or isinstance(reid, dict) else reid,
        }
        self.tracker = MapSort(**kwargs, **landmarks) if enabled else BotSort(**kwargs)
        self.tracks: list[Track] = []

    def __call__(
        self,
        frame_idx: int,
        frame: np.ndarray,
        detections: list[Detection],
        landmarks: list[LandmarkObservation | None],
    ) -> Track:
        dets = np.array([[*d.bbox, d.confidence, d.class_id] for d in detections], dtype=np.float32).reshape(-1, 6)
        if isinstance(self.tracker, MapSort):
            self.tracker.observations = landmarks or []

        # BoT-SORT's ReID encoder and camera-motion compensation expect BGR; pipeline frames are RGB.
        outputs = np.asarray(self.tracker.update(dets, np.ascontiguousarray(frame[..., ::-1]))).reshape(-1, 8)

        # Nx8 rows: x1, y1, x2, y2, track_id, conf, cls, det_ind
        track = Track(
            frame_id=frame_idx,
            detections=[detections[i] for i in outputs[:, 7].astype(int)],
            track_ids=outputs[:, 4].astype(int).tolist(),
        )
        self.tracks.append(track)
        return track
