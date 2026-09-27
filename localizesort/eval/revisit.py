"""Revisit-restricted identity metrics (docs/problem.md, parent Axis 2).

This is the one part of the protocol neither TrackEval nor boxmot provides, and
it is the measurement the project's claim rests on: when the camera comes back to
a stationary object after a gap, does its identity survive?

A **revisit event** is a ground-truth identity reappearing after being absent for
at least `min_gap` frames. At each event the identity either survives, switches,
or the object goes unmatched:

    preserved  the tracker gives it the same id it had before the gap
    switch     the tracker gives it a different id
    unmatched  no tracker detection matches it at reappearance

The matching rule mirrors `CLEAR`'s exactly -- same Hungarian assignment, same
bias toward the previous frame's pairing, same threshold -- so these counts are a
classification of the `IDSW` that CLEAR itself reports, split by the gap that
preceded them, rather than an independent and possibly inconsistent measurement.
CLEAR compares against the last frame an identity was *present*, however long
ago (see its own note at `clear.py:62`), which is what makes this decomposition
exact: every revisit switch counted here is one of CLEAR's IDSWs.
"""

from __future__ import annotations

import tempfile
from collections.abc import Sequence as SequenceABC
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..data import Sequence
from .mot_io import filter_frames
from .tracking import PRIMARY_THRESHOLD, evaluate, preprocess

DEFAULT_MIN_GAP = 10  # frames an identity must be absent for a return to count
DEFAULT_WINDOW = 30  # frames scored after each reappearance
DEFAULT_GRACE = 5  # frames allowed for the tracker to re-acquire a returning object


@dataclass(frozen=True)
class RevisitEvent:
    """One ground-truth identity returning after an absence."""

    gt_id: int
    frame: int  # 0-based timestep of reappearance
    gap: int  # frames absent
    id_before: int | None  # tracker id before the gap, None if never matched
    id_after: int | None  # tracker id once re-acquired, None if never matched
    matched_at: int | None  # frame the match happened, within the grace period

    @property
    def outcome(self) -> str:
        if self.id_after is None or self.id_before is None:
            return "unmatched"
        return "preserved" if self.id_after == self.id_before else "switch"


def assignments(
    data: dict[str, Any], *, threshold: float = PRIMARY_THRESHOLD
) -> list[dict[int, int]]:
    """Per-frame ground-truth to tracker id assignment, exactly as CLEAR makes it.

    Returns one `{gt_id: tracker_id}` per timestep. The assignment is
    order-dependent -- it is biased toward keeping the previous frame's pairing --
    so it has to be replayed forwards over the whole sequence.
    """
    num_gt = data["num_gt_ids"]
    prev_timestep_tracker_id = np.full(num_gt, np.nan)
    eps = np.finfo("float").eps
    out: list[dict[int, int]] = []

    for t, (gt_ids, tracker_ids) in enumerate(
        zip(data["gt_ids"], data["tracker_ids"])
    ):
        matched_gt_ids = np.array([], dtype=int)
        matched_tracker_ids = np.array([], dtype=int)

        if len(gt_ids) > 0 and len(tracker_ids) > 0:
            similarity = data["similarity_scores"][t]
            # Identical to CLEAR: prefer keeping the previous frame's pairing,
            # break ties on similarity, and refuse matches below the threshold.
            score_mat = (
                tracker_ids[np.newaxis, :]
                == prev_timestep_tracker_id[gt_ids[:, np.newaxis]]
            )
            score_mat = 1000 * score_mat + similarity
            score_mat[similarity < threshold - eps] = 0

            rows, cols = linear_sum_assignment(-score_mat)
            keep = score_mat[rows, cols] > eps
            matched_gt_ids = gt_ids[rows[keep]]
            matched_tracker_ids = tracker_ids[cols[keep]]

        prev_timestep_tracker_id[:] = np.nan
        prev_timestep_tracker_id[matched_gt_ids] = matched_tracker_ids
        out.append(dict(zip(matched_gt_ids.tolist(), matched_tracker_ids.tolist())))

    return out


def find_events(
    data: dict[str, Any],
    *,
    threshold: float = PRIMARY_THRESHOLD,
    min_gap: int = DEFAULT_MIN_GAP,
    grace: int = DEFAULT_GRACE,
) -> list[RevisitEvent]:
    """Revisit events in one sequence's preprocessed data.

    `grace` is the number of frames the tracker is allowed to re-acquire a
    returning object before the event counts as unmatched. It exists because
    trackers confirm a new track only after several hits -- StrongSORT's
    `n_init` is 3 -- so a returning object emits nothing at all for its first
    few frames. Scoring only the reappearance frame would report every revisit as
    unmatched and hide whether the identity was preserved, which is the thing
    being measured. The grace window is clipped to frames where the object is
    still actually present.
    """
    per_frame = assignments(data, threshold=threshold)
    present = [set(ids.tolist()) for ids in data["gt_ids"]]
    num_timesteps = data["num_timesteps"]

    # The tracker id each gt id last carried, and the last frame it was present.
    last_id: dict[int, int] = {}
    last_present: dict[int, int] = {}

    events: list[RevisitEvent] = []
    for t in range(num_timesteps):
        for gt_id in sorted(present[t]):
            gap = t - last_present.get(gt_id, -1) - 1
            if gt_id in last_present and gap >= min_gap:
                # Look forward for the first frame the tracker re-acquires it.
                id_after, matched_at = None, None
                for u in range(t, min(t + grace, num_timesteps)):
                    if gt_id not in present[u]:
                        break
                    if gt_id in per_frame[u]:
                        id_after, matched_at = per_frame[u][gt_id], u
                        break
                events.append(
                    RevisitEvent(
                        gt_id=gt_id,
                        frame=t,
                        gap=gap,
                        id_before=last_id.get(gt_id),
                        id_after=id_after,
                        matched_at=matched_at,
                    )
                )
            last_present[gt_id] = t

        for gt_id, tracker_id in per_frame[t].items():
            last_id[gt_id] = tracker_id

    return events


def _event_summary(events: list[RevisitEvent]) -> dict[str, Any]:
    outcomes = [e.outcome for e in events]
    preserved = outcomes.count("preserved")
    switches = outcomes.count("switch")
    decided = preserved + switches
    return {
        "revisit_events": len(events),
        "revisit_preserved": preserved,
        "revisit_switches": switches,
        "revisit_unmatched": outcomes.count("unmatched"),
        # Share of *decided* events that kept their identity; events where the
        # object was not detected at all say nothing about association.
        "revisit_id_preservation": 100.0 * preserved / decided if decided else float("nan"),
        "revisit_mean_gap": float(np.mean([e.gap for e in events])) if events else float("nan"),
    }


def evaluate_revisits(
    sequences: SequenceABC[Sequence],
    tracker_files: dict[str, Path],
    *,
    threshold: float = PRIMARY_THRESHOLD,
    min_gap: int = DEFAULT_MIN_GAP,
    grace: int = DEFAULT_GRACE,
    window: int = DEFAULT_WINDOW,
    tracker_name: str = "localizesort",
    gt_files: dict[str, Path] | None = None,
    do_preproc: bool = False,
) -> dict[str, Any]:
    """Revisit event counts, plus the standard metrics restricted to revisits.

    The restricted metrics are the ordinary CLEAR/HOTA/Identity numbers computed
    over only the frames in `[reappearance, reappearance + window)` of some event,
    produced by filtering the MOT files down to those frames and handing them
    back to the same engine, so no metric is reimplemented.

    Read them with care. IDF1 solves a global identity assignment over whatever
    it is given, so on a subset it re-solves it there: a tracker that assigns a
    brand-new id to every returning object still scores well, because that new id
    is used consistently *within* the window. The restricted numbers therefore
    measure local identity consistency during revisit periods, not identity
    persistence across the gap. The event counts above are what measure the
    latter, and they are the metric the project's claim rests on.
    """
    sequences = list(sequences)
    data = preprocess(
        sequences,
        tracker_files,
        tracker_name=tracker_name,
        gt_files=gt_files,
        do_preproc=do_preproc,
    )

    events = {
        name: find_events(
            seq_data, threshold=threshold, min_gap=min_gap, grace=grace
        )
        for name, seq_data in data.items()
    }

    # Frames to score, as 1-based MOT frame numbers.
    windows = {
        name: {
            frame
            for event in seq_events
            for frame in range(event.frame + 1, event.frame + 1 + window)
        }
        for name, seq_events in events.items()
    }

    per_sequence = {name: _event_summary(evs) for name, evs in events.items()}
    combined = _event_summary([e for evs in events.values() for e in evs])

    restricted: dict[str, Any] | None = None
    if any(windows.values()):
        with tempfile.TemporaryDirectory(prefix="localizesort-revisit-") as tmp:
            stage = Path(tmp)
            gt_subset, tracker_subset = {}, {}
            for seq in sequences:
                keep = windows[seq.name]
                gt_source = (gt_files or {}).get(seq.name, seq.gt_path)
                gt_subset[seq.name] = filter_frames(
                    gt_source, stage / "gt" / f"{seq.name}.txt", keep
                )
                tracker_subset[seq.name] = filter_frames(
                    tracker_files[seq.name], stage / "trk" / f"{seq.name}.txt", keep
                )
            restricted = evaluate(
                sequences,
                tracker_subset,
                thresholds=(threshold,),
                tracker_name=tracker_name,
                gt_files=gt_subset,
                do_preproc=do_preproc,
            )

    return {
        "combined": combined,
        "per_sequence": per_sequence,
        "restricted": restricted,
        "events": {
            name: [
                {
                    "gt_id": e.gt_id,
                    "frame": e.frame,
                    "gap": e.gap,
                    "outcome": e.outcome,
                    "id_before": e.id_before,
                    "id_after": e.id_after,
                }
                for e in evs
            ]
            for name, evs in events.items()
        },
        "config": {
            "threshold": threshold,
            "min_gap": min_gap,
            "grace": grace,
            "window": window,
        },
    }
