"""Tracking metrics, computed by TrackEval.

No metric is implemented here. TrackEval's four metric families cover the whole
protocol -- `CLEAR` gives MOTA and ID switches, `HOTA` gives HOTA and HOTA(0),
`Identity` gives IDF1, and `Count` gives the predicted and GT identity counts --
so this module only stages files into the layout TrackEval reads and flattens
its output.

The CLEAR match threshold is a config value, so MOTA at an IoU of 0.25 needs no
code of its own: it is another entry in `thresholds`. HOTA takes no threshold,
since it integrates over alphas 0.05..0.95; `HOTA(0)` is the alpha-to-zero
limit. Fields from the primary threshold keep their plain names, and any
additional threshold is suffixed, e.g. `MOTA@0.25`.
"""

from __future__ import annotations

import tempfile
from collections.abc import Sequence as SequenceABC
from pathlib import Path
from typing import Any

import numpy as np
import trackeval

from ..data import Sequence

# GT class 1 is 'pedestrian' in MOTChallenge's class map, which is what our
# single-class crop datasets use. TrackEval accepts no other class here.
CLASS = "pedestrian"

# Which threshold's fields keep their plain names; the protocol owns the value,
# this is the fallback for direct library use.
PRIMARY_THRESHOLD = 0.5


def _stage(
    sequences: SequenceABC[Sequence],
    tracker_files: dict[str, Path],
    stage_dir: Path,
    tracker_name: str,
    gt_files: dict[str, Path] | None = None,
) -> tuple[Path, Path]:
    """Symlink GT and tracker files into the layout TrackEval expects.

    TrackEval wants all sequences under one GT folder, but ours are split across
    `mot/train` and `mot/test`, and restricted-subset evaluation needs
    substitute GT. Staging symlinks handles both without copying or mutating
    anything.
    """
    gt_root = stage_dir / "gt"
    tracker_root = stage_dir / "trackers" / tracker_name
    tracker_root.mkdir(parents=True, exist_ok=True)

    for seq in sequences:
        gt_dest = gt_root / seq.name / "gt"
        gt_dest.mkdir(parents=True, exist_ok=True)
        gt_src = (gt_files or {}).get(seq.name, seq.gt_path)
        (gt_dest / "gt.txt").symlink_to(gt_src.resolve())
        (tracker_root / f"{seq.name}.txt").symlink_to(
            tracker_files[seq.name].resolve()
        )

    return gt_root, tracker_root


def _flatten(metric: Any, res: dict[str, Any], suffix: str = "") -> dict[str, Any]:
    """One metric's results as plain JSON-able scalars.

    Array fields are per-alpha; the headline HOTA number is their mean, which is
    what the HOTA paper reports. Rates are scaled to percentages, matching how
    the LocalizeSORT paper and boxmot both report them.
    """
    out: dict[str, Any] = {}
    for field in getattr(metric, "float_array_fields", []):
        out[field + suffix] = float(np.mean(res[field])) * 100
    for field in getattr(metric, "integer_array_fields", []):
        out[field + suffix] = int(np.sum(res[field]))
    for field in getattr(metric, "float_fields", []):
        # MOTP_sum is an accumulator, not a rate, so it is not a percentage.
        scale = 1.0 if field.endswith("_sum") else 100.0
        out[field + suffix] = float(res[field]) * scale
    for field in getattr(metric, "integer_fields", []):
        out[field + suffix] = int(res[field])
    return out


def _metrics(thresholds: SequenceABC[float]) -> list[tuple[Any, str]]:
    """The metric objects to run, each with the suffix its fields get.

    HOTA and Count are threshold-free, so they run once. CLEAR and Identity run
    once per threshold.
    """
    quiet = {"PRINT_CONFIG": False}
    built: list[tuple[Any, str]] = [
        (trackeval.metrics.HOTA(quiet), ""),
        (trackeval.metrics.Count(quiet), ""),
    ]
    for threshold in thresholds:
        suffix = "" if threshold == PRIMARY_THRESHOLD else f"@{threshold:g}"
        config = {"THRESHOLD": threshold, **quiet}
        built.append((trackeval.metrics.CLEAR(config), suffix))
        built.append((trackeval.metrics.Identity(config), suffix))
    return built


def preprocess(
    sequences: SequenceABC[Sequence],
    tracker_files: dict[str, Path],
    *,
    tracker_name: str = "localizesort",
    gt_files: dict[str, Path] | None = None,
    do_preproc: bool = False,
) -> dict[str, dict[str, Any]]:
    """Per-sequence metric-ready data, as TrackEval's dataset class produces it.

    Each value holds `gt_ids`, `tracker_ids` and `similarity_scores` per
    timestep. Exposed so that metrics TrackEval does not provide (see `revisit`)
    can reuse its file parsing and IoU computation rather than redoing them.

    `gt_files` overrides which GT a sequence is scored against, which is how
    restricted subsets are evaluated.

    `do_preproc` is off by default. MOTChallenge preprocessing drops GT marked as
    distractor classes or below a visibility cutoff; our single-class crop data
    has neither, so leaving it on would only risk silently filtering rows.
    """
    sequences = list(sequences)
    missing = [s.name for s in sequences if s.name not in tracker_files]
    if missing:
        raise ValueError(f"no tracker file for sequences: {missing}")

    with tempfile.TemporaryDirectory(prefix="localizesort-eval-") as tmp:
        stage_dir = Path(tmp)
        gt_root, tracker_root = _stage(
            sequences, tracker_files, stage_dir, tracker_name, gt_files
        )

        dataset = trackeval.datasets.MotChallenge2DBox(
            {
                "GT_FOLDER": str(gt_root),
                "TRACKERS_FOLDER": str(tracker_root.parent),
                "OUTPUT_FOLDER": str(stage_dir / "out"),
                "TRACKERS_TO_EVAL": [tracker_name],
                "TRACKER_SUB_FOLDER": "",
                "OUTPUT_SUB_FOLDER": "",
                "CLASSES_TO_EVAL": [CLASS],
                "SEQ_INFO": {s.name: s.num_frames for s in sequences},
                "SKIP_SPLIT_FOL": True,
                "DO_PREPROC": do_preproc,
                "PRINT_CONFIG": False,
            }
        )

        return {
            seq.name: dataset.get_preprocessed_seq_data(
                dataset.get_raw_seq_data(tracker_name, seq.name), CLASS
            )
            for seq in sequences
        }


def evaluate(
    sequences: SequenceABC[Sequence],
    tracker_files: dict[str, Path],
    *,
    thresholds: SequenceABC[float] = (0.5, 0.25),
    tracker_name: str = "localizesort",
    gt_files: dict[str, Path] | None = None,
    do_preproc: bool = False,
) -> dict[str, Any]:
    """Score tracker files against ground truth.

    `tracker_files` maps sequence name to a MOT-format file.

    Returns `{"combined": {...}, "per_sequence": {seq: {...}}, "config": {...}}`.
    """
    sequences = list(sequences)
    metrics = _metrics(thresholds)

    # Drive the metrics directly rather than through trackeval.Evaluator, which
    # would also write its own CSV report into the staging directory.
    data = preprocess(
        sequences,
        tracker_files,
        tracker_name=tracker_name,
        gt_files=gt_files,
        do_preproc=do_preproc,
    )
    per_sequence_raw = {
        name: [metric.eval_sequence(seq_data) for metric, _ in metrics]
        for name, seq_data in data.items()
    }

    per_sequence = {
        name: {
            k: v
            for (metric, suffix), res in zip(metrics, results)
            for k, v in _flatten(metric, res, suffix).items()
        }
        for name, results in per_sequence_raw.items()
    }

    combined: dict[str, Any] = {}
    for i, (metric, suffix) in enumerate(metrics):
        merged = metric.combine_sequences(
            {name: results[i] for name, results in per_sequence_raw.items()}
        )
        combined.update(_flatten(metric, merged, suffix))

    return {
        "combined": combined,
        "per_sequence": per_sequence,
        "config": {
            "engine": f"trackeval {trackeval.__version__ if hasattr(trackeval, '__version__') else '1.3.0'}",
            "thresholds": list(thresholds),
            "primary_threshold": PRIMARY_THRESHOLD,
            "do_preproc": do_preproc,
            "sequences": [f"{s.dataset}/{s.split}/{s.name}" for s in sequences],
        },
    }


# The metrics the protocol reports, in the order they belong in a table.
HEADLINE_FIELDS = (
    "MOTA",
    "MOTA@0.25",
    "HOTA",
    "HOTA(0)",
    "IDF1",
    "IDSW",
    "IDs",
    "GT_IDs",
)


def summary_table(result: dict[str, Any]) -> str:
    """The headline fields as a fixed-width table, combined row last."""
    rows = [(name, res) for name, res in sorted(result["per_sequence"].items())]
    rows.append(("COMBINED", result["combined"]))

    header = f"{'seq':10s}" + "".join(f"{f:>11s}" for f in HEADLINE_FIELDS)
    lines = [header, "-" * len(header)]
    for name, res in rows:
        cells = "".join(
            f"{res[f]:>11.2f}" if isinstance(res.get(f), float) else f"{res.get(f, '-'):>11}"
            for f in HEADLINE_FIELDS
        )
        lines.append(f"{name:10s}{cells}")
    return "\n".join(lines)
