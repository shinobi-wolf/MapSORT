"""Evaluation. Reads run artifacts from disk; never imports the pipeline.

See docs/problem.md for the protocol. Tracking metrics come from TrackEval
(`tracking`); localization metrics are not implemented yet.
"""

from . import protocol
from .mot_io import filter_frames, read_rows, write_tracks
from .revisit import evaluate_revisits, find_events
from .tracking import HEADLINE_FIELDS, evaluate, preprocess, summary_table

__all__ = [
    "HEADLINE_FIELDS",
    "evaluate",
    "evaluate_revisits",
    "filter_frames",
    "find_events",
    "preprocess",
    "protocol",
    "read_rows",
    "summary_table",
    "write_tracks",
]
