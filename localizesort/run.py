"""Run the pipeline over a dataset split, optionally scoring and displaying it.

    localizesort --split test --detections gt --tag baseline --eval
    localizesort --seqs 0011 --viz --viz-compare --no-save --eval

Writes a run directory (see `runs`):

    runs/<stamp>_tracker_<tag>/
      manifest.json          what ran: git state, configs, sequences, timing
      artifacts/<seq>.txt    MOT-format tracker output, one file per sequence
      metrics.json           with --eval

`--no-save` keeps nothing: output goes to a temporary directory, metrics are
printed, and the whole lot is discarded on exit.

`--eval` scores the artifacts straight after the run. Scoring is seconds where
running is minutes, so a saved run can also be re-scored later without repeating
it:

    python -m localizesort.eval --run runs/<run_id>
"""

from __future__ import annotations

import argparse
import contextlib
import tempfile
import time
from pathlib import Path

from . import runs
from .data import Sequence, sequences
from .eval import protocol
from .eval.mot_io import filter_frames, write_tracks
from .eval.revisit import evaluate_revisits
from .eval.tracking import evaluate, summary_table
from .main import DEFAULT_CONFIG, LocalizeSORT


def run_sequence(
    seq: Sequence,
    config_path: Path,
    *,
    detections: str,
    limit: int | None = None,
    window=None,
    colour_by: str = "match",
    compare: bool = False,
    video_path: Path | None = None,
) -> tuple[list, float]:
    """Run a fresh pipeline over one sequence. Returns its tracks and duration.

    The pipeline is rebuilt per sequence: the tracker carries state across
    frames, so reusing one would leak identities between sequences.

    If `window` is given, each frame's overlay is displayed and the run stops
    early when the window asks it to -- so the caller must treat the number of
    tracks returned as the number of frames actually processed, not `limit`.
    """
    pipeline = LocalizeSORT(config_path)
    drawing = window is not None or video_path is not None

    # Ground truth is needed for the detections themselves, and for every
    # drawing mode except plain per-id colouring.
    gt_dets, gt_ids = None, None
    if detections == "gt" or drawing:
        gt_dets, gt_ids = seq.gt_detections(limit=limit)

    video = None
    started = time.perf_counter()
    try:
        for frame_idx, frame in enumerate(seq.iter_frames(limit=limit)):
            frame_dets = gt_dets[frame_idx] if detections == "gt" else None
            track = pipeline.step(frame_idx, frame, frame_dets)

            if not drawing:
                continue

            from . import viz  # local: configures Qt on import

            position = f"{seq.name}  frame {frame_idx + 1}/{seq.num_frames}"
            canvas = viz.draw_tracks(
                frame,
                track.detections,
                track.track_ids,
                gt_detections=gt_dets[frame_idx] if gt_dets else None,
                colour_by=colour_by,
                label="" if compare else position,
            )
            if compare:
                canvas = viz.side_by_side(
                    [
                        (viz.draw_gt(frame, gt_dets[frame_idx], gt_ids[frame_idx]),
                         "ground truth"),
                        (canvas, "localizesort"),
                    ]
                )
                canvas = viz.add_header(canvas, position)
            if video_path is not None and video is None:
                video = viz.writer(
                    video_path, canvas.shape[:2], float(seq.info["frameRate"])
                )
            if video is not None:
                video.write(canvas)
            if window is not None and not window.show(canvas):
                break
    finally:
        if video is not None:
            video.release()
    elapsed = time.perf_counter() - started

    return pipeline.tracker.tracks, elapsed


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dataset", default="lettucemots")
    ap.add_argument("--split", default="test")
    ap.add_argument("--seqs", nargs="*", help="sequences to run (default: the whole split)")
    ap.add_argument(
        "--detections",
        choices=("gt", "detector"),
        default="gt",
        help="ground-truth boxes, or the pipeline's own Detector. Stock YOLO "
        "weights detect no lettuce, so 'gt' isolates tracking quality "
        "(default: gt)",
    )
    ap.add_argument("--limit", type=int, help="frames per sequence, for a quick check")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="pipeline config")
    ap.add_argument("--tag", default="run", help="short label for the run directory")
    ap.add_argument("--eval", action="store_true", help="score the run when it finishes")
    ap.add_argument("--protocol", type=Path, help="eval protocol (default: config/eval.yaml)")
    ap.add_argument(
        "--no-save",
        action="store_true",
        help="keep nothing: output goes to a temporary directory and metrics are "
        "printed only",
    )
    ap.add_argument(
        "--viz",
        action="store_true",
        help="show the box and track-id overlay in a window. q/Esc quits, space "
        "pauses and steps",
    )
    ap.add_argument(
        "--viz-ids",
        action="store_true",
        help="colour boxes by track id instead of match status, so an ID switch "
        "shows up as a box changing colour",
    )
    ap.add_argument(
        "--viz-compare",
        action="store_true",
        help="show ground truth and prediction as side-by-side panels",
    )
    ap.add_argument(
        "--viz-height", type=int, default=1000, help="window height in pixels (default: 1000)"
    )
    ap.add_argument(
        "--viz-save",
        type=Path,
        help="write the overlay to an mp4 instead of needing a display. With "
        "several sequences the name is suffixed per sequence",
    )
    ap.add_argument("--task-id", help="project tracker task this run belongs to, e.g. T06")
    ap.add_argument("--hypothesis", help="what this run is meant to show")
    ap.add_argument("--baseline-run", help="run id this one should be compared against")
    args = ap.parse_args()

    seqs = sequences(args.dataset, args.split, args.seqs)

    # Imported only when drawing is asked for: it configures Qt on import.
    viz_module = None
    if args.viz or args.viz_save:
        from . import viz as viz_module

    if args.viz:
        ok, why = viz_module.Window.available()
        if not ok:
            raise SystemExit(
                f"--viz needs a display but none is usable: {why}\n"
                "Use --viz-save out.mp4 to write the overlay to a file instead."
            )

    if args.detections == "gt" and not args.viz_ids and (args.viz or args.viz_save):
        print(
            "note: with --detections gt every prediction is a ground-truth box, "
            "so match colouring draws them all black; --viz-ids colours by track "
            "id instead, which is what shows ID switches"
        )

    run_config = {
        "dataset": args.dataset,
        "split": args.split,
        "sequences": [s.name for s in seqs],
        "detections": args.detections,
        "limit": args.limit,
        "pipeline_config": str(args.config),
        "pipeline": runs.read_yaml(args.config),
    }

    with contextlib.ExitStack() as stack:
        if args.no_save:
            # Nothing is kept, but evaluation reads files, so the artifacts still
            # need somewhere to live for the length of the run.
            run_dir = Path(
                stack.enter_context(tempfile.TemporaryDirectory(prefix="localizesort-nosave-"))
            )
            (run_dir / "artifacts").mkdir()
            print("running without saving (--no-save)")
        else:
            run_dir = runs.create(
                "tracker",
                args.tag,
                task_id=args.task_id,
                hypothesis=args.hypothesis,
                baseline_run=args.baseline_run,
                config=run_config,
            )
            print(f"run {run_dir.name}")

        window = (
            stack.enter_context(viz_module.Window(height=args.viz_height))
            if args.viz
            else None
        )

        tracker_files: dict[str, Path] = {}
        timings: dict[str, float] = {}
        frames_run: dict[str, int] = {}
        for seq in seqs:
            video_path = None
            if args.viz_save:
                video_path = (
                    args.viz_save
                    if len(seqs) == 1
                    else args.viz_save.with_name(
                        f"{args.viz_save.stem}_{seq.name}{args.viz_save.suffix}"
                    )
                )

            tracks, elapsed = run_sequence(
                seq,
                args.config,
                detections=args.detections,
                limit=args.limit,
                window=window,
                colour_by="id" if args.viz_ids else "match",
                compare=args.viz_compare,
                video_path=video_path,
            )
            tracker_files[seq.name] = write_tracks(
                tracks, run_dir / "artifacts" / f"{seq.name}.txt"
            )
            timings[seq.name] = elapsed
            frames_run[seq.name] = len(tracks)
            print(
                f"  {seq.name}  {len(tracks):4d} frames  {elapsed:7.1f}s  "
                f"{len(tracks) / elapsed:5.1f} fps"
                + (f"  -> {video_path}" if video_path else "")
            )
            if window is not None and window.stopped:
                print("  stopped early")
                break

        seqs = [s for s in seqs if s.name in tracker_files]

        if not args.no_save:
            # Record timing now that it is known; the manifest is the run's record.
            manifest = runs.read_json(run_dir / "manifest.json")
            manifest["timing"] = {
                "seconds_per_sequence": timings,
                "total_seconds": sum(timings.values()),
            }
            manifest["frames_run"] = frames_run
            runs.write_json(run_dir / "manifest.json", manifest)

        if not args.eval:
            if args.no_save:
                print("\nnothing written (--no-save); add --eval for metrics")
            else:
                print(f"\nartifacts in {run_dir / 'artifacts'}")
                print(f"score later with: python -m localizesort.eval --run {run_dir}")
            return

        proto = protocol.load(args.protocol)

        tmp = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="localizesort-gt-")))
        # A run can cover fewer frames than the sequence has -- via --limit, or by
        # quitting the viz window early -- so the ground truth is cut to exactly
        # the frames that ran. Otherwise every unrun frame scores as a miss.
        gt_files = None
        partial = {
            seq.name: frames_run[seq.name]
            for seq in seqs
            if frames_run[seq.name] < seq.num_frames
        }
        if partial:
            gt_files = {
                seq.name: filter_frames(
                    seq.gt_path,
                    tmp / f"{seq.name}.txt",
                    set(range(1, frames_run[seq.name] + 1)),
                )
                for seq in seqs
            }
            covered = ", ".join(f"{k}:{v}" for k, v in sorted(partial.items()))
            print(f"\nnote: partial run, scoring only the frames covered ({covered})")

        result = evaluate(
            seqs,
            tracker_files,
            thresholds=proto["tracking"]["thresholds"],
            do_preproc=proto["tracking"]["do_preproc"],
            gt_files=gt_files,
        )
        result["revisit"] = evaluate_revisits(
            seqs,
            tracker_files,
            min_gap=proto["revisit"]["min_gap"],
            grace=proto["revisit"]["grace"],
            window=proto["revisit"]["window"],
            do_preproc=proto["tracking"]["do_preproc"],
            gt_files=gt_files,
        )
        result["config"]["protocol"] = proto

        print()
        print(summary_table(result))
        combined = result["revisit"]["combined"]
        print(
            f"\nrevisits (gap>={proto['revisit']['min_gap']}, "
            f"grace={proto['revisit']['grace']}): "
            f"{combined['revisit_events']} events, "
            f"{combined['revisit_switches']} switched, "
            f"{combined['revisit_preserved']} preserved, "
            f"{combined['revisit_unmatched']} unmatched -> "
            f"{combined['revisit_id_preservation']:.1f}% identity preservation"
        )

        # Written inside the stack: under --no-save run_dir is a temporary
        # directory that is removed as soon as it closes.
        if args.no_save:
            print("\nnothing written (--no-save)")
        else:
            runs.write_json(run_dir / "metrics.json", result)
            print(f"\nwrote {run_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
