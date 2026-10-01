"""Score MOT tracker files and write the result into a run directory.

    # score a directory of <seq>.txt files
    python -m mapsort.eval --tracker-dir path/to/results --split test

    # score a run's own artifacts, writing metrics.json back into it
    python -m mapsort.eval --run runs/20260926-120000_tracker_baseline

Sequences come from the split in config/datasets.yaml, restricted to those with a
matching tracker file.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .. import runs
from ..data import sequences
from . import protocol
from .revisit import evaluate_revisits
from .tracking import evaluate, summary_table


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--tracker-dir", type=Path, help="directory of <seq>.txt files")
    src.add_argument(
        "--run",
        type=Path,
        help="existing run directory; reads artifacts/ and writes metrics.json",
    )
    ap.add_argument("--dataset", default="lettucemots")
    ap.add_argument("--split", default="test")
    ap.add_argument("--seqs", nargs="*", help="sequences to score (default: all present)")
    ap.add_argument(
        "--config", type=Path, help="config holding the protocol (default: config/mapsort.yaml)"
    )
    ap.add_argument("--no-revisit", action="store_true", help="skip revisit metrics")
    ap.add_argument(
        "--out",
        type=Path,
        help="where to write metrics.json (default: the run dir, or nowhere)",
    )
    args = ap.parse_args()

    tracker_dir = args.run / "artifacts" if args.run else args.tracker_dir
    if not tracker_dir.is_dir():
        raise SystemExit(f"no such directory: {tracker_dir}")

    seqs = sequences(args.dataset, args.split, args.seqs)
    tracker_files = {
        s.name: tracker_dir / f"{s.name}.txt"
        for s in seqs
        if (tracker_dir / f"{s.name}.txt").exists()
    }
    if not tracker_files:
        raise SystemExit(f"no <seq>.txt files in {tracker_dir} for {args.dataset}/{args.split}")

    seqs = [s for s in seqs if s.name in tracker_files]
    if args.seqs:
        skipped = sorted(set(args.seqs) - set(tracker_files))
        if skipped:
            print(f"warning: no tracker file for {skipped}")

    proto = protocol.load(args.config)
    result = evaluate(
        seqs,
        tracker_files,
        thresholds=proto["tracking"]["thresholds"],
        do_preproc=proto["tracking"]["do_preproc"],
    )
    result["config"]["protocol"] = proto
    print(summary_table(result))

    if not args.no_revisit:
        rev = evaluate_revisits(
            seqs,
            tracker_files,
            min_gap=proto["revisit"]["min_gap"],
            grace=proto["revisit"]["grace"],
            window=proto["revisit"]["window"],
            do_preproc=proto["tracking"]["do_preproc"],
        )
        result["revisit"] = rev
        c = rev["combined"]
        print(
            f"\nrevisits (gap>={proto['revisit']['min_gap']}, "
            f"grace={proto['revisit']['grace']}): "
            f"{c['revisit_events']} events, {c['revisit_switches']} switched, "
            f"{c['revisit_preserved']} preserved, {c['revisit_unmatched']} unmatched "
            f"-> {c['revisit_id_preservation']:.1f}% identity preservation"
        )

    out = args.out or (args.run / "metrics.json" if args.run else None)
    if out:
        runs.write_json(out, result)
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
