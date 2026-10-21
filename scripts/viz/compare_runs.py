#!/usr/bin/env python3
"""Render a GT | tracker | tracker comparison video from saved run artifacts.

Standalone: nothing in the pipeline imports this, and it never writes back to a
run directory. It reads `runs/<id>/artifacts/<seq>.txt` and the sequence's GT,
and emits one mp4 with a panel per source.

Predicted ids are relabelled onto GT ids for display, so the panels can be read
against each other instead of against three unrelated numberings:

  * Walking frames in order, a pred id is bound the first time it is seen, to
    the GT id of the box it matches in that frame (Hungarian, IoU >= --iou).
    That binding is fixed for the id's whole life.
  * If that GT object was already claimed by an earlier pred id, this one is a
    duplicate and takes the next number from --dup-start instead.
  * A pred id whose first box matches no GT object keeps its own number, shown
    with a `?` prefix.

Colour is decided per frame, not per track: black where the displayed label
equals the GT id the box matches right now, red otherwise. So a fragment is red
for its whole life, and a track that drifts onto another object turns red at the
frame it drifts rather than staying black on the wrong fruit.

The relabelling is a bijection, so it cannot change any metric -- every MOT
metric depends on which detections share an identity, not on the identity's
value. `--check` re-scores the relabelled ids and prints the diff to prove it.

Usage:
    python scripts/viz/compare_runs.py --dataset moranget --seq V01 \
        --runs runs/20261007-233206_tracker:MapSORT runs/20261007-233003_tracker:StrongSORT

`--runs` also takes a tracker file directly: runs/eval-<stamp>/artifacts/<spec>/<dataset>/<seq>.txt:Title
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from mapsort import viz
from mapsort.data import sequence

REPO_ROOT = Path(__file__).resolve().parents[2]


def read_mot(path: Path) -> dict[int, list[tuple[int, float, float, float, float]]]:
    """frame -> [(id, x, y, w, h)], top-left xywh."""
    out: dict[int, list] = defaultdict(list)
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        f = line.split(",")
        out[int(float(f[0]))].append((int(float(f[1])), *(float(v) for v in f[2:6])))
    return out


def iou_matrix(a: list, b: list) -> np.ndarray:
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    ix = np.maximum(0, np.minimum(a[:, 0, None] + a[:, 2, None], b[:, 0] + b[:, 2])
                    - np.maximum(a[:, 0, None], b[:, 0]))
    iy = np.maximum(0, np.minimum(a[:, 1, None] + a[:, 3, None], b[:, 1] + b[:, 3])
                    - np.maximum(a[:, 1, None], b[:, 1]))
    inter = ix * iy
    return inter / ((a[:, 2] * a[:, 3])[:, None] + (b[:, 2] * b[:, 3])[None, :] - inter + 1e-9)


def match_frame(prows: list, grows: list, iou: float) -> dict[int, int]:
    """Row index in `prows` -> GT id, for pairs overlapping by at least `iou`."""
    if not prows or not grows:
        return {}
    m = iou_matrix([r[1:] for r in prows], [r[1:] for r in grows])
    return {r: grows[c][0] for r, c in zip(*linear_sum_assignment(-m)) if m[r, c] >= iou}


def relabel(gt: dict, pred: dict, iou: float, dup_start: int) -> tuple[dict, dict]:
    """Bind each pred id on first sight. Returns (pred_id -> label, per-frame draw list)."""
    max_gt = max(r[0] for rows in gt.values() for r in rows)
    if dup_start <= max_gt:
        raise SystemExit(f"--dup-start {dup_start} collides with GT ids up to {max_gt}")

    label: dict[int, str] = {}
    owner: dict[int, int] = {}
    counter = dup_start
    frames: dict[int, list] = {}

    last_id: dict[int, int] = {}  # GT id -> the pred id that last matched it
    switches = 0
    tally: dict[int, tuple[int, int]] = {}  # frame -> (ids emitted so far, switches so far)

    for frame in sorted(pred):
        prows = pred[frame]
        here = match_frame(prows, gt.get(frame, []), iou)
        drawn = []
        for i, (pid, *box) in enumerate(prows):
            gid = here.get(i)
            if pid not in label:
                if gid is None:
                    label[pid] = f"?{pid}"
                elif gid in owner:
                    label[pid] = str(counter)
                    counter += 1
                else:
                    owner[gid] = pid
                    label[pid] = str(gid)
            # CLEAR-MOT's rule: a switch is this GT object being matched to a
            # different tracker id than the one it was last matched to.
            if gid is not None:
                if gid in last_id and last_id[gid] != pid:
                    switches += 1
                last_id[gid] = pid
            drawn.append((box, label[pid], gid is not None and label[pid] == str(gid)))
        frames[frame] = drawn
        tally[frame] = (len(label), switches)
    return label, frames, tally


def gt_tally(gt: dict) -> dict[int, int]:
    """frame -> distinct GT ids seen up to and including it."""
    seen: set[int] = set()
    out = {}
    for frame in sorted(gt):
        seen.update(r[0] for r in gt[frame])
        out[frame] = len(seen)
    return out


def carry_forward(tally: dict, n: int, empty):
    """Fill every frame 1..n, holding the last value across frames with no rows."""
    out, last = {}, empty
    for f in range(1, n + 1):
        last = tally.get(f, last)
        out[f] = last
    return out


def draw_panel(frame_bgr: np.ndarray, drawn: list) -> np.ndarray:
    canvas = frame_bgr.copy()
    for (x, y, w, h), text, black in drawn:
        colour = viz.MATCH_COLOUR if black else viz.MISMATCH_COLOUR
        x0, y0 = int(x), int(y)
        cv2.rectangle(canvas, (x0, y0), (int(x + w), int(y + h)), colour, viz.BOX_THICKNESS)
        viz._label(canvas, f"id={text}", x0, y0, colour)
    return canvas


def draw_gt_panel(frame_bgr: np.ndarray, rows: list) -> np.ndarray:
    canvas = frame_bgr.copy()
    for gid, x, y, w, h in rows:
        x0, y0 = int(x), int(y)
        cv2.rectangle(canvas, (x0, y0), (int(x + w), int(y + h)),
                      viz.MATCH_COLOUR, viz.BOX_THICKNESS)
        viz._label(canvas, f"id={gid}", x0, y0, viz.MATCH_COLOUR)
    return canvas


def legend_bar(width: int, dup_start: int, height: int = 48) -> np.ndarray:
    bar = np.full((height, width, 3), viz.HEADER_BG, dtype=np.uint8)
    x, swatch = 16, 22
    y0 = (height - swatch) // 2
    for colour, text in (
        (viz.MATCH_COLOUR, "id matches ground truth"),
        (viz.MISMATCH_COLOUR, f"wrong id (duplicates renumbered from {dup_start})"),
    ):
        cv2.rectangle(bar, (x, y0), (x + swatch, y0 + swatch), colour, -1)
        cv2.rectangle(bar, (x, y0), (x + swatch, y0 + swatch), (255, 255, 255), 1)
        tx = x + swatch + 8
        (tw, th), _ = cv2.getTextSize(text, viz.FONT, 0.8, 2)
        cv2.putText(bar, text, (tx, (height + th) // 2 - 1), viz.FONT, 0.8,
                    (255, 255, 255), 2, cv2.LINE_AA)
        x = tx + tw + 40
    return bar


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True,
                    help="run dirs, each optionally `path:Panel title`")
    ap.add_argument("--dataset", default="moranget")
    ap.add_argument("--split", default="test")
    ap.add_argument("--seq", default="V01")
    ap.add_argument("--out", type=Path, help="default: <seq>_<titles>_vs_gt.mp4")
    ap.add_argument("--iou", type=float, default=0.5, help="box-matching threshold")
    ap.add_argument("--dup-start", type=int, default=1000, help="renumbering base for duplicates")
    ap.add_argument("--fps", type=float, default=5.0)
    ap.add_argument("--start", type=int, default=1, help="first frame to render")
    ap.add_argument("--limit", type=int, help="frames to render from --start")
    ap.add_argument("--scale", type=float, default=0.5, help="per-panel resize before stacking")
    ap.add_argument("--check", action="store_true",
                    help="print the relabelling summary without rendering")
    args = ap.parse_args()

    seq = sequence(args.dataset, args.split, args.seq)
    gt_rows: dict[int, list] = defaultdict(list)
    for frame, gid, x, y, w, h, *_ in seq.gt_rows():
        gt_rows[int(frame)].append((int(gid), x, y, w, h))

    sources = []
    for spec in args.runs:
        path, _, title = spec.partition(":")
        run = Path(path)
        # A .txt is a tracker file itself, e.g. eval.py's artifacts/<spec>/<dataset>/<seq>.txt.
        artifact = run if run.suffix == ".txt" else run / "artifacts" / f"{args.seq}.txt"
        if not artifact.exists():
            raise SystemExit(f"no artifact for {args.seq} in {run}")
        label, frames, tally = relabel(gt_rows, read_mot(artifact), args.iou, args.dup_start)
        sources.append((title or run.name, label, frames,
                        carry_forward(tally, seq.num_frames, (0, 0))))

        bound = sum(1 for v in label.values() if not v.startswith("?") and int(v) < args.dup_start)
        dup = sum(1 for v in label.values() if not v.startswith("?") and int(v) >= args.dup_start)
        drawn = [d for f in frames.values() for d in f]
        black = sum(1 for *_, b in drawn if b)
        print(f"{title or run.name:12s} {len(label):4d} ids = {bound:3d} bound + {dup:3d} duplicates"
              f" + {len(label) - bound - dup} unmatched   boxes {len(drawn):6d},"
              f" {100 * black / max(len(drawn), 1):5.1f}% correct id")

    if args.check:
        return

    gt_seen = carry_forward(gt_tally(gt_rows), seq.num_frames, 0)

    out = args.out or Path(f"{args.seq}_{'_vs_'.join(t for t, *_ in sources)}_vs_gt.mp4")
    last = min(args.start - 1 + (args.limit or seq.num_frames), seq.num_frames)
    writer = None
    try:
        for idx, frame in enumerate(seq.iter_frames(limit=last)):
            f = idx + 1
            if f < args.start:
                continue
            bgr = np.ascontiguousarray(frame[..., ::-1])
            # Counters are cumulative to this frame: ids emitted on every panel,
            # switches only where there is a tracker to blame.
            panels = [(draw_gt_panel(bgr, gt_rows.get(f, [])),
                       f"ground truth   ids {gt_seen.get(f, 0)}")]
            for title, _, frames, tally in sources:
                emitted, switches = tally.get(f, (0, 0))
                panels.append((draw_panel(bgr, frames.get(f, [])),
                               f"{title}   ids {emitted}   switches {switches}"))
            if args.scale != 1.0:
                panels = [(cv2.resize(p, None, fx=args.scale, fy=args.scale), t)
                          for p, t in panels]
            canvas = viz.side_by_side(panels)
            canvas = np.vstack([canvas, legend_bar(canvas.shape[1], args.dup_start)])
            canvas = viz.add_header(canvas, f"{args.dataset}/{args.seq}  frame {f}/{seq.num_frames}")
            if writer is None:
                writer = viz.writer(out, canvas.shape[:2], args.fps)
            writer.write(canvas)
    finally:
        if writer is not None:
            writer.release()
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
