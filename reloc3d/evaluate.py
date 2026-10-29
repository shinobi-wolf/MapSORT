"""Run Reloc3D on the stored detections of every test sequence and score it as scripts/eval/eval.py scores the paper's trackers.

    python -m reloc3d.evaluate                         # every dataset in datasets.yaml
    python -m reloc3d.evaluate --datasets moranget --max-geom-error 20

Writes reloc3d/runs/<stamp>_eval/: artifacts/<dataset>/<seq>.txt, metrics.json, and prints the tab_main columns.
Never touches the paper's results.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import time

import numpy as np
import yaml

from mapsort import runs
from mapsort.data import registry
from mapsort.eval.mot_io import write_tracks
from mapsort.schemas import Track

from . import Params, Reloc3D, load_cameras
from .reproduce import RUNS

_spec = importlib.util.spec_from_file_location("paper_eval", runs.REPO_ROOT / "scripts" / "eval" / "eval.py")
paper_eval = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(paper_eval)


def track(seq, params: Params) -> list[Track]:
    """Reloc3D on the stored detections and the cameras of recon/<seq>/model_txt."""
    dets = seq.stored_detections()
    cameras = load_cameras(seq.recon_path / "model_txt", [p.name for p in seq.image_paths()])
    boxes = [np.array([d.bbox for d in frame], dtype=float).reshape(-1, 4) for frame in dets]
    out = [Track(frame_id=i, detections=[], track_ids=[]) for i in range(len(dets))]
    for track_id, t in enumerate(Reloc3D(cameras, params).run(boxes), 1):
        for f, j in zip(t.frames, t.indices):
            out[f].detections.append(dets[f][j])
            out[f].track_ids.append(track_id)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", help="default: every dataset in datasets.yaml")
    ap.add_argument("--tag", default="", help="suffix of the run directory")
    for f in dataclasses.fields(Params):
        flag = "--" + f.name.replace("_", "-")
        if isinstance(f.default, bool):
            ap.add_argument(flag, action=argparse.BooleanOptionalAction, default=f.default)
        else:
            ap.add_argument(flag, type=type(f.default), default=f.default)
    args = ap.parse_args()

    from mapsort.eval.revisit import evaluate_revisits
    from mapsort.eval.tracking import evaluate

    metrics_cfg = yaml.safe_load((runs.REPO_ROOT / "config" / "eval.yaml").read_text())["metrics"]
    iou, preproc, rev = (metrics_cfg["tracking"]["iou_threshold"], metrics_cfg["tracking"]["do_preproc"],
                         metrics_cfg["revisit"])
    params = Params(**{f.name: getattr(args, f.name) for f in dataclasses.fields(Params)})
    test = {ds: paper_eval.present(ds, spec["splits"]["test"]) for ds, spec in registry().items()
            if not args.datasets or ds in args.datasets}
    run_dir = runs.create("eval" + (f"_{args.tag}" if args.tag else ""), runs_dir=RUNS, config={
        "params": dataclasses.asdict(params), "metrics": metrics_cfg,
        "sequences": {ds: [s.name for s in seqs] for ds, seqs in test.items()},
    })

    metrics, rows = {}, []
    for ds, seqs in test.items():
        files, timing = {}, {}
        for seq in seqs:
            started = time.perf_counter()
            tracks = track(seq, params)
            timing[seq.name] = {"seconds": time.perf_counter() - started, "frames": len(tracks)}
            files[seq.name] = run_dir / "artifacts" / ds / f"{seq.name}.txt"
            write_tracks(tracks, files[seq.name])
            print(f"{ds}/{seq.name}: {len(tracks)} frames in {timing[seq.name]['seconds']:.1f}s", flush=True)
        tracking = evaluate(seqs, files, thresholds=[iou], do_preproc=preproc)
        revisit = evaluate_revisits(seqs, files, threshold=iou, min_gap=rev["min_gap"], grace=rev["grace"],
                                    window=0, do_preproc=preproc)
        metrics[ds] = {"tracking": tracking, "revisit": revisit, "timing": timing}
        rows.append({"dataset": paper_eval.DATASET_NAMES[ds],
                     **paper_eval.metric_row(tracking["combined"], revisit["combined"], iou)})
    runs.write_json(run_dir / "metrics.json", metrics)

    if rows:
        cols = list(rows[0])
        print("\n" + "".join(f"{c:>13s}" for c in cols))
        for r in rows:
            print("".join(f"{r[c]:>13s}" for c in cols))
    print(f"\nrun dir: {run_dir}")


if __name__ == "__main__":
    main()
