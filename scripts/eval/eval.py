"""Run every model in config/eval.yaml over the test data and fill the paper's result tables.

    python scripts/eval/eval.py                          # all tables, new run dir
    python scripts/eval/eval.py --run runs/eval-<stamp>  # resume: finished sequences are reused
    python scripts/eval/eval.py --tables main ablation   # a subset

Each run lives in runs/eval-<UTC stamp>/: manifest.json, eval.log (everything printed, appended
on resume), metrics/, and tracker files at artifacts/<spec>/<dataset>/<seq>.txt, one <spec> per
distinct (tracker, impl, hp, landmark source, live), so identical runs are shared between
tables. Scores merge into docs/precision-ag-paper/results/*.csv by row key; rows not computed
here are left as they are.

Inputs are the stored detections and the cached ReID embeddings. SAM2MOT takes the GT masks in
masks/<seq>/ for now, standing in for SAM 2, and each detection's filled box where a sequence
or detection has none. The runtime table alone runs live, computing ReID inside the loop.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from dataclasses import fields
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from mapsort import runs  # noqa: E402
from mapsort.data import DEFAULT_REGISTRY, Sequence, data_root, registry, sequence  # noqa: E402
from mapsort.eval.mot_io import write_tracks  # noqa: E402
from mapsort.schemas import Track  # noqa: E402

RESULTS = REPO / "docs" / "precision-ag-paper" / "results"
BOXMOT_V25 = REPO / "boxmot-v25.0.0"

DATASET_NAMES = {
    "lettucemots": "LettuceMOTS",
    "moranget": "MOrangeT",
    "grapemots": "GrapeMOTS",
    "applemots": "AppleMOTS",
}
MODEL_NAMES = {"botsort": "BoT-SORT", "sam2mot": "SAM2MOT", "strongsort": "StrongSORT", "mapsort": "MapSORT"}
# dataset -> (sequence, ablation column prefix)
REPRESENTATIVE = {
    "lettucemots": ("0011", "lettuce0011"),
    "moranget": ("V01", "orange_V01"),
    "grapemots": ("PathPlanning_7", "grape_PP7"),
    "applemots": ("0008", "apple0008"),
}
HARDWARE = {"shinobi-owl": "L", "shinobi-wolf": "W"}

# Overrides of the main MapSORT block; `source` picks landmarks/<seq>/<source>.npz.
EUCLIDEAN_GATE_RADII = 1.0  # TODO: prior work's threshold, in median object radii (sfm_point.npz `scale`)
ABLATIONS = {
    "(a) BoT-SORT": {"hp": {"landmarks": {"enabled": False}}},
    "(b) + landmark revival": {"hp": {"landmarks": {"reprojection": None, "map_motion": {"enabled": False}}}},
    "(c) + camera revival": {"hp": {"landmarks": {"map_motion": {"enabled": False}}}},
    "(d) + map motion (MapSORT)": {},
    "(e) (d), point landmark": {
        "hp": {"landmarks": {"distance": "euclidean", "gate": EUCLIDEAN_GATE_RADII, "short_gate": EUCLIDEAN_GATE_RADII}},
        "source": "sfm_point",
    },
}
GATES = [7.8147, 25, 50, 100, 200, 400, 800, 1600]  # first is chi2inv95[3]
TABLES = ("main", "sequences", "ablation", "gate", "gap", "runtime")


# ── specs ──────────────────────────────────────────────────────────────────────


def merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        out[k] = merge(out.get(k, {}), v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def make_spec(tracker: str, model: dict, *, source: str | None = None, live: bool = False) -> dict:
    """What a run depends on, beyond the data."""
    return {"tracker": tracker, "impl": model["impl"], "hp": model.get("hp") or {}, "source": source, "live": live}


def spec_key(spec: dict) -> str:
    digest = hashlib.sha1(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:8]
    return f"{spec['tracker']}-{digest}"


def mapsort_variant(models: dict, override: dict, *, live: bool = False) -> dict:
    model = merge(models["mapsort"], {"hp": override.get("hp", {})})
    return make_spec("mapsort", model, source=override.get("source", "sfm"), live=live)


def model_spec(models: dict, name: str, *, live: bool = False) -> dict:
    if name == "mapsort":
        return mapsort_variant(models, {}, live=live)
    # SAM2MOT computes nothing live while GT masks stand in for SAM 2, so its runtime reuses its main run.
    return make_spec(name, models[name], live=live and name != "sam2mot")


def suffix(iou: float) -> str:
    """Field suffix mapsort.eval gives CLEAR/Identity at a non-primary IoU."""
    from mapsort.eval.tracking import PRIMARY_THRESHOLD

    return "" if iou == PRIMARY_THRESHOLD else f"@{iou:g}"


# ── inputs ─────────────────────────────────────────────────────────────────────


def present(dataset: str, names: list[str]) -> list[Sequence]:
    """The listed sequences whose MOT view exists on this machine."""
    out = []
    for name in names:
        try:
            out.append(sequence(dataset, "test", name))
        except FileNotFoundError:
            pass
    return out


def reid_model(seq: Sequence) -> str:
    models = (registry()[seq.dataset].get("reid") or {}).get(seq.name)
    if not models:
        raise SystemExit(f"{seq.dataset}/{seq.name}: no ReID model under `reid` in datasets.yaml")
    return models[0]


def det_rows(seq: Sequence) -> np.ndarray:
    return np.atleast_2d(np.loadtxt(seq.dets_path, delimiter=","))


def cached_embeddings(seq: Sequence, model: str) -> list[np.ndarray]:
    """reid/<seq>/<model>.npz split per 0-based frame, rows parallel to stored_detections."""
    path = data_root() / seq.dataset / "reid" / seq.name / f"{model}.npz"
    cache = np.load(path)
    digest = hashlib.sha256(seq.dets_path.read_bytes()).hexdigest()
    if str(cache["dets_sha256"]) != digest:
        raise SystemExit(f"{path} is stale: dets/{seq.name}.txt changed; rerun scripts/reid/reid_cache.py")
    out = [np.empty((0, cache["embedding"].shape[1]), np.float32) for _ in range(seq.num_frames)]
    for frame in np.unique(cache["frame"]):
        if frame <= seq.num_frames:
            out[frame - 1] = cache["embedding"][cache["frame"] == frame]
    return out


def detection_masks(seq: Sequence, frame_idx: int, boxes: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """bool [n, H, W] for one frame's detections: GT masks, else (or where empty) the filled box."""
    from PIL import Image

    masks = np.zeros((len(boxes), *shape), dtype=bool)
    if seq.name in (registry()[seq.dataset].get("masks") or []):
        labels = np.array(Image.open(data_root() / seq.dataset / "masks" / seq.name / f"{frame_idx + 1:06d}.png"))
        masks = labels[None] == np.arange(1, len(boxes) + 1, dtype=labels.dtype)[:, None, None]
    for k, (x1, y1, x2, y2) in enumerate(boxes):
        if not masks[k].any():
            masks[k, max(int(y1), 0) : max(int(np.ceil(y2)), int(y1) + 1), max(int(x1), 0) : max(int(np.ceil(x2)), int(x1) + 1)] = True
    return masks


class CachedEncoder:
    """AppearanceEncoder serving cached embeddings, matched to the tracker's detections by box."""

    def __init__(self, boxes: list[np.ndarray], embeddings: list[np.ndarray]) -> None:
        self.boxes, self.embeddings, self.frame = boxes, embeddings, 0

    @property
    def embedding_dim(self) -> int:
        return int(self.embeddings[0].shape[1])

    @property
    def requirements(self):
        from boxmot.reid.protocols import EncoderRequirements

        return EncoderRequirements()

    def encode(self, frames, detections):
        import torch

        out = []
        for dets in detections:
            boxes = dets.geometry.values.cpu().numpy()
            # Trackers may encode a confidence-filtered subset; match each row back to the file's.
            rows = [int(np.flatnonzero(np.all(np.isclose(self.boxes[self.frame], b), axis=1))[0]) for b in boxes]
            out.append(torch.from_numpy(self.embeddings[self.frame][rows].astype(np.float32)))
        return out


# ── runners: each returns the per-frame Tracks ──────────────────────────────────


def frame_boxes(seq: Sequence) -> list[np.ndarray]:
    """Per-frame xyxy float32 boxes in file order, as the trackers receive them."""
    rows = det_rows(seq)
    out = [np.empty((0, 4), np.float32) for _ in range(seq.num_frames)]
    for frame in np.unique(rows[:, 0].astype(int)):
        r = rows[rows[:, 0].astype(int) == frame]
        out[frame - 1] = np.stack([r[:, 2], r[:, 3], r[:, 2] + r[:, 4], r[:, 3] + r[:, 5]], axis=1).astype(np.float32)
    return out


def frames(seq: Sequence):
    """RGB frames, zero-padded bottom/right to the largest one where sizes vary (MOrangeT V05).

    Frame-to-frame camera-motion compensation needs one size; padding leaves box coordinates as they are.
    """
    from PIL import Image

    sizes = {Image.open(p).size for p in seq.image_paths()}
    if len(sizes) == 1:
        yield from seq.iter_frames()
        return
    width, height = max(w for w, _ in sizes), max(h for _, h in sizes)
    for frame in seq.iter_frames():
        canvas = np.zeros((height, width, 3), dtype=frame.dtype)
        canvas[: frame.shape[0], : frame.shape[1]] = frame
        yield canvas


def encoder_for(seq: Sequence, live: bool):
    model = reid_model(seq)
    if live:
        return {"model": model}
    return CachedEncoder(frame_boxes(seq), cached_embeddings(seq, model))


def run_mapsort(seq: Sequence, spec: dict) -> list[Track]:
    from mapsort.tracker import Tracker

    hp = spec["hp"]
    landmarks = merge({"enabled": True}, hp.get("landmarks", {}))
    stored = None
    if landmarks["enabled"]:
        stored = seq.stored_landmarks(spec["source"])
        if landmarks.get("distance") == "euclidean":
            scale = float(np.load(seq.landmark_path(spec["source"]))["scale"])
            for key in ("gate", "short_gate"):
                if key in landmarks:
                    landmarks[key] *= scale

    reid = encoder_for(seq, spec["live"])
    tracker = Tracker(botsort=hp.get("botsort"), reid=reid, landmarks=landmarks, frame_rate=int(seq.info["frameRate"]))
    dets = seq.stored_detections()
    for i, frame in enumerate(frames(seq)):
        if isinstance(reid, CachedEncoder):
            reid.frame = i
        tracker(i, frame, dets[i], stored[i] if stored is not None else None)
    return tracker.tracks


def run_boxmot(seq: Sequence, spec: dict) -> list[Track]:
    """A vendored-boxmot tracker at its config defaults plus `hp`."""
    import importlib

    from boxmot import ReIDConfig, create_tracker

    name = spec["tracker"]
    config_cls = getattr(
        importlib.import_module(f"boxmot.trackers.{name}.config"),
        {"strongsort": "StrongSortConfig", "botsort": "BotSortConfig"}[name],
    )
    params = dict(spec["hp"])
    if "frame_rate" in {f.name for f in fields(config_cls)}:
        params["frame_rate"] = int(seq.info["frameRate"])
    reid = encoder_for(seq, spec["live"])
    tracker = create_tracker(
        name, config=config_cls(**params), reid=ReIDConfig(**reid) if isinstance(reid, dict) else reid
    )

    dets = seq.stored_detections()
    tracks = []
    for i, frame in enumerate(frames(seq)):
        if isinstance(reid, CachedEncoder):
            reid.frame = i
        arr = np.array([[*d.bbox, d.confidence, d.class_id] for d in dets[i]], np.float32).reshape(-1, 6)
        out = np.asarray(tracker.update(arr, np.ascontiguousarray(frame[..., ::-1]))).reshape(-1, 8)
        tracks.append(
            Track(frame_id=i, detections=[dets[i][j] for j in out[:, 7].astype(int)], track_ids=out[:, 4].astype(int).tolist())
        )
    return tracks


def run_sam2mot(seq: Sequence, spec: dict) -> list[Track]:
    """Sam2Mot from boxmot-v25.0.0; only valid inside the --worker subprocess."""
    import torch
    from boxmot.structures import Detections, MaskBatch
    from boxmot.structures.geometry import Boxes
    from boxmot.trackers.multimodal.sam2mot.tracker import Sam2Mot

    params = dict(spec["hp"])
    params.pop("segmentor")  # unused while GT masks stand in for SAM 2
    tracker = Sam2Mot(**params)

    dets = seq.stored_detections()
    tracks = []
    for i, frame in enumerate(seq.iter_frames()):
        bgr = np.ascontiguousarray(frame[..., ::-1])
        boxes = np.array([d.bbox for d in dets[i]], np.float32).reshape(-1, 4)
        batch = Detections(
            geometry=Boxes(torch.from_numpy(boxes)),
            scores=torch.tensor([d.confidence for d in dets[i]], dtype=torch.float32),
            class_ids=torch.tensor([d.class_id for d in dets[i]], dtype=torch.int64),
            sample_id=f"{seq.name}/{i}",
        )
        masks = MaskBatch(torch.from_numpy(detection_masks(seq, i, boxes, bgr.shape[:2])))
        out = tracker.update(batch.with_masks(masks), bgr)
        idx = out.detection_indices.cpu().numpy().astype(int)
        ids = out.track_ids.cpu().numpy().astype(int)
        # Coasting tracks come back with index -1; like the other trackers, report matched rows only.
        keep = idx >= 0
        tracks.append(Track(frame_id=i, detections=[dets[i][j] for j in idx[keep]], track_ids=ids[keep].tolist()))
    return tracks


def run(seq: Sequence, spec: dict, out: Path) -> dict:
    """Write one sequence's tracker file and its timing, reusing them if already written."""
    timing = out.with_suffix(".json")
    if out.exists() and timing.exists():
        return runs.read_json(timing)

    if spec["impl"] == "boxmot-v25.0.0":
        # boxmot-v25.0.0 cannot share a process with the vendored boxmot.
        env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(BOXMOT_V25), str(REPO)]), "PYTHONUNBUFFERED": "1"}
        cmd = [sys.executable, __file__, "--worker", json.dumps(spec), seq.dataset, seq.split, seq.name, str(out)]
        # Relayed line by line so the worker's output reaches the run log too.
        with subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as proc:
            for line in proc.stdout:
                sys.stdout.write(line)
        if proc.returncode:
            raise subprocess.CalledProcessError(proc.returncode, cmd)
        return runs.read_json(timing)

    runner = {"mapsort": run_mapsort, "boxmot": run_boxmot}[spec["impl"]]
    started = time.perf_counter()
    tracks = runner(seq, spec)
    result = {"seconds": time.perf_counter() - started, "frames": len(tracks)}
    write_tracks(tracks, out)
    runs.write_json(timing, result)
    return result


def worker(argv: list[str]) -> None:
    spec, dataset, split, name, out = json.loads(argv[0]), *argv[1:]
    seq = sequence(dataset, split, name)
    started = time.perf_counter()
    tracks = run_sam2mot(seq, spec)
    seconds = time.perf_counter() - started
    write_tracks(tracks, Path(out))
    runs.write_json(Path(out).with_suffix(".json"), {"seconds": seconds, "frames": len(tracks)})


# ── run dir and log ────────────────────────────────────────────────────────────


def stamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def create_run_dir(config: dict) -> Path:
    """runs/eval-<UTC stamp>/ with its manifest, as mapsort.runs.create but eval-prefixed."""
    created = datetime.now(timezone.utc)
    run_dir = runs.RUNS_DIR / f"eval-{created.strftime('%Y%m%d-%H%M%S')}"
    (run_dir / "artifacts").mkdir(parents=True, exist_ok=False)
    runs.write_json(
        run_dir / "manifest.json",
        {"run_id": run_dir.name, "module": "eval", "created": created.isoformat(), "git": runs.git_state(), "config": config},
    )
    return run_dir


class Tee:
    """Mirror a stream into the run's log file."""

    def __init__(self, stream, log) -> None:
        self.stream, self.log = stream, log

    def write(self, text: str) -> int:
        self.log.write(text)
        self.log.flush()
        return self.stream.write(text)

    def flush(self) -> None:
        self.stream.flush()
        self.log.flush()

    def __getattr__(self, name: str):
        return getattr(self.stream, name)


# ── scoring ────────────────────────────────────────────────────────────────────


class Scorer:
    """Runs and scores (spec, dataset) pairs once each, however many tables use them."""

    def __init__(self, run_dir: Path, metrics_cfg: dict, no_run: bool = False) -> None:
        self.run_dir, self.cfg, self.cache, self.no_run = run_dir, metrics_cfg, {}, no_run

    def done(self, spec: dict, seqs: list[Sequence]) -> bool:
        base = self.run_dir / "artifacts" / spec_key(spec)
        return all((base / s.dataset / f"{s.name}.json").exists() for s in seqs)

    def files(self, spec: dict, seqs: list[Sequence]) -> dict[str, Path]:
        base = self.run_dir / "artifacts" / spec_key(spec)
        runs.write_json(base / "spec.json", spec)
        out = {}
        for seq in seqs:
            path = base / seq.dataset / f"{seq.name}.txt"
            reused = path.exists() and path.with_suffix(".json").exists()
            print(f"{stamp()}  {spec_key(spec):<22} {seq.dataset}/{seq.name} ...", flush=True)
            t = run(seq, spec, path)
            note = "reused" if reused else "ran"
            print(f"{stamp()}  {spec_key(spec):<22} {seq.dataset}/{seq.name} {note}: "
                  f"{t['frames']} frames in {t['seconds']:.1f}s", flush=True)
            out[seq.name] = path
        return out

    def score(self, spec: dict, seqs: list[Sequence]) -> dict | None:
        """None under --no-run when any of the sequences has no finished output yet."""
        from mapsort.eval.revisit import evaluate_revisits
        from mapsort.eval.tracking import evaluate

        key = (spec_key(spec), tuple(f"{s.dataset}/{s.name}" for s in seqs))
        if self.no_run and not self.done(spec, seqs):
            return None
        if key not in self.cache:
            files = self.files(spec, seqs)
            iou, preproc = self.cfg["tracking"]["iou_threshold"], self.cfg["tracking"]["do_preproc"]
            rev = self.cfg["revisit"]
            self.cache[key] = {
                "tracking": evaluate(seqs, files, thresholds=[iou], do_preproc=preproc),
                # window=0 skips the restricted metrics, which eval.yaml does not report.
                "revisit": evaluate_revisits(
                    seqs, files, threshold=iou, min_gap=rev["min_gap"], grace=rev["grace"], window=0, do_preproc=preproc
                ),
                "timing": {s.name: runs.read_json(files[s.name].with_suffix(".json")) for s in seqs},
            }
            name = key[0] + "_" + hashlib.sha1("|".join(key[1]).encode()).hexdigest()[:6]
            runs.write_json(self.run_dir / "metrics" / f"{name}.json", {"spec": spec, "sequences": key[1], **self.cache[key]})
        return self.cache[key]


def fmt(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def metric_row(tracking: dict, revisit: dict, iou: float) -> dict[str, str]:
    """The tab_main / tab_sequences columns from one (pooled or per-sequence) result."""
    sfx = suffix(iou)
    ce =100.0 * (tracking["IDs"] - tracking["GT_IDs"]) / tracking["GT_IDs"]
    rip = revisit["revisit_id_preservation"]
    return {
        "HOTA": fmt(tracking["HOTA"]),
        "AssA": fmt(tracking["AssA"]),
        "DetA": fmt(tracking["DetA"]),
        "MOTA": fmt(tracking[f"MOTA{sfx}"]),
        "IDF1": fmt(tracking[f"IDF1{sfx}"]),
        "IDSW": str(int(tracking[f"IDSW{sfx}"])),
        "CE": "0.0" if ce == 0 else f"{ce:+.1f}",
        "RIP": "n/a" if np.isnan(rip) else fmt(rip, 1),
    }


def pooled_rip(events: list[dict]) -> float:
    preserved = sum(e["outcome"] == "preserved" for e in events)
    decided = preserved + sum(e["outcome"] == "switch" for e in events)
    return 100.0 * preserved / decided if decided else float("nan")


def all_events(result: dict) -> list[dict]:
    return [e for evs in result["revisit"]["events"].values() for e in evs]


# ── tables ─────────────────────────────────────────────────────────────────────


def merge_csv(name: str, keys: list[str], rows: list[dict[str, str]]) -> None:
    """Update rows matching on `keys`, append the rest, keep every other row and column."""
    path = RESULTS / name
    header, existing = list(keys), []
    if path.exists():
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            header, existing = list(reader.fieldnames), list(reader)
    index = {tuple(r[k] for k in keys): r for r in existing}
    for row in rows:
        for col in row:
            if col not in header:
                header.append(col)
        k = tuple(row[c] for c in keys)
        if k in index:
            index[k].update(row)
        else:
            existing.append(row)
            index[k] = row
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header, restval="")
        writer.writeheader()
        writer.writerows(existing)
    print(f"wrote {path.relative_to(REPO)} ({len(rows)} rows)")


def gap_bin(gap: int, edges: list[int]) -> str | None:
    for lo, hi in zip(edges, edges[1:]):
        if lo <= gap < hi:
            return f"{lo}-{hi - 1}"
    return f">={edges[-1]}" if gap >= edges[-1] else None


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        return worker(sys.argv[2:])

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=REPO / "config" / "eval.yaml")
    ap.add_argument("--run", type=Path, help="existing run dir to resume")
    ap.add_argument("--tables", nargs="+", choices=TABLES, default=list(TABLES))
    ap.add_argument("--models", nargs="+", help="subset of eval.yaml models (default: all)")
    ap.add_argument("--datasets", nargs="+", help="subset of datasets.yaml datasets (default: all)")
    ap.add_argument("--no-run", action="store_true", help="score finished outputs only; skip the rest")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    if (REPO / cfg["data"]["config"]).resolve() != DEFAULT_REGISTRY.resolve():
        raise SystemExit(f"mapsort.data reads {DEFAULT_REGISTRY} only, not {cfg['data']['config']}")
    models, metrics = cfg["models"], cfg["metrics"]
    run_models = args.models or list(models)
    iou = metrics["tracking"]["iou_threshold"]

    run_dir = args.run or create_run_dir({"eval": cfg, "tables": args.tables})
    log = (run_dir / "eval.log").open("a")
    sys.stdout, sys.stderr = Tee(sys.stdout, log), Tee(sys.stderr, log)
    print(f"{stamp()}  {socket.gethostname()}  {' '.join(sys.argv)}  -> {run_dir}", flush=True)
    scorer = Scorer(run_dir, metrics, no_run=args.no_run)
    test, complete = {}, {}
    for ds, spec in registry().items():
        if args.datasets and ds not in args.datasets:
            continue
        test[ds] = present(ds, spec["splits"]["test"])
        complete[ds] = len(test[ds]) == len(spec["splits"]["test"])
        if not complete[ds]:
            print(f"{ds}: {len(test[ds])}/{len(spec['splits']['test'])} test sequences on this machine; "
                  "pooled tables (main, gap) skip it", flush=True)
    reps = {
        ds: seq for ds, seqs in test.items() if ds in REPRESENTATIVE
        for seq in seqs if seq.name == REPRESENTATIVE[ds][0]
    }
    main_specs = {name: model_spec(models, name) for name in run_models}
    if "mapsort" not in run_models:
        args.tables = [t for t in args.tables if t not in ("ablation", "gate")]

    if "main" in args.tables:
        rows = []
        for ds, seqs in test.items():
            if not complete[ds]:
                continue
            for name, spec in main_specs.items():
                r = scorer.score(spec, seqs)
                if r is None:
                    continue
                row = metric_row(r["tracking"]["combined"], r["revisit"]["combined"], iou)
                rows.append({"dataset": DATASET_NAMES[ds], "tracker": MODEL_NAMES[name], **row})
        merge_csv("tab_main.csv", ["dataset", "tracker"], rows)

    if "sequences" in args.tables:
        rows = []
        for ds, seq in reps.items():
            for name, spec in main_specs.items():
                # Shared with tab_main; per-sequence scores do not depend on the rest of the split.
                r = scorer.score(spec, test[ds]) or scorer.score(spec, [seq])
                if r is None:
                    continue
                row = metric_row(
                    r["tracking"]["per_sequence"][seq.name], r["revisit"]["per_sequence"][seq.name], iou
                )
                rows.append({"sequence": f"{DATASET_NAMES[ds]} {seq.name}", "tracker": MODEL_NAMES[name], **row})
        merge_csv("tab_sequences.csv", ["sequence", "tracker"], rows)

    if "ablation" in args.tables:
        rows = []
        for variant, override in ABLATIONS.items():
            spec, row, events = mapsort_variant(models, override), {"variant": variant}, []
            results = {ds: scorer.score(spec, [seq]) for ds, seq in reps.items()}
            for ds, r in results.items():
                if r is None:
                    continue
                prefix = REPRESENTATIVE[ds][1]
                row[f"{prefix}_HOTA"] = fmt(r["tracking"]["combined"]["HOTA"])
                row[f"{prefix}_IDF1"] = fmt(r["tracking"]["combined"][f"IDF1{suffix(iou)}"])
                events += all_events(r)
            if all(results.values()):
                rip = pooled_rip(events)
                row["RIP"] = "n/a" if np.isnan(rip) else fmt(rip, 1)
            rows.append(row)
        merge_csv("tab_ablation.csv", ["variant"], rows)

    if "gate" in args.tables:
        rows = []
        for gate in GATES:
            spec = mapsort_variant(models, {"hp": {"landmarks": {"gate": gate}}})
            for ds, seq in reps.items():
                r = scorer.score(spec, [seq])
                if r is None:
                    continue
                rip = r["revisit"]["combined"]["revisit_id_preservation"]
                rows.append({
                    "sequence": f"{DATASET_NAMES[ds]} {seq.name}",
                    "gate": f"{gate:g}",
                    "HOTA": fmt(r["tracking"]["combined"]["HOTA"]),
                    "RIP": "n/a" if np.isnan(rip) else fmt(rip, 1),
                })
        merge_csv("fig_gate.csv", ["sequence", "gate"], rows)

    if "gap" in args.tables and not all(complete.values()):
        print("gap: skipped, pooled over every test sequence")
    elif "gap" in args.tables:
        edges, rows = metrics["revisit"]["gap_bins"], []
        for name, spec in main_specs.items():
            results = [scorer.score(spec, seqs) for seqs in test.values()]
            if not all(results):
                continue  # pooled over every test sequence or not at all
            events = [e for r in results for e in all_events(r)]
            for b in [gap_bin(lo, edges) for lo in edges]:
                inside = [e for e in events if gap_bin(e["gap"], edges) == b]
                rip = pooled_rip(inside)
                rows.append({
                    "tracker": MODEL_NAMES[name],
                    "gap_bin": b,
                    "events": str(len(inside)),
                    "preserved": str(sum(e["outcome"] == "preserved" for e in inside)),
                    "switches": str(sum(e["outcome"] == "switch" for e in inside)),
                    "RIP": "n/a" if np.isnan(rip) else fmt(rip, 1),
                })
        merge_csv("fig_gap.csv", ["tracker", "gap_bin"], rows)

    if "runtime" in args.tables:
        rows = []
        for ds, seq in reps.items():
            row = {"sequence": f"{DATASET_NAMES[ds]} {seq.name}", "hardware": HARDWARE.get(socket.gethostname(), "?")}
            for name in run_models:
                r = scorer.score(model_spec(models, name, live=True), [seq])
                if r is None:
                    continue
                t = r["timing"][seq.name]
                row[MODEL_NAMES[name]] = fmt(t["frames"] / t["seconds"])
            rows.append(row)
        merge_csv("tab_runtime.csv", ["sequence"], rows)

    print(f"run dir: {run_dir}")


if __name__ == "__main__":
    main()
