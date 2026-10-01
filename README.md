# MapSORT: Online Landmark Association for Real-Time Stationary Object Tracking

LocalizeSORT improved multi-object tracking in precision agriculture by augmenting appearance- and motion-based association with an explicit world-coordinate metric, exploiting the stationary nature of agricultural produce to substantially reduce ID switches and improve yield-counting accuracy over DeepSORT and StrongSORT. That signal depends on RLoc, an offline mapping module requiring Structure-from-Motion over a complete video sequence, which restricts the method to post-hoc analysis and precludes real-time deployment. We propose MapSORT, which makes the geometric cue both online and probabilistic. First, RLoc is replaced by a feed-forward pointmap module that recovers per-frame camera pose and per-pixel 3D geometry causally from monocular RGB, without optimization over the full sequence. Second, where RLoc reduced each detection to the single triangulated keypoint nearest its bounding-box centre, MapSORT emits a *landmark observation* — a Gaussian over the object's map-frame position, carrying the uncertainty about its centre separately from the object's physical extent. Tracks fuse these observations into persistent landmarks whose centre estimate sharpens with every sighting, so association becomes a Mahalanobis gate that is invariant to the arbitrary scale of a monocular reconstruction, removing the hand-tuned distance threshold the original method depended on. Because a stationary object's map-frame position is unchanged by an arbitrarily long absence, a track may also be revived after the frame-to-frame tracker would have deleted it — the regime in which appearance galleries have decayed and image-space motion prediction is worthless. We evaluate MapSORT against LocalizeSORT on sequences with existing offline SfM reconstructions, holding detection and appearance fixed and varying the geometry source, and report MOTA, HOTA, IDF1 and ID switches alongside a direct analysis of mapping drift and of whether the landmark covariance is calibrated, with particular attention to re-identification when the camera revisits a previously observed location. We hypothesize that a sufficiently drift-controlled online mapping module preserves most of LocalizeSORT's identity-persistence gains while enabling real-time operation, closing the gap toward practical deployment in precision agriculture tasks — targeted spraying, selective harvesting, and yield estimation — that require tracking to run live rather than after the fact.


---

# Repo guide

Read [CLAUDE.md](CLAUDE.md) for working rules.

## Pipeline

```
I_t ─> Detector ─> D_t ─┬─> Tracker ─> T (running tracks)
                        │      ^
                        └─> Mapping ─> P_t (landmark observations)
```

[mapsort/main.py](mapsort/main.py) wires the modules. `MapSORT.step(frame_idx, frame, detections=None)` advances one frame; passing `detections` bypasses the Detector (how GT boxes are fed in). `run(frames, detections)` loops `step`.

## Layout

| Path | What |
|---|---|
| [mapsort/main.py](mapsort/main.py) | `MapSORT`: config, modules, per-frame loop. |
| [mapsort/detector.py](mapsort/detector.py) | `Detector`: Ultralytics YOLO wrapper. |
| [mapsort/mapping.py](mapsort/mapping.py) | `Mapping`: online landmark estimator. **Stub** — the `sfm` backend's data exists on disk but nothing reads it yet. |
| [mapsort/tracker.py](mapsort/tracker.py) | `Tracker`: boxmot `StrongSort`. Ignores landmarks for now. |
| [mapsort/schemas.py](mapsort/schemas.py) | `Detection`, `LandmarkObservation`, `Track`. |
| [mapsort/data.py](mapsort/data.py) | Dataset access from [config/datasets.yaml](config/datasets.yaml): `sequence(dataset, split, name)`. |
| [mapsort/run.py](mapsort/run.py) | The `mapsort` command: run a split, `--eval` to score, `--viz` to watch. |
| [mapsort/viz.py](mapsort/viz.py) | Track overlays and the imshow window. |
| [mapsort/runs.py](mapsort/runs.py) | Run directories: `runs/<stamp>_<module>/`. |
| [mapsort/eval/](mapsort/eval/) | Scores run artifacts from disk; never imports the pipeline. |
| [config/](config/) | `mapsort.yaml`: one file, three sections -- `run` (CLI flag defaults; flags override), `pipeline` (module kwargs), `protocol` (scoring). Plus `datasets.yaml`. |
| [scripts/data/](scripts/data/) | Builds `mot/` and `dets/` from a dataset's native annotations. |
| [scripts/landmarks/](scripts/landmarks/) | Builds `landmarks/` — `sfm_mapping.py` extracts them from a COLMAP model. |
| [lab/](lab/) | Scratch notebook. |
| [docs/problem.md](docs/problem.md) | Problem statement and evaluation protocol. |

Progress is tracked in the GitHub project (MapSORT, #5).

## External dependencies

Git-ignored: `boxmot/` (vendored, editable), `CUT3R/` (symlink to `../CUT3R`), `models/` (weights). Reference projects under `~/IISc/`: `MapSORT/`, `TrackEval/`.

## Data

[data/](data/) is git-ignored except [data/README.md](data/README.md) (layout, setup on a new machine). Root is `$MAPSORT_DATA` if set, else `./data`; [config/datasets.yaml](config/datasets.yaml) lists splits and which sequences have dets, landmarks and recon.

```
data/<dataset>/
  raw                      symlink -> the original download; never written to
  mot/<split>/<seq>/       canonical MOT view: img1/, gt/gt.txt, seqinfo.ini
  dets/<seq>.txt           detections, MOT det format
  landmarks/<seq>/<backend>.npz  per-detection landmark observations
  recon/<seq>/             offline SfM reconstruction (RLoc-style pseudo-GT)
```

A **landmark observation** is a Gaussian over an object's map-frame position —
mean, a covariance for the uncertainty about that centre, and a separate
`extent` for the object's physical spread. `sfm_mapping.py` builds one per
detection from the COLMAP model: the triangulated keypoints inside the bbox,
minus background points rejected by camera-frame depth. Detections enclosing no
usable point carry `has_landmark = False` and degrade to appearance-only
association. Array schema in [data/README.md](data/README.md).

LettuceMOTS: all 12 sequences have GT and stored detections, only 0011 has a reconstruction. GT is MOT (top-left `xywh`, 1-based frames); `Detection.bbox` is `xyxy`. Stock `yolov8n.pt` detects no lettuce, so use `--detections stored`. Outputs go in `runs/<run_id>/artifacts/`.

## Evaluation

Metrics come from **TrackEval** (`pip install --no-deps trackeval==1.3.0`; `--no-deps` because its metadata wants numpy ≥ 2.3 and this env pins 1.26). Don't use the clone at `~/IISc/TrackEval`, which uses `np.float`.

`CLEAR` gives MOTA and IDSW, `HOTA` gives HOTA and HOTA(0), `Identity` gives IDF1. Fields at the primary threshold (0.5) keep plain names, others are suffixed (`MOTA@0.25`). The protocol is frozen in the `protocol` section of [config/mapsort.yaml](config/mapsort.yaml).

```bash
mapsort --split test --detections stored --eval   # run + score
python -m mapsort.eval --run runs/<run_id>                   # re-score a run
python -m mapsort.eval --tracker-dir path/to/results         # score external output
```

A run writes `runs/<stamp>_tracker/` with `manifest.json`, `artifacts/<seq>.txt` and, with `--eval`, `metrics.json`. `--no-save` discards everything after printing metrics. `--detections stored` reads `data/<dataset>/dets/<seq>.txt`, isolating association quality; `live` runs the Detector, which finds no lettuce yet.

### Watching a run

```bash
mapsort --seqs 0011 --viz --viz-compare --no-save   # GT | prediction
mapsort --seqs 0011 --viz-save overlay.mp4          # no display needed
```

- Default colours: **black** = prediction matches a GT object (one-to-one Hungarian, IoU ≥ 0.5), **red** = false positive.
- `--viz-compare`: GT and prediction in side-by-side panels.
- `--limit N` or quitting early cuts GT to the frames that ran.

### Eval code

| File | What |
|---|---|
| [eval/tracking.py](mapsort/eval/tracking.py) | Stages files into TrackEval's layout, flattens its output. |
| [eval/revisit.py](mapsort/eval/revisit.py) | Revisit-restricted identity metrics (not in TrackEval or boxmot). |
| [eval/mot_io.py](mapsort/eval/mot_io.py) | MOT-format read, write, frame filtering. |
| [eval/protocol.py](mapsort/eval/protocol.py) | Loads the `protocol` section of `mapsort.yaml`. |
| [eval/\_\_main\_\_.py](mapsort/eval/__main__.py) | Re-scoring CLI. |

**Revisit metrics** ask whether an object's identity survives the camera returning after a gap. Each event is `preserved`, `switch` or `unmatched`; the counts decompose CLEAR's `IDSW` (at `min_gap=0, grace=1` the switch count equals it). `min_gap` (10) is the absence needed to count as a revisit; `grace` (5) is the re-acquisition window, needed because StrongSORT confirms a track only after `n_init=3` hits.

### Baseline

StrongSORT with stored (GT-derived) detections, test split:

| seq | MOTA | HOTA | IDF1 | IDSW | IDs | GT_IDs |
|---|---|---|---|---|---|---|
| 0003 | 86.77 | 56.89 | 45.94 | 120 | 116 | 40 |
| 0007 | 92.85 | 75.36 | 62.47 | 24 | 60 | 37 |
| 0011 | 92.97 | 76.33 | 64.03 | 23 | 60 | 37 |
| **combined** | **90.82** | **69.97** | **57.41** | **167** | 236 | 114 |

Of 70 revisit events, **2 preserved identity (2.9%)**: 236 predicted identities for 114 real ones. Detection is near-perfect and association fails; 2.9% is the number to beat.

Checks: re-scoring the external StrongSORT result for 0011 matches `eval_results/0011_metrics.json` exactly, and the run is within ~1 point of the boxmot v22 baseline (HOTA 68.94, MOTA 90.73, IDF1 57.62, IDSW 166).

`MOTA@0.25` equals `MOTA` here because GT detections match GT boxes at IoU ≈ 1. The `restricted` block measures local consistency within revisit windows, not persistence across the gap (fresh ids still score 97%); the event counts are the real metric.

## Conventions

- Conda env `nerfly`.
- Frames are RGB `np.ndarray` (H, W, 3, uint8); Detector and Tracker flip to BGR internally.
- Causality: frame `t` may use frames `1..t` only ([docs/problem.md](docs/problem.md)).
- Bboxes are `xyxy` in image coordinates. Landmarks are `p_t^i in R^3` in one map frame `M`, with a 3x3 covariance over that position and a separate 3x3 extent for the object's size. The map frame's scale is arbitrary.

## Running

```bash
conda activate nerfly
pip install -e .            # boxmot is installed editable separately
pip install --no-deps trackeval==1.3.0 tabulate
mapsort --split test --detections stored --eval
```

## Status

| Module | State |
|---|---|
| Detector | Working (YOLOv8); needs crop-specific weights. |
| Tracker | Working (StrongSORT); ignores landmarks. |
| Mapping | Stub. `sfm` landmarks built for 0011 from COLMAP; no backend reads them yet. CUT3R ran offline on 0011, not integrated. |
| Evaluation | Tracking metrics done; mapping metrics not started (only 0011 has a reconstruction). |
| Data | All 12 LettuceMOTS sequences in the MOT view, dets from GT. LFSD and mango not added. |
| Tests | Paused; `tests/` is stale. |

Next: give `Mapping` an `sfm` backend that reads `landmarks/<seq>/sfm.npz`, with a
backend switch so CUT3R can slot in as the online one. Then landmark association
in the Tracker, and mapping metrics (Umeyama alignment via `evo`, position error,
revisit drift, NEES) from [docs/problem.md](docs/problem.md).
