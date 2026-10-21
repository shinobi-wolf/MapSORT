# MapSORT: Landmark Association for Identity-Persistent Tracking of Stationary Crops and Fruit

Precision agriculture needs multi-object tracking that keeps one identity per plant or fruit through occlusion, visual ambiguity, and a camera that leaves an object and later returns to it. Frame-to-frame trackers such as BoT-SORT lose identity there by construction: crops look alike, and a track unmatched for more than a few dozen frames is deleted. MapSORT exploits the one structural fact these methods ignore: agricultural produce doesn't move.

MapSORT is BoT-SORT plus a scene map. A structure-from-motion reconstruction of the sequence gives every detection a *landmark observation*, a Gaussian over its centre in the map frame, and every track fuses its observations into a landmark. Each frame, the pairs BoT-SORT has just matched supply landmark-to-box correspondences, and a robust resection (RANSAC, full DLT camera or a plane homography for near-coplanar scenes) recovers the frame's camera from them. With that camera every stationary object, tracked, lost or long gone, has a predicted position in the image. MapSORT uses it twice:

- **Map motion:** reliable tracks that BoT-SORT's first stage left unmatched are re-matched where the camera places them.
- **Revival:** lost tracks, and tracks kept *dormant* instead of being deleted, are revived where their landmark projects. Prediction boxes are clipped to the frame, so an object re-entering at the edge can be revived, and a new track at the edge waits a few frames for that revival before it is reported.

When no camera can be fitted, revival falls back to a Mahalanobis gate on the 3D landmarks. One configuration serves every dataset. The scene map is currently built offline with COLMAP; an online mapper (CUT3R) is the planned next geometry source.

On the paper's test data (LettuceMOTS, MOrangeT, GrapeMOTS, AppleMOTS; 21 sequences), MapSORT cuts identity switches from 1154 (BoT-SORT) to 217, preserves identity in 92.0% of 1043 revisit events against 9.1%, and has fewer identity switches than BoT-SORT on 19 sequences and as few on the other two. Full results: [docs/precision-ag-paper/results/](docs/precision-ag-paper/results/).

---

# Repo guide

Read [CLAUDE.md](CLAUDE.md) for working rules.

## Pipeline

```
I_t ─> Detector ─> D_t ─┬─> Tracker ─> T (running tracks)
                        │      ^
                        └─> Mapping ─> P_t (landmark observations)
```

[mapsort/main.py](mapsort/main.py) wires the modules. `MapSORT.step(frame_idx, frame, detections=None)` advances one frame; passing `detections` bypasses the Detector (how stored detections are fed in). `run(frames, detections)` loops `step`.

Inside the Tracker, one frame is:

1. BoT-SORT's own association (Kalman prediction with camera-motion compensation, fused IoU and appearance).
2. Camera resection from its matches plus mutual-nearest landmark pairs.
3. Map motion: unmatched reliable tracks re-solved at their projected position.
4. Revival of lost and dormant tracks by clipped reprojection error (or by the 3D gate without a camera).
5. New tracks; those at the frame edge where a revival is expected wait before confirming.

## Layout

| Path | What |
|---|---|
| [mapsort/main.py](mapsort/main.py) | `MapSORT`: config, modules, per-frame loop. |
| [mapsort/detector.py](mapsort/detector.py) | `Detector`: Ultralytics YOLO wrapper. |
| [mapsort/mapping.py](mapsort/mapping.py) | `Mapping`: online landmark estimator. **Stub**; landmarks are read from `landmarks/<seq>/sfm.npz` instead. |
| [mapsort/tracker.py](mapsort/tracker.py) | `Tracker`: `MapSort`, boxmot `BotSort` plus the camera, map-motion and revival stages; `landmarks.enabled: false` is plain BoT-SORT. |
| [mapsort/camera.py](mapsort/camera.py) | Per-frame camera resection: RANSAC over a DLT camera and a plane homography. |
| [mapsort/association.py](mapsort/association.py) | `LandmarkState` (information-form fusion) and the Mahalanobis/Euclidean landmark cost. |
| [mapsort/schemas.py](mapsort/schemas.py) | `Detection`, `LandmarkObservation`, `Track`. |
| [mapsort/data.py](mapsort/data.py) | Dataset access from [config/datasets.yaml](config/datasets.yaml): `sequence(dataset, split, name)`. |
| [mapsort/run.py](mapsort/run.py) | The `mapsort` command: run a split, `--eval` to score, `--viz` to watch. |
| [mapsort/viz.py](mapsort/viz.py) | Track overlays and the imshow window. |
| [mapsort/runs.py](mapsort/runs.py) | Run directories: `runs/<stamp>_<module>/`. |
| [mapsort/eval/](mapsort/eval/) | Scores run artifacts from disk; never imports the pipeline. |
| [config/](config/) | `mapsort.yaml` (`run` flag defaults, `pipeline` module kwargs, `protocol`), `eval.yaml` (paper evaluation), `datasets.yaml`, `landmark_filters.yaml`, `recon_plan.yaml`. |
| [scripts/data/](scripts/data/) | Builds `mot/`, `dets/` and `masks/` from each dataset's native annotations. |
| [scripts/colmap/](scripts/colmap/) | The reconstruction queue that builds `recon/`. |
| [scripts/landmarks/](scripts/landmarks/) | `sfm_mapping.py` extracts landmark observations from a COLMAP model. |
| [scripts/reid/](scripts/reid/) | `reid_cache.py` caches per-detection ReID embeddings. |
| [scripts/eval/](scripts/eval/) | `eval.py` runs every model over the test data; `fill_paper.py` copies the result CSVs into the paper. |
| [scripts/viz/](scripts/viz/) | `compare_runs.py`: GT and tracker panels side by side, as a video. |
| [docs/precision-ag-paper/](docs/precision-ag-paper/) | The paper, its result CSVs and review comments. |
| [lab/](lab/) | Scratch notebooks. |
| [docs/problem.md](docs/problem.md) | Problem statement and evaluation protocol. |

Progress is tracked in the GitHub project (MapSORT, #5).

## External dependencies

Git-ignored: `boxmot/` (vendored upstream master, editable install), `boxmot-v25.0.0/` (only for the SAM2MOT baseline; see [CLAUDE.md](CLAUDE.md)), `CUT3R/` (symlink to `../CUT3R`), `models/` (weights). Reference projects under `~/IISc/`: `MapSORT/`, `TrackEval/`.

## Data

[data/](data/) is git-ignored except [data/README.md](data/README.md), which covers the layout, the landmark and ReID cache formats, per-dataset notes and setup on a new machine. Root is `$MAPSORT_DATA` if set, else `./data`.

| Dataset | Test sequences | Notes |
|---|---|---|
| LettuceMOTS | 0003, 0007, 0011 | Field robot, nadir; every plant seen twice. |
| MOrangeT | V01, V02, V05 to V12 | Handheld sweeps of orange trees; ships COLMAP models and 3D fruit GT. |
| GrapeMOTS | PathPlanning_1, _2, _3, _7 | UAV, multi-perspective; 4K for PP2 and PP7. |
| AppleMOTS | 0006, 0007, 0008, 0010 | Dense apples, no revisits; 0011 and 0012 duplicate 0007 and 0008 and are excluded. |

Every test sequence has stored detections (`dets/`, currently derived from GT until per-dataset detectors are trained), a COLMAP reconstruction, `sfm` and `sfm_point` landmarks and cached OSNet embeddings. Stock `yolov8n.pt` detects no crops, so runs use `--detections stored`.

## Evaluation

Metrics come from **TrackEval** (`pip install --no-deps trackeval==1.3.0`; `--no-deps` because its metadata wants numpy ≥ 2.3 and this env pins 1.26). Don't use the clone at `~/IISc/TrackEval`, which uses `np.float`.

`CLEAR` gives MOTA and IDSW, `HOTA` gives HOTA, AssA and DetA, `Identity` gives IDF1, all at IoU 0.5. Two more address agricultural use:

- **Count error (CE):** signed relative error of the number of identities issued, the quantity yield estimation consumes. Average |CE| per sequence rather than pooling signed CE, which cancels.
- **Revisit identity preservation (RIP):** [mapsort/eval/revisit.py](mapsort/eval/revisit.py). A revisit event is a GT object reappearing after at least `min_gap` (10) frames out of view; it is `preserved`, `switch` or `unmatched`, read from CLEAR's own per-frame assignment within a `grace` window (5 frames) that allows for track confirmation. The switch counts are a subset of CLEAR's IDSW. RIP = preserved / (preserved + switch). AppleMOTS has no revisit events, and GrapeMOTS only in PathPlanning_7.

### Paper evaluation

[scripts/eval/eval.py](scripts/eval/eval.py) runs the models in [config/eval.yaml](config/eval.yaml) (BoT-SORT, SAM2MOT, MapSORT) over every test sequence and writes the paper's tables to `docs/precision-ag-paper/results/*.csv`:

```bash
python scripts/eval/eval.py                                    # all models, all tables, new run dir
python scripts/eval/eval.py --models mapsort --tables main     # a subset
python scripts/eval/eval.py --run runs/eval-<stamp>            # resume: finished sequences are reused
python scripts/eval/fill_paper.py                              # CSVs -> sn-article.tex tables and plots
```

Tables: `main` (pooled per dataset), `sequences` (one representative sequence each), `ablation`, `gate` (landmark-gate sweep), `gap` (RIP by absence length, pooled over all test data; needs every dataset complete), `runtime`. A run lives in `runs/eval-<UTC stamp>/` with `manifest.json` (commit, dirty flag, config), `eval.log`, `metrics/` and `artifacts/<spec>/<dataset>/<seq>.txt`; only the manifest and metrics are committed. Identical (tracker, hp, landmark source) specs share artifacts across tables, so resuming or adding tables never re-runs a finished sequence. Run with a clean tree so the manifest pins the code.

SAM2MOT runs in a subprocess from `boxmot-v25.0.0` and takes the stored instance masks in place of SAM 2. It is slow on CPU (hours on 4K GrapeMOTS); split it across processes with `--run` and `--datasets`.

### Quick runs

```bash
mapsort --split test --detections stored --eval   # run + score
python -m mapsort.eval --run runs/<run_id>                   # re-score a run
python -m mapsort.eval --tracker-dir path/to/results         # score external output
```

A run writes `runs/<stamp>_tracker/` with `manifest.json`, `artifacts/<seq>.txt` and, with `--eval`, `metrics.json`. `--no-save` discards everything after printing metrics.

### Watching a run

```bash
mapsort --seqs 0011 --viz --viz-compare --no-save   # GT | prediction
mapsort --seqs 0011 --viz-save overlay.mp4          # no display needed
python scripts/viz/compare_runs.py --dataset moranget --seq V01 \
    --runs runs/eval-<stamp>/artifacts/<spec>/moranget/V01.txt:MapSORT runs/eval-<stamp>/artifacts/<spec>/moranget/V01.txt:BoT-SORT
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

## Conventions

- Conda env `nerfly`.
- Frames are RGB `np.ndarray` (H, W, 3, uint8); Detector and Tracker flip to BGR internally.
- Bboxes are `xyxy` in image coordinates. Landmarks live in one map frame `M` of arbitrary scale: a mean, a 3x3 covariance over that centre, and a separate 3x3 extent for the object's size.
- Tracker settings are identical across datasets; only the landmark point filters ([config/landmark_filters.yaml](config/landmark_filters.yaml)) vary.

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
| Detector | YOLO wrapper works; per-dataset crop weights not trained yet, so stored detections are used. |
| Tracker | Working: BoT-SORT plus the per-frame camera, map motion and edge-aware revival of lost and dormant tracks. |
| Mapping | Stub. Landmarks come from offline COLMAP reconstructions (`sfm.npz`); CUT3R ran on LettuceMOTS 0011 but is not integrated. |
| Evaluation | Tracking, count and revisit metrics on all four datasets; paper tables generated by `scripts/eval/`. MangoYield (yield counts only) not added. |
| Tests | Paused; `tests/` is stale. |

Next: per-dataset YOLO detectors and fine-tuned ReID, MangoYield yield counts, and a `Mapping` backend that produces landmarks online (CUT3R) in place of the offline SfM map.
