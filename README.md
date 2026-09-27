# LocalizeSORTv2: Online World-Coordinate Association for Real-Time Stationary Object Tracking

LocalizeSORT improved multi-object tracking in precision agriculture by augmenting appearance- and motion-based association with an explicit world-coordinate (WC) metric, exploiting the stationary nature of agricultural produce to substantially reduce ID switches and improve yield-counting accuracy over DeepSORT and StrongSORT. However, LocalizeSORT's WC signal depends on RLoc, an offline localization module that requires Structure-from-Motion reconstruction over a complete video sequence, restricting the method to post-hoc analysis and precluding real-time deployment. We propose LocalizeSORTv2, which replaces RLoc with an online, feed-forward pointmap-based localization module that recovers per-frame camera pose and per-pixel 3D scene geometry causally, directly from monocular RGB video, without offline optimization over the full sequence. The rest of the LocalizeSORT pipeline — its weighted appearance/motion/WC association cost and Kalman-filter-based track management — is held fixed, isolating a single variable: whether the identity-persistence gains driven by the WC metric survive the transition from globally-optimized, whole-sequence geometry to causally-estimated, frame-by-frame geometry. We evaluate LocalizeSORTv2 against LocalizeSORT on sequences with existing offline SfM reconstructions, comparing online- and offline-derived world coordinates under identical detection and association settings, and report downstream tracking metrics (MOTA, HOTA, IDF1, ID switches) alongside a direct analysis of localization drift, with particular attention to re-identification scenarios in which the camera revisits a previously observed location. We hypothesize that a sufficiently drift-controlled online localization module preserves most of LocalizeSORT's identity-persistence gains while enabling real-time operation, closing the gap toward practical deployment in precision agriculture tasks — targeted spraying, selective harvesting, and yield estimation — that require tracking to run live rather than after the fact.


---

# Repo guide

Read [CLAUDE.md](CLAUDE.md) for working rules.

## Pipeline

```
I_t ─> Detector ─> D_t ─┬─> Tracker ─> T (running tracks)
                        │      ^
                        └─> Localization ─> P_t (map-frame poses)
```

[localizesort/main.py](localizesort/main.py) wires the modules. `LocalizeSORT.step(frame_idx, frame, detections=None)` advances one frame; passing `detections` bypasses the Detector (how GT boxes are fed in). `run(frames, detections)` loops `step`.

## Layout

| Path | What |
|---|---|
| [localizesort/main.py](localizesort/main.py) | `LocalizeSORT`: config, modules, per-frame loop. |
| [localizesort/detector.py](localizesort/detector.py) | `Detector`: Ultralytics YOLO wrapper. |
| [localizesort/localization.py](localizesort/localization.py) | `Localization`: online pose estimator. **Stub** until the CUT3R module lands. |
| [localizesort/tracker.py](localizesort/tracker.py) | `Tracker`: boxmot `StrongSort`. Ignores poses for now. |
| [localizesort/schemas.py](localizesort/schemas.py) | `Detection`, `ObjectPose`, `Track`. |
| [localizesort/data.py](localizesort/data.py) | Dataset access from [config/datasets.yaml](config/datasets.yaml): `sequence(dataset, split, name)`. |
| [localizesort/run.py](localizesort/run.py) | The `localizesort` command: run a split, `--eval` to score, `--viz` to watch. |
| [localizesort/viz.py](localizesort/viz.py) | Track overlays and the imshow window. |
| [localizesort/runs.py](localizesort/runs.py) | Run directories: `runs/<stamp>_<module>/`. |
| [localizesort/eval/](localizesort/eval/) | Scores run artifacts from disk; never imports the pipeline. |
| [config/](config/) | `localizesort.yaml` (module kwargs), `datasets.yaml`, `eval.yaml`, `run.yaml` (CLI flag defaults; flags override). |
| [scripts/data/](scripts/data/) | Builds `mot/` and `dets/` from a dataset's native annotations. |
| [lab/](lab/) | Scratch notebook. |
| [docs/problem.md](docs/problem.md) | Problem statement and evaluation protocol. |

Progress is tracked in the GitHub project (LocalizeSORT, #5).

## External dependencies

Git-ignored: `boxmot/` (vendored, editable), `CUT3R/` (symlink to `../CUT3R`), `models/` (weights). Reference projects under `~/IISc/`: `LocalizeSORT/`, `TrackEval/`.

## Data

[data/](data/) is git-ignored except [data/README.md](data/README.md) (layout, setup on a new machine). Root is `$LOCALIZESORT_DATA` if set, else `./data`; [config/datasets.yaml](config/datasets.yaml) lists splits and which sequences have dets, pose and recon.

```
data/<dataset>/
  raw                      symlink -> the original download; never written to
  mot/<split>/<seq>/       canonical MOT view: img1/, gt/gt.txt, seqinfo.ini
  dets/<seq>.txt           detections, MOT det format
  pose/<seq>.txt           object world coordinates, one row per detection
  recon/<seq>/             offline SfM reconstruction (RLoc-style pseudo-GT)
```

LettuceMOTS: all 12 sequences have GT and stored detections, only 0011 has a reconstruction. GT is MOT (top-left `xywh`, 1-based frames); `Detection.bbox` is `xyxy`. Stock `yolov8n.pt` detects no lettuce, so use `--detections stored`. Outputs go in `runs/<run_id>/artifacts/`.

## Evaluation

Metrics come from **TrackEval** (`pip install --no-deps trackeval==1.3.0`; `--no-deps` because its metadata wants numpy ≥ 2.3 and this env pins 1.26). Don't use the clone at `~/IISc/TrackEval`, which uses `np.float`.

`CLEAR` gives MOTA and IDSW, `HOTA` gives HOTA and HOTA(0), `Identity` gives IDF1. Fields at the primary threshold (0.5) keep plain names, others are suffixed (`MOTA@0.25`). The protocol is frozen in [config/eval.yaml](config/eval.yaml).

```bash
localizesort --split test --detections stored --eval   # run + score
python -m localizesort.eval --run runs/<run_id>                   # re-score a run
python -m localizesort.eval --tracker-dir path/to/results         # score external output
```

A run writes `runs/<stamp>_tracker/` with `manifest.json`, `artifacts/<seq>.txt` and, with `--eval`, `metrics.json`. `--no-save` discards everything after printing metrics. `--detections stored` reads `data/<dataset>/dets/<seq>.txt`, isolating association quality; `live` runs the Detector, which finds no lettuce yet.

### Watching a run

```bash
localizesort --seqs 0011 --viz --viz-compare --no-save   # GT | prediction
localizesort --seqs 0011 --viz-save overlay.mp4          # no display needed
```

- Default colours: **black** = prediction matches a GT object (one-to-one Hungarian, IoU ≥ 0.5), **red** = false positive.
- `--viz-compare`: GT and prediction in side-by-side panels.
- `--limit N` or quitting early cuts GT to the frames that ran.

### Eval code

| File | What |
|---|---|
| [eval/tracking.py](localizesort/eval/tracking.py) | Stages files into TrackEval's layout, flattens its output. |
| [eval/revisit.py](localizesort/eval/revisit.py) | Revisit-restricted identity metrics (not in TrackEval or boxmot). |
| [eval/mot_io.py](localizesort/eval/mot_io.py) | MOT-format read, write, frame filtering. |
| [eval/protocol.py](localizesort/eval/protocol.py) | Loads `eval.yaml`. |
| [eval/\_\_main\_\_.py](localizesort/eval/__main__.py) | Re-scoring CLI. |

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
- Bboxes are `xyxy` in image coordinates. Poses are `p_t^i in R^3` in one map frame `M` with a 3x3 covariance.

## Running

```bash
conda activate nerfly
pip install -e .            # boxmot is installed editable separately
pip install --no-deps trackeval==1.3.0 tabulate
localizesort --split test --detections stored --eval
```

## Status

| Module | State |
|---|---|
| Detector | Working (YOLOv8); needs crop-specific weights. |
| Tracker | Working (StrongSORT); ignores poses. |
| Localization | Stub. CUT3R ran offline on LettuceMOTS/0011, not integrated. |
| Evaluation | Tracking metrics done; localization metrics not started (only 0011 has a reconstruction). |
| Data | All 12 LettuceMOTS sequences in the MOT view, dets from GT. LFSD and mango not added. |
| Tests | Paused; `tests/` is stale. |

Next: integrate CUT3R as `Localization`, then add localization metrics (Umeyama alignment via `evo`, position error, revisit drift, NEES) from [docs/problem.md](docs/problem.md).
