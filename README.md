# LocalizeSORTv2: Online World-Coordinate Association for Real-Time Stationary Object Tracking

LocalizeSORT improved multi-object tracking in precision agriculture by augmenting appearance- and motion-based association with an explicit world-coordinate (WC) metric, exploiting the stationary nature of agricultural produce to substantially reduce ID switches and improve yield-counting accuracy over DeepSORT and StrongSORT. However, LocalizeSORT's WC signal depends on RLoc, an offline localization module that requires Structure-from-Motion reconstruction over a complete video sequence, restricting the method to post-hoc analysis and precluding real-time deployment. We propose LocalizeSORTv2, which replaces RLoc with an online, feed-forward pointmap-based localization module that recovers per-frame camera pose and per-pixel 3D scene geometry causally, directly from monocular RGB video, without offline optimization over the full sequence. The rest of the LocalizeSORT pipeline — its weighted appearance/motion/WC association cost and Kalman-filter-based track management — is held fixed, isolating a single variable: whether the identity-persistence gains driven by the WC metric survive the transition from globally-optimized, whole-sequence geometry to causally-estimated, frame-by-frame geometry. We evaluate LocalizeSORTv2 against LocalizeSORT on sequences with existing offline SfM reconstructions, comparing online- and offline-derived world coordinates under identical detection and association settings, and report downstream tracking metrics (MOTA, HOTA, IDF1, ID switches) alongside a direct analysis of localization drift, with particular attention to re-identification scenarios in which the camera revisits a previously observed location. We hypothesize that a sufficiently drift-controlled online localization module preserves most of LocalizeSORT's identity-persistence gains while enabling real-time operation, closing the gap toward practical deployment in precision agriculture tasks — targeted spraying, selective harvesting, and yield estimation — that require tracking to run live rather than after the fact.


---

# Repo guide

Start here if you are a new contributor or a fresh AI agent session. Also read [CLAUDE.md](CLAUDE.md) for working rules.

## Pipeline at a glance

```
I_t ─> Detector ─> D_t ─┬─> Tracker ─> T (running tracks)
                        │      ^
                        └─> Localization ─> P_t (map-frame poses)
```

[localizesort/main.py](localizesort/main.py) wires the three modules. `LocalizeSORT.step(frame_idx, frame, detections=None)` advances one frame. Passing `detections` bypasses the Detector, which is how ground-truth boxes are fed in. `LocalizeSORT.run(frames, detections)` loops `step` over a sequence.

## Layout

| Path | What it is |
|---|---|
| [localizesort/](localizesort/) | The package (installed editable via [pyproject.toml](pyproject.toml)). |
| [localizesort/main.py](localizesort/main.py) | `LocalizeSORT`: loads the config, builds the modules, and runs the per-frame loop. |
| [localizesort/detector.py](localizesort/detector.py) | `Detector`: Ultralytics YOLO wrapper. RGB frame in, `list[Detection]` out. |
| [localizesort/localization.py](localizesort/localization.py) | `Localization`: online pose estimator. **Stub**: returns no poses until the CUT3R-based module lands. |
| [localizesort/tracker.py](localizesort/tracker.py) | `Tracker`: wraps boxmot's `StrongSort`. Ignores poses for now (appearance and image-space motion only). |
| [localizesort/schemas.py](localizesort/schemas.py) | Dataclasses passed between modules: `Detection`, `ObjectPose`, `Track`. |
| [localizesort/data.py](localizesort/data.py) | Dataset access driven by [config/datasets.yaml](config/datasets.yaml). `sequence(dataset, split, name)` gives frames, GT detections and the recon path. |
| [localizesort/run.py](localizesort/run.py) | The `localizesort` command: runs the pipeline over a split, with `--eval` to score it and `--viz` to watch it. |
| [localizesort/viz.py](localizesort/viz.py) | Track overlay rendering and the imshow window. Black/red match status, or per-id colours under `--viz-ids`. |
| [localizesort/runs.py](localizesort/runs.py) | Run directories: `runs/<stamp>_<module>_<tag>/` with `manifest.json`, `metrics.json` and `artifacts/`. |
| [localizesort/eval/](localizesort/eval/) | Evaluation. Reads run artifacts from disk; never imports the pipeline, so any run can be re-scored later. |
| [config/localizesort.yaml](config/localizesort.yaml) | Pipeline config. Top-level keys (`detector`, `localization`, `tracker`) become each module's constructor kwargs. `tracker.strongsort` maps to boxmot's `StrongSortConfig` and `tracker.reid` to `ReIDConfig`. |
| [lab/run_sequence.ipynb](lab/run_sequence.ipynb) | Scratch notebook: runs the pipeline on a LettuceMOTS sequence with GT detections and visualises the tracks. Outputs go to `lab/outputs/` (git-ignored). |
| [scripts/data/](scripts/data/) | Dataset adapters that build the canonical `mot/` view from a dataset's native annotation format. |
| [tests/](tests/) | pytest suite. |
| [docs/problem.md](docs/problem.md) | Problem statement, constraints, and evaluation protocol. Source of truth for what the project must do. |
| [docs/modules.yaml](docs/modules.yaml), [docs/schema.yaml](docs/schema.yaml) | Design specs cited by module docstrings. Agents are told to ignore them (see CLAUDE.md). |
| `docs/LocalizeSORTv2_Project_Tracker.xlsx` | Project task tracker (phases, task IDs `T01`...). Read it only when asked (see CLAUDE.md). |

## External dependencies (not tracked in this repo)

These are git-ignored. Two are symlinks to sibling directories under `~/IISc/`.

| Path | What it is |
|---|---|
| `boxmot/` | Vendored clone of boxmot, installed editable. Provides `StrongSort` and the ReID models. It has its own git history and docs. |
| `CUT3R/` | Symlink to `../CUT3R`. The feed-forward online 3D reconstruction backbone that will power `Localization`. |
| `models/` | Local model weights (git-ignored, as are `*.pt`, `*.pth` and similar). |

Sibling projects under `~/IISc/` that are useful references: `LocalizeSORT/` (the original offline pipeline and its results), `TrackEval/`.

## Data

[data/](data/) is the dataset root — git-ignored apart from [data/README.md](data/README.md), which documents the layout and how to recreate it on a new machine. [config/datasets.yaml](config/datasets.yaml) is the registry of datasets, splits and path templates. The root is `$LOCALIZESORT_DATA` if set, else `./data`.

Every dataset has the same shape, so eval code never branches on which one it is running against:

```
data/<dataset>/
  raw                      symlink -> the original download; never written to
  mot/<split>/<seq>/       canonical MOT view: img1/, gt/gt.txt, seqinfo.ini
  recon/<seq>/             offline SfM reconstruction (RLoc-style pseudo-GT)
```

For LettuceMOTS: `raw` points at the upstream download, all 12 sequences have GT, and only 0011 has a reconstruction. The `mot/` view is generated by [scripts/data/lettucemots_to_mot.py](scripts/data/lettucemots_to_mot.py) from the dataset's MOTS mask annotations. Generated GT is standard MOT (`frame, track_id, x, y, w, h, conf, class, visibility`, 1-based frames, top-left `xywh`), while `Detection.bbox` is `xyxy` — convert. The stock `yolov8n.pt` is COCO-trained and detects nothing in the crops, so use GT detections until crop-specific weights exist. Derived outputs belong in `runs/<run_id>/artifacts/`, not here.

## Evaluation

The protocol is in [docs/problem.md](docs/problem.md). Metrics come from **TrackEval**, installed as `pip install --no-deps trackeval==1.3.0` — `--no-deps` because its metadata demands numpy ≥ 2.3 while this env pins 1.26 for torch and CUT3R; the code itself is fine on 1.26. Do not use the clone at `~/IISc/TrackEval`, which is upstream master and still uses `np.float`.

No metric is implemented by hand. `CLEAR` gives MOTA and IDSW, `HOTA` gives HOTA and HOTA(0), `Identity` gives IDF1, `Count` gives predicted vs GT identity counts. MOTA at IoU 0.25 is a config value, not code, so both thresholds are reported: fields at the primary threshold (0.5) keep their plain names and others are suffixed, e.g. `MOTA@0.25`.

### Workflow

One command runs the pipeline and, with `--eval`, scores it in place:

```bash
localizesort --split test --detections gt --tag baseline --eval
```

That writes `runs/<stamp>_tracker_<tag>/` containing `manifest.json` (git state, both configs, sequences, timing), `artifacts/<seq>.txt` (MOT-format output) and, with `--eval`, `metrics.json`.

Running is minutes; scoring is seconds. So a run can be re-scored without repeating it — after changing a threshold, or fixing a metric:

```bash
python -m localizesort.eval --run runs/<run_id>            # re-score a run
python -m localizesort.eval --tracker-dir path/to/results  # score someone else's output
```

`--no-save` keeps nothing — output goes to a temporary directory, metrics print to the terminal, and it is all discarded on exit. Use it with `--eval` for a quick number, or alone to just watch a sequence run.

`--detections gt` uses ground-truth boxes, which isolates association quality; `detector` uses the pipeline's YOLO, which currently finds no lettuce. The choice lands in the manifest because it changes every number.

### Watching a run

```bash
localizesort --seqs 0011 --viz --viz-compare --no-save   # GT | prediction, side by side
localizesort --seqs 0011 --viz --viz-ids --no-save       # colour by track id
localizesort --seqs 0011 --viz-save overlay.mp4          # no display needed
```

Boxes are sized to match LocalizeSORT's own comparison figures (`FONT_SCALE=1.0`, `BOX_THICKNESS=3`) so overlays from the two projects read side by side.

**Colour means match status by default** — **black** where the prediction corresponds to a ground-truth object, **red** where it does not. Matching is one-to-one Hungarian on IoU ≥ 0.5, the same rule CLEAR-MOT uses, so a red box is one the metrics also count as a false positive.

**`--viz-ids`** colours by track id instead. An identity keeps its colour as long as the tracker keeps calling it the same thing, so an ID switch shows up as a box changing colour. Under `--detections gt` this is the mode you want: every prediction *is* a ground-truth box, so match colouring draws everything black and tells you nothing. The CLI says so when you hit that combination.

**`--viz-compare`** puts ground truth and prediction in titled panels side by side, as `visualize_comparison.py` does. The id labels line up per object, so drift is visible directly — GT `id=13` sitting opposite predicted `id=14`.

`--viz-height` sets the window height (default 1000); the window sizes itself from the first frame's aspect ratio rather than opening at OpenCV's arbitrary default.

`--viz` fails up front with a clear message if no display is usable, rather than running blind. Opening a window here also emits ~130 `QObject::moveToThread` warnings from a Qt plugin conflict elsewhere in the env; [viz.py](localizesort/viz.py) silences them with `QT_LOGGING_RULES=default.warning=false`, which is the category they come through.

A run can cover fewer frames than the sequence has — via `--limit N`, or by quitting the window early. The ground truth is then cut to exactly the frames that ran, with a note, so the numbers stay self-consistent instead of scoring every unrun frame as a miss.

The protocol lives in [config/eval.yaml](config/eval.yaml) — thresholds, `min_gap`, `grace`, `window`, `do_preproc` — separate from invocation, so changing what is measured is a visible commit rather than a flag someone passes differently. Dataset and split stay CLI arguments.

| File | What |
|---|---|
| [eval/tracking.py](localizesort/eval/tracking.py) | Stages files into TrackEval's layout and flattens its output. `preprocess()` exposes its parsed, IoU-matched data for reuse. |
| [eval/revisit.py](localizesort/eval/revisit.py) | Revisit-restricted identity metrics — the one part neither TrackEval nor boxmot provides. |
| [eval/mot_io.py](localizesort/eval/mot_io.py) | MOT-format reading, writing and frame filtering. |
| [eval/protocol.py](localizesort/eval/protocol.py) | Loads [config/eval.yaml](config/eval.yaml). |
| [eval/\_\_main\_\_.py](localizesort/eval/__main__.py) | The re-scoring CLI. |

**Revisit metrics** answer the project's actual question: when the camera returns to a stationary object after a gap, does its identity survive? Each event is classified `preserved`, `switch` or `unmatched`. The matching rule mirrors CLEAR's exactly, so these counts are a decomposition of the `IDSW` CLEAR reports rather than an independent measurement — verified: at `min_gap=0, grace=1` the switch count equals CLEAR's IDSW exactly.

Two parameters matter. `min_gap` (default 10) is how long an identity must be absent to count as a revisit. `grace` (default 5) is how long the tracker has to re-acquire the object, and it is not optional: StrongSORT confirms a track only after `n_init=3` hits, so scoring the reappearance frame alone reports every revisit as `unmatched` and hides whether identity was preserved.

### Baseline

StrongSORT with GT detections on the test split (`--tag strongsort-gtdets`):

| seq | MOTA | HOTA | IDF1 | IDSW | IDs | GT_IDs |
|---|---|---|---|---|---|---|
| 0003 | 86.77 | 56.89 | 45.94 | 120 | 116 | 40 |
| 0007 | 92.85 | 75.36 | 62.47 | 24 | 60 | 37 |
| 0011 | 92.97 | 76.33 | 64.03 | 23 | 60 | 37 |
| **combined** | **90.82** | **69.97** | **57.41** | **167** | 236 | 114 |

Of 70 revisit events, **2 preserved identity — 2.9%**. Returning objects get fresh ids: 236 predicted identities for 114 real ones. Detection is near-perfect and association is what fails, which is exactly the gap LocalizeSORTv2's world-coordinate cue is meant to close. 2.9% is the number to beat.

Two independent checks that the stack is sound. Re-scoring the existing external StrongSORT result for 0011 reproduces `eval_results/0011_metrics.json` field-for-field, worst absolute difference 0.00e+00. And this run lands within ~1 point of the separately-measured boxmot v22 baseline (HOTA 68.94, MOTA 90.73, IDF1 57.62, IDSW 166), despite a different detection source and a different metrics path.

`MOTA@0.25` equals `MOTA` on every sequence here, and will until a real detector is in the loop: GT detections match GT boxes exactly, so every true positive has IoU ≈ 1 and lowering the threshold changes nothing.

Note the `restricted` block in the output measures local consistency within revisit windows, *not* persistence across the gap — IDF1 re-solves its identity assignment on whatever subset it is given, so fresh ids still score well there (97% on the baseline). The event counts are the metric that answers the question.

## Conventions

- **Environment:** conda env `nerfly`.
- **Frames are RGB** `np.ndarray` (H, W, 3, uint8) throughout the pipeline. Detector and Tracker flip to BGR internally where Ultralytics and boxmot expect it.
- **Causality:** everything at frame `t` may use frames `1..t` only (see [docs/problem.md](docs/problem.md)).
- **Bboxes** are `xyxy` in image coordinates. Poses are `p_t^i in R^3` in a single map frame `M` with a 3x3 covariance.

## Running

```bash
conda activate nerfly
pip install -e .            # package (boxmot is installed editable separately)
pip install --no-deps trackeval==1.3.0 tabulate   # eval engine; --no-deps keeps numpy at 1.26

localizesort --split test --detections gt --tag baseline --eval   # run + score
jupyter lab lab/run_sequence.ipynb                                # scratch
```

## Status

| Module | State |
|---|---|
| Detector | Working (YOLOv8). Needs crop-specific weights. |
| Tracker | Working (StrongSORT via boxmot). Does not use poses yet. |
| Localization | Stub. CUT3R has been run offline on LettuceMOTS/0011, but it is not integrated. |
| Evaluation | Tracking metrics done (T06), protocol locked in [config/eval.yaml](config/eval.yaml). Localization metrics not started — only 0011 has a reconstruction to score against. |
| Data | All 12 LettuceMOTS sequences in the canonical MOT view. LFSD and mango not added. |
| Runs | Run directories with manifests and metrics (T07). |
| Tests | Paused by decision — see CLAUDE.md. `tests/` is stale; ignore it. |

Next: integrate CUT3R as the `Localization` module, then add the localization metrics (Umeyama alignment via `evo`, position error, revisit drift, NEES) from [docs/problem.md](docs/problem.md).
