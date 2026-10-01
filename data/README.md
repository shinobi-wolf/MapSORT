# Data

This directory is the project's dataset root. Everything in it is git-ignored
except this file — the contents are symlinks into machine-specific download
locations, plus derived views built from them.

[config/datasets.yaml](../config/datasets.yaml) lists each dataset's splits and
which sequences have dets, landmarks and recon. Code resolves the root from
`$MAPSORT_DATA`, falling back to `./data`.

## Layout

Every dataset has exactly these directories:

```
data/<dataset>/
  raw                      symlink -> the original download; never written to
  mot/<split>/<seq>/       canonical MOT view: img1/, gt/gt.txt, seqinfo.ini
  dets/<seq>.txt           detections, MOT det format: frame,-1,x,y,w,h,conf,class
  landmarks/<seq>/<backend>.npz  per-detection landmark observations
  recon/<seq>/             offline SfM reconstruction (RLoc-style pseudo-GT)
```

`landmarks/` is per sequence and then per mapping backend — `sfm.npz` for the
offline COLMAP baseline, later `cut3r.npz` for the online one — so the two are
directly comparable on the same detections.

### Landmark format

A landmark observation is a **Gaussian over the object's position in the map
frame**, not a single point. Each `.npz` holds parallel arrays, one row per line
of `dets/<seq>.txt`, **in that order** — so row `i` is that file's line `i`, and
alignment is a length check rather than a key lookup:

| key | shape | what |
|---|---|---|
| `has_landmark` | `bool[N]` | false where the detection yielded no usable point |
| `position` | `f32[N,3]` | the Gaussian's mean; `nan` where absent |
| `covariance` | `f32[N,3,3]` | uncertainty **about the centre** — shrinks with more points |
| `extent` | `f32[N,3,3]` | the object's **physical spread** — does not shrink |
| `frame`, `det_idx` | `i32[N]` | 1-based frame, 0-based index within it |
| `n_inside`, `n_kept` | `i32[N]` | keypoints in the bbox, and after depth filtering |
| `mad` | `f32` | the depth-filter threshold these rows were built with; a huge value means none was applied |
| `green` | `f32` | the excess-green threshold these rows were built with; `nan` means no colour filter |

The depth filter drops points more than `mad` MADs from the median camera-frame
depth, and **is off by default on these rows** (`mad` is huge). At the old
default of 3 it discarded ~8% of in-box points on 0011, but those were mostly
the plant, not noise: 72% sat within two plant-widths of the centre, and three
quarters were only 3-5 MADs out — a tail being sliced, not an outlier
population. A box is a cone, so the camera sees a solid plant's near-facing
surface and the depths are skewed toward the lens (83% of boxes); a symmetric
cut then shaves that near face off. Measured, it shrank the spread *along* the
viewing axis to 82% while leaving the spread *across* it at 100%.

`--green` is the better-targeted filter: COLMAP gives every 3D point a colour,
and excess green (`2g - r - b` on chromaticity) separates leaf from soil cleanly
-- soil sits at a median of -0.005 against +0.048 for in-box points, so a cut at
0.02 discards 91% of soil. Unlike the depth filter it is isotropic, so it removes
soil showing between the leaves without shaving the plant's near face.

Turning it off costs nothing in `position` (the median is robust) but raises
`n_kept` ~13%, which *tightens* `covariance = extent / n_kept` and so narrows
the tracker's Mahalanobis gate. That divisor, not the filter, is the real
problem: it treats hundreds of keypoints from one reconstruction as independent
measurements of the centre, claiming ~0.003 when the same plant's estimated
position moves ~0.29 between camera passes.

`covariance` and `extent` are deliberately separate and differ by roughly 16x on
0011 (centre sd ~0.025 against extent sd ~0.39). Conflating them is the easy
mistake: fusing *extent* across observations would claim sub-plant precision,
because only centre uncertainty shrinks as `extent / n_kept`.

Detections that enclose no usable point carry `has_landmark = False` rather than
a sentinel coordinate — load-bearing, since those degrade to appearance-only
association downstream. `position` is `nan` there, never zero, which is itself a
plausible coordinate.

`raw` is a pointer, so the download stays wherever it already lives and is
shared with other projects untouched. The rest are ours. `dets/`, `landmarks/` and
`recon/` are flat by sequence; the split comes from `splits` in the yaml.

Derived outputs — predictions, visualisations, point clouds — do **not** belong
here. They go in `runs/<run_id>/artifacts/`.

## Setting up on a new machine

Create the `raw` symlink for each dataset:

```bash
mkdir -p data/lettucemots/{dets,landmarks,recon}
ln -s /path/to/LettuceMOTS data/lettucemots/raw
```

Then build the derived views:

```bash
python scripts/data/lettucemots_to_mot.py    # writes mot/{train,test}/<seq>/
ln -s ../raw/colmap/0011 data/lettucemots/recon/0011
python scripts/data/gt_to_dets.py                # writes dets/<seq>.txt from mot GT
python scripts/landmarks/sfm_mapping.py --seq 0011  # writes landmarks/<seq>/sfm.npz from recon + dets
```

The `recon/` symlink is relative, so the tree survives being moved as long as
`raw` still resolves. `mot/` is regenerable at any time — delete it and re-run.

## LettuceMOTS

Source: <https://github.com/NanH5837/LettuceMOTS> (dataset on Google Drive).

The download is not a clean dataset — it is the upstream code repo with the data
inside it, and it has since accumulated derived outputs. Only these subpaths are
genuine source data:

| Path under `raw/` | What |
|---|---|
| `images/<seq>/` | Frames, `%06d.png`, **0-based**. |
| `instances/<split>/<seq>.txt` | MOTS masks — the actual ground truth. |
| `labels/<seq>/` | YOLO detection labels, upstream train sequences only. |

Ignore everything else in there: `segment/`, `tracking/`, `models/`, `utils/`,
`export.py` are upstream code, and `pred_viz/`, `gt_viz/`, `comparison_viz/`,
`eval_results/` are another project's outputs. `raw/mot/` is an upstream partial
conversion covering three sequences; we generate our own, so don't read it.

**Ground truth exists for all 12 sequences.** It ships as MOTS instance files,
`frame id class height width RLE`, with 0-based frames and ids encoded as
`class * 1000 + n`. `scripts/data/lettucemots_to_mot.py` decodes the masks to
tight bounding boxes and writes the MOT view — its output is byte-identical to
the three sequences converted upstream, which `--verify` checks.

- **Splits.** `mot/test/` holds the upstream held-out sequences — 0003, 0007,
  0011, the long ones at 223–245 frames, and what LocalizeSORT reported numbers
  on. That is the headline eval set. `mot/train/` holds the other nine (45–117
  frames); we never train on them, so they serve as a development set for any
  threshold that must not be fitted on the test set.
- **GT format** (generated): `frame, track_id, x, y, w, h, conf, class,
  visibility`, 1-based frames, top-left `xywh`. `Detection.bbox` is `xyxy`, so
  convert.
- **Reconstruction:** only 0011 has one (COLMAP, under `raw/colmap/0011`,
  exposed as `recon/0011`), so the mapping sub-task is limited to that one
  sequence regardless of how many have tracking GT. `landmarks/0011/sfm.npz`
  covers 3032/3073 detections over all 223 registered frames, with a median of
  240 keypoints behind each landmark; the remaining 41 enclose no triangulated
  keypoint. Note the COLMAP scale is arbitrary, so distances in this file are in
  reconstruction units, not metres.
- **Detector weights:** stock `yolov8n.pt` is COCO-trained and detects no
  lettuce. Use GT detections until crop-specific weights exist. Any detector
  fine-tuned on LettuceMOTS has seen `mot/train/`, so detector-in-the-loop
  numbers on those sequences are contaminated.

## Planned

`lfsd/` and a mango orchard dataset, added with the same layout via an adapter
that writes the canonical `mot/` view.
