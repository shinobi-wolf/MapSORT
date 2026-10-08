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
| `n_inside`, `n_kept` | `i32[N]` | keypoints in the bbox, and after filtering |
| `green` | `f32` | the excess-green threshold these rows were built with; `nan` means no colour filter |
| `min_track` | `i32` | points seen in fewer images than this were dropped; `0` means no track filter |

Two filters can trim the in-box points before the median is taken, and each
`.npz` records which were active.

`--green` keeps points whose excess green (`2g - r - b` on chromaticity) exceeds
a threshold. On LettuceMOTS it separates leaf from soil cleanly — soil sits at a
median of -0.005 against +0.048 for in-box points, so a cut at 0.02 discards 91%
of soil. It is useless on MOrangeT, for reasons set out in that section below.

`--min-track` keeps points COLMAP saw in at least N images. A point triangulated
from two views is mostly noise; one carried across many views is on something
real and persistent. This is the only filter measurably worth having on
MOrangeT: at `--min-track 5` the median per-fruit localisation error falls from
1.12 to 0.93 fruit radii (paired 1.09 -> 0.93 over the fruits both configurations
localise, better on 37/104, Wilcoxon p=0.023) and costs 2 of 106 fruits. It is
off on LettuceMOTS, which has no 3D ground truth to tune it against.

**A symmetric depth filter used to run here and has been removed.** It dropped
points more than `mad` MADs from the median camera-frame depth. On LettuceMOTS
the points it discarded were mostly the plant, not noise: a box is a cone, so
the camera sees a solid plant's near-facing surface and depths skew toward the
lens (83% of boxes), and a symmetric cut then shaves that near face off —
measured, it shrank the spread *along* the viewing axis to 82% while leaving the
spread *across* it at 100%. On MOrangeT it does nothing at all: `depth_inliers`
returned all-true below three points and the median box holds two, and end to
end `mad=3` against no filter was 1.12 -> 1.13 radii at p=0.667. A *one-sided*
near-side cut is a different proposition and did show a small real gain
(1.12 -> 1.09, p=0.016, no cost); it is not implemented.

`covariance` and `extent` are deliberately separate and differ by roughly 13x on
0011 (centre sd ~0.027 against extent sd ~0.35). Conflating them is the easy
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
python scripts/landmarks/sfm_mapping.py --seq 0011 --green 0.02  # landmarks/<seq>/sfm.npz
```

The `recon/` symlink is relative, so the tree survives being moved as long as
`raw` still resolves. `mot/` is regenerable at any time — delete it and re-run.

MOrangeT ships MOT-shaped already, so its adapter only normalises; its COLMAP
model ships as `.bin` and has to be converted to the `.txt` the mapping script
reads:

```bash
mkdir -p data/moranget/{dets,landmarks,recon}
ln -s /path/to/MOrangeT data/moranget/raw

python scripts/data/moranget_to_mot.py              # writes mot/test/<seq>/
ln -s ../../raw/V01/geom data/moranget/recon/V01/sparse
colmap model_converter --input_path data/moranget/recon/V01/sparse \
  --output_path data/moranget/recon/V01/model_txt --output_type TXT
python scripts/data/gt_to_dets.py --dataset moranget
python scripts/landmarks/sfm_mapping.py --dataset moranget --split test --seq V01 --min-track 5
```

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
  covers 3030/3073 detections over all 223 registered frames, with a median of
  162 keypoints behind each landmark; the remaining 43 enclose no keypoint that
  survives the green filter. It is built with `--green 0.02` and no track filter
  — the `.npz` records both, so check there rather than trusting this paragraph. Note the COLMAP scale is arbitrary, so distances in this file are in
  reconstruction units, not metres.
- **Detector weights:** stock `yolov8n.pt` is COCO-trained and detects no
  lettuce. Use GT detections until crop-specific weights exist. Any detector
  fine-tuned on LettuceMOTS has seen `mot/train/`, so detector-in-the-loop
  numbers on those sequences are contaminated.

## MOrangeT

Source: <https://github.com/AgtechFundecitrus/MOrangeT>. Orange trees scanned by
hand-held phone during the 2022 Fundecitrus harvest survey: 12 sequences, 1198
tracked fruits, portrait 1083x1944. **Only V01 is downloaded** (408 frames, 110
oranges); the rest go in `config/datasets.yaml` as they arrive.

This is the closest thing we have to a purpose-built MapSORT benchmark: every
sequence ships a COLMAP reconstruction *and* `oranges-3d.txt`, a per-fruit 3D
centre and radius. Unlike LettuceMOTS, the mapping sub-task is not limited to
one sequence, and there is real 3D ground truth to score landmarks against.

| Path under `raw/<seq>/` | What |
|---|---|
| `images/frame-%04d.jpg` | Frames, **1-based**, so `img1` is a plain directory symlink. |
| `gt/gt.txt` | MOT16 GT, visible boxes only — the evaluation GT. |
| `gt/gt.all.txt` | The same plus boxes >50% occluded (column 9 is visibility). |
| `geom/*.bin` | COLMAP model, all frames registered. |
| `oranges-3d.txt` | `x, y, z, radius` per fruit, in the COLMAP frame. |

- **Splits.** Upstream's seqmap calls all 12 `test`; we keep that, so there is
  no development split to tune thresholds on. If that bites, carve one out of
  the sequences still to be downloaded rather than out of V01.
- **`seqinfo.ini` is wrong upstream** — it claims 1080x1940, but the JPEGs and
  the GT boxes (max corner 1082x1943) are 1083x1944, as is the COLMAP camera.
  `moranget_to_mot.py` rewrites it from the actual frames.
- **GT vs dets.** `dets/V01.txt` comes from `gt.txt`, so the 6068 heavily
  occluded boxes in `gt.all.txt` are absent from both GT and detections. A real
  detector would fire on some of them.
- **No colour filter works on V01, and `--green` must stay off.** The fruit in
  this sequence is *unripe* — uniformly green, not orange (see any GT crop).
  Measured on image pixels, the fruit core and the background differ by almost
  nothing in colour: mean excess green **+0.215 vs +0.221**, mean RGB
  (149,170,109) against (114,125,87). The fruit is *lighter*, not differently
  hued.

  Labelling COLMAP points by the GT fruit spheres in `oranges-3d.txt` and asking
  which colour statistic separates fruit from the rest of its own bbox, the best
  single one is intensity at **AUC 0.67**; excess green inverts (fruit is *less*
  green than the foliage around it) at 0.65, and a logistic regression over
  seven colour features does no better than intensity alone. Compare the lettuce
  case, where a cut at 0.02 discards 91% of soil.

  Scored end to end against `oranges-3d.txt`, every candidate filter is neutral
  or harmful. An `exg < 0.15` cut looks like a win unpaired (median 1.05 vs 1.13
  radii) but that is survivorship: it drops 42% of detections and 16 fruits
  entirely, and on the 90 fruits both configurations localise it is *worse* —
  0.97 to 1.05 radii, better on only 30/90, Wilcoxon p=0.004. So MOrangeT
  landmarks are built with no colour filter at all.

  This is a statement about V01, not the dataset. Other sequences are different
  varieties and years, and ripe orange fruit would separate easily — re-run the
  check when one is downloaded.
- **Landmarks are thin.** `landmarks/V01/sfm.npz` is built with `--min-track 5`
  and covers 6037/7848 detections, but the median box still encloses only **2**
  surviving keypoints against LettuceMOTS' 162 — a 40px fruit is a far smaller
  target than a lettuce plant. `extent` from two points is close to degenerate,
  and the `covariance = extent / n_kept` divisor means even less here than it
  does there. Median extent sd is 0.083 against a median GT fruit radius of
  0.113, so the scale is at least plausible.
- **Mapping accuracy, scored against `oranges-3d.txt`.** Row `i` of that file is
  track id `i+1` — confirmed, since each fruit's distance to its own row matches
  its distance to the nearest row of any index. Fusing a track's observations by
  median gives **0.93 fruit radii** (p90 4.11) over 104 of 110 fruits, with 7
  beyond 5 radii. Error falls with views fused (Spearman -0.47): under 20
  observations it is 3.5 radii, at 20 or more 0.89. Note this is the *fused*
  error and it uses GT association, so it is an upper bound on what the tracker
  would see. `lab/landmark_overlay.ipynb` reproduces it.
- COLMAP scale is arbitrary here too; distances are reconstruction units.

## Planned

`lfsd/`, and the remaining eleven MOrangeT sequences.
