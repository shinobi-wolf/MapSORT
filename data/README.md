# Data

This directory is the project's dataset root. Everything in it is git-ignored
except this file — the contents are symlinks into machine-specific download
locations, plus derived views built from them.

[config/datasets.yaml](../config/datasets.yaml) lists each dataset's splits and
which sequences have dets, pose and recon. Code resolves the root from
`$LOCALIZESORT_DATA`, falling back to `./data`.

## Layout

Every dataset has exactly these directories:

```
data/<dataset>/
  raw                      symlink -> the original download; never written to
  mot/<split>/<seq>/       canonical MOT view: img1/, gt/gt.txt, seqinfo.ini
  dets/<seq>.txt           detections, MOT det format: frame,-1,x,y,w,h,conf,class
  pose/<seq>.txt           object world coordinates, one row per detection
  recon/<seq>/             offline SfM reconstruction (RLoc-style pseudo-GT)
```

`raw` is a pointer, so the download stays wherever it already lives and is
shared with other projects untouched. The rest are ours. `dets/`, `pose/` and
`recon/` are flat by sequence; the split comes from `splits` in the yaml.

Derived outputs — predictions, visualisations, point clouds — do **not** belong
here. They go in `runs/<run_id>/artifacts/`.

## Setting up on a new machine

Create the `raw` symlink for each dataset:

```bash
mkdir -p data/lettucemots/{dets,pose,recon}
ln -s /path/to/LettuceMOTS data/lettucemots/raw
```

Then build the derived views:

```bash
python scripts/data/lettucemots_to_mot.py    # writes mot/{train,test}/<seq>/
ln -s ../raw/colmap/0011 data/lettucemots/recon/0011
python scripts/data/gt_to_dets.py                # writes dets/<seq>.txt from mot GT
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
  frames); we never train on them, so they serve as a development set for tuning
  things like the WC gate threshold that must not be fitted on the test set.
- **GT format** (generated): `frame, track_id, x, y, w, h, conf, class,
  visibility`, 1-based frames, top-left `xywh`. `Detection.bbox` is `xyxy`, so
  convert.
- **Reconstruction:** only 0011 has one (COLMAP, under `raw/colmap/0011`,
  exposed as `recon/0011`), so the localization sub-task is limited to that one
  sequence regardless of how many have tracking GT.
- **Detector weights:** stock `yolov8n.pt` is COCO-trained and detects no
  lettuce. Use GT detections until crop-specific weights exist. Any detector
  fine-tuned on LettuceMOTS has seen `mot/train/`, so detector-in-the-loop
  numbers on those sequences are contaminated.

## Planned

`lfsd/` and a mango orchard dataset, added with the same layout via an adapter
that writes the canonical `mot/` view.
