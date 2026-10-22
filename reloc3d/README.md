# reloc3d: Santos et al.'s tracker, as a side exercise

**What.** A reimplementation of the multiple-orange tracker in Santos et al., *Multiple orange
detection and tracking with 3-D fruit relocalization and neural-net based yield regression in
commercial sweet orange orchards* (arXiv:2312.16724v1, Sec. 3.3; the MOrangeT paper). It works in three steps:

1. **Frame-to-frame matching.** Boxes in consecutive frames are linked by IoU with Hungarian assignment.
2. **Sphere fit.** Each track with at least 5 boxes gets a 3-D sphere, fitted by a RANSAC DLT that uses the
   dataset's COLMAP cameras.
3. **Relocalization.** A lost track is relocalized when its sphere reprojects onto a new box.
   Only tracks that have a sphere are counted.

**Why.** It is the closest prior work to MapSORT: monocular video of stationary fruit, with
identities recovered through a 3-D map. We built and evaluated it only as insurance, to know how
MapSORT compares if a reviewer asks. It is **not** a baseline in the paper. The paper cites it in
related work only, and none of its numbers, rows or figures come from here.

**Isolation.** Everything about it lives in this directory. Nothing outside imports it, and
`scripts/eval/eval.py`, `config/eval.yaml` and `docs/precision-ag-paper/results/` don't know it exists. It only
*reads* from the project: `mapsort.data` sequences, stored detections, `recon/<seq>/model_txt`
cameras, and the scorers in `mapsort.eval` and `scripts/eval/eval.py`.

## Layout

| Path | What |
| --- | --- |
| `tracker.py`, `geometry.py`, `cameras.py` | The tracker (`Reloc3D`, `Params`), its sphere/ray geometry, and the COLMAP camera loader. |
| `reproduce.py` | `python -m reloc3d`: the paper's Tables 3 and 4 (GT boxes kept with probability `rate`) on MOrangeT. |
| `paper_tables.csv` | The paper's Tables 3 and 4, for comparison. |
| `evaluate.py` | `python -m reloc3d.evaluate`: runs it on the stored detections of every test sequence and scores it with the paper's metrics. |
| `runs/` | Its run directories (only `manifest.json` and `metrics.json` are tracked). |
| `data/` | Git-ignored symlinks to MOrangeT V03 and V04 (see below). |

## Data

`reproduce.py` uses all 12 MOrangeT sequences, as the paper does. The project itself uses only
V01, V02 and V05–V12, which it reads from `data/moranget/raw/`. V03 and V04 are linked here
instead, so that no project script picks them up:

    ln -s ../../../data/moranget/V03/V03 reloc3d/data/V03   # from the project root, on SW
    ln -s ../../../data/moranget/V04/V04 reloc3d/data/V04

Their zips come from the MOrangeT release and need a resumable download (`curl -C -`), because the
server drops long transfers.

## Results so far

| Run | What it shows |
| --- | --- |
| `runs/20261022-025222_reloc3d_all12` | Paper reproduction, 12 sequences, GT boxes. With relocalization: HOTA 96.3 (paper 93.5), 1217 counted vs 1198 GT (paper 1170). Without: HOTA 73.9 (paper 43.4). |
| `runs/20261022-005612_reloc3d` | The same on the 10 sequences the project uses. |
| `runs/*_reloc3d_sweep-*` | Algorithm 1's open thresholds (`max_geom_error`, `inliers_ratio`, `min_boxes`, order, `keep_failed`). Only `min_boxes = 5` matters: it alone reproduces the paper's 60% and 40% rows. |
| `runs/eval-20261022-005918` | All four test sets on stored detections, default `Params` (made with the old `scripts/eval/eval.py` hook, now `evaluate.py`). Pooled HOTA: MOrangeT 96.3, GrapeMOTS 77.0, AppleMOTS 79.7, LettuceMOTS 90.3. MapSORT scores 97.9, 94.4, 94.7 and 95.1. |
| `runs/20261022-011000_reloc3d_threshold` | `max_geom_error` (px) per dataset. Unchanged on MOrangeT (96.3–96.4 HOTA). At 100 px instead of 10 px, scores rise on the others: LettuceMOTS 90.3 → 99.0, AppleMOTS 79.7 → 83.5, GrapeMOTS 77.0 → 79.2. |

In short, Reloc3D is close to MapSORT on MOrangeT, the dataset it was designed for, and well behind on
GrapeMOTS and AppleMOTS. On LettuceMOTS, with a loose threshold, it is ahead.
