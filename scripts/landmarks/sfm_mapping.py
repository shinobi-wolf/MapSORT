"""Extract per-detection landmark observations from a COLMAP model (RLoc pseudo-GT).

Each detection's bbox encloses a set of triangulated keypoints. Rejecting depth
outliers (~8% on 0011, almost all nearer than the object body: raised leaf tips
and triangulation noise) leaves a point set whose median and scatter give a
Gaussian over the object's map-frame position.

Writes `landmarks/<seq>/sfm.npz`, one row per line of `dets/<seq>.txt`, in that
order, with `has_landmark` marking the detections that enclose no usable point.

Usage:
    python scripts/landmarks/sfm_mapping.py [--dataset lettucemots] [--split test] [--seq 0011]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from mapsort.data import data_root, sequence


def read_keypoints(images_txt: Path) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Image name -> (xy[N,2], point3D_id[N], world-to-camera R, t), triangulated points only."""
    lines = [l for l in images_txt.read_text().splitlines() if not l.startswith("#")]
    out = {}
    for header, points in zip(lines[::2], lines[1::2]):
        fields = header.split()
        qw, qx, qy, qz = (float(v) for v in fields[1:5])
        t = np.array(fields[5:8], dtype=float)
        R = np.array([
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ])
        kp = np.array(points.split(), dtype=float).reshape(-1, 3)
        kp = kp[kp[:, 2] != -1]
        out[fields[-1]] = (kp[:, :2], kp[:, 2].astype(int), R, t)
    return out


def read_points3D(points_txt: Path) -> dict[int, np.ndarray]:
    """point3D_id -> XYZ."""
    out = {}
    for line in points_txt.read_text().splitlines():
        if line.startswith("#"):
            continue
        fields = line.split()
        out[int(fields[0])] = np.array(fields[1:4], dtype=float)
    return out


def read_excess_green(points_txt: Path) -> dict[int, float]:
    """point3D_id -> excess green, 2g - r - b on chromaticity. Soil is negative, leaf positive."""
    out = {}
    for line in points_txt.read_text().splitlines():
        if line.startswith("#"):
            continue
        fields = line.split()
        rgb = np.array(fields[4:7], dtype=float)
        total = rgb.sum()
        r, g, b = rgb / total if total else (0.0, 0.0, 0.0)
        out[int(fields[0])] = 2 * g - r - b
    return out


def read_focal(cameras_txt: Path) -> float:
    for line in cameras_txt.read_text().splitlines():
        if not line.startswith("#"):
            return float(line.split()[4])
    raise ValueError(f"no camera in {cameras_txt}")


def read_dets(dets_txt: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """dets rows in file order as (frame[N], det_idx[N], bbox[N,4] xyxy)."""
    frames, boxes = [], []
    for line in dets_txt.read_text().splitlines():
        if not line.strip():
            continue
        fields = line.split(",")
        x, y, w, h = (float(v) for v in fields[2:6])
        frames.append(int(float(fields[0])))
        boxes.append([x, y, x + w, y + h])

    frames = np.array(frames, dtype=np.int32)
    seen: dict[int, int] = {}
    det_idx = np.empty(len(frames), dtype=np.int32)
    for i, frame in enumerate(frames):
        det_idx[i] = seen.get(frame, 0)
        seen[frame] = det_idx[i] + 1
    return frames, det_idx, np.array(boxes, dtype=float).reshape(-1, 4)


def pick_keypoint(bbox: np.ndarray, xy: np.ndarray, ids: np.ndarray) -> tuple[int, np.ndarray] | None:
    """The bbox's keypoint nearest its centre, as (point3D_id, xy), or None if it holds none."""
    inside = np.all((xy >= bbox[:2]) & (xy <= bbox[2:]), axis=1)
    if not inside.any():
        return None
    cand, cand_ids = xy[inside], ids[inside]
    centre = (bbox[:2] + bbox[2:]) / 2
    best = np.argmin(((cand - centre) ** 2).sum(axis=1))
    return int(cand_ids[best]), cand[best]


def depth_inliers(depths: np.ndarray, k: float = 3.0) -> np.ndarray:
    """Points within k MADs of the median depth. Mostly matters for the scatter, not the median."""
    if len(depths) < 3:
        return np.ones(len(depths), dtype=bool)
    median = np.median(depths)
    mad = np.median(np.abs(depths - median))
    if mad == 0:
        return np.ones(len(depths), dtype=bool)
    return np.abs(depths - median) <= k * mad


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="lettucemots")
    ap.add_argument("--split", default="test")
    ap.add_argument("--seq", default="0011")
    ap.add_argument("--mode", choices=("gaussian", "nearest"), default="gaussian",
                    help="gaussian: median of inlier points; nearest: LocalizeSORT's nearest-to-centre keypoint")
    ap.add_argument("--mad", type=float, default=3.0, help="depth inlier threshold, in MADs")
    ap.add_argument("--green", type=float, default=None,
                    help="keep only points with excess green above this (try 0.02); omit to keep every colour")
    ap.add_argument("--model", type=Path, help="COLMAP text model dir (default: recon/<seq>/model_txt)")
    ap.add_argument("--out", type=Path, help="default: landmarks/<seq>/sfm.npz")
    args = ap.parse_args()

    seq = sequence(args.dataset, args.split, args.seq)
    root = data_root() / args.dataset
    model = args.model or root / "recon" / args.seq / "model_txt"
    out = args.out or root / "landmarks" / args.seq / "sfm.npz"

    keypoints = read_keypoints(model / "images.txt")
    points3D = read_points3D(model / "points3D.txt")
    focal = read_focal(model / "cameras.txt")
    excess_green = read_excess_green(model / "points3D.txt") if args.green is not None else {}
    frames, det_idx, boxes = read_dets(seq.dets_path or root / "dets" / f"{args.seq}.txt")
    # COLMAP was run on the MOT view, so its NAMEs are that sequence's frames.
    frame_image = {i + 1: p.name for i, p in enumerate(seq.image_paths())}

    n = len(frames)
    has = np.zeros(n, dtype=bool)
    position = np.full((n, 3), np.nan, dtype=np.float32)
    covariance = np.full((n, 3, 3), np.nan, dtype=np.float32)
    extent = np.full((n, 3, 3), np.nan, dtype=np.float32)
    n_inside = np.zeros(n, dtype=np.int32)
    n_kept = np.zeros(n, dtype=np.int32)

    for i in range(n):
        entry = keypoints.get(frame_image[frames[i]])
        if entry is None:
            continue
        xy, ids, R, t = entry
        bbox = boxes[i]

        inside = np.all((xy >= bbox[:2]) & (xy <= bbox[2:]), axis=1)
        n_inside[i] = inside.sum()
        if not inside.any():
            continue

        point_ids = ids[inside]
        # Colour first, so the depth statistics below see plant points only.
        if args.green is not None:
            point_ids = point_ids[
                np.array([excess_green[j] for j in point_ids]) > args.green
            ]
            if not len(point_ids):
                continue

        pts = np.array([points3D[j] for j in point_ids])
        keep = depth_inliers((pts @ R.T + t)[:, 2], args.mad)
        pts = pts[keep]
        n_kept[i] = len(pts)
        if not len(pts):
            continue

        if args.mode == "nearest":
            hit = pick_keypoint(bbox, xy, ids)
            position[i] = points3D[hit[0]]
        else:
            position[i] = np.median(pts, axis=0)

        # Isotropic floor keeps the spread usable when too few points survive to
        # estimate one: the object's apparent size at its own depth.
        depth = float(np.median((pts @ R.T + t)[:, 2]))
        radius = 0.25 * np.hypot(*(bbox[2:] - bbox[:2])) * depth / focal
        spread = np.cov(pts.T) if len(pts) >= 3 else np.zeros((3, 3))
        spread = spread + radius**2 * np.eye(3)

        extent[i] = spread
        covariance[i] = spread / len(pts)
        has[i] = True

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        has_landmark=has,
        position=position,
        covariance=covariance,
        extent=extent,
        frame=frames,
        det_idx=det_idx,
        n_inside=n_inside,
        n_kept=n_kept,
        mad=np.float32(args.mad),  # recorded so readers need not assume the default
        green=np.float32(np.nan if args.green is None else args.green),
    )
    registered = len({int(f) for f in frames if frame_image[f] in keypoints})
    print(
        f"{args.seq}  landmarks={has.sum()}/{n} dets  "
        f"frames={registered}/{len(set(frames.tolist()))} registered  "
        f"median kept={np.median(n_kept[has]):.0f}  -> {out}"
    )


if __name__ == "__main__":
    main()
