"""Per-frame 3x4 projections P_i from a COLMAP model (text or binary), pinhole part only as in the paper."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .geometry import camera_center

# COLMAP model id -> (name, number of params); single-focal models list f, cx, cy first, the rest fx, fy, cx, cy.
MODELS = {
    0: ("SIMPLE_PINHOLE", 3), 1: ("PINHOLE", 4), 2: ("SIMPLE_RADIAL", 4), 3: ("RADIAL", 5),
    4: ("OPENCV", 8), 5: ("OPENCV_FISHEYE", 8), 6: ("FULL_OPENCV", 12), 7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4), 9: ("RADIAL_FISHEYE", 5), 10: ("THIN_PRISM_FISHEYE", 12),
}
SINGLE_FOCAL = {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL", "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE"}


@dataclass(frozen=True)
class Camera:
    P: np.ndarray  # [3, 4]
    C: np.ndarray  # [3], projection centre
    focal: float  # px
    size: tuple[int, int]  # width, height the model was calibrated at


def _intrinsics(model: str, params: np.ndarray) -> np.ndarray:
    if model in SINGLE_FOCAL:
        f, cx, cy = params[:3]
        fx = fy = f
    else:
        fx, fy, cx, cy = params[:4]
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])


def _rotation(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ])


def _read_text(model: Path) -> tuple[dict, dict]:
    cams, images = {}, {}
    for line in (model / "cameras.txt").read_text().splitlines():
        if line and not line.startswith("#"):
            f = line.split()
            cams[int(f[0])] = (f[1], int(f[2]), int(f[3]), np.array(f[4:], dtype=float))
    lines = [l for l in (model / "images.txt").read_text().splitlines() if not l.startswith("#")]
    for header in lines[::2]:
        f = header.split()
        images[f[9]] = (np.array(f[1:5], dtype=float), np.array(f[5:8], dtype=float), int(f[8]))
    return cams, images


def _read_binary(model: Path) -> tuple[dict, dict]:
    cams, images = {}, {}
    with open(model / "cameras.bin", "rb") as fh:
        for _ in range(struct.unpack("<Q", fh.read(8))[0]):
            cam_id, model_id, w, h = struct.unpack("<iiQQ", fh.read(24))
            name, n = MODELS[model_id]
            cams[cam_id] = (name, w, h, np.array(struct.unpack(f"<{n}d", fh.read(8 * n))))
    with open(model / "images.bin", "rb") as fh:
        for _ in range(struct.unpack("<Q", fh.read(8))[0]):
            fields = struct.unpack("<i7di", fh.read(64))
            name = b""
            while (ch := fh.read(1)) != b"\x00":
                name += ch
            fh.seek(24 * struct.unpack("<Q", fh.read(8))[0], 1)
            images[name.decode()] = (np.array(fields[1:5]), np.array(fields[5:8]), fields[8])
    return cams, images


def load_cameras(model: Path, image_names: list[str]) -> list[Camera | None]:
    """One Camera per frame, frames named `image_names` in order; None where COLMAP registered no image."""
    model = Path(model)
    cams, images = _read_text(model) if (model / "images.txt").exists() else _read_binary(model)
    out: list[Camera | None] = []
    for name in image_names:
        if name not in images:
            out.append(None)
            continue
        q, t, cam_id = images[name]
        cam_model, w, h, params = cams[cam_id]
        K = _intrinsics(cam_model, params)
        P = K @ np.c_[_rotation(*q), t]
        out.append(Camera(P=P, C=camera_center(P), focal=float((K[0, 0] + K[1, 1]) / 2), size=(w, h)))
    return out
