"""Online camera resection: the frame's 3x4 projection, fitted to tracked landmarks and their boxes."""

from __future__ import annotations

import numpy as np


def _normalize(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Hartley normalisation: homogeneous points and the similarity that produced them."""
    c = pts.mean(0)
    s = np.sqrt(pts.shape[1]) / max(np.mean(np.linalg.norm(pts - c, axis=1)), 1e-12)
    T = np.eye(pts.shape[1] + 1)
    T[:-1, :-1] *= s
    T[:-1, -1] = -s * c
    return np.c_[pts, np.ones(len(pts))] @ T.T, T


def dlt(X: np.ndarray, x: np.ndarray) -> np.ndarray:
    """3x4 projection from >= 6 3D-2D correspondences (direct linear transform)."""
    Xn, TX = _normalize(X)
    xn, Tx = _normalize(x)
    u, v = xn[:, :1] / xn[:, 2:], xn[:, 1:2] / xn[:, 2:]
    z = np.zeros_like(Xn)
    A = np.vstack([np.hstack([z, -Xn, v * Xn]), np.hstack([Xn, z, -u * Xn])])
    return np.linalg.inv(Tx) @ np.linalg.svd(A)[2][-1].reshape(3, 4) @ TX


def homography(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """3x3 map from >= 4 2D-2D correspondences."""
    an, Ta = _normalize(a)
    bn, Tb = _normalize(b)
    u, v = bn[:, :1] / bn[:, 2:], bn[:, 1:2] / bn[:, 2:]
    z = np.zeros_like(an)
    A = np.vstack([np.hstack([z, -an, v * an]), np.hstack([an, z, -u * an])])
    return np.linalg.inv(Tb) @ np.linalg.svd(A)[2][-1].reshape(3, 3) @ Ta


def plane_frame(X: np.ndarray) -> np.ndarray:
    """3x4 map from homogeneous map points to homogeneous coordinates in their best-fit plane."""
    c = X.mean(0)
    B = np.linalg.svd(X - c, full_matrices=False)[2][:2].T
    F = np.zeros((3, 4))
    F[:2, :3], F[:2, 3], F[2, 3] = B.T, -B.T @ c, 1.0
    return F


def project(P: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Pixel coordinates of map points X [N, 3]; nan behind the camera's plane at infinity."""
    h = np.c_[X, np.ones(len(X))] @ P.T
    with np.errstate(divide="ignore", invalid="ignore"):
        return h[:, :2] / h[:, 2:3]


def errors(P: np.ndarray, X: np.ndarray, x: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Reprojection errors in units of `scale`; inf where a point does not project."""
    err = np.linalg.norm(project(P, X) - x, axis=1) / scale
    return np.nan_to_num(err, nan=np.inf)


def _fit(X: np.ndarray, x: np.ndarray, F: np.ndarray | None) -> np.ndarray:
    """A full camera or, given a plane frame F, a homography from that plane lifted to 3x4."""
    if F is None:
        return dlt(X, x)
    return homography(project(F, X), x) @ F


def _ransac(X, x, scale, rng, F, iters, inlier, min_inliers, prior, refits):
    """Best camera and inlier mask for one model; the winner is refit on its inliers while that helps."""
    k = 6 if F is None else 4
    best_P, best = None, None
    if prior is not None:
        best_P, best = prior, errors(prior, X, x, scale) < inlier
    for _ in range(iters):
        idx = rng.choice(len(X), k, replace=False)
        try:
            P = _fit(X[idx], x[idx], F)
        except np.linalg.LinAlgError:
            continue
        inl = errors(P, X, x, scale) < inlier
        if best is None or inl.sum() > best.sum():
            best_P, best = P, inl
    if best is None or best.sum() < min_inliers:
        return None, None
    for _ in range(refits):
        try:
            P = _fit(X[best], x[best], F)
        except np.linalg.LinAlgError:
            break
        inl = errors(P, X, x, scale) < inlier
        if inl.sum() < best.sum():
            break
        settled = np.array_equal(inl, best)
        best_P, best = P, inl
        if settled:
            break
    return best_P, best


def resect(
    X: np.ndarray,
    x: np.ndarray,
    scale: np.ndarray,
    rng: np.random.Generator,
    *,
    iters: int = 300,
    inlier: float = 0.5,
    min_inliers: int = 8,
    prior: np.ndarray | None = None,
    refits: int = 5,
    planar_margin: float = 0.1,
) -> np.ndarray | None:
    """RANSAC camera, errors in units of `scale` (box sizes); None when too few points agree.

    A full DLT camera competes with a homography from the points' best-fit plane, which stays well
    conditioned when the landmarks are near coplanar; the plane wins unless the full camera explains
    clearly more points. The previous frame's camera competes with the random hypotheses of both.
    """
    if len(X) < max(6, min_inliers):
        return None
    P3, in3 = _ransac(X, x, scale, rng, None, iters, inlier, min_inliers, prior, refits)
    P2, in2 = _ransac(X, x, scale, rng, plane_frame(X), iters, inlier, min_inliers, prior, refits)
    n3 = 0 if in3 is None else int(in3.sum())
    n2 = 0 if in2 is None else int(in2.sum())
    return P2 if P2 is not None and n2 >= (1.0 - planar_margin) * n3 else P3
